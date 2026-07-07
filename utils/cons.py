import os
from typing import Optional, Dict, Set, Tuple, List
import numpy as np
import pandas as pd
import cvxpy as cp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil

def load_map(csv_path: str) -> Dict[int, float]:
    df = pd.read_csv(csv_path)
    df['Code'] = df['Code'].astype(int)
    return dict(zip(df['Code'], df['Value']))


class ConstraintPostProcessor:
    def __init__(
        self,
        num_regions: int,
        num_variables: int,
        var_constraints: Optional[Dict[int, List[int]]] = None,
        non_negative_vars: Optional[Set[int]] = None,
        solver: str = "OSQP",
        solver_opts: Optional[Dict] = None,
        verbose: bool = False,
    ):
        self.num_regions = int(num_regions)
        self.num_variables = int(num_variables)
        self.var_constraints = var_constraints or {
            0: [1, 2, 44],
            3: [4, 5, 47],
            8: [39, 40, 41, 43],
            9: [10, 38, 42, 46],
            28: [34, 35, 36, 37, 45],
            11: [12, 13, 14, 15, 16, 29, 30, 31, 32],
            17: [18, 19, 20, 21, 22, 23, 24, 25, 26]
        }
        self.non_negative_vars = non_negative_vars or {
            26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
            32, 31, 16, 15, 30, 29, 14, 13, 11, 38, 
            46, 42, 10, 9, 43, 41, 40, 39, 8, 37, 
            36, 45, 35, 34, 28
        }
        self.solver = solver
        self.solver_opts = solver_opts or {"eps_abs": 1e-6, "eps_rel": 1e-6}
        self.verbose = verbose

        # 优化变量和参数
        self.x = cp.Variable((self.num_regions, self.num_variables))
        self.x_pred_param = cp.Parameter((self.num_regions, self.num_variables))

        # 目标：最小化修正前后差异
        objective = cp.Minimize(cp.sum_squares(self.x - self.x_pred_param))
        constraints = self._build_constraints()
        self.problem = cp.Problem(objective, constraints)

        if self.verbose:
            print("CVXPY 优化问题已定义完成。")

    def _build_constraints(self):
        constraints = []
        # 非负约束
        for var_id in self.non_negative_vars:
            if var_id < self.num_variables:
                constraints.append(self.x[:, var_id] >= 0)
        # 变量求和约束
        for total_var, comp_vars in self.var_constraints.items():
            if total_var < self.num_variables and all(v < self.num_variables for v in comp_vars):
                constraints.append(self.x[:, total_var] == cp.sum([self.x[:, v] for v in comp_vars]))
        # 地区求和约束：第0个地区等于其它地区的总和
        if self.num_regions > 1:
            constraints.append(self.x[0, :] == cp.sum(self.x[1:, :], axis=0))
        return constraints

    @staticmethod
    def _frame_to_full_matrix_no_dup(frame_t: np.ndarray, num_regions: int, num_variables: int) -> np.ndarray:
        """
        将某个时刻的 (groups, 5) 数据转换为 (num_regions, num_variables) 的预测矩阵。
        同一时刻没有重复的 (region, variable) 组合。
        列为: [model, category, region, variable, value]
        """
        r_idx = frame_t[:, 2].astype(np.int64)
        v_idx = frame_t[:, 3].astype(np.int64)
        vals = frame_t[:, 4].astype(np.float64)
        full = np.zeros((num_regions, num_variables), dtype=np.float64)
        full[r_idx, v_idx] = vals
        return full

    def _solve_one_matrix(self, full_x_pred: np.ndarray, warm_start_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        self.x_pred_param.value = full_x_pred
        # warm start
        try:
            self.x.value = warm_start_matrix if (warm_start_matrix is not None and
                                                 warm_start_matrix.shape == (self.num_regions, self.num_variables)) else full_x_pred
        except Exception:
            pass

        try:
            self.problem.solve(
                solver=getattr(cp, self.solver),
                warm_start=True,
                verbose=self.verbose,
                **self.solver_opts
            )
        except Exception:
            return full_x_pred

        if self.problem.status in ("optimal", "optimal_inaccurate"):
            return np.asarray(self.x.value, dtype=np.float64)
        else:
            return full_x_pred

    def _process_single_sample_values(self, sample_data: np.ndarray) -> np.ndarray:
        """
        处理单个 sample 的数据，返回修正后的 value 列，shape = (groups, timesteps)
        """
        groups, n_timesteps, feat_dim = sample_data.shape
        assert feat_dim == 5
        corrected_values = np.empty((groups, n_timesteps), dtype=np.float64)

        prev_solution = None
        for t in range(n_timesteps):
            frame_t = sample_data[:, t, :]  # (groups, 5)
            # print(frame_t.shape)
            full_x_pred = self._frame_to_full_matrix_no_dup(frame_t, self.num_regions, self.num_variables)
            corrected_matrix = self._solve_one_matrix(full_x_pred, warm_start_matrix=prev_solution)
            prev_solution = corrected_matrix

            # 回写到 groups 维（仅 value 列）
            r_idx = frame_t[:, 2].astype(np.int64)
            v_idx = frame_t[:, 3].astype(np.int64)
            corrected_values[:, t] = corrected_matrix[r_idx, v_idx]

        return corrected_values.astype(sample_data.dtype, copy=False)


def _process_sample_from_path_values_only(
    pre_path: str,
    sample_idx: int,
    num_regions: int,
    num_variables: int,
    var_constraints: Dict[int, List[int]],
    non_negative_vars: Set[int],
    solver: str,
    solver_opts: Dict,
) -> Tuple[int, np.ndarray]:
    """
    进程池工作函数：从 memmap 读取一个 sample，返回 (sample_idx, corrected_values)
    corrected_values: shape = (groups, timesteps)，仅包含修正后的 value 列
    """
    pre_mem = np.load(pre_path, mmap_mode='r')
    sample = pre_mem[sample_idx]  # (groups, timesteps, 5)
    processor = ConstraintPostProcessor(
        num_regions=num_regions,
        num_variables=num_variables,
        var_constraints=var_constraints,
        non_negative_vars=non_negative_vars,
        solver=solver,
        solver_opts=solver_opts,
        verbose=False,
    )
    corrected_values = processor._process_single_sample_values(sample)
    return sample_idx, corrected_values


def process_memmap_in_batches(
    pre_path: str,
    out_path: str,
    num_regions: Optional[int] = None,
    num_variables: Optional[int] = None,
    batch_size: int = 8,
    parallel_workers: Optional[int] = None,
    solver: str = "OSQP",
    solver_opts: Optional[Dict] = None,
    var_constraints: Optional[Dict[int, List[int]]] = None,
    non_negative_vars: Optional[Set[int]] = None,
    show_progress: bool = True,
) -> str:
    """
    对 memmap 格式的预测数组按 batch 进行后处理，并写入 out_path（memmap 文件）。
    仅修改 value 列（索引 4），其他列原样拷贝。
    """
    solver_opts = solver_opts or {"eps_abs": 1e-6, "eps_rel": 1e-6}

    # 打开输入 memmap

    pre_mem = np.load(pre_path, mmap_mode='r')
    n_samples, n_groups, n_timesteps, feat_dim = pre_mem.shape
    assert feat_dim == 5

    # 推断区域与变量数量
    if num_regions is None:
        num_regions = int(pre_mem[:, :, :, 2].max()) + 1
    if num_variables is None:
        num_variables = int(pre_mem[:, :, :, 3].max()) + 1

    if show_progress:
        print(f"数据信息: {n_samples} 个样本, {n_groups} 个组, {n_timesteps} 个时间步, "
              f"{num_regions} 个地区, {num_variables} 个变量")

    # 创建输出 memmap
    if os.path.exists(out_path):
        out_mem = np.load(out_path, mmap_mode='r+')
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
        shutil.copyfile(pre_path, out_path) # OS 级别拷贝（通常更快，可能用到 sendfile/零拷贝）
        out_mem = np.load(out_path, mmap_mode='r+') # 直接在拷贝后的文件上原地修改

    # 顺序或并行处理
    if not parallel_workers or parallel_workers <= 1:
        if show_progress:
            pbar = tqdm(range(n_samples), desc="Processing samples")
        else:
            pbar = range(n_samples)

        # 单实例复用（减少构建 QP 的开销）
        processor = ConstraintPostProcessor(
            num_regions=num_regions,
            num_variables=num_variables,
            var_constraints=var_constraints,
            non_negative_vars=non_negative_vars,
            solver=solver,
            solver_opts=solver_opts,
            verbose=False,
        )
        for s in pbar:
            corrected_values = processor._process_single_sample_values(pre_mem[s])
            out_mem[s, :, :, 4] = corrected_values
    else:
        # 并行：进程内自行构建实例；主进程按 batch 投递，收集后只写 value 列
        indices = list(range(n_samples))
        if show_progress:
            outer_pbar = tqdm(range(0, n_samples, batch_size), desc="Processing batches")
        else:
            outer_pbar = range(0, n_samples, batch_size)

        var_constraints = var_constraints or {
            0: [1, 2, 44],
            3: [4, 5, 47],
            8: [39, 40, 41, 43],
            9: [10, 38, 42, 46],
            28: [34, 35, 36, 37, 45],
            11: [12, 13, 14, 15, 16, 29, 30, 31, 32],
            17: [18, 19, 20, 21, 22, 23, 24, 25, 26]
        }

        non_negative_vars = non_negative_vars or {
            26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
            32, 31, 16, 15, 30, 29, 14, 13, 11, 38, 
            46, 42, 10, 9, 43, 41, 40, 39, 8, 37, 
            36, 45, 35, 34, 28
        }

        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            for start in outer_pbar:
                end = min(start + batch_size, n_samples)
                idx_batch = indices[start:end]
                futures = [
                    executor.submit(
                        _process_sample_from_path_values_only,
                        pre_path,
                        idx,
                        num_regions,
                        num_variables,
                        var_constraints,
                        non_negative_vars,
                        solver,
                        solver_opts,
                    )
                    for idx in idx_batch
                ]
                for fut in as_completed(futures):
                    idx, corrected_values = fut.result()
                    out_mem[idx, :, :, 4] = corrected_values

    if show_progress:
        print("修正完成 ✅ 写入文件:", out_path)
    return out_path