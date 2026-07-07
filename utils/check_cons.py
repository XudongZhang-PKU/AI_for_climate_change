
import argparse
import os
from typing import Dict, Optional, List, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============== 约束定义 ==============
VAR_CONSTRAINTS: Dict[int, List[int]] = {
            # CCS 总量 = 生物质 + 化石 + 工业过程
            0:  [1, 2, 45],

            # CO2 总排放 = AFOLU + 能源与工业过程 + 其他
            # （保持原口径，不纳入供需明细/分部门拆分）
            3:  [4, 5, 33],

            # 居民/商业终端能源 = 电 + 气 + 液体 + 固体
            8:  [40, 41, 42, 44],

            # 交通终端能源 = 电 + 液体(油) + 气体 + 氢
            9:  [10, 39, 43, 47],

            # 工业终端能源 = 电 + 气 + 液体 + 固体 + 氢
            28: [35, 36, 37, 38, 46],

            # 一次能源总量 = 生物质 + 煤 + 气 + 核 + 油 + 地热 + 水电 + 太阳能 + 风能
            11: [12, 13, 14, 15, 16, 29, 30, 31, 32],

            # 二次能源（电力）= 各电源分项
            17: [18, 19, 20, 21, 22, 23, 24, 25, 26],
        }
NON_NEGATIVE_VARS: Set[int] = {
            0, 1, 2, 45,
            # 二次能源（电力）及子项
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26,

            # 一次能源及子项
            11, 12, 13, 14, 15, 16, 29, 30, 31, 32,

            # 终端能源：居民/商业、交通、工业 及其子项
            8, 40, 41, 42, 44,
            9, 10, 39, 43, 47,
            28, 35, 36, 37, 38, 46,
        }





def _process_one_sample(
    pre_path: str,
    sample_idx: int,
    var_constraints: Dict[int, List[int]],
    non_negative_vars: Set[int],
    tol: float = 1e-6,
) -> Tuple[
    Dict[int, Tuple[int, int, float, float]],  # nonneg_stats: var -> (total_count, viol_count, viol_sum, min_viol_val)
    Dict[int, Tuple[float, float, int]],       # varsum_stats: total_var -> (sum_abs_err, sum_rel_err, N)
    Dict[int, Tuple[float, float, int]]        # regionsum_stats: var -> (sum_abs_err, sum_rel_err, N)
]:
    """
    进程池工作函数：处理一个 sample 的三个约束统计并返回。
    - 输入 memmap 路径，内部用 np.load(..., mmap_mode='r') 打开，避免大内存
    - 返回三个字典，主进程汇总
    """
    pre_mem = np.load(pre_path, mmap_mode='r')        # shape: (S, G, T, 5)
    sample = pre_mem[sample_idx]                      # (G, T, 5)

    groups, n_timesteps, feat_dim = sample.shape
    assert feat_dim == 5

    # 累计容器
    nonneg_stats: Dict[int, List[float]] = {}   # var -> [total_count, viol_count, viol_sum, min_viol_val]
    varsum_stats: Dict[int, List[float]] = {}   # total_var -> [sum_abs_err, sum_rel_err, N]
    regionsum_stats: Dict[int, List[float]] = {}# var -> [sum_abs_err, sum_rel_err, N]

    # 逐 timestep 处理，内存占用低
    for t in range(n_timesteps):
        frame = sample[:, t, :]  # (G, 5) -> [model, category, region, variable, value]
        # 基本列
        cat = frame[:, 1].astype(np.int64)
        reg = frame[:, 2].astype(np.int64)
        var = frame[:, 3].astype(np.int64)
        val = frame[:, 4].astype(np.float64)

        # ---------- 1) 非负约束 ----------
        mask_nn = np.isin(var, list(non_negative_vars))
        total_vars = var[mask_nn]
        total_vals = val[mask_nn]
        # total_count
        if total_vars.size > 0:
            # 统计每个变量出现次数
            for v_i in total_vars:
                st = nonneg_stats.setdefault(int(v_i), [0.0, 0.0, 0.0, np.inf])  # [tot, viol, sum, min]
                st[0] += 1.0

            # 违反项
            mask_viol = total_vals < -tol
            if np.any(mask_viol):
                viol_vars = total_vars[mask_viol]
                viol_vals = total_vals[mask_viol]
                # 逐条累加（变量种类不多，开销可接受）
                for v_i, vv in zip(viol_vars.tolist(), viol_vals.tolist()):
                    st = nonneg_stats.setdefault(int(v_i), [0.0, 0.0, 0.0, np.inf])
                    st[1] += 1.0
                    st[2] += vv
                    st[3] = vv if vv < st[3] else st[3]  # 最小（最负）

        # 将本 timestep 构成 DataFrame（用于两个求和约束的透视）
        df = pd.DataFrame({
            "category": cat,
            "region": reg,
            "variable": var,
            "value": val
        })

        # ---------- 2) 变量求和 total_var = sum(comp_vars)（按 category, region 逐行检查） ----------
        # 只做一次透视，后续复用各个 total_var
        df_pivot_var = df.pivot_table(index=['category', 'region'], columns='variable', values='value')
        if not df_pivot_var.empty:
            for total_var, comp_vars in var_constraints.items():
                needed = [total_var] + comp_vars
                # 要求该 pivot 含有所有所需列
                if all((c in df_pivot_var.columns) for c in needed):
                    sub = df_pivot_var[needed].dropna()
                    if not sub.empty:
                        total_vals_vec = sub[total_var].to_numpy()
                        comp_sum_vec = sub[comp_vars].to_numpy().sum(axis=1)
                        err = total_vals_vec - comp_sum_vec
                        abs_err_sum = np.abs(err).sum()
                        rel_err_sum = (np.abs(err) / (np.abs(total_vals_vec) + 1e-9)).sum()
                        N = sub.shape[0]
                        st = varsum_stats.setdefault(int(total_var), [0.0, 0.0, 0.0])
                        st[0] += float(abs_err_sum)
                        st[1] += float(rel_err_sum)
                        st[2] += float(N)

        # ---------- 3) 地区求和约束：region 0 = sum(other regions)（按 category, variable 检查） ----------
        df_pivot_reg = df.pivot_table(index=['category', 'variable'], columns='region', values='value')
        if not df_pivot_reg.empty and (0 in df_pivot_reg.columns):
            comp_regions = [c for c in df_pivot_reg.columns if c != 0]
            if len(comp_regions) > 0:
                df2 = df_pivot_reg.dropna(subset=[0] + comp_regions)
                if not df2.empty:
                    total_vals_vec = df2[0].to_numpy()
                    comp_sum_vec = df2[comp_regions].to_numpy().sum(axis=1)
                    err = total_vals_vec - comp_sum_vec
                    abs_err_sum = np.abs(err).sum()
                    rel_err_sum = (np.abs(err) / (np.abs(total_vals_vec) + 1e-9)).sum()
                    N = df2.shape[0]
                    # 分变量统计：从 index 取出 variable 列
                    # df2 的 index 是 MultiIndex(category, variable)
                    variables_here = df2.index.get_level_values('variable').to_numpy()
                    # 将误差按变量汇总
                    # 注意：这里需要按 variable 累加
                    # 为避免 groupby 再运算，直接循环（变量数量有限）
                    for v_i, aerr, terr in zip(variables_here.tolist(), np.abs(err).tolist(),
                                               (np.abs(err) / (np.abs(total_vals_vec) + 1e-9)).tolist()):
                        st = regionsum_stats.setdefault(int(v_i), [0.0, 0.0, 0.0])
                        st[0] += float(aerr)
                        st[1] += float(terr)
                        st[2] += 1.0

    # 将 list 转换为不可变 tuple 便于 pickling
    nonneg_stats_fin = {k: (int(v[0]), int(v[1]), float(v[2]), (float(v[3]) if v[3] != np.inf else np.nan))
                        for k, v in nonneg_stats.items()}
    varsum_stats_fin = {k: (float(v[0]), float(v[1]), int(v[2])) for k, v in varsum_stats.items()}
    regionsum_stats_fin = {k: (float(v[0]), float(v[1]), int(v[2])) for k, v in regionsum_stats.items()}

    return nonneg_stats_fin, varsum_stats_fin, regionsum_stats_fin


def _merge_nonneg_agg(
    agg: Dict[int, List[float]],
    local: Dict[int, Tuple[int, int, float, float]]
):
    for var, (tot, viol, vsum, vmin) in local.items():
        st = agg.setdefault(var, [0.0, 0.0, 0.0, np.inf])
        st[0] += float(tot)
        st[1] += float(viol)
        st[2] += float(vsum)
        if not np.isnan(vmin):
            st[3] = vmin if vmin < st[3] else st[3]


def _merge_sum_agg(
    agg: Dict[int, List[float]],
    local: Dict[int, Tuple[float, float, int]]
):
    for key, (abs_sum, rel_sum, n) in local.items():
        st = agg.setdefault(key, [0.0, 0.0, 0.0])
        st[0] += float(abs_sum)
        st[1] += float(rel_sum)
        st[2] += float(n)


def _build_nonneg_report(
    agg: Dict[int, List[float]],
    var_map: Dict[int, str]
) -> pd.DataFrame:
    if not agg:
        print("✅ 非负约束：未发现可检查的数据或均满足。")
        return pd.DataFrame(columns=['variable', 'variable_name', 'violation_ratio', 'violation_count', 'mean_violation', 'min_violation'])

    rows = []
    for v, (tot, viol, vsum, vmin) in agg.items():
        if tot <= 0:
            continue
        viol_ratio = (viol / tot) if tot > 0 else 0.0
        mean_viol = (vsum / viol) if viol > 0 else 0.0
        min_viol = (vmin if vmin != np.inf else np.nan)
        rows.append((v, var_map.get(v, str(v)), viol_ratio, int(viol), mean_viol, min_viol))
    df = pd.DataFrame(rows, columns=['variable', 'variable_name', 'violation_ratio', 'violation_count', 'mean_violation', 'min_violation'])
    df = df.sort_values(by='violation_ratio', ascending=False, kind='mergesort')
    print("\n" + "=" * 20 + " 检查: 1. 非负约束 " + "=" * 20)
    if df.empty:
        print("✅ 非负约束：未发现违反。")
    else:
        print(df.to_string(index=False))
    return df


def _build_varsum_report(
    agg: Dict[int, List[float]],
    var_map: Dict[int, str]
) -> pd.DataFrame:
    if not agg:
        print("✅ 变量求和约束：未发现可检查的约束组。")
        return pd.DataFrame(columns=['total_var', 'variable_name', 'mean_abs_error', 'mean_relative_error'])

    rows = []
    for total_var, (abs_sum, rel_sum, n) in agg.items():
        if n <= 0:
            continue
        rows.append((
            total_var,
            var_map.get(total_var, str(total_var)),
            abs_sum / n,
            rel_sum / n
        ))
    df = pd.DataFrame(rows, columns=['total_var', 'variable_name', 'mean_abs_error', 'mean_relative_error'])
    df = df.sort_values(by='mean_relative_error', ascending=False, kind='mergesort')
    print("\n" + "=" * 20 + " 检查: 2. 变量求和约束 " + "=" * 20)
    if df.empty:
        print("✅ 变量求和约束：未发现违反。")
    else:
        print(df.to_string(index=False))
    return df


def _build_regionsum_report(
    agg: Dict[int, List[float]],
    var_map: Dict[int, str]
) -> pd.DataFrame:
    if not agg:
        print("✅ 地区求和约束：未发现可检查的约束组。")
        return pd.DataFrame(columns=['variable', 'variable_name', 'mean_abs_error', 'mean_relative_error'])

    rows = []
    for v, (abs_sum, rel_sum, n) in agg.items():
        if n <= 0:
            continue
        rows.append((v, var_map.get(v, str(v)), abs_sum / n, rel_sum / n))
    df = pd.DataFrame(rows, columns=['variable', 'variable_name', 'mean_abs_error', 'mean_relative_error'])
    df = df.sort_values(by='mean_relative_error', ascending=False, kind='mergesort')
    print("\n" + "=" * 20 + " 检查: 3. 地区求和约束 " + "=" * 20)
    if df.empty:
        print("✅ 地区求和约束：未发现违反。")
    else:
        print(df.to_string(index=False))
    return df


# ============== 对外 API（可 import 调用） ==============
def run_checks(
    input_path,
    workers,
    sample_limit,
    var_map,
    region_map,  # 目前报告按变量维度输出，region_map 仅预留
    tqdm_disable: bool = False
):
    """
    对 corrected.npy 进行并行约束检查，返回三个报告 DataFrame:
      - 'nonneg': 非负约束报告
      - 'var_sum': 变量求和约束报告
      - 'region_sum': 地区求和约束报告
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    pre_mem = np.load(input_path, mmap_mode='r')  # (S, G, T, 5)
    if pre_mem.ndim != 4 or pre_mem.shape[-1] != 5:
        raise ValueError("输入数组形状应为 (samples, groups, timesteps, 5)")

    n_samples = pre_mem.shape[0]
    if sample_limit is not None:
        n_samples = min(n_samples, int(sample_limit))

    # 汇总容器（主进程）
    nonneg_agg: Dict[int, List[float]] = {}
    varsum_agg: Dict[int, List[float]] = {}
    regionsum_agg: Dict[int, List[float]] = {}

    indices = list(range(n_samples))
    if workers is None or workers <= 1:
        it = tqdm(indices, desc="Checking samples", disable=tqdm_disable)
        for idx in it:
            nn, vs, rs = _process_one_sample(
                input_path, idx, VAR_CONSTRAINTS, NON_NEGATIVE_VARS
            )
            _merge_nonneg_agg(nonneg_agg, nn)
            _merge_sum_agg(varsum_agg, vs)
            _merge_sum_agg(regionsum_agg, rs)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_process_one_sample, input_path, idx, VAR_CONSTRAINTS, NON_NEGATIVE_VARS): idx
                for idx in indices
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Checking samples (parallel)", disable=tqdm_disable):
                nn, vs, rs = fut.result()
                _merge_nonneg_agg(nonneg_agg, nn)
                _merge_sum_agg(varsum_agg, vs)
                _merge_sum_agg(regionsum_agg, rs)

    # 生成报告
    rep_nonneg = _build_nonneg_report(nonneg_agg, var_map)
    rep_varsum = _build_varsum_report(varsum_agg, var_map)
    rep_regionsum = _build_regionsum_report(regionsum_agg, var_map)

    return {
        "nonneg": rep_nonneg,
        "var_sum": rep_varsum,
        "region_sum": rep_regionsum,
    }





