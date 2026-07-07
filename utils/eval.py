# -*- coding: utf-8 -*-
import torch

from typing import Optional, Sequence, Dict, Tuple
from tqdm.auto import tqdm
from contextlib import contextmanager
from collections import defaultdict
import numpy as np
from .inspection import (
    load_maps as _vet_load_maps,
    load_baselines as _vet_load_baselines,
    check_one_sample as _vet_check,
)

import csv, os
import pandas as pd

def _pack_generated_to_gtk5(label_cpu: torch.Tensor, y_s_orig_cpu: torch.Tensor, default_model_code: int = 0) -> np.ndarray:
    """
    label_cpu:  [B,G,T,5] (CPU)
    y_s_orig_cpu: [B,S,G,T] (CPU) —— 生成值（原空间）
    返回: samples [B*S, G, T, 5]，通道: [model, category, region, variable, value]
    """
    B, G, T, _ = label_cpu.shape
    S = y_s_orig_cpu.shape[1]

    cat = label_cpu[:, :, 0, 1].numpy().astype(np.float32)        # [B,G]
    reg = label_cpu[:, :, 0, 2].numpy().astype(np.float32)
    var = label_cpu[:, :, 0, 3].numpy().astype(np.float32)

    cat = np.broadcast_to(cat[:, None, :, None], (B, S, G, T))
    reg = np.broadcast_to(reg[:, None, :, None], (B, S, G, T))
    var = np.broadcast_to(var[:, None, :, None], (B, S, G, T))

    val = y_s_orig_cpu.numpy().astype(np.float32)                 # [B,S,G,T]
    mdl = np.full_like(val, float(default_model_code))

    samples = np.stack([mdl, cat, reg, var, val], axis=-1)        # [B,S,G,T,5]
    samples = samples.reshape(B * S, G, T, 5)
    return samples


def append_result_row(config, params, split, epoch, ckpt_path, metrics, avg_train_loss=None):
    os.makedirs(os.path.dirname(config.results_csv) or ".", exist_ok=True)

    def _safe_get(d, *keys, default=float("nan")):
        try:
            x = d
            for k in keys:
                x = x[k]
            return x
        except Exception:
            return default

    row = {
        # identifiers
        "exp_tag": config.get("exp_tag", "default"),
        "split": split,                  # "val" | "test" | "final_test"
        "epoch": epoch,
        "seed": params.get("seed", ""),
        "split_seed": params.get("split_seed", ""),
        "save_suffix": params.get("save_suffix", ""),
        "ckpt": os.path.basename(ckpt_path) if ckpt_path else "",
        "train_loss_avg": float(avg_train_loss) if avg_train_loss is not None else float("nan"),

        # core hparams
        "hidden_size": params["hidden_size"],
        "N": params["N"],
        "beta_end": params["beta_end"],
        "sample_steps": params["sample_steps"],
        "n_blocks": params["n_blocks"],
        "lr": params["lr"],
        "train_batch_size": params["train_batch_size"],
        "n_samples": params["n_samples"],
        "train_fraction": params.get("train_fraction", 1.0),

        # constraint (train)
        "lambda_region": params["lambda_region"],
        "lambda_var":    params["lambda_var"],
        "lambda_nn":     params["lambda_nn"],

        # soft guidance (sampling)
        "use_soft_guidance": params["use_soft_guidance"],
        "soft_sched": params["soft_guidance_schedule"],
        "soft_eta_min": params["soft_eta_min"],
        "soft_eta_max": params["soft_eta_max"],

        # ratio adaptation
        "target_ratio": params["constraint_target_ratio"],
        "ratio_smooth": params["constraint_ratio_smooth"],
        "gain_min": params["constraint_gain_min"],
        "gain_max": params["constraint_gain_max"],
        "warmup":   params["constraint_warmup_steps"] ,

        # post-process
        "reverse_push": params["reverse_projection_scale"],

        # metrics: micro (点估计)
        "micro_MAE": _safe_get(metrics, "overall", "MAE"),
        "micro_RMSE": _safe_get(metrics, "overall", "RMSE"),

        # metrics: macro
        "macro_MAE": _safe_get(metrics, "overall_macro", "MAE"),
        "macro_RMSE": _safe_get(metrics, "overall_macro", "RMSE"),
        "macro_MMD2": _safe_get(metrics, "overall_macro", "MMD2"),

        # vet pass rate（仅保留 VetPass，已删除 physics Pass）
        "vet_pass_rate": metrics.get("vet_pass_rate", float("nan")),
    }

    write_header = (not os.path.exists(config.results_csv)) or os.path.getsize(config.results_csv) == 0
    with open(config.results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@contextmanager
def evaluating(model, use_inference_mode: bool = False):
    """切到 eval 状态；按需开启 inference_mode（默认不开）。"""
    was_training = model.training
    try:
        model.eval()
        if use_inference_mode:
            with torch.inference_mode():
                yield
        else:
            yield
    finally:
        model.train(was_training)


# =========================
#   Pairwise metrics
# =========================

@torch.no_grad()
def mmd_rbf_vs_point(X: torch.Tensor, y: torch.Tensor,
                     sigmas: Sequence[float]) -> float:
    """
    逐样本 MMD^2（V-statistic，多σ RBF），X~P, y~Q(单点)：
      MMD^2 = E k(X,X') + k(y,y) - 2 E k(X,y)
    X: [S, D]   该样本的 S 条生成向量（标准化空间）
    y: [D]      该样本的真实向量（标准化空间）
    """
    assert X.dim() == 2 and y.dim() == 1 and X.size(1) == y.size(0)
    if X.size(0) == 0:
        return 0.0

    # CPU、float64 提高稳定性
    X = X.detach().cpu().to(torch.float64)
    y = y.detach().cpu().to(torch.float64)

    sig = torch.as_tensor(list(sigmas), dtype=X.dtype, device=X.device).view(1, 1, -1)  # [1,1,K]

    # Kxx.mean（多σ平均）
    dist_xx2  = torch.cdist(X, X, p=2).pow(2)                 # [S,S]
    Kxx       = torch.exp(-dist_xx2.unsqueeze(-1) / (2.0 * sig * sig)).mean(-1).mean()

    # k(y,y)=1（与 X 同 device，避免 CPU/CUDA 混用）
    Kyy       = torch.tensor(1.0, dtype=X.dtype, device=X.device)

    # Kxy.mean
    dist_xy2  = torch.cdist(X, y.view(1, -1), p=2).pow(2)     # [S,1]
    Kxy       = torch.exp(-dist_xy2.unsqueeze(-1) / (2.0 * sig * sig)).mean(-1).mean()

    mmd2 = float((Kxx + Kyy - 2.0 * Kxy).clamp_min(0.0).item())
    return mmd2


@torch.no_grad()
def wasserstein_vs_point(X: torch.Tensor, y: torch.Tensor, p: int = 2):
    """
    逐样本标准 Wasserstein：集合 X 与单点 y 的距离。
    闭式：W_p^p = E_i ||x_i - y||_p^p
    返回: (W_p, W_p^p)
    """
    assert X.dim() == 2 and y.dim() == 1 and X.size(1) == y.size(0)
    if X.size(0) == 0:
        return 0.0, 0.0

    X = X.detach().cpu().to(torch.float64)
    y = y.detach().cpu().to(torch.float64)

    diff = X - y.unsqueeze(0)
    if p == 2:
        Wp_p = float(diff.pow(2).sum(dim=1).mean().item())
    else:
        Wp_p = float(diff.abs().pow(p).sum(dim=1).mean().item())
    return float(Wp_p ** (1.0 / p)), Wp_p


@torch.no_grad()
def energy_score_ensemble(samples: torch.Tensor, target: torch.Tensor,
                          ord: int = 2, pair_block: Optional[int] = None) -> float:
    """
    多变量 CRPS 的等价形式（Energy Score）：
      ES = E ||X - y|| - 0.5 * E ||X - X'||
    samples: [S, D]（标准化空间，D=T）
    target : [D]    （标准化空间）
    """
    assert samples.dim() == 2 and target.dim() == 1 and samples.size(1) == target.size(0)
    S = samples.size(0)
    if S == 0:
        return 0.0

    X = samples.detach().cpu().to(torch.float64)
    y = target.detach().cpu().to(torch.float64)

    term1 = torch.norm(X - y.unsqueeze(0), p=ord, dim=1).mean()
    if pair_block is None or S <= (pair_block or S):
        term2 = 0.5 * torch.cdist(X, X, p=ord).mean()
    else:
        # 大 S 时块状累加避免一次性 SxS
        acc, cnt = 0.0, 0
        for i in range(0, S, pair_block):
            Xi = X[i:i + pair_block]
            Dij = torch.cdist(Xi, X, p=ord)
            acc += Dij.sum().item()
            cnt += Xi.size(0) * S
        term2 = 0.5 * (acc / max(cnt, 1))
    return float((term1 - term2).item())


# =========================
#   Helpers
# =========================

@torch.no_grad()
def sample_predict(model, label, n_samples: int, split: int = 10) -> torch.Tensor:
    """
    调用模型做采样；返回值放 CPU，避免占 GPU 显存。
    label : [B,G,T,5]（值列已标准化）
    out   : [B,S,G,T]（原空间）
    """
    model.eval()
    B, _, _, _ = label.shape
    parts = []
    for off in range(0, n_samples, split):
        ns = min(split, n_samples - off)
        xhat = model(label, ns)[..., 0]          # [B,ns,G,T]
        parts.append(xhat.detach().cpu())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(parts, dim=1)               # [B,S,G,T]


def get_mu_sd_from_label(label, means, stds, eps: float = 1e-8):
    """
    从编码取每个 (b,g) 的 μ、σ。输出与 label 同设备。
    label: [B,G,T,5]
    means/stds: 任意（numpy.ndarray 或 torch.Tensor），形状 [R,V] 或 [C,R,V]
    返回: mu, sd 形状 [B,G,1]（与 label 同设备/类型）
    """
    # 1) 统一成 torch，并搬到 label.device
    if isinstance(means, np.ndarray):
        means = torch.from_numpy(means)
    if isinstance(stds, np.ndarray):
        stds = torch.from_numpy(stds)

    means = means.to(device=label.device, dtype=label.dtype, non_blocking=True)
    stds  = stds.to(device=label.device, dtype=label.dtype,  non_blocking=True)

    # 2) 取索引（索引必须是 long）
    cat_code    = label[:, :, 0, 1].long()
    region_code = label[:, :, 0, 2].long()
    var_code    = label[:, :, 0, 3].long()

    # 3) 索引并扩展到 [B,G,1]
    if means.dim() == 3:
        mu = means[cat_code, region_code, var_code].unsqueeze(-1)
        sd = (stds[cat_code, region_code, var_code] + eps).unsqueeze(-1)
    else:
        mu = means[region_code, var_code].unsqueeze(-1)
        sd = (stds[region_code, var_code] + eps).unsqueeze(-1)
    return mu, sd


def _compute_physics_per_sample(sample_gtk5, var_constraints=None, non_negative_vars=None):
    """单条生成样本 (G,T,5) 物理空间：EqResidual、NegRatio。服务于 evaluate_sampling_metrics。"""
    from utils.check_cons import VAR_CONSTRAINTS, NON_NEGATIVE_VARS
    vc = var_constraints if var_constraints is not None else VAR_CONSTRAINTS
    nnv = non_negative_vars if non_negative_vars is not None else NON_NEGATIVE_VARS
    G, T, _ = sample_gtk5.shape
    val = sample_gtk5[:, :, 4]
    var = sample_gtk5[:, 0, 3].astype(int)
    res_region = 0.0
    res_var = 0.0
    res_var_dict = defaultdict(float)
    for t in range(T):
        frame = sample_gtk5[:, t, :]
        df = pd.DataFrame({
            "category": frame[:, 1], "region": frame[:, 2], "variable": frame[:, 3], "value": frame[:, 4]
        })
        piv_r = df.pivot_table(index=["category", "variable"], columns="region", values="value")
        if 0 in piv_r.columns and piv_r.shape[1] > 1:
            others = [c for c in piv_r.columns if c != 0]
            world = piv_r[0].to_numpy()
            rest = piv_r[others].sum(axis=1).to_numpy()
            res_region += np.sum((world - rest) ** 2)
        piv_v = df.pivot_table(index=["category", "region"], columns="variable", values="value")
        for tot, comps in vc.items():
            if tot not in piv_v.columns or not all(c in piv_v.columns for c in comps):
                continue
            tot_val = piv_v[tot].to_numpy()
            comp_sum = piv_v[comps].sum(axis=1).to_numpy()
            err_sq = (tot_val - comp_sum) ** 2
            res_var += np.sum(err_sq)
            res_var_dict[tot] += np.sum(err_sq)
    eq_residual = float(np.sqrt(res_region + res_var + 1e-20))
    eq_residual_region = float(np.sqrt(res_region + 1e-20))
    eq_residual_var = float(np.sqrt(res_var + 1e-20))
    neg_count = 0
    total_nn = 0
    for v in np.unique(var):
        if int(v) in nnv:
            total_nn += np.sum(var == v)
            neg_count += np.sum(val[var == v, :] < -1e-5)
    neg_ratio = (neg_count / total_nn) if total_nn > 0 else 0.0
    return {
        "eq_residual": eq_residual,
        "eq_residual_region": eq_residual_region,
        "eq_residual_var": eq_residual_var,
        "neg_ratio": neg_ratio,
        "res_var_dict": dict(res_var_dict),
    }


# =========================
#   Main evaluation
# =========================

class _Agg:
    __slots__ = ("sum", "cnt")
    def __init__(self):
        self.sum = 0.0
        self.cnt = 0
    def add(self, val, n=1):
        self.sum += float(val)
        self.cnt += int(n)
    def mean(self):
        return float(self.sum / max(self.cnt, 1))


@torch.no_grad()
def evaluate_sampling_metrics(model, data_loader, full_mask, config, means, stds,
                              n_samples: int,
                              split_n: int = 10,
                              desc: str = "val",
                              mmd_sigmas: Sequence[float] = (0.5, 1.0, 2.0),
                              es_pair_block: Optional[int] = None,  # 保留未用
                              region_map: Optional[dict] = None,
                              var_map: Optional[dict] = None,
                              vet_region_csv: Optional[str] = None,
                              vet_variable_csv: Optional[str] = None,
                              vet_variable_baseline_csv: Optional[str] = None,
                              vet_region_baseline_csv: Optional[str] = None,
                              vet_category_csv: Optional[str] = None,
                              vet_default_model_code: int = 0):
    """
    评估逻辑：
      1) 逐批生成，收集 (X_std, y_std) 与点估计累加
      2) 结束后在三类分组上算 MMD2、MAE、RMSE；overall/overall_macro 只保留 MAE、RMSE
      3) physics：eq_residual、neg_ratio 等均值（已移除 pass）
    返回:
      overall: {MAE, RMSE}, overall_macro: {MAE, RMSE, MMD2},
      groups: by_category/region/variable 每组 mean: {MMD2, MAE, RMSE}, physics: {eq_residual, neg_ratio, ...}, vet_pass_rate
    """
    # VET 资源
    _reg_c2n, reg_n2c, _var_c2n, var_n2c = _vet_load_maps(vet_region_csv, vet_variable_csv)
    VAR_REL_ERR, REG_REL_ERR = _vet_load_baselines(vet_variable_baseline_csv, vet_region_baseline_csv)
    vet_pass, vet_total = 0, 0
    vet_condition_counts = {}  # Dict[str, int] 各条件通过数

    # -------- 点估计类（overall micro 的逐点累加） --------
    mae_sum, rmse_sq_sum, n_points_total = 0.0, 0.0, 0

    # -------- overall_macro（先系列内，后 across (r,v)）--------
    macro_mae_list, macro_rmse_list = [], []

    # -------- 三类分组分布性指标的“收集桶” --------
    def _mk_group_bucket():
        return {'pairs': []}  # 列表元素是 (X_std_t[ S,Teff ], y_bg_t[ Teff ])

    group_cat = defaultdict(_mk_group_bucket)  # 按 label[...,1]
    group_reg = defaultdict(_mk_group_bucket)  # 按 label[...,2]
    group_var = defaultdict(_mk_group_bucket)  # 按 label[...,3]

    # -------- 物理约束（EqResidual/NegRatio/Pass）--------
    physics_list = []

    total_batches = len(data_loader) if hasattr(data_loader, "__len__") else None
    pbar = tqdm(total=total_batches, desc=f"{desc} (batches)", leave=False)

    for step, batch in enumerate(data_loader):
        label, index = batch                     # [B,G,T,5]
        B, G, T, _ = label.shape
        label = label.to(torch.float32).to(config.device)
        mask_b = full_mask[index].to(config.device).bool()   # [B,G]

        # 采样（原空间，返回 CPU）
        y_s_orig = sample_predict(model, label, n_samples=n_samples, split=split_n)   # [B,S,G,T] (CPU)

        # VET 打包与检查
        samples_gtk5 = _pack_generated_to_gtk5(label.detach().cpu(), y_s_orig, default_model_code=vet_default_model_code)  # [B*S,G,T,5]
        for i in range(samples_gtk5.shape[0]):
            ok, _, cond = _vet_check(samples_gtk5[i], reg_n2c, var_n2c, VAR_REL_ERR, REG_REL_ERR)
            vet_pass += int(bool(ok)); vet_total += 1
            for k, v in cond.items():
                vet_condition_counts[k] = vet_condition_counts.get(k, 0) + int(bool(v))

        # 物理约束（复用 samples_gtk5，不重复采样）
        try:
            for i in range(samples_gtk5.shape[0]):
                physics_list.append(_compute_physics_per_sample(samples_gtk5[i], None, None))
        except Exception as e:
            import warnings
            warnings.warn(f"[evaluate_sampling_metrics] physics: {e}")

        # GT（标准化）与 μ/σ（CPU）
        y_t_std = label[..., 4].detach().cpu()                                        # [B,G,T]
        mu, sd  = get_mu_sd_from_label(label, means, stds)
        mu, sd  = mu.detach().cpu(), sd.detach().cpu()                                # [B,G,1]

        # 标准化后的生成
        y_s_std = (y_s_orig - mu.unsqueeze(1)) / sd.unsqueeze(1)                      # [B,S,G,T]

        # 有效时间点 mask
        mask3 = mask_b.detach().cpu().unsqueeze(-1).expand(B, G, T)                   # [B,G,T]

        # ------ 逐 (n=b, g) 收集 & 点估计累加 ------
        for b in range(B):
            g_idx = torch.nonzero(mask_b[b], as_tuple=False).flatten().tolist()
            for g in g_idx:
                X_std = y_s_std[b, :, g, :]      # [S,T] (CPU,float32)
                y_bg  = y_t_std[b, g, :]         # [T]   (CPU,float32)
                m_t   = mask3[b, g]              # [T]   (CPU,bool)
                if not torch.any(m_t):
                    continue
                X_std_t = X_std[:, m_t]          # [S,Teff]
                y_bg_t  = y_bg[m_t]              # [Teff]

                # ==== 点估计类：逐点累加（micro） ====
                y_pred_std_bg = X_std_t.mean(dim=0)  # [Teff]
                diff_std_bg   = y_pred_std_bg - y_bg_t
                Teff = int(y_bg_t.numel())
                mae_sum      += float(diff_std_bg.abs().sum().item())
                rmse_sq_sum  += float(diff_std_bg.pow(2).sum().item())
                n_points_total += Teff

                # ==== macro：按系列先求 MAE/RMSE，再 across (n,r,v) 求均值 ====
                mae_g  = float(diff_std_bg.abs().mean().item())
                rmse_g = float(diff_std_bg.pow(2).mean().sqrt().item())
                macro_mae_list.append(mae_g)
                macro_rmse_list.append(rmse_g)

                # ==== 分布一致性：仅“收集”，不在批内计算 ====
                X_pair = X_std_t.to(torch.float64).contiguous()
                y_pair = y_bg_t.to(torch.float64).contiguous()

                cat_code = int(label[b, g, 0, 1].item())
                reg_code = int(label[b, g, 0, 2].item())
                var_code = int(label[b, g, 0, 3].item())

                group_cat[cat_code]['pairs'].append((X_pair, y_pair))
                group_reg[reg_code]['pairs'].append((X_pair, y_pair))
                group_var[var_code]['pairs'].append((X_pair, y_pair))

        pbar.update(1)

    pbar.close()

    # ---------- 结束后统一计算：overall / overall_macro ----------
    overall_out = {
        'MAE':  mae_sum / max(n_points_total, 1),
        'RMSE': float((rmse_sq_sum / max(n_points_total, 1)) ** 0.5),
    }
    overall_macro_out = {
        'MAE':  float(np.mean(macro_mae_list)) if len(macro_mae_list) else float('nan'),
        'RMSE': float(np.mean(macro_rmse_list)) if len(macro_rmse_list) else float('nan'),
    }

    # ---------- 结束后统一计算：三类分组的分布一致性 + 组内 MAE/RMSE ----------
    def _compute_group_stats(group_bucket: Dict[int, dict]) -> dict:
        per_group = {}
        for code, bkt in group_bucket.items():
            mmd2_vals, mae_vals, mse_vals = [], [], []

            for (X_pair, y_pair) in bkt['pairs']:
                mmd2 = mmd_rbf_vs_point(X_pair, y_pair, sigmas=mmd_sigmas)
                mmd2_vals.append(mmd2)
                y_hat = X_pair.mean(dim=0)
                diff = y_hat - y_pair
                n_eff = max(int(y_pair.numel()), 1)
                mae_vals.append(float(diff.abs().mean().item()))
                mse_vals.append(float((diff * diff).sum().item()) / n_eff)

            per_group[int(code)] = {
                'MMD2': float(np.mean(mmd2_vals)) if mmd2_vals else float('nan'),
                'MAE':  float(np.mean(mae_vals)) if mae_vals else float('nan'),
                'RMSE': float(np.sqrt(np.mean(mse_vals))) if mse_vals else float('nan'),
            }

        keys = list(per_group.keys())
        def _mean_key(k):
            vals = [per_group[c][k] for c in keys if np.isfinite(per_group[c][k])]
            return float(np.mean(vals)) if vals else float('nan')
        mean_all = {'MMD2': _mean_key('MMD2'), 'MAE': _mean_key('MAE'), 'RMSE': _mean_key('RMSE')}
        return {'per_group': per_group, 'mean': mean_all}

    groups_out = {
        'by_category': _compute_group_stats(group_cat),
        'by_region':   _compute_group_stats(group_reg),
        'by_variable': _compute_group_stats(group_var),
    }

    # 总体 MMD2：对所有 (b,g) 对求 MMD2 再取均值（非分组均值的平均）
    all_pairs = []
    for bkt in group_cat.values():
        all_pairs.extend(bkt['pairs'])
    all_mmd2 = [mmd_rbf_vs_point(X, y, sigmas=mmd_sigmas) for (X, y) in all_pairs]
    overall_macro_out['MMD2'] = float(np.mean(all_mmd2)) if all_mmd2 else float('nan')

    # 物理约束汇总（已移除 pass，仅保留 eq_residual / neg_ratio）
    physics_out = {
        'eq_residual': float('nan'), 'eq_residual_region': float('nan'), 'eq_residual_var': float('nan'),
        'neg_ratio': float('nan'),
    }
    if physics_list:
        physics_out = {
            'eq_residual': float(np.mean([p['eq_residual'] for p in physics_list])),
            'eq_residual_region': float(np.mean([p['eq_residual_region'] for p in physics_list])),
            'eq_residual_var': float(np.mean([p['eq_residual_var'] for p in physics_list])),
            'neg_ratio': float(np.mean([p['neg_ratio'] for p in physics_list])),
        }

    out =  {
        'overall': overall_out,
        'overall_macro': overall_macro_out,
        'groups': groups_out,
        'physics': physics_out,
    }
    out['vet_pass_rate'] = (vet_pass / vet_total) if vet_total else float('nan')
    out['vet_condition_pass_rate'] = {k: v / vet_total for k, v in vet_condition_counts.items()} if vet_total else {}
    return out
