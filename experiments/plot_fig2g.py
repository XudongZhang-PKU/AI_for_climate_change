# -*- coding: utf-8 -*-
"""
================================================================================
Fig.2(g) Ablation summary (Radar only) — Nature-ish FINAL (POLISHED)
================================================================================

改进点：
1) 引入 Nature 经典调色板 (NPG Palette)。
2) 全局字体替换为无衬线字体 (Arial/Helvetica)，符合顶刊排版规范。
3) 自动转换模型名称为标准学术消融格式 (w/ 和 w/o)。
4) [核心修改] 将 xtick_pad 大幅缩小至 12，使 MMD 等文字标签紧贴雷达圈。
5) [核心修改] 画板高度压缩为 7.5，图例高度下调，去除上下多余留白，整体极其紧凑。
================================================================================
"""

import os
import sys
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.inspection import load_maps as _vet_load_maps, load_baselines as _vet_load_baselines, check_one_sample as _vet_check

R = 11
V = 48

VAR_CONSTRAINTS = {
    0: [1, 2, 45],
    3: [4, 5, 33],
    8: [40, 41, 42, 44],
    9: [10, 39, 43, 47],
    28: [35, 36, 37, 38, 46],
    11: [12, 13, 14, 15, 16, 29, 30, 31, 32],
    17: [18, 19, 20, 21, 22, 23, 24, 25, 26],
}

NON_NEGATIVE_VARS = {
    0, 1, 2, 45,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    11, 12, 13, 14, 15, 16, 29, 30, 31, 32,
    8, 40, 41, 42, 44,
    9, 10, 39, 43, 47,
    28, 35, 36, 37, 38, 46,
}

def _to_float_or_nan(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _parse_model_name(run_dir):
    parent = os.path.basename(os.path.dirname(run_dir))
    m = re.match(r"fig2_data_prepare_(.+?)_\d+$", parent)
    return m.group(1) if m else parent

def _discover_run_dirs(data_dir_arg):
    if data_dir_arg:
        return [data_dir_arg]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return sorted(glob.glob(os.path.join(repo_root, "fig2_data_prepare_*_1234", "fig2_data")))

def _read_training_record_metrics(run_dir):
    train_csv = os.path.join(os.path.dirname(run_dir), "训练记录.csv")
    out = {"macro_MMD2": np.nan, "macro_MAE": np.nan, "macro_RMSE": np.nan}
    if not os.path.isfile(train_csv):
        return out
    df = pd.read_csv(train_csv)
    for c in ["macro_MMD2", "macro_MAE", "macro_RMSE"]:
        if c not in df.columns:
            return out

    dd = df.copy()
    dd["macro_MAE"] = pd.to_numeric(dd["macro_MAE"], errors="coerce")
    if dd["macro_MAE"].notna().sum() == 0:
        return out
    row = dd.loc[dd["macro_MAE"].idxmin()]

    out["macro_MMD2"] = _to_float_or_nan(row["macro_MMD2"])
    out["macro_MAE"] = _to_float_or_nan(row["macro_MAE"])
    out["macro_RMSE"] = _to_float_or_nan(row["macro_RMSE"])
    return out

def _summary_ts(values_bt):
    return float(np.sum(np.asarray(values_bt)))

def _load_crv_stats_lut(run_dir):
    p = os.path.join(run_dir, "fig2a_real_syn.npz")
    if not os.path.isfile(p):
        return {}
    d = np.load(p)
    keys = np.asarray(d["keys"], dtype=np.int64)
    means = np.asarray(d["train_means"], dtype=np.float64)
    stds = np.asarray(d["train_stds"], dtype=np.float64)
    n = min(len(keys), len(means), len(stds))
    lut = {}
    for i in range(n):
        c, r, v = int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2])
        lut[(c, r, v)] = (float(means[i]), float(stds[i]))
    return lut

def _build_real_test_from_raw(run_dir, n_vars):
    raw_path = os.path.join(run_dir, "fig2_raw_test.npz")
    if not os.path.isfile(raw_path):
        return None, None
    lut = _load_crv_stats_lut(run_dir)
    if not lut:
        return None, None

    raw = np.load(raw_path)
    test_label = np.asarray(raw["test_label"])
    n_test, g_num, _, _ = test_label.shape

    x_rows, y_rows = [], []
    for i in range(n_test):
        lab = test_label[i]
        row = []
        for vc in range(n_vars):
            val = np.nan
            for g in range(g_num):
                c = int(lab[g, 0, 1]); r = int(lab[g, 0, 2]); v = int(lab[g, 0, 3])
                if v != vc:
                    continue
                key = (c, r, v)
                if key not in lut:
                    continue
                mu, sd = lut[key]
                val = _summary_ts(lab[g, :, 4] * sd + mu)
                break
            row.append(val)
        x_rows.append(row)
        y_rows.append(int(lab[0, 0, 1]))
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.int64)

def _compute_gap_real_minus_syn(run_dir):
    xtr_p = os.path.join(run_dir, "X_train.npy")
    ytr_p = os.path.join(run_dir, "y_train.npy")
    xsy_p = os.path.join(run_dir, "X_syn.npy")
    ysy_p = os.path.join(run_dir, "y_syn.npy")
    for p in [xtr_p, ytr_p, xsy_p, ysy_p]:
        if not os.path.isfile(p):
            return np.nan

    X_train = np.nan_to_num(np.load(xtr_p), nan=0.0)
    y_train = np.load(ytr_p)
    X_syn = np.nan_to_num(np.load(xsy_p), nan=0.0)
    y_syn = np.load(ysy_p)

    X_real, y_real = _build_real_test_from_raw(run_dir, n_vars=X_train.shape[1])
    if X_real is None or y_real is None:
        return np.nan
    X_real = np.nan_to_num(X_real, nan=0.0)

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_train)
    Xsy_s = sc.transform(X_syn)
    Xre_s = sc.transform(X_real)

    clf = NearestCentroid()
    clf.fit(Xtr_s, y_train)
    f1_syn = float(f1_score(y_syn, clf.predict(Xsy_s), average="macro"))
    f1_real = float(f1_score(y_real, clf.predict(Xre_s), average="macro"))
    return f1_real - f1_syn

def _build_rv_index(label_i):
    g_num = label_i.shape[0]
    out = np.full((R, V), -1, dtype=np.int64)
    reg = label_i[:, 0, 2].astype(int)
    var = label_i[:, 0, 3].astype(int)
    for g in range(g_num):
        r = reg[g]; v = var[g]
        if 0 <= r < R and 0 <= v < V and out[r, v] < 0:
            out[r, v] = g
    return out

def _trimmed_mean(vals, q_low=1.0, q_high=99.0):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if arr.size < 8:
        return float(np.mean(arr))
    lo = np.percentile(arr, q_low)
    hi = np.percentile(arr, q_high)
    keep = arr[(arr >= lo) & (arr <= hi)]
    if keep.size == 0:
        return float(np.mean(arr))
    return float(np.mean(keep))

def _compute_constraint_triplet_and_vet_trimmed(
    run_dir, reg_n2c, var_n2c, VAR_REL_ERR, REG_REL_ERR,
    apply_trim=False, q_low=1.0, q_high=99.0
):
    raw_path = os.path.join(run_dir, "fig2_raw_test.npz")
    if not os.path.isfile(raw_path):
        return {"eq_region_mse": np.nan, "eq_variable_mse": np.nan,
                "neg_violation_mean": np.nan, "vet_pass_rate": np.nan}

    raw = np.load(raw_path)
    label = np.asarray(raw["test_label"])   
    syn = np.asarray(raw["test_syn"])       
    n_test, s_num, _, t_num = syn.shape

    eq_region_list = []
    eq_var_list = []
    neg_ss = 0.0
    neg_den = 0
    vet_pass = 0
    vet_total = 0

    for i in range(n_test):
        label_i = label[i]
        rv_idx = _build_rv_index(label_i)
        sample_code = label_i[:, :, :4]

        var_codes = label_i[:, 0, 3].astype(int)
        nn_mask_g = np.isin(var_codes, list(NON_NEGATIVE_VARS))

        for s in range(s_num):
            vals_gt = syn[i, s]  
            sample = np.concatenate([sample_code, vals_gt[..., None]], axis=-1)  

            ok, _, _ = _vet_check(sample, reg_n2c, var_n2c, VAR_REL_ERR, REG_REL_ERR)
            vet_pass += int(bool(ok))
            vet_total += 1

            vals_rvt = np.full((R, V, t_num), np.nan, dtype=np.float64)
            for r in range(R):
                for v in range(V):
                    g = rv_idx[r, v]
                    if g >= 0:
                        vals_rvt[r, v, :] = vals_gt[g, :]

            world = vals_rvt[0, :, :]
            rest = np.nansum(vals_rvt[1:, :, :], axis=0)
            d = world - rest
            m = np.isfinite(d)
            if np.any(m):
                eq_region_list.append(float(np.mean(d[m] ** 2)))

            d2_all = []
            for tot, comps in VAR_CONSTRAINTS.items():
                if tot >= V:
                    continue
                valid_comps = [c for c in comps if c < V]
                if not valid_comps:
                    continue
                tot_v = vals_rvt[:, tot, :]
                comp_sum = np.nansum(vals_rvt[:, valid_comps, :], axis=1)
                d2 = (tot_v - comp_sum)
                m2 = np.isfinite(d2)
                if np.any(m2):
                    d2_all.append(d2[m2].reshape(-1))
            if d2_all:
                xx = np.concatenate(d2_all, axis=0)
                eq_var_list.append(float(np.mean(xx ** 2)))

            if np.any(nn_mask_g):
                vv = vals_gt[nn_mask_g, :]
                neg_ss += float(np.sum(np.maximum(-vv, 0.0)))
                neg_den += int(vv.size)

    if apply_trim:
        eq_region = _trimmed_mean(eq_region_list, q_low=q_low, q_high=q_high)
        eq_var = _trimmed_mean(eq_var_list, q_low=q_low, q_high=q_high)
    else:
        eq_region = float(np.mean(eq_region_list)) if len(eq_region_list) > 0 else np.nan
        eq_var = float(np.mean(eq_var_list)) if len(eq_var_list) > 0 else np.nan

    return {
        "eq_region_mse": eq_region,
        "eq_variable_mse": eq_var,
        "neg_violation_mean": (neg_ss / neg_den) if neg_den > 0 else np.nan,
        "vet_pass_rate": (vet_pass / vet_total) if vet_total > 0 else np.nan,
    }


def _plot_radar_nature_polished(
    df_raw, metric_cols, out_path,
    show_delta=False,
    delta_thresh=0.08,
    floor=0.08,                # [恢复原值] 不改比例
    ceiling=1.0,               
    gap_tau_floor=0.03,
    gap_boost_gamma=0.35,      
    xtick_pad=30,
    group_title_r=1.23,
    delta_r=1.17,
    show_yticks=True,
    figsize=(9.5, 8.5),
    dpi=300,
):
    metric_cols = [m for m in metric_cols if m != "gap_real_minus_syn"]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 21,
        "axes.unicode_minus": False,
    })
    
    # 学术规范消融命名映射
    rename_map = {
        "Full": "Full",
        "Full_ConstraintLoss": "w ConstraintLoss",
        "Full_NoCatCond": "w/o CatCond",
        "Full_NoRegionCond": "w/o RegionCond",
        "Full_NoVarCond": "w/o VarCond",
        "Full_TinyUGnet": "w/ TinyUGnet",
        "Full_UGnetNoTCN": "w/o TCN",
        "NoConstraint": "w/o Constraint",
        "NoGraph": "w/o Graph",
    }
    
    raw_names = df_raw["model_name"].astype(str).tolist()
    names = [rename_map.get(n, n) for n in raw_names]
    
    vals_raw = df_raw[metric_cols].to_numpy(dtype=np.float64)

    if "Full" not in names:
        raise RuntimeError("需要存在名为 'Full' 的行作为归一化锚点。")
    i_full = names.index("Full")
    full_ref = vals_raw[i_full].copy()

    short = {
        "macro_MMD2": "\nMMD$^2$", "macro_MAE": "MAE", "macro_RMSE": "RMSE",
        "vet_pass_rate": "Vet-Pass",
        "neg_violation_mean": "Neg-Viol", "eq_region_mse": "Eq-Region",
        "eq_variable_mse": "Eq-Variable",
    }

    lower_better = {
        "macro_MMD2", "macro_MAE", "macro_RMSE",
        "eq_region_mse", "eq_variable_mse", "neg_violation_mean",
    }
    higher_better = {"vet_pass_rate"}

    EPS = 1e-12
    vals = np.full_like(vals_raw, np.nan, dtype=np.float64)

    for j, m in enumerate(metric_cols):
        col = vals_raw[:, j].astype(np.float64)
        finite = np.isfinite(col)
        ref = full_ref[j]
        if not np.isfinite(ref): 
            continue

        if m in lower_better:
            score = ref / (col + EPS)
        elif m in higher_better:
            score = col / (ref + EPS)
        else:
            score = col / (ref + EPS)

        score = np.clip(score, 0.0, 1.2)

        if m in ("eq_region_mse", "eq_variable_mse"):
            score = np.sqrt(score) * 0.5 + score * 0.5

        mapped_score = floor + (ceiling - floor) * score
        vals[finite, j] = mapped_score[finite]

    n = len(metric_cols)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_c = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=figsize)
    # 强制居中对齐，并稍微放大绘图区，避免标签被裁剪
    ax = fig.add_axes([0.18, 0.12, 0.64, 0.62], polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    max_val = max(1.05, np.nanmax(vals) + 0.05)
    ax.set_ylim(0.0, max_val) 

    ax.grid(True, color='#D3D3D3', linestyle='-', linewidth=0.7, alpha=0.8)
    ax.spines["polar"].set_color('#BBBBBB')

    ax.set_xticks(angles)
    ax.set_xticklabels([short.get(m, m) for m in metric_cols], fontsize=19.5)
    # pad=xtick_pad 确保外圈文字紧贴圆圈但不遮挡
    ax.tick_params(axis="x", pad=xtick_pad)

    # 绘制基准参考线 (Full = 1.0)
    ax.plot(angles_c, np.full_like(angles_c, 1.0), ls="--", lw=1.5, color="#555555", alpha=0.8, zorder=1)

    if show_yticks:
        ticks = [0.4, 0.6, 0.8, 1.0]
        if max_val >= 1.15: ticks.append(1.2)
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks], fontsize=18, color="#555555")
        
        label_angle = 18.0
        ax.set_rlabel_position(label_angle) 
        
        for label in ax.get_yticklabels():
            label.set_zorder(15)

    npg_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']
    
    palette = {"Full": "#E64B35", "w/o Constraint": "#4DBBD5", "w/o Graph": "#00A087"}
    color_idx = 3
    for name in names:
        if name not in palette:
            palette[name] = npg_colors[color_idx % len(npg_colors)]
            color_idx += 1

    model_handles = []
    for i, name in enumerate(names):
        v = np.nan_to_num(vals[i], nan=floor)
        v = np.concatenate([v, [v[0]]])
        color = palette.get(name, "0.25")
        is_full = (name == "Full")
        
        ax.plot(angles_c, v, lw=2.5 if is_full else 1.8, color=color, alpha=1.0 if is_full else 0.85, 
                zorder=10 if is_full else 5,
                marker='o', markersize=7 if is_full else 5, 
                mfc='white' if is_full else color, 
                mec=color, mew=1.5 if is_full else 0.5, label=name)
        
        if is_full: 
            ax.fill(angles_c, v, color=color, alpha=0.1)

        if is_full:
            handle = Line2D([0], [0], marker='o', linestyle='-', color=color,
                           markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5,
                           markersize=9, label=name, linewidth=2.5)
        else:
            handle = Line2D([0], [0], marker='o', linestyle='-', color=color,
                           markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.5,
                           markersize=7, label=name, linewidth=1.8)
        model_handles.append(handle)

    n_models = len(model_handles)
    ncol = (n_models + 1) // 2 
    ordered_handles = []
    for i in range(ncol):
        for j in range(i, n_models, ncol):
            if j < n_models:
                ordered_handles.append(model_handles[j])

    # [核心修改] 将 bbox_to_anchor 下移至 1.12，使图例紧贴雷达图，不留多余空白
    ax.legend(handles=ordered_handles, loc="lower center", bbox_to_anchor=(0.5, 1.10),
              ncol=ncol, frameon=False, fontsize=18,
              handletextpad=0.9,
              columnspacing=1.8,
              labelspacing=1.0)
    
    # ax.text(0.5, -0.15, "* Relative Retention (Outer = Better, dashed line = Full Model baseline)", 
    #         transform=ax.transAxes, fontsize=10, color="#666666", style='italic', ha="center")

    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close()


def _add_bool_arg(parser, name, default=False, help_text=""):
    parser.add_argument(name, dest=name.lstrip("-").replace("-", "_"), action="store_true", help=help_text)
    parser.add_argument(
        "--no-" + name.lstrip("-").replace("_", "-"),
        dest=name.lstrip("-").replace("-", "_"),
        action="store_false",
    )
    parser.set_defaults(**{name.lstrip("-").replace("-", "_"): default})


def main():
    p = argparse.ArgumentParser(description="Fig.2(g) ablation radar (Nature-ish polished)")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--summary_csv", type=str, default=None)
    p.add_argument("--force_recompute", action="store_true")

    _add_bool_arg(p, "--trim_noconstraint", default=False)
    p.add_argument("--trim_q_low", type=float, default=1.0)
    p.add_argument("--trim_q_high", type=float, default=99.0)

    p.add_argument(
        "--metrics",
        type=str,
        default="macro_MMD2,macro_MAE,macro_RMSE,vet_pass_rate,neg_violation_mean,eq_region_mse,eq_variable_mse",
    )

    p.add_argument("--show_delta", action="store_true", help="是否绘制 Δ%（默认不画，避免重叠）")
    p.add_argument("--delta_thresh", type=float, default=0.08)
    p.add_argument("--floor", type=float, default=0.08)
    p.add_argument("--ceiling", type=float, default=1.0)
    p.add_argument("--gap_tau_floor", type=float, default=0.03)
    p.add_argument("--gap_boost_gamma", type=float, default=0.55, help="Sem-Gap 饱满增强：<1 更鼓")
    _add_bool_arg(p, "--show_yticks", default=True)
    args = p.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = args.out or os.path.join(repo_root, "figs", "fig2g", "fig2g_ablation_radar_nature_polished.svg")
    summary_csv = args.summary_csv or os.path.join(repo_root, "figs", "fig2g", "fig2g_ablation_metrics_summary.csv")
    excluded_model_keys = {
        "linear_denoise",
        "no_cat_cond", "no_region_cond", "no_var_cond",
        "full_no_cat_cond", "full_no_region_cond", "full_no_var_cond",
        "nocatcond", "noregioncond", "novarcond",
        "full_nocatcond", "full_noregioncond", "full_novarcond",
    }

    def _norm_model_key(name):
        return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

    if (not args.force_recompute) and os.path.isfile(summary_csv):
        df_cached = pd.read_csv(summary_csv)
        if "model_name" in df_cached.columns:
            keep_mask = ~df_cached["model_name"].map(_norm_model_key).isin(excluded_model_keys)
            df_cached = df_cached.loc[keep_mask].reset_index(drop=True)
        req = [m.strip() for m in args.metrics.split(",") if m.strip()]
        metric_cols = [c for c in req if c in df_cached.columns and np.any(np.isfinite(pd.to_numeric(df_cached[c], errors="coerce")))]
        metric_cols = [c for c in metric_cols if c != "gap_real_minus_syn"]
        if len(metric_cols) >= 2:
            _plot_radar_nature_polished(
                df_cached[["model_name"] + metric_cols],
                metric_cols,
                out_path,
                show_delta=args.show_delta,
                delta_thresh=args.delta_thresh,
                floor=args.floor,
                ceiling=args.ceiling,
                gap_tau_floor=args.gap_tau_floor,
                gap_boost_gamma=args.gap_boost_gamma,
                show_yticks=args.show_yticks,
            )
            print(f"[plot_fig2g] cache hit: {summary_csv}")
            print(f"[plot_fig2g] saved: {out_path}")
            return

    run_dirs = _discover_run_dirs(args.data_dir)
    if not run_dirs:
        print("[plot_fig2g] 未找到 fig2_data 目录")
        return

    reg_csv = os.path.join(repo_root, "data", "dataset", "SPM", "变量编码对应表", "dim_Region_encoding.csv")
    var_csv = os.path.join(repo_root, "data", "dataset", "SPM", "变量编码对应表", "dim_Variable_encoding.csv")
    vb_csv = os.path.join(repo_root, "data", "dataset", "SPM", "variable_baseline.csv")
    rb_csv = os.path.join(repo_root, "data", "dataset", "SPM", "region_baseline.csv")
    _, reg_n2c, _, var_n2c = _vet_load_maps(reg_csv, var_csv)
    VAR_REL_ERR, REG_REL_ERR = _vet_load_baselines(vb_csv, rb_csv)

    rows = []
    for rd in run_dirs:
        model_name = _parse_model_name(rd)
        if _norm_model_key(model_name) in excluded_model_keys:
            continue

        row = {"model_name": model_name}
        row.update(_read_training_record_metrics(rd))
        row["gap_real_minus_syn"] = _compute_gap_real_minus_syn(rd)

        do_trim = bool(args.trim_noconstraint and model_name == "NoConstraint")
        row.update(
            _compute_constraint_triplet_and_vet_trimmed(
                rd, reg_n2c, var_n2c, VAR_REL_ERR, REG_REL_ERR,
                apply_trim=do_trim, q_low=args.trim_q_low, q_high=args.trim_q_high
            )
        )
        rows.append(row)

    if not rows:
        raise RuntimeError("过滤后无可用模型（已跳过 LinearDenoise/NoCatCond/NoRegionCond/NoVarCond）。")

    df_raw = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    df_raw.to_csv(summary_csv, index=False)

    req = [m.strip() for m in args.metrics.split(",") if m.strip()]
    metric_cols = [c for c in req if c in df_raw.columns and np.any(np.isfinite(pd.to_numeric(df_raw[c], errors="coerce")))]
    metric_cols = [c for c in metric_cols if c != "gap_real_minus_syn"]
    
    if len(metric_cols) < 2:
        raise RuntimeError(f"有效指标不足：{metric_cols}")

    _plot_radar_nature_polished(
        df_raw[["model_name"] + metric_cols],
        metric_cols,
        out_path,
        show_delta=args.show_delta,
        delta_thresh=args.delta_thresh,
        floor=args.floor,
        ceiling=args.ceiling,
        gap_tau_floor=args.gap_tau_floor,
        gap_boost_gamma=args.gap_boost_gamma,
        show_yticks=args.show_yticks,
    )

    print(f"[plot_fig2g] saved radar: {out_path}")
    print(f"[plot_fig2g] saved summary: {summary_csv}")

if __name__ == "__main__":
    main()