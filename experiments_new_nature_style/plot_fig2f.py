# -*- coding: utf-8 -*-
"""
================================================================================
Fig.2(f) Category prediction (Semantic validity) — ONE FIG (nested circles + ribbons)
================================================================================

读取：
- X_train.npy, y_train.npy   (real-train: summary features + category)
- X_syn.npy, y_syn.npy       (syn-test:  summary features + category)
- fig2_raw_test.npz + fig2a_real_syn.npz 现场构造 real-test (summary features + category)

训练：
- NearestCentroid on real-train (with StandardScaler)

评估：
- syn-test  + real-test (if available)

输出：
- ONE figure: figs/fig2f/fig2f_onefig_nestedCircles_ribbons.png
- CSV:        figs/fig2f/fig2f_onefig_nestedCircles_ribbons_model_feature_sweep.csv

说明（Nature Publication Style）：
1) Confusion matrix 的每个 cell 用双圆 Dodge 排布：
   - 左圆 = Real-test 的 row-normalized 概率（颜色 + 大小）
   - 右圆 = Syn-test  的 row-normalized 概率（颜色 + 大小）
2) 不写 cell 数字（避免样本量不一致带来的困扰）
3) 不画图标题
4) 两个边缘采用 Ribbon/Profile 带状分布代替柱状图（折线+半透明填充）
5) 采用 Okabe-Ito 色盲友好配色和 Arial 无衬线字体，满足顶刊要求。

运行：
python plot_fig2f_onefig.py --data_dir /path/to/fig2_data
或直接不传 --data_dir，让它自动在 repo_root 下找候选目录。
================================================================================
"""

import os
import sys
import argparse
import csv
import numpy as np

from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.font_manager as fm

from nature_plot_style import save_dual


# =============================================================================
# Feature build for real-test
# =============================================================================
def _summary_ts(values_bt):
    return float(np.sum(np.asarray(values_bt)))


def _load_crv_stats_lut(data_dir):
    stats_path = os.path.join(data_dir, "fig2a_real_syn.npz")
    if not os.path.exists(stats_path):
        return {}
    d = np.load(stats_path)
    keys = np.asarray(d["keys"], dtype=np.int64)
    means = np.asarray(d["train_means"], dtype=np.float64)
    stds = np.asarray(d["train_stds"], dtype=np.float64)
    lut = {}
    n = min(len(keys), len(means), len(stds))
    for i in range(n):
        c, r, v = int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2])
        lut[(c, r, v)] = (float(means[i]), float(stds[i]))
    return lut


def _build_real_test_from_raw(data_dir, n_vars):
    raw_path = os.path.join(data_dir, "fig2_raw_test.npz")
    if not os.path.exists(raw_path):
        return None, None
    lut = _load_crv_stats_lut(data_dir)
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
                c = int(lab[g, 0, 1])
                r = int(lab[g, 0, 2])
                v = int(lab[g, 0, 3])
                if v != vc:
                    continue
                key = (c, r, v)
                if key not in lut:
                    continue
                mu, sd = lut[key]
                real_ts = lab[g, :, 4] * sd + mu
                val = _summary_ts(real_ts)
                break
            row.append(val)
        x_rows.append(row)
        y_rows.append(int(lab[0, 0, 1]))
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.int64)


# =============================================================================
# Path resolve
# =============================================================================
def _resolve_data_dir(arg_data_dir):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = []
    if arg_data_dir:
        candidates.append(arg_data_dir)
    candidates.extend([
        os.path.join(repo_root, "fig2_data_prepare_Full_1234", "fig2_data"),
        os.path.join(repo_root, "fig2_data_prepare_Full_1234", "output", "fig2_data"),
        os.path.join(repo_root, "output", "fig2_data"),
    ])
    required = ["X_train.npy", "y_train.npy", "X_syn.npy", "y_syn.npy"]
    for d in candidates:
        if all(os.path.exists(os.path.join(d, n)) for n in required):
            return d
    raise FileNotFoundError(
        "找不到 Fig2(f) 所需数据目录。请传 --data_dir，且目录需包含: "
        + ", ".join(required)
        + "。已检查路径: "
        + "; ".join(candidates)
    )


def _resolve_output_dir():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, "figs", "fig2f")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# =============================================================================
# Feature transform & stats
# =============================================================================
def _build_models():
    return {
        "nearest_centroid": {
            "estimator": NearestCentroid(),
            "use_scaler": True,
        },
    }


def _transform_features(x, mode):
    x = np.asarray(x, dtype=np.float64)
    if mode == "sum":
        return x
    sign_log = np.sign(x) * np.log1p(np.abs(x))
    if mode == "signed_log1p":
        return sign_log
    sign_sqrt = np.sign(x) * np.sqrt(np.abs(x))
    if mode == "signed_sqrt":
        return sign_sqrt
    if mode == "sum_plus_log":
        return np.concatenate([x, sign_log], axis=1)
    if mode == "sum_plus_log_sqrt":
        return np.concatenate([x, sign_log, sign_sqrt], axis=1)
    raise ValueError(f"未知 feature_mode: {mode}")


def _safe_percentile(v, q):
    if v.size == 0:
        return float("nan")
    return float(np.percentile(v, q))


def _log_feature_stats(tag, x):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    finite_mask = np.isfinite(x)
    finite_cnt = int(np.sum(finite_mask))
    nan_cnt = int(np.sum(np.isnan(x)))
    inf_cnt = int(np.sum(np.isinf(x)))
    finite_vals = x[finite_mask]
    mean_v = float(np.mean(finite_vals)) if finite_cnt > 0 else float("nan")
    std_v = float(np.std(finite_vals)) if finite_cnt > 0 else float("nan")
    p01 = _safe_percentile(finite_vals, 1)
    p50 = _safe_percentile(finite_vals, 50)
    p99 = _safe_percentile(finite_vals, 99)
    pass # 省略冗长的输出


# =============================================================================
# ONE-FIG plotter (nested circles + ribbons) - NATURE STYLE
# =============================================================================
def plot_fig2f_onefig_nested_circles_ribbons(
    cm_real_counts: np.ndarray,
    cm_syn_counts: np.ndarray,
    out_path: str,
    class_names=None,
    top_band_h=1.00,
    right_band_w=1.30,
    bubble_alpha=0.85,
    bubble_edge_lw=0.6,
    dodge=0.18,                 
    r_min=0.045,
    r_max=0.38,
    gamma=0.62,
    show_grid=True,
    grid_lw=0.5,
    tick_fs=13.5,
    label_fs=15,
    dpi=600, # 顶刊要求的 600 DPI
):
    """
    Nature Style Plot:
    - Ribbon/Profile bands for Precision and Recall
    - Okabe-Ito Colorblind Safe Palette
    - Arial / Sans-serif globally
    """
    # --- 全局设定 Nature 期刊样式 ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts for PDF
    plt.rcParams['ps.fonttype'] = 42

    cm_real = np.asarray(cm_real_counts, dtype=np.float64)
    cm_syn  = np.asarray(cm_syn_counts, dtype=np.float64)
    n = cm_real.shape[0]

    def _row_normalize(cm, eps=1e-12):
        s = cm.sum(axis=1, keepdims=True)
        return cm / (s + eps)

    def _safe_div(a, b, eps=1e-12):
        return a / (b + eps)

    def _prec_recall_from_cm_counts(cm_counts):
        tp = np.diag(cm_counts)
        fp = cm_counts.sum(axis=0) - tp
        fn = cm_counts.sum(axis=1) - tp
        prec = _safe_div(tp, tp + fp)
        rec  = _safe_div(tp, tp + fn)
        return prec, rec

    def _value_to_radius(v):
        v = float(np.clip(v, 0.0, 1.0))
        return r_min + (r_max - r_min) * (v ** gamma)

    R = _row_normalize(cm_real) 
    S = _row_normalize(cm_syn)   
    prec_r, rec_r = _prec_recall_from_cm_counts(cm_real)
    prec_s, rec_s = _prec_recall_from_cm_counts(cm_syn)

    # Nature/Science 色盲友好搭配 (Okabe-Ito Palette)
    REAL_COLOR = '#0072B2'  # 纯粹科学蓝
    SYN_COLOR  = '#009E73'  # 纯粹科学青绿
    EDGE_COLOR = '#333333'  # 深灰描边

    fig = plt.figure(figsize=(8.5, 7.5))
    ax = fig.add_axes([0.16, 0.20, 0.62, 0.62])
    ax.set_aspect("equal")

    ax.set_xlim(-0.35, n + right_band_w + 0.85)
    ax.set_ylim(n + 0.55, -top_band_h - 0.55)

    # =========================
    # Main matrix: TWO bubbles per cell
    # =========================
    for i in range(n):
        for j in range(n):
            vr, vs = float(R[i, j]), float(S[i, j])
            rr, rs = _value_to_radius(vr), _value_to_radius(vs)
            cx, cy = j + 0.5, i + 0.5
            
            # Real
            ax.add_patch(patches.Circle(
                (cx - dodge, cy), radius=rr, facecolor=REAL_COLOR, 
                edgecolor=EDGE_COLOR, lw=bubble_edge_lw, alpha=bubble_alpha
            ))
            # Syn
            ax.add_patch(patches.Circle(
                (cx + dodge, cy), radius=rs, facecolor=SYN_COLOR, 
                edgecolor=EDGE_COLOR, lw=bubble_edge_lw, alpha=bubble_alpha
            ))

    # Grid - 极其纤细干净的网格
    if show_grid:
        for k in range(n + 1):
            ax.plot([0, n], [k, k], color='#E0E0E0', lw=grid_lw, zorder=0)
            ax.plot([k, k], [0, n], color='#E0E0E0', lw=grid_lw, zorder=0)

    # Ticks
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(class_names, fontsize=tick_fs)
    ax.set_yticklabels(class_names, fontsize=tick_fs)
    ax.tick_params(axis='x', which='both', length=0, pad=1)
    ax.tick_params(axis='y', which='both', length=0, pad=1)
    ax.set_xlabel("Predicted Class", fontsize=label_fs, labelpad=4)
    ax.set_ylabel("True Class", fontsize=label_fs, labelpad=4)

    # =========================
    # Top band: Precision RIBBON (带状图)
    # =========================
    y_pad_top = 0.1
    y_pad_bottom = 0.1
    y_base = 0.0 - y_pad_bottom
    y_top  = -top_band_h + y_pad_top
    band_h = y_base - y_top

    x_centers = np.arange(n) + 0.5
    y_prec_r = y_base - np.clip(prec_r, 0, 1) * band_h
    y_prec_s = y_base - np.clip(prec_s, 0, 1) * band_h

    # Ribbon - Real (Line + Shaded Area)
    ax.plot(x_centers, y_prec_r, color=REAL_COLOR, lw=1.5, marker='o', markersize=3)
    ax.fill_between(x_centers, y_base, y_prec_r, facecolor=REAL_COLOR, alpha=0.2)
    # Ribbon - Syn
    ax.plot(x_centers, y_prec_s, color=SYN_COLOR, lw=1.5, marker='s', markersize=3)
    ax.fill_between(x_centers, y_base, y_prec_s, facecolor=SYN_COLOR, alpha=0.2)

    ax.text((x_centers[0] + x_centers[-1]) / 2, y_top - 0.28, "Precision", fontsize=label_fs-1, ha="center", va="center")

    # =========================
    # Right band: Recall RIBBON (带状图)
    # =========================
    x_pad_left = 0.25
    x0 = n + x_pad_left
    x1 = n + right_band_w - 0.1
    band_w = x1 - x0

    y_centers = np.arange(n) + 0.5
    x_rec_r = x0 + np.clip(rec_r, 0, 1) * band_w
    x_rec_s = x0 + np.clip(rec_s, 0, 1) * band_w

    # Ribbon - Real
    ax.plot(x_rec_r, y_centers, color=REAL_COLOR, lw=1.5, marker='o', markersize=3)
    ax.fill_betweenx(y_centers, x0, x_rec_r, facecolor=REAL_COLOR, alpha=0.2)
    # Ribbon - Syn
    ax.plot(x_rec_s, y_centers, color=SYN_COLOR, lw=1.5, marker='s', markersize=3)
    ax.fill_betweenx(y_centers, x0, x_rec_s, facecolor=SYN_COLOR, alpha=0.2)

    ax.text(x1 + 0.24, (y_centers[0] + y_centers[-1]) / 2, "Recall", fontsize=label_fs-1, ha="center", va="center", rotation=90)

    # =========================
    # 极简 Nature 风图例
    # =========================
    lx = n + 0.30
    ly = -top_band_h - 0.22
    
    # Real 标记
    ax.plot([lx, lx+0.2], [ly, ly], color=REAL_COLOR, lw=1.5, marker='o', markersize=4)
    ax.text(lx + 0.3, ly, "Real", fontsize=label_fs-1, ha="left", va="center")
    
    # Syn 标记
    ly += 0.38
    ax.plot([lx, lx+0.2], [ly, ly], color=SYN_COLOR, lw=1.5, marker='s', markersize=4)
    ax.text(lx + 0.3, ly, "Syn", fontsize=label_fs-1, ha="left", va="center")

    # 隐藏所有 spine
    for sp in ax.spines.values():
        sp.set_visible(False)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    saved = save_dual(fig, out_path, dpi=dpi, transparent=False)
    plt.close(fig)
    return saved

# =============================================================================
# Main (保留你原始的逻辑，仅在最后的绘图调用部分配合修改即可)
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Fig.2(f) train on real / test on syn + real (ONE FIG)")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--out_path", type=str, default=None, help="output svg path (optional)")
    p.add_argument("--format", type=str, default="svg", choices=["svg", "png", "pdf"], help="output format when --out_path is not set")
    args = p.parse_args()

    data_dir = _resolve_data_dir(args.data_dir)
    out_dir = _resolve_output_dir()

    # load
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_syn   = np.load(os.path.join(data_dir, "X_syn.npy"))
    y_syn   = np.load(os.path.join(data_dir, "y_syn.npy"))

    X_test_real, y_test_real = _build_real_test_from_raw(data_dir, n_vars=X_train.shape[1])
    has_real_test = X_test_real is not None and y_test_real is not None

    X_train_base = np.nan_to_num(X_train, nan=0.0)
    X_syn_base   = np.nan_to_num(X_syn, nan=0.0)
    if has_real_test:
        X_test_real_base = np.nan_to_num(X_test_real, nan=0.0)

    feature_modes = ["sum"]
    models = _build_models()

    results = []
    for fmode in feature_modes:
        X_train_f = _transform_features(X_train_base, fmode)
        X_syn_f   = _transform_features(X_syn_base, fmode)
        if has_real_test:
            X_test_real_f = _transform_features(X_test_real_base, fmode)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_f)
        X_syn_s   = scaler.transform(X_syn_f)
        if has_real_test:
            X_test_real_s = scaler.transform(X_test_real_f)

        for name, spec in models.items():
            try:
                clf = clone(spec["estimator"])
                xtr = X_train_s if spec["use_scaler"] else X_train_f
                xsy = X_syn_s   if spec["use_scaler"] else X_syn_f

                clf.fit(xtr, y_train)

                y_pred_syn = clf.predict(xsy)
                syn_f1 = float(f1_score(y_syn, y_pred_syn, average="macro"))

                row = {
                    "feature_mode": fmode, "model": name, "syn_f1": syn_f1,
                    "y_pred_syn": y_pred_syn, "clf": clf,
                    "scaler": scaler if spec["use_scaler"] else None,
                    "use_scaler": spec["use_scaler"],
                }

                if has_real_test:
                    xte = X_test_real_s if spec["use_scaler"] else X_test_real_f
                    y_pred_real = clf.predict(xte)
                    real_f1 = float(f1_score(y_test_real, y_pred_real, average="macro"))
                    row["real_f1"] = real_f1
                    row["gap_real_minus_syn"] = real_f1 - syn_f1
                    row["y_pred_real"] = y_pred_real

                results.append(row)
            except Exception as e:
                print(f"Skipped: {e}")

    out_csv = os.path.join(out_dir, "fig2f_onefig_nestedCircles_ribbons_model_feature_sweep.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        if has_real_test:
            w.writerow(["feature_mode", "model", "macro_f1_syn_test", "macro_f1_real_test", "gap_real_minus_syn"])
            for row in sorted(results, key=lambda x: x["syn_f1"], reverse=True):
                w.writerow([row["feature_mode"], row["model"], row["syn_f1"], row["real_f1"], row["gap_real_minus_syn"]])
        else:
            w.writerow(["feature_mode", "model", "macro_f1_syn_test"])
            for row in sorted(results, key=lambda x: x["syn_f1"], reverse=True):
                w.writerow([row["feature_mode"], row["model"], row["syn_f1"]])

    best_syn = max(results, key=lambda x: x["syn_f1"])
    cm_syn = confusion_matrix(y_syn, best_syn["y_pred_syn"])

    if not has_real_test:
        return

    best_real = max(results, key=lambda x: x.get("real_f1", -1e9))
    cm_real = confusion_matrix(y_test_real, best_real["y_pred_real"])

    class_names = [f"C{i + 1}" for i in range(cm_syn.shape[0])]
    if args.out_path:
        out_path = args.out_path
    else:
        ext = args.format
        out_path = os.path.join(out_dir, f"fig2f_onefig_nestedCircles_ribbons.{ext}")

    saved = plot_fig2f_onefig_nested_circles_ribbons(
        cm_real_counts=cm_real,
        cm_syn_counts=cm_syn,
        out_path=out_path,
        class_names=class_names,
        dpi=600, # 确保输出高分辨率
    )

    print("[plot_fig2f] 已保存 ONE FIG (Nature Style): " + ", ".join(saved))


if __name__ == "__main__":
    main()
