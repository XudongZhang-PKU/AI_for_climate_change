# -*- coding: utf-8 -*-
"""
================================================================================
Fig.2(b) Sector Circular Heatmaps (DOUBLE-RING) — Nature Style Edition
================================================================================
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

# ==========================================
# Nature 风格全局设定
# ==========================================
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Roboto', 'DejaVu Sans']
mpl.rcParams['font.size'] = 21
mpl.rcParams['pdf.fonttype'] = 42  # 保证在 Illustrator 中文字可编辑
mpl.rcParams['ps.fonttype'] = 42

# -------- fixed paths --------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORR_REAL_PATH = os.path.join(REPO_ROOT, "fig2_data_prepare_Full_1234", "fig2_data", "corr_real.npy")
CORR_SYN_PATH  = os.path.join(REPO_ROOT, "fig2_data_prepare_Full_1234", "fig2_data", "corr_syn.npy")
OUT_DIR        = os.path.join(REPO_ROOT, "figs", "fig2b")

REGION_CSV = os.path.join(REPO_ROOT, "data", "dataset", "SPM", "变量编码对应表", "dim_Region_encoding.csv")
VAR_CSV    = os.path.join(REPO_ROOT, "data", "dataset", "SPM", "变量编码对应表", "dim_Variable_encoding.csv")

R = 11
V = 48
N = R * V

# contrast knob for |rho|
ROBUST_P = 99.5
ABS_FLOOR = 0.08  
COLOR_MIN = 0.20  

# sector layout
SECTOR_GAP_RAD = np.deg2rad(1.8)  
RING_REAL = (0.4, 0.68)          
RING_SYN  = (0.7, 0.98)          

# label density
LABEL_EVERY_VARIABLE = 2          
LABEL_EVERY_REGION   = 1          

# colorbar placement 
CB_LEFT = 0.91
CB_W = 0.018
CB_H = 0.26
CB_GAP = 0.06
CB_BOTTOM_REAL = 0.20
CB_BOTTOM_SYN = CB_BOTTOM_REAL + CB_H + CB_GAP


def rgb255(r, g, b):
    return (r/255.0, g/255.0, b/255.0)


def green_5_cmap():
    """
    Nature 风格顺序色条 (Real 环)
    基于 Nature Publishing Group (NPG) 经典青绿色 #00A087
    """
    cols = [
        rgb255(244, 250, 249),  # 极浅青灰 (底色)
        rgb255(184, 225, 217),  # 浅青绿
        rgb255(92,  192, 176),  # 中青绿
        rgb255(0,   160, 135),  # NPG 经典青绿 (#00A087)
        rgb255(0,   102, 86),   # 极暗深绿 
    ]
    return LinearSegmentedColormap.from_list("nature_npg_green", cols, N=256)


def blue_5_cmap():
    """
    Nature 风格顺序色条 (Syn 环)
    基于 Nature Publishing Group (NPG) 经典深空蓝 #3C5488
    """
    cols = [
        rgb255(242, 244, 247),  # 极浅蓝灰 (底色)
        rgb255(191, 201, 217),  # 浅灰蓝
        rgb255(125, 142, 176),  # 中灰蓝
        rgb255(60,  84,  136),  # NPG 经典深蓝 (#3C5488)
        rgb255(36,  50,  82),   # 极暗深蓝
    ]
    return LinearSegmentedColormap.from_list("nature_npg_blue", cols, N=256)


def robust_abs_vmax(vals, p=99.5, floor=0.08):
    x = np.asarray(vals, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    vmax = float(np.percentile(np.abs(x), p))
    return max(vmax, floor)


def load_value_labels(csv_path, expected_n):
    df = pd.read_csv(csv_path)
    assert "Code" in df.columns and "Value" in df.columns, f"Bad mapping file: {csv_path}"
    df = df.sort_values("Code")
    codes = df["Code"].to_numpy().astype(int)
    vals  = df["Value"].astype(str).to_list()

    out = [""] * (codes.max() + 1)
    for c, v in zip(codes, vals):
        if 0 <= c < len(out):
            out[c] = v
    if len(out) < expected_n:
        out = out + [f"UNK{i}" for i in range(len(out), expected_n)]
    return out[:expected_n]


def short_label(s, max_len=12):
    s = str(s)
    if "|" in s:
        s = s.split("|")[-1]
    s = s.strip()
    return s if len(s) <= max_len else (s[:max_len-1] + "…")


def clean_region_label(s):
    s = str(s)
    if "|" in s:
        s = s.split("|")[-1]
    s = s.strip()
    s = re.sub(r"^R10", "", s, flags=re.IGNORECASE)
    return s


def idx_rv(r, v):
    return r * V + v


def within_region_indices(r):
    return np.array([idx_rv(r, v) for v in range(V)], dtype=int)


def within_variable_indices(v):
    return np.array([idx_rv(r, v) for r in range(1, R)], dtype=int)


def pcolormesh_sector_matrix(ax, M, theta0, theta1, r_in, r_out, cmap, norm):
    K = M.shape[0]
    theta_edges = np.linspace(theta0, theta1, K + 1)
    r_edges = np.linspace(r_in, r_out, K + 1)
    TH, RR = np.meshgrid(theta_edges, r_edges, indexing="ij")
    ax.pcolormesh(TH, RR, M, cmap=cmap, norm=norm, shading="auto", linewidth=0.0, rasterized=True)


def add_sector_name_radial(ax, theta0, theta1, name, r_label=1.01, fontsize=12, alpha=0.9):
    theta_mid = 0.5 * (theta0 + theta1)
    rot = np.degrees(theta_mid) - 90.0
    if rot > 90.0:
        rot -= 180.0
    elif rot < -90.0:
        rot += 180.0
    ax.text(
        theta_mid, r_label, name,
        ha="center", va="center",
        rotation=rot, rotation_mode="anchor",
        fontsize=fontsize, alpha=alpha,
    )


def style_colorbar(cb, label):
    """提取统一的 colorbar 美化逻辑"""
    cb.set_ticks([0.0, 0.4, 0.6, 0.8, 1.0])
    cb.set_label(label, fontsize=18, labelpad=12)
    cb.ax.tick_params(labelsize=18, length=0) # length=0 去除刻度短线
    cb.outline.set_visible(False) # 去除黑框，极致轻盈


def plot_sector_double_by_region_abs(corr_real, corr_syn, out_path, region_names, cmap_real, cmap_syn, norm):
    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_axes([0.04, 0.05, 0.82, 0.90], projection="polar")
    ax.set_axis_off()

    regions = list(range(1, R))  
    n_data = len(regions)
    sector_width = 2 * np.pi / n_data

    for i, r in enumerate(regions):
        base0 = i * sector_width
        base1 = (i + 1) * sector_width
        gap = min(SECTOR_GAP_RAD, 0.5 * (base1 - base0) * 0.9)
        theta0 = base0 + gap / 2
        theta1 = base1 - gap / 2

        I = within_region_indices(r)
        M_real = np.abs(corr_real[np.ix_(I, I)])
        M_syn  = np.abs(corr_syn [np.ix_(I, I)])
        region_label = clean_region_label(region_names[r])

        pcolormesh_sector_matrix(ax, M_real, theta0, theta1, RING_REAL[0], RING_REAL[1], cmap_real, norm)
        pcolormesh_sector_matrix(ax, M_syn,  theta0, theta1, RING_SYN[0],  RING_SYN[1],  cmap_syn, norm)

        if (i % LABEL_EVERY_REGION) == 0:
            add_sector_name_radial(
                ax,
                theta0,
                theta1,
                region_label,
                r_label=1.02,
                fontsize=18,
                alpha=0.9
            )

    # 渲染 Real 侧 Colorbar
    sm_real = plt.cm.ScalarMappable(cmap=cmap_real, norm=norm)
    sm_real.set_array([])
    cax_real = fig.add_axes([CB_LEFT, CB_BOTTOM_REAL, CB_W, CB_H])
    cb_real = plt.colorbar(sm_real, cax=cax_real)
    style_colorbar(cb_real, r"Real $|\rho|$")

    # 渲染 Syn 侧 Colorbar
    sm_syn = plt.cm.ScalarMappable(cmap=cmap_syn, norm=norm)
    sm_syn.set_array([])
    cax_syn = fig.add_axes([CB_LEFT, CB_BOTTOM_SYN, CB_W, CB_H])
    cb_syn = plt.colorbar(sm_syn, cax=cax_syn)
    style_colorbar(cb_syn, r"Syn $|\rho|$")

    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    region_values = load_value_labels(REGION_CSV, expected_n=R)
    var_values    = load_value_labels(VAR_CSV, expected_n=V)

    corr_real = np.load(CORR_REAL_PATH)
    corr_syn  = np.load(CORR_SYN_PATH)
    
    # stabilize + clip
    corr_real = np.clip((corr_real + corr_real.T) / 2.0, -1.0, 1.0)
    corr_syn  = np.clip((corr_syn  + corr_syn.T)  / 2.0, -1.0, 1.0)

    # compute vmax 
    vals = []
    for r in range(1, R):  
        I = within_region_indices(r)
        vals.append(np.abs(corr_real[np.ix_(I, I)]).ravel())
        vals.append(np.abs(corr_syn [np.ix_(I, I)]).ravel())
    for v in range(V):
        I = within_variable_indices(v)
        vals.append(np.abs(corr_real[np.ix_(I, I)]).ravel())
        vals.append(np.abs(corr_syn [np.ix_(I, I)]).ravel())
    vals = np.concatenate(vals, axis=0)

    vmax = robust_abs_vmax(vals, p=ROBUST_P, floor=ABS_FLOOR)
    vmax = max(vmax, COLOR_MIN + 1e-6)
    
    norm = PowerNorm(gamma=2.2, vmin=0.0, vmax=vmax)
    cmap_real = green_5_cmap()
    cmap_syn = blue_5_cmap()

    # ---- by region ----
    out1 = os.path.join(OUT_DIR, "fig2b_sector_by_region_double_abs.svg")
    fig1 = plot_sector_double_by_region_abs(corr_real, corr_syn, out1, region_values, cmap_real, cmap_syn, norm)
    
    # 增加透明度和高 DPI 支持，确保出图极致干净
    fig1.savefig(out1, bbox_inches="tight", transparent=False, dpi=300)
    plt.close(fig1)
    print(f"[Nature Style Saved]: {out1}")

if __name__ == "__main__":
    main()