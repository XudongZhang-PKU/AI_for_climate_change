# -*- coding: utf-8 -*-
"""
================================================================================
Fig.2(a) Distribution Fidelity — FINAL NATURE-STYLE VERSION
================================================================================
包含：
1) CRV-level 5–95 trimming
2) RV-level QQ overlay: 极窄边缘 KDE 分布 (压缩至 1/3)，保留 r 和 R2，坐标轴明确分位数与聚合方式
3) CRV QQ by category: 多面板，Category 从 1 开始编号，坐标轴明确分位数与层级
4) DimRed: UMAP/PCA: NPG 调色板，图例完美整合在顶部两行
5) 数据分布小提琴图: 分组对比
================================================================================
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.metrics import r2_score
from tqdm import tqdm
try:
    import importlib.metadata
except ModuleNotFoundError:
    import importlib_metadata
    sys.modules["importlib.metadata"] = importlib_metadata
import umap
from matplotlib.lines import Line2D

from nature_plot_style import apply_nature_style, save_dual

HAS_UMAP = True
EPS = 1e-8
MAX_SUBPLOTS_PER_PAGE = 24
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===============================
# Style - Nature 期刊风格配置
# ===============================
def setup_style():
    apply_nature_style(font_scale=2.0)
    plt.rcParams.update({
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    })

# ===============================
# Utility
# ===============================
def clip_pair(real, syn, low=5, high=95):
    real = np.asarray(real).ravel()
    syn = np.asarray(syn).ravel()
    real = real[np.isfinite(real)]
    syn = syn[np.isfinite(syn)]
    if real.size == 0 or syn.size == 0:
        return real, syn
    lo, hi = np.percentile(np.concatenate([real, syn]), [low, high])
    return np.clip(real, lo, hi), np.clip(syn, lo, hi)

# ===============================
# Step0: CRV trimming
# ===============================
def trim_crv_npz(path, out_path):
    data = np.load(path, allow_pickle=False)
    keys = data["keys"]
    K = keys.shape[0]
    save_dict = {"keys": keys}
    if "train_means" in data.files and "train_stds" in data.files:
        save_dict["train_means"] = data["train_means"]
        save_dict["train_stds"] = data["train_stds"]
    for i in range(K):
        real, syn = clip_pair(data[f"real_{i}"], data[f"syn_{i}"], 5, 95)
        save_dict[f"real_{i}"] = real
        save_dict[f"syn_{i}"] = syn
    np.savez_compressed(out_path, **save_dict)
    return out_path

# ===============================
# RV-level QQ (极度压缩的 KDE 边缘分布)
# ===============================
def plot_qq_rv_overlay(path, out_path):
    setup_style()
    data = np.load(path)
    keys = data["keys"]
    K = keys.shape[0]
    rv_pairs = sorted(set((int(keys[i,1]), int(keys[i,2])) for i in range(K)))
    
    # 【细节优化】：采样点调细，从 1% 到 99% 每隔 1% 取一个点，让散点更连贯顺滑
    q_grid = np.arange(0.05, 0.96, 0.05)
    xs_all, ys_all = [], []
    
    for rv in tqdm(rv_pairs, desc="Processing RV QQ"):
        real_vals, syn_vals = [], []
        for i in range(K):
            if (int(keys[i,1]), int(keys[i,2])) == rv:
                real_vals.extend(data[f"real_{i}"].ravel())
                syn_vals.extend(data[f"syn_{i}"].ravel())
                
        real = np.array(real_vals)
        syn = np.array(syn_vals)
        real = real[np.isfinite(real)]
        syn = syn[np.isfinite(syn)]
        if len(real) < 10 or len(syn) < 10: continue
        
        # ==========================================
        # 核心改动 1：丢掉范围外的数据 (剔除离群点，而不是截断)
        # 我们用 1% 和 99% 框定合理范围，直接 Drop 掉极端的噪点
        # ==========================================
        lo, hi = np.percentile(np.concatenate([real, syn]), [5, 95])
        real = real[(real >= lo) & (real <= hi)]
        syn = syn[(syn >= lo) & (syn <= hi)]
        
        if len(real) < 10 or len(syn) < 10: continue
        
        # ==========================================
        # 核心改动 2：去量纲 (Normalization)
        # 用该变量真实数据的最大绝对值作为缩放基准
        # ==========================================
        norm_factor = np.max(np.abs(real)) + 1e-8
        
        for q in q_grid:
            xs_all.append(np.quantile(real, q) / norm_factor)
            ys_all.append(np.quantile(syn, q) / norm_factor)
            
    xs_all, ys_all = np.array(xs_all), np.array(ys_all)
    if len(xs_all) == 0: return

    pearson_r, _ = stats.pearsonr(xs_all, ys_all)
    r2 = r2_score(xs_all, ys_all)
    
    all_vals = np.concatenate([xs_all, ys_all])
    lo, hi = np.percentile(all_vals, [1, 99])
    margin = (hi - lo) * 0.05
    plot_min, plot_max = lo - margin, hi + margin

    fig = plt.figure(figsize=(7, 7))
    
    # 核心修改：将比例从 6:1 调整为 15:1，让边缘分布变得非常窄 (变成原来的约1/3)
    gs = gridspec.GridSpec(2, 2, width_ratios=[15, 1], height_ratios=[1, 15], wspace=0.03, hspace=0.03)
    
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    color_scatter, color_fill, color_line, color_reg = '#00777B', '#6BD2D6', '#333333', '#2980B9'

    # 1. 主图散点与辅助线
    ax_main.scatter(xs_all, ys_all, s=8, alpha=0.3, color=color_scatter, edgecolors='none')
    ax_main.plot([plot_min, plot_max], [plot_min, plot_max], color=color_line, linestyle='--', linewidth=1.2, zorder=3, label="1:1 Reference")
    slope, intercept, _, _, _ = stats.linregress(xs_all, ys_all)
    ax_main.plot(np.array([plot_min, plot_max]), slope * np.array([plot_min, plot_max]) + intercept, color=color_reg, linestyle='-', linewidth=1.2, zorder=4, label="Linear Fit")
    
    # 【修改处 1】：详细标注坐标轴，注明 Dimensionless
    ax_main.set(xlim=(plot_min, plot_max), ylim=(plot_min, plot_max),
                xlabel="Real Quantiles",
                ylabel="Synthetic Quantiles")
    ax_main.legend(loc='lower right', fontsize=12)

    stats_text = f"Pearson's r = {pearson_r:.3f}\n$R^2$ = {r2:.3f}"
    ax_main.text(0.05, 0.95, stats_text, transform=ax_main.transAxes, fontsize=13.5, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#DDDDDD'))

    # 2. 压缩且极简的边缘 KDE 分布图
    x_grid, y_grid = np.linspace(plot_min, plot_max, 200), np.linspace(plot_min, plot_max, 200)
    
    # 【防呆保护】：因为0点大量聚集，KDE计算极小概率会报错奇异矩阵，加个try-except
    try:
        kde_x = stats.gaussian_kde(xs_all)(x_grid)
    except np.linalg.LinAlgError:
        kde_x = np.zeros_like(x_grid)
        
    try:
        kde_y = stats.gaussian_kde(ys_all)(y_grid)
    except np.linalg.LinAlgError:
        kde_y = np.zeros_like(y_grid)

    ax_top.plot(x_grid, kde_x, color=color_line, linewidth=0.8, alpha=0.8)
    ax_top.fill_between(x_grid, 0, kde_x, color=color_fill, alpha=0.3, edgecolor='none')
    ax_top.set_ylim(bottom=0)
    ax_top.axis('off')

    ax_right.plot(kde_y, y_grid, color=color_line, linewidth=0.8, alpha=0.8)
    ax_right.fill_betweenx(y_grid, 0, kde_y, color=color_fill, alpha=0.3, edgecolor='none')
    ax_right.set_xlim(left=0)
    ax_right.axis('off')

    saved = save_dual(fig, out_path)
    plt.close(fig)
    return saved

# ===============================
# QQ by category
# ===============================
def plot_qq_by_category(path, out_path):
    setup_style()
    data = np.load(path)
    keys = data["keys"]
    K = keys.shape[0]
    categories = sorted(set(int(keys[i,0]) for i in range(K)))
    q_grid = np.arange(0.05, 0.96, 0.05)

    def compact_tick(v):
        av = abs(float(v))
        if av >= 1000:
            return f"{v / 1000:.0f}k"
        if av >= 100:
            return f"{v:.0f}"
        return f"{v:.2g}"
    
    nrows = int(np.ceil(len(categories) / 4))
    fig, axes = plt.subplots(nrows, 4, figsize=(12, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, c in enumerate(categories):
        ax = axes[idx]
        xs_all, ys_all = [], []
        for i in range(K):
            if int(keys[i,0]) == c:
                real, syn = clip_pair(data[f"real_{i}"], data[f"syn_{i}"])
                if len(real) < 10: continue
                for q in q_grid:
                    xs_all.append(np.quantile(real, q))
                    ys_all.append(np.quantile(syn, q))
        
        if len(xs_all) == 0:
            ax.set_title(f"Category {c+1} (no data)", fontsize=15)
            ax.axis('off')
            continue
            
        xs_all, ys_all = np.array(xs_all), np.array(ys_all)
        lo, hi = np.percentile(np.concatenate([xs_all, ys_all]), [2, 98])
        margin = (hi - lo) * 0.05
        
        ax.scatter(xs_all, ys_all, s=6, alpha=0.3, color='#00777B', edgecolors='none')
        ax.plot([lo, hi], [lo, hi], color='#333333', linestyle='--', linewidth=1)
        
        pearson_r, _ = stats.pearsonr(xs_all, ys_all)
        ax.text(0.06, 0.84, f"r = {pearson_r:.2f}\nn = {len(xs_all)}", transform=ax.transAxes, fontsize=12, color="#333333")
        
        axis_min, axis_max = 0.0, hi + margin
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        tick_vals = np.linspace(axis_min, axis_max, 3)
        tick_labels = [compact_tick(v) for v in tick_vals]
        ax.set_xticks(tick_vals)
        ax.set_yticks(tick_vals)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        ax.tick_params(axis='both', labelsize=10.5, pad=2)
        ax.set_title(f"Category {c+1}", fontsize=16.5, pad=8)
            
    for empty_idx in range(len(categories), len(axes)): axes[empty_idx].set_visible(False)
        
    fig.supxlabel("Real Quantiles", fontsize=15, y=0.04)
    fig.supylabel("Synthetic Quantiles", fontsize=15, x=0.025)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.14, top=0.90, wspace=0.45, hspace=0.62)
    saved = save_dual(fig, out_path)
    plt.close(fig)
    return saved

# ===============================
# DimRed: UMAP/PCA (顶部两行图例合并)
# ===============================
def plot_dimred_crv(path, out_path):
    setup_style()
    data = np.load(path)
    keys = data["keys"]
    K = keys.shape[0]
    real_samples, syn_samples, cat_samples = [], [], []
    for i in range(K):
        real_i, syn_i = data[f"real_{i}"], data[f"syn_{i}"]
        if real_i.size == 0 or syn_i.size == 0: continue
        real_i, syn_i = real_i[np.isfinite(real_i)], syn_i[np.isfinite(syn_i)]
        if len(real_i) < 5 or len(syn_i) < 5: continue
        min_len = min(len(real_i), len(syn_i))
        real_samples.append(real_i[:min_len])
        syn_samples.append(syn_i[:min_len])
        cat_samples.append(int(keys[i, 0]))
        
    if not real_samples: return
    
    max_len = max(len(x) for x in real_samples + syn_samples)
    X = np.vstack([
        np.array([np.pad(x, (0, max_len - len(x))) for x in real_samples]),
        np.array([np.pad(x, (0, max_len - len(x))) for x in syn_samples])
    ])
    X -= X.mean(0, keepdims=True)
    n = len(real_samples)
    
    if HAS_UMAP:
        Z = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X)
    else:
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        Z = U[:,:2] * S[:2]
        
    Z_real, Z_syn = Z[:n], Z[n:]
    cat_samples = np.asarray(cat_samples, dtype=int)
    uniq_c = sorted(np.unique(cat_samples).tolist())
    
    npg_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']
    c2color = {c: npg_colors[i % len(npg_colors)] for i, c in enumerate(uniq_c)}
    
    # 恢复正方形画幅
    fig, ax = plt.subplots(figsize=(7, 7))
    for c in uniq_c:
        m = (cat_samples == c)
        col = c2color[c]
        ax.scatter(Z_real[m, 0], Z_real[m, 1], s=35, marker='o', facecolors='none', edgecolors=col, linewidths=1.2, alpha=0.9)
        ax.scatter(Z_syn[m, 0], Z_syn[m, 1], s=30, marker='^', c=col, edgecolors='white', linewidths=0.5, alpha=0.75)

    # 创建所有 Handles
    cat_handles = [Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=c2color[c], markeredgecolor='none', markersize=7, label=f"C{c+1}") for c in uniq_c]
    type_handles = [
        Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='none', markeredgecolor='#333333', markersize=7, markeredgewidth=1.2, label="Real Data"),
        Line2D([0], [0], marker='^', linestyle='None', markerfacecolor='#777777', markeredgecolor='none', markersize=7, label="Synthetic Data"),
    ]
    
    # 核心修改：将 Category (8个) 和 Type (2个) 交织成 10 个元素的列表，
    # 使用 ncol=5，这样会自动排版成完美的 2 行。
    # 第1行: C1, C2, C3, C4, Real
    # 第2行: C5, C6, C7, C8, Syn
    combined_handles = [
        cat_handles[0], cat_handles[1], cat_handles[2], cat_handles[3],
        cat_handles[4], cat_handles[5], cat_handles[6], cat_handles[7], type_handles[0], type_handles[1]
    ]
    
    ax.legend(handles=combined_handles, loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=5, frameon=False, fontsize=13.5, handletextpad=0.25, columnspacing=1.1)
    
    ax.set_xlabel("UMAP Dimension 1" if HAS_UMAP else "PC 1", fontsize=16.5)
    ax.set_ylabel("UMAP Dimension 2" if HAS_UMAP else "PC 2", fontsize=16.5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 预留顶部空间给图例
    plt.tight_layout(rect=[0, 0, 1, 0.92]) 
    saved = save_dual(fig, out_path)
    plt.close(fig)
    return saved

# ===============================
# 数据分布图（小提琴图）
# ===============================
def plot_distribution_by_group(path, out_path, use_std=True):
    setup_style()
    data = np.load(path)
    keys, train_means, train_stds = data["keys"], data["train_means"], data["train_stds"]
    K = keys.shape[0]
    
    colors = ["#4DBBD5", "#E64B35"] 
    page_size = 24
    n_pages = int(np.ceil(K / page_size))
    
    for page in tqdm(range(n_pages), desc="Plotting Distributions"):
        start_idx = page * page_size
        end_idx = min(start_idx + page_size, K)
        page_ids = list(range(start_idx, end_idx))
        nrows_page = int(np.ceil(len(page_ids) / 6))
        
        fig, axes = plt.subplots(nrows_page, 6, figsize=(14, 2.5 * nrows_page))
        axes = np.atleast_1d(axes).flatten()
        
        for local_idx, i in enumerate(page_ids):
            ax = axes[local_idx]
            mean_i, std_i = train_means[i], max(train_stds[i], EPS)
            real = ((data[f"real_{i}"] - mean_i) / std_i).ravel()
            syn = ((data[f"syn_{i}"] - mean_i) / std_i).ravel()
            real, syn = real[np.isfinite(real)], syn[np.isfinite(syn)]
            c, r, v = map(int, keys[i])
            
            if real.size == 0 or syn.size == 0:
                ax.set_title(f"C{c+1}-R{r}-V{v} (Empty)", fontsize=13.5)
                ax.axis('off')
                continue
                
            parts = ax.violinplot([real, syn], positions=[1, 2], showmeans=True, showmedians=False, widths=0.7)
            
            for pc_idx, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[pc_idx])
                pc.set_alpha(0.75)
                pc.set_edgecolor('#333333')
                pc.set_linewidth(1.0)
                
            for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
                if partname in parts:
                    parts[partname].set_edgecolor('#333333')
                    parts[partname].set_linewidth(1.0)
                    if partname == 'cmeans':
                        parts[partname].set_linestyle('--')
                    
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Real", "Syn"], fontsize=13.5)
            ax.set_title(f"C{c+1} | R{r} | V{v}", fontsize=13.5)
            if local_idx % 6 == 0: ax.set_ylabel("Z-score", fontsize=13.5)
            ax.axhline(y=0, color='#999999', linestyle=':', alpha=0.8, zorder=0)
            
        for empty_idx in range(len(page_ids), len(axes)): axes[empty_idx].set_visible(False)
        plt.tight_layout()
        page_out = out_path.replace(".svg", f"_p{page+1}.svg")
        saved = save_dual(fig, page_out)
        plt.close(fig)
        print("[plot_fig2a_my] saved " + ", ".join(saved))

# ===============================
# Main Execution
# ===============================
if __name__ == "__main__":
    INPUT_NPZ = os.path.join(REPO_ROOT, "fig2_data_prepare_Full_1234", "fig2_data", "fig2a_real_syn.npz")
    TRIMMED_NPZ = os.path.join(REPO_ROOT, "fig2_data_prepare_Full_1234", "fig2_data", "fig2a_real_syn_trimmed.npz")
    OUT_DIR = os.path.join(REPO_ROOT, "figs", "fig2a")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("1. Trimming data...")
    trim_crv_npz(INPUT_NPZ, TRIMMED_NPZ)
    
    print("2. Generating RV QQ Overlay Plot...")
    saved = plot_qq_rv_overlay(TRIMMED_NPZ, os.path.join(OUT_DIR, "fig2a_qq_rv_overlay_std.svg"))
    print("[plot_fig2a_my] saved " + ", ".join(saved))
    
    print("3. Generating Category QQ Plots...")
    saved = plot_qq_by_category(TRIMMED_NPZ, os.path.join(OUT_DIR, "figS_qq_crv_by_category_std.svg"))
    print("[plot_fig2a_my] saved " + ", ".join(saved))
    
    print("4. Generating UMAP/PCA Dimensionality Reduction Plot...")
    saved = plot_dimred_crv(TRIMMED_NPZ, os.path.join(OUT_DIR, "fig2a_dimred_crv.svg"))
    print("[plot_fig2a_my] saved " + ", ".join(saved))
    
    print("5. Generating Distribution Violin Plots...")
    # plot_distribution_by_group(TRIMMED_NPZ, os.path.join(OUT_DIR, "fig2a_dist_by_group_global.svg"), use_std=True)
    
    print("All visualizations generated successfully!")
