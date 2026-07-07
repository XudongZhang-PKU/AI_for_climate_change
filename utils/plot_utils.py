# -*- coding: utf-8 -*-
"""
Fig.2 绘图共用工具：箱线图、小提琴图、双 heatmap、收敛曲线、UMAP、混淆矩阵、分组柱状图等。

所有函数均使用 Agg 后端，适合无界面环境保存图片；风格通过 setup_figure 统一。
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def setup_figure(style: str = "default"):
    """统一 matplotlib 全局风格（字体、标题、DPI），在每次绘图前调用。"""
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["figure.dpi"] = 150


def box_or_violin_by_category(
    data_dict: dict,
    out_path: str,
    ylabel: str = "Value",
    title: str = "",
    use_violin: bool = False,
    threshold_line: float | None = None,
):
    """
    Fig.2(a) 用：按 category 分组的箱线图或小提琴图。
    data_dict: { "C0": array, "C1": array, ... }，键为类别标签，值为该类别下的指标数组。
    threshold_line: 若给定，在图上画一条水平虚线（如 1e-5 阈值）。
    """
    setup_figure()
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    cats = list(data_dict.keys())
    vals = [np.asarray(data_dict[c]).ravel() for c in cats]
    if use_violin:
        parts = ax.violinplot(vals, positions=range(len(cats)), showmeans=True, showmedians=True)
    else:
        bp = ax.boxplot(vals, positions=range(len(cats)), patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
    if threshold_line is not None:
        ax.axhline(y=threshold_line, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def heatmap_pair(
    mat_left: np.ndarray,
    mat_right: np.ndarray,
    out_path: str,
    title_left: str = "Real",
    title_right: str = "Synthetic",
    corr_distance: float | None = None,
    vmin: float = -1,
    vmax: float = 1,
):
    """Fig.2(b) 用：左右并排两个相关矩阵 heatmap（Real / Synthetic），图下方可标注 corr_distance = mean(|corr_real - corr_syn|)。"""
    setup_figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mat, title in [(ax1, mat_left, title_left), (ax2, mat_right, title_right)]:
        im = ax.imshow(np.asarray(mat), cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    if corr_distance is not None:
        fig.text(0.5, 0.02, f"corr_distance = {corr_distance:.4f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def violin_or_box_three(
    eq_residual: np.ndarray,
    neg_ratio: np.ndarray,
    pass_arr: np.ndarray | None = None,
    out_path: str = "",
    threshold: float = 1e-5,
):
    """Fig.2(c) 用：(c1) EqResidual 分布 violin + 阈值虚线，(c2) NegRatio 分布；pass_arr 已废弃不再使用。"""
    setup_figure()
    n_panels = 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6, 3))
    eq_residual = np.asarray(eq_residual).ravel()
    neg_ratio = np.asarray(neg_ratio).ravel()

    axes[0].violinplot([eq_residual], positions=[0])
    axes[0].axhline(y=threshold, color="gray", linestyle="--")
    axes[0].set_ylabel("EqResidual")
    axes[0].set_title("(c1) Eq residual")

    axes[1].violinplot([neg_ratio], positions=[0])
    axes[1].set_ylabel("NegRatio")
    axes[1].set_title("(c2) Neg ratio")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def convergence_plot(
    n_list: list | np.ndarray,
    ci_widths: list | np.ndarray,
    out_path: str,
    xlabel: str = "log10(n)",
    ylabel: str = "CI width",
):
    """Fig.2(d) 用：横轴 log10(n)（采样量），纵轴 CI width（如 bootstrap 95% CI 宽度），展示收敛。"""
    setup_figure()
    n_list = np.asarray(n_list).ravel()
    ci_widths = np.asarray(ci_widths).ravel()
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(np.log10(np.maximum(n_list, 1)), ci_widths, "o-")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def umap_scatter(
    embeddings: np.ndarray,
    categories: np.ndarray,
    out_path: str,
    title: str = "Embedding by category",
    sample_per_cat: int | None = 500,
):
    """Fig.2(e) 用：将 embeddings 降维到 2D（优先 UMAP，无则 PCA），按 categories 着色；样本过多时每类最多 sample_per_cat 个点。"""
    X = np.asarray(embeddings)
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        emb_2d = reducer.fit_transform(X)
    except ImportError:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        emb_2d = reducer.fit_transform(X)

    cats = np.asarray(categories).ravel()
    if sample_per_cat and emb_2d.shape[0] > sample_per_cat * len(np.unique(cats)):
        idx = []
        for c in np.unique(cats):
            mask = cats == c
            ii = np.where(mask)[0]
            if len(ii) > sample_per_cat:
                ii = np.random.RandomState(42).choice(ii, sample_per_cat, replace=False)
            idx.extend(ii.tolist())
        idx = np.array(idx)
        emb_2d = emb_2d[idx]
        cats = cats[idx]

    setup_figure()
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for c in np.unique(cats):
        mask = cats == c
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=f"C{c}", alpha=0.6, s=10)
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def adjacency_heatmap(adjacency: np.ndarray, out_path: str, top_k: int = 20):
    """Fig.2(e) 用：邻接矩阵 (G,G) 的 heatmap；top_k>0 时只保留绝对值最大的 top_k 条边，其余置 0 便于可视化。"""
    A = np.asarray(adjacency)
    if top_k > 0 and A.size > top_k:
        flat = np.abs(A).ravel()
        flat.sort()
        thresh = flat[-min(top_k, len(flat))] if len(flat) >= top_k else 0
        A = np.where(np.abs(A) >= thresh, A, 0)
    setup_figure()
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    im = ax.imshow(A, cmap="viridis")
    ax.set_title("Adjacency (top-k)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def confusion_matrix_plot(cm: np.ndarray, out_path: str, class_names: list | None = None):
    """Fig.2(f) 用：绘制混淆矩阵 heatmap，可选 class_names 作为轴标签（如 C0..C7）。"""
    setup_figure()
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    if class_names:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def grouped_bar(
    model_names: list,
    metrics: dict,
    out_path: str,
    normalize: bool = False,
):
    """
    Fig.2(g) 用：分组柱状图，每根组对应一个模型，每组内多根 bar 对应不同指标。
    metrics: { "MMD2": [v1,v2,...], "corr_dist": [...] }，每个列表长度 = len(model_names)。
    normalize=True 时对每个指标做 min-max 归一化再画，便于量纲差异大时对比。
    """
    setup_figure()
    n_models = len(model_names)
    metric_names = list(metrics.keys())
    if normalize:
        for k in metric_names:
            arr = np.array(metrics[k])
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                metrics[k] = ((arr - mn) / (mx - mn)).tolist()
            else:
                metrics[k] = arr.tolist()

    x = np.arange(n_models)
    width = 0.8 / len(metric_names)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i, name in enumerate(metric_names):
        offset = (i - len(metric_names) / 2) * width + width / 2
        ax.bar(x + offset, metrics[name], width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
