# -*- coding: utf-8 -*-
"""
================================================================================
Fig.2(d) 采样量敏感性（Why 1e6） — NATURE-STYLE SINGLE PANEL FINAL
================================================================================
改进点：
1) 目标安全误差阈值 (Target Error Bound) 严格设定为 1% (0.01)。
2) 动态计算并精准标注 Required n (红线首次跌破 1% 时的横坐标)。
3) 蒙特卡洛理论衰减线 O(n^-0.5) 贯穿整个对数坐标轴，完美贴合红线趋势。
4) 保持极值不确定性阴影的统计严谨性 (q95_median 到 q95_worst)。
================================================================================
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --------------------------
# Utilities (保持不变)
# --------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def infer_time_axis(arr_shape, T_expected=91):
    candidates = [i for i, s in enumerate(arr_shape) if s == T_expected]
    if len(candidates) == 1: return candidates[0]
    if len(candidates) > 1: return candidates[-1]
    raise RuntimeError(f"Cannot infer time axis: no dim == {T_expected}, shape={arr_shape}")

def infer_rv_axes(arr_shape, R=11, V=48, time_axis=None):
    axes = list(range(len(arr_shape)))
    if time_axis is not None: axes.remove(time_axis)
    r_axes = [i for i in axes if arr_shape[i] == R]
    v_axes = [i for i in axes if arr_shape[i] == V]
    if len(r_axes) != 1 or len(v_axes) != 1:
        raise RuntimeError(f"Cannot infer R/V axes: shape={arr_shape}, r_axes={r_axes}, v_axes={v_axes}")
    return r_axes[0], v_axes[0]

def parse_n_list(raw, N):
    n_list = [int(x) for x in raw.split(",") if x.strip()]
    n_list = [min(n, N) for n in n_list]
    return sorted(list(dict.fromkeys(n_list)))

def build_Y_rv_from_big(big, shape, s_axis, r_axis, v_axis, time_axis, N, R, V, chunkN, dtype_y):
    Y_rv = np.empty((N, R * V), dtype=dtype_y)
    for s0 in range(0, N, chunkN):
        s1 = min(N, s0 + chunkN)
        sl = [slice(None)] * len(shape)
        sl[s_axis] = slice(s0, s1)
        chunk = big[tuple(sl)]
        chunk = np.moveaxis(chunk, (s_axis, r_axis, v_axis, time_axis), (0, 1, 2, 3))
        y_chunk = np.sum(chunk, axis=3, dtype=np.float64) 
        Y_rv[s0:s1, :] = y_chunk.reshape(s1 - s0, R * V).astype(dtype_y, copy=False)
        if (s0 // chunkN) % 10 == 0:
            print(f"[Y_rv] processed {s1}/{N}")
    return Y_rv

def build_flat_features_from_big(big, shape, s_axis, N, chunkN, dtype_y):
    other_axes = [ax for ax in range(len(shape)) if ax != s_axis]
    feature_shape = tuple(shape[ax] for ax in other_axes)
    F = int(np.prod(feature_shape))
    Y_rv = np.empty((N, F), dtype=dtype_y)
    for s0 in range(0, N, chunkN):
        s1 = min(N, s0 + chunkN)
        sl = [slice(None)] * len(shape)
        sl[s_axis] = slice(s0, s1)
        chunk = big[tuple(sl)]
        chunk = np.moveaxis(chunk, s_axis, 0)
        Y_rv[s0:s1, :] = chunk.reshape(s1 - s0, F).astype(dtype_y, copy=False)
        if (s0 // chunkN) % 10 == 0:
            print(f"[flat] processed {s1}/{N}")
    return Y_rv, feature_shape

def build_Y_rv_from_gtk5(big, shape, s_axis, g_axis, t_axis, f_axis, N, chunkN, dtype_y):
    G = shape[g_axis]
    Y_rv = np.empty((N, G), dtype=dtype_y)
    for s0 in range(0, N, chunkN):
        s1 = min(N, s0 + chunkN)
        sl = [slice(None)] * len(shape)
        sl[s_axis] = slice(s0, s1)
        chunk = big[tuple(sl)]
        chunk = np.moveaxis(chunk, (s_axis, g_axis, t_axis, f_axis), (0, 1, 2, 3))
        val = chunk[..., 4]  
        y_chunk = np.sum(val, axis=2, dtype=np.float64) 
        Y_rv[s0:s1, :] = y_chunk.astype(dtype_y, copy=False)
        if (s0 // chunkN) % 10 == 0:
            print(f"[gtk5] processed {s1}/{N}")
    return Y_rv

def bootstrap_logvar_iqr(Y_rv, n_list, B=200, seed=0, chunk_rv=512, dtype=np.float32):
    rng = np.random.default_rng(seed)
    N, RV = Y_rv.shape
    eps = 1e-12
    out_p25 = np.zeros((len(n_list), RV), dtype=dtype)
    out_p50 = np.zeros((len(n_list), RV), dtype=dtype)
    out_p75 = np.zeros((len(n_list), RV), dtype=dtype)
    out_iqr = np.zeros((len(n_list), RV), dtype=dtype)

    for ti, n in enumerate(n_list):
        n = int(min(n, N))
        chunk_eff = int(chunk_rv if n <= 200000 else min(chunk_rv, 128))
        for s in range(0, RV, chunk_eff):
            e = min(RV, s + chunk_eff)
            lv = np.empty((B, e - s), dtype=np.float32)
            for b in range(B):
                idx_b = rng.integers(0, N, size=n, endpoint=False)
                Xb = Y_rv[idx_b, s:e]  
                vb = np.var(Xb, axis=0, ddof=1)
                lv[b] = np.log(vb + eps)
            out_p25[ti, s:e] = np.percentile(lv, 25, axis=0).astype(dtype)
            out_p50[ti, s:e] = np.percentile(lv, 50, axis=0).astype(dtype)
            out_p75[ti, s:e] = np.percentile(lv, 75, axis=0).astype(dtype)
            out_iqr[ti, s:e] = (out_p75[ti, s:e] - out_p25[ti, s:e]).astype(dtype)
        print(f"[bootstrap] n={n} done ({ti+1}/{len(n_list)})")
    return out_p25, out_p50, out_p75, out_iqr

def bootstrap_ci_stat(Y_rv, n_list, B=200, seed=0, stat="mean", q=0.95, chunk_rv=512, dtype=np.float32):
    rng = np.random.default_rng(seed)
    N, RV = Y_rv.shape
    out_center = np.zeros((len(n_list), RV), dtype=dtype)
    out_lo = np.zeros((len(n_list), RV), dtype=dtype)
    out_hi = np.zeros((len(n_list), RV), dtype=dtype)
    out_w = np.zeros((len(n_list), RV), dtype=dtype)

    for ti, n in enumerate(n_list):
        n = int(min(n, N))
        chunk_eff = int(chunk_rv if n <= 200000 else min(chunk_rv, 128))
        for s in range(0, RV, chunk_eff):
            e = min(RV, s + chunk_eff)
            st = np.empty((B, e - s), dtype=np.float32)
            for b in range(B):
                idx_b = rng.integers(0, N, size=n, endpoint=False)
                Xb = Y_rv[idx_b, s:e]  
                if stat == "mean":
                    st[b] = np.mean(Xb, axis=0)
                elif stat == "quantile":
                    st[b] = np.quantile(Xb, q, axis=0)
                else:
                    raise ValueError(stat)
            out_center[ti, s:e] = np.mean(st, axis=0).astype(dtype)
            lo = np.percentile(st, 2.5, axis=0)
            hi = np.percentile(st, 97.5, axis=0)
            out_lo[ti, s:e] = lo.astype(dtype)
            out_hi[ti, s:e] = hi.astype(dtype)
            out_w[ti, s:e] = (hi - lo).astype(dtype)
        print(f"[bootstrap_ci] stat={stat} n={n} done ({ti+1}/{len(n_list)})")
    return out_center, out_lo, out_hi, out_w

# --------------------------
# Nature-Style Plotting
# --------------------------
def setup_nature_single_panel():
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 21,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Roboto", "DejaVu Sans"],
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
    })

def plot_single_bottleneck_convergence(n_list, mean_c, mean_w, q95_c, q95_w, out_path, target_error=0.01):
    """
    Nature-style SINGLE panel proving the necessity of 1e6 samples.
    """
    setup_nature_single_panel()
    x = np.array(n_list, dtype=np.float64)
    
    # 1. 计算相对不确定性 (Relative CI Width = Width / Abs(Center))
    rel_w_mean = mean_w / (np.abs(mean_c) + 1e-9)
    rel_w_q95 = q95_w / (np.abs(q95_c) + 1e-9)
    
    # 过滤异常值
    valid_mask_m = (np.abs(mean_c) > 1e-3) & np.isfinite(rel_w_mean)
    valid_mask_q = (np.abs(q95_c) > 1e-3) & np.isfinite(rel_w_q95)
    
    # 2. 提取统计聚合量
    mean_typ = np.zeros(len(x))
    q95_worst = np.zeros(len(x))
    q95_median = np.zeros(len(x))
    
    for i in range(len(x)):
        m_vals = rel_w_mean[i][valid_mask_m[i]]
        q_vals = rel_w_q95[i][valid_mask_q[i]]
        
        mean_typ[i] = np.nanmedian(m_vals) if len(m_vals) > 0 else np.nan
        q95_worst[i] = np.nanpercentile(q_vals, 95) if len(q_vals) > 0 else np.nan
        q95_median[i] = np.nanmedian(q_vals) if len(q_vals) > 0 else np.nan

    # 3. 开始绘图
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.18, top=0.94)
    
    color_mean = "#4DBBD5"  # Teal
    color_tail = "#E64B35"  # Red
    color_bound = "#333333" # Dark Grey
    
    # 绘制均值轨迹
    ax.plot(x, mean_typ, color=color_mean, marker='o', markersize=4, lw=2, alpha=0.9, 
            label="Mean uncertainty")
    
    # 绘制尾部瓶颈轨迹 (严格保持极值的内部阴影，不连蓝线)
    ax.fill_between(x, q95_median, q95_worst, color=color_tail, alpha=0.15, edgecolor="none", 
                    label="Tail dispersion")
    ax.plot(x, q95_worst, color=color_tail, marker='s', markersize=4, lw=2.5, alpha=0.9, 
            label=r"Tail uncertainty ($q_{95}$)")
    
    # 【修改点 4】理论辅助线 O(n^-0.5)，贯穿全图
    valid_idx = np.where(~np.isnan(q95_worst))[0]
    if len(valid_idx) > 0:
        anchor_idx = valid_idx[-1] # 锚定在最后一个有效点，确保右侧贴合
        ref_x = x
        ref_y = q95_worst[anchor_idx] * (ref_x / x[anchor_idx]) ** (-0.5)
        ax.plot(ref_x, ref_y, linestyle="--", color=color_bound, lw=1.5, alpha=0.6,
                label=r"$\mathcal{O}(n^{-0.5})$ theory")

    # 【修改点 1】判决边界: 目标误差 1%
    ax.axhline(y=target_error, color="gray", linestyle=":", lw=1.5, zorder=0)
    ax.text(
        0.08,
        0.36,
        f"{target_error*100:.0f}% bound",
        color="gray",
        fontsize=18,
        transform=ax.transAxes
    )
    
    # 【修改点 2】动态计算 Required n
    required_n = None
    for i, err in enumerate(q95_worst):
        if not np.isnan(err) and err <= target_error:
            required_n = x[i]
            break
            
    if required_n is not None:
        ax.axvline(x=required_n, color="gray", linestyle=":", lw=1.5, zorder=0)
        
        # 优化文本显示，比如 1000000 显示为 10^6
        # 使用 numpy log10 安全转换为指数
        exp_n = int(np.round(np.log10(required_n)))
        display_text = f"Required $n \\approx 10^{exp_n}$" if 10**exp_n == required_n else f"Required $n \\approx {required_n:g}$"
        
        ax.text(
            0.52,
            0.74,
            f"{display_text}\nfor Tail Safety", 
            color="gray",
            fontsize=18,
            transform=ax.transAxes
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{y*100:g}%"))
    
    ax.set_xlabel("Sample size $n$", fontsize=19.5)
    ax.set_ylabel("Relative uncertainty", fontsize=19.5)
    
    ax.legend(
        loc="lower left",
        fontsize=18,
        frameon=True,
        framealpha=0.88,
        edgecolor="none",
        facecolor="white",
        handlelength=1.4,
        borderpad=0.35,
        labelspacing=0.35,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    out_svg = out_path.replace(".png", ".svg")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close()
    print(f"[fig] Nature-style single panel convergence plot saved to:\n  -> {out_path}\n  -> {out_svg}")

# --------------------------
# Main pipeline
# --------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    default_big_npy = "/lustre/home/2501111516/work/co2-0204-best-for-gen/output/forecast/最终生成_250.npy"
    default_out_dir = os.path.join(project_root, "figs", "fig2d")

    ap = argparse.ArgumentParser()
    ap.add_argument("--big_npy", type=str, default=default_big_npy)
    ap.add_argument("--out_dir", type=str, default=default_out_dir)
    ap.add_argument("--R", type=int, default=11)
    ap.add_argument("--V", type=int, default=48)
    ap.add_argument("--T_expected", type=int, default=91)
    ap.add_argument("--chunkN", type=int, default=128)
    ap.add_argument("--dtype_y", type=str, default="float32", choices=["float32", "float64"])
    ap.add_argument("--force_recompute_metrics", dest="force_recompute_metrics", action="store_true")
    ap.add_argument("--reuse_cached_stats", dest="force_recompute_metrics", action="store_false")

    # bootstrap configs
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_list", type=str, default="1000,2000,3000,5000,8000,10000,15000,20000,30000,50000,80000,100000,150000,200000,300000,500000,800000,1000000,1500000,2000000,3000000,5000000,8000000,10000000")
    ap.add_argument("--topK", type=int, default=12)

    # force execution for required metrics
    ap.set_defaults(force_recompute_metrics=True, do_mean_ci=True, do_q95_ci=True)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    dtype_y = np.float32 if args.dtype_y == "float32" else np.float64
    stat_dir = os.path.join(args.out_dir, "stats")
    fig_dir = args.out_dir
    ensure_dir(stat_dir)
    ensure_dir(fig_dir)

    n_list_path = os.path.join(stat_dir, "n_list.npy")
    mean_center_path = os.path.join(stat_dir, "mean_center.npy")
    mean_w_path = os.path.join(stat_dir, "mean_ci_width.npy")
    q95_center_path = os.path.join(stat_dir, "q95_center.npy")
    q95_w_path = os.path.join(stat_dir, "q95_ci_width.npy")
    
    labels_path = os.path.join(stat_dir, "labels.npy")
    topk_path = os.path.join(stat_dir, "topk_iqr_idx.npy")
    cat_ids_path = os.path.join(stat_dir, "category_ids.npy")
    p25_path = os.path.join(stat_dir, "logvar_p25.npy")
    p50_path = os.path.join(stat_dir, "logvar_p50.npy")
    p75_path = os.path.join(stat_dir, "logvar_p75.npy")
    iqr_path = os.path.join(stat_dir, "logvar_iqr.npy")
    mean_lo_path = os.path.join(stat_dir, "mean_lo.npy")
    mean_hi_path = os.path.join(stat_dir, "mean_hi.npy")
    q95_lo_path = os.path.join(stat_dir, "q95_lo.npy")
    q95_hi_path = os.path.join(stat_dir, "q95_hi.npy")

    has_base_stats = all(os.path.isfile(p) for p in [n_list_path, labels_path, topk_path, p25_path, p50_path, p75_path, iqr_path])
    has_mean_stats = all(os.path.isfile(p) for p in [mean_center_path, mean_lo_path, mean_hi_path, mean_w_path])
    has_q95_stats = all(os.path.isfile(p) for p in [q95_center_path, q95_lo_path, q95_hi_path, q95_w_path])
    can_plot_from_cache = has_base_stats and has_mean_stats and has_q95_stats
    must_compute = args.force_recompute_metrics or (not can_plot_from_cache)

    if must_compute:
        if not os.path.isfile(args.big_npy):
            raise FileNotFoundError(f"Raw file not found: {args.big_npy}.")
        print(f"[load] mmap big npy: {args.big_npy}")
        big = np.load(args.big_npy, mmap_mode="r")
        shape = big.shape

        try:
            time_axis = infer_time_axis(shape, args.T_expected)
            r_axis, v_axis = infer_rv_axes(shape, args.R, args.V, time_axis=time_axis)
            axes = set(range(len(shape)))
            s_axis = list(axes - {time_axis, r_axis, v_axis})[0]
            N = shape[s_axis]
            n_list = parse_n_list(args.n_list, N)
            labels = np.array([f"R{r}-V{v}" for r in range(args.R) for v in range(args.V)], dtype=object)
            Y_rv = build_Y_rv_from_big(big, shape, s_axis, r_axis, v_axis, time_axis, N, args.R, args.V, args.chunkN, dtype_y)
        except RuntimeError:
            f_axes = [i for i, s in enumerate(shape) if s == 5]
            g_axes = [i for i, s in enumerate(shape) if s == args.R * args.V]
            t_axes = [i for i, s in enumerate(shape) if s == 10]
            if len(f_axes) == 1 and len(g_axes) == 1 and len(t_axes) == 1:
                f_axis, g_axis, t_axis = f_axes[0], g_axes[0], t_axes[0]
                s_axis = list(set(range(len(shape))) - {f_axis, g_axis, t_axis})[0]
                N = shape[s_axis]
                n_list = parse_n_list(args.n_list, N)
                Y_rv = build_Y_rv_from_gtk5(big, shape, s_axis, g_axis, t_axis, f_axis, N, args.chunkN, dtype_y)
                labels = np.array([f"RV{i}" for i in range(Y_rv.shape[1])], dtype=object)
            else:
                s_axis = int(np.argmax(shape))
                N = shape[s_axis]
                n_list = parse_n_list(args.n_list, N)
                Y_rv, _ = build_flat_features_from_big(big, shape, s_axis, N, args.chunkN, dtype_y)
                labels = np.array([f"F{i}" for i in range(Y_rv.shape[1])], dtype=object)

        logvar_p25, logvar_p50, logvar_p75, logvar_iqr = bootstrap_logvar_iqr(Y_rv, n_list, B=args.B, seed=args.seed, chunk_rv=512, dtype=np.float32)
        np.save(p25_path, logvar_p25); np.save(p50_path, logvar_p50)
        np.save(p75_path, logvar_p75); np.save(iqr_path, logvar_iqr)
        top_idx = np.argsort(-logvar_iqr[-1])[:min(args.topK, logvar_iqr[-1].shape[0])].astype(np.int64)
        np.save(topk_path, top_idx)
        np.save(n_list_path, np.array(n_list, dtype=np.int64))
        np.save(labels_path, labels)

        c, lo, hi, w = bootstrap_ci_stat(Y_rv, n_list, B=args.B, seed=args.seed, stat="mean", chunk_rv=512, dtype=np.float32)
        np.save(mean_center_path, c); np.save(mean_lo_path, lo)
        np.save(mean_hi_path, hi); np.save(mean_w_path, w)

        c, lo, hi, w = bootstrap_ci_stat(Y_rv, n_list, B=args.B, seed=args.seed, stat="quantile", q=0.95, chunk_rv=512, dtype=np.float32)
        np.save(q95_center_path, c); np.save(q95_lo_path, lo)
        np.save(q95_hi_path, hi); np.save(q95_w_path, w)
        del Y_rv
    else:
        print("[cache] reuse full cached stats (skip reading raw big_npy)")
        n_list = np.load(n_list_path).astype(np.int64).tolist()

    # ---------- Single Panel Plotting Execution ----------
    out_single = os.path.join(fig_dir, "fig2d_sampling_necessity_single.png")
    
    try:
        m_c = np.load(mean_center_path, mmap_mode="r")
        m_w = np.load(mean_w_path, mmap_mode="r")
        q_c = np.load(q95_center_path, mmap_mode="r")
        q_w = np.load(q95_w_path, mmap_mode="r")
        
        # 【应用修改点 1】: Target error 设为 0.01
        plot_single_bottleneck_convergence(
            n_list=n_list, 
            mean_c=m_c, mean_w=m_w, 
            q95_c=q_c, q95_w=q_w, 
            out_path=out_single,
            target_error=0.01 
        )
    except FileNotFoundError as e:
        print(f"Error loading required cache files. Please re-run with --force_recompute_metrics. Detail: {e}")

    print("[done] The ultimate convergence plot has been generated successfully.")

if __name__ == "__main__":
    main()