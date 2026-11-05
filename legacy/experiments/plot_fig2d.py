# -*- coding: utf-8 -*-
"""
Fig.2(d) 采样量敏感性（Why 1e6）。
本脚本用于验证“大规模采样”的统计合理性：随着采样规模 n 增加，对生成分布的关键统计量（均值、分位数、方差等）的估计会逐步收敛，其不确定性（bootstrap 置信区间宽度、方差估计的 IQR 等）按统计规律下降，从而量化“需要多少样本才稳定”。

考虑到原始生成结果文件体积巨大（>100GB），脚本采用流式/内存映射读取（mmap），先对每个情景在 2010–2100 年维度做逐年累计求和，将每个样本的时间序列压缩为 (region, variable) 的标量 Y（即每个 region×variable 的累计量）。Y 仅在内存中使用，不会写入中间文件。

在收敛评估中，脚本对一组采样规模 n（对数间隔）进行 bootstrap 有放回抽样（默认 200 次），并输出多类“采样规模—不确定性”曲线与统计表，包括：
1）方差估计不确定性：对每个 (region, variable) 计算 bootstrap 下 log(Var-hat) 的 IQR（P75−P25），并在最大 n 时选取 IQR 最大的 Top-K 组绘制 IQR(n) 曲线，用于展示“最不稳定的组”如何随 n 增大而变稳定。
2）均值/尾部分位数的收敛：对 mean(Y) 或 q95(Y) 计算 bootstrap 95% 置信区间宽度随 n 的下降，并可绘制 log–log 图拟合收敛斜率（SE ~ n^{-1/2}）。

最终，脚本会将“画图所需指标”全部落盘（不存中间 Y），保证后续重画无需再次扫描 100GB 原始文件。
"""
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------
# Utilities
# --------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def infer_time_axis(arr_shape, T_expected=91):
    """
    Try to find a time axis with length 91 (2010-2100 inclusive).
    Return axis index or raise.
    """
    candidates = [i for i, s in enumerate(arr_shape) if s == T_expected]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # prefer last-ish axis
        return candidates[-1]
    raise RuntimeError(f"Cannot infer time axis: no dim == {T_expected}, shape={arr_shape}")

def infer_rv_axes(arr_shape, R=11, V=48, time_axis=None):
    """
    Infer region and variable axes (dims with size 11 and 48).
    """
    axes = list(range(len(arr_shape)))
    if time_axis is not None:
        axes.remove(time_axis)
    r_axes = [i for i in axes if arr_shape[i] == R]
    v_axes = [i for i in axes if arr_shape[i] == V]
    if len(r_axes) != 1 or len(v_axes) != 1:
        raise RuntimeError(f"Cannot infer R/V axes: shape={arr_shape}, r_axes={r_axes}, v_axes={v_axes}")
    return r_axes[0], v_axes[0]

def move_axes_to(arr, src_axes, dst_axes):
    return np.moveaxis(arr, src_axes, dst_axes)

def parse_n_list(raw, N):
    n_list = [int(x) for x in raw.split(",") if x.strip()]
    n_list = [min(n, N) for n in n_list]
    return sorted(list(dict.fromkeys(n_list)))

def build_Y_rv_from_big(big, shape, s_axis, r_axis, v_axis, time_axis, N, R, V, chunkN, dtype_y):
    """
    Stream from large array and build Y_rv=[N, R*V] in memory.
    """
    Y_rv = np.empty((N, R * V), dtype=dtype_y)
    for s0 in range(0, N, chunkN):
        s1 = min(N, s0 + chunkN)
        sl = [slice(None)] * len(shape)
        sl[s_axis] = slice(s0, s1)
        chunk = big[tuple(sl)]
        chunk = np.moveaxis(chunk, (s_axis, r_axis, v_axis, time_axis), (0, 1, 2, 3))
        y_chunk = np.sum(chunk, axis=3, dtype=np.float64)  # [nchunk, R, V]
        Y_rv[s0:s1, :] = y_chunk.reshape(s1 - s0, R * V).astype(dtype_y, copy=False)
        if (s0 // chunkN) % 10 == 0:
            print(f"[Y_rv] processed {s1}/{N}")
    return Y_rv

def build_flat_features_from_big(big, shape, s_axis, N, chunkN, dtype_y):
    """
    Generic fallback when no explicit time axis exists:
    flatten all non-sample dims into feature axis.
    Returns Y_rv=[N, F], feature_shape tuple.
    """
    other_axes = [ax for ax in range(len(shape)) if ax != s_axis]
    feature_shape = tuple(shape[ax] for ax in other_axes)
    F = int(np.prod(feature_shape))
    Y_rv = np.empty((N, F), dtype=dtype_y)
    for s0 in range(0, N, chunkN):
        s1 = min(N, s0 + chunkN)
        sl = [slice(None)] * len(shape)
        sl[s_axis] = slice(s0, s1)
        chunk = big[tuple(sl)]
        # move sample axis to front, then flatten the rest
        chunk = np.moveaxis(chunk, s_axis, 0)
        Y_rv[s0:s1, :] = chunk.reshape(s1 - s0, F).astype(dtype_y, copy=False)
        if (s0 // chunkN) % 10 == 0:
            print(f"[flat] processed {s1}/{N}")
    return Y_rv, feature_shape

def build_Y_rv_from_gtk5(big, shape, s_axis, g_axis, t_axis, f_axis, N, chunkN, dtype_y):
    """
    GTK5-like layout: feature dim contains [model, category, region, variable, value].
    We only use value channel (index=4), then sum over time -> Y_rv=[N, G].
    """
    G = shape[g_axis]
    Y_rv = np.empty((N, G), dtype=dtype_y)
    for s0 in range(0, N, chunkN):
        s1 = min(N, s0 + chunkN)
        sl = [slice(None)] * len(shape)
        sl[s_axis] = slice(s0, s1)
        chunk = big[tuple(sl)]
        # align to [Nchunk, G, T, F]
        chunk = np.moveaxis(chunk, (s_axis, g_axis, t_axis, f_axis), (0, 1, 2, 3))
        val = chunk[..., 4]  # [Nchunk, G, T]
        y_chunk = np.sum(val, axis=2, dtype=np.float64)  # [Nchunk, G]
        Y_rv[s0:s1, :] = y_chunk.astype(dtype_y, copy=False)
        if (s0 // chunkN) % 10 == 0:
            print(f"[gtk5] processed {s1}/{N}")
    return Y_rv

def bootstrap_logvar_iqr(Y_rv, n_list, B=200, seed=0, chunk_rv=512, dtype=np.float32):
    """
    Y_rv: [N, RV] float32/float64
    For each n in n_list:
      - bootstrap with replacement B times
      - compute sample variance (ddof=1) for each RV
      - take log(var + eps)
      - return p25/p50/p75/iqr arrays of shape [len(n_list), RV]

    Memory strategy:
      - process RV in chunks to avoid storing [B, RV] huge tensors
    """
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

        # process RV in chunks
        for s in range(0, RV, chunk_eff):
            e = min(RV, s + chunk_eff)
            lv = np.empty((B, e - s), dtype=np.float32)
            # bootstrap one-by-one to avoid allocating [B, n, chunk]
            for b in range(B):
                idx_b = rng.integers(0, N, size=n, endpoint=False)
                Xb = Y_rv[idx_b, s:e]      # [n, chunk]
                vb = np.var(Xb, axis=0, ddof=1)
                lv[b] = np.log(vb + eps)

            out_p25[ti, s:e] = np.percentile(lv, 25, axis=0).astype(dtype)
            out_p50[ti, s:e] = np.percentile(lv, 50, axis=0).astype(dtype)
            out_p75[ti, s:e] = np.percentile(lv, 75, axis=0).astype(dtype)
            out_iqr[ti, s:e] = (out_p75[ti, s:e] - out_p25[ti, s:e]).astype(dtype)

        print(f"[bootstrap] n={n} done ({ti+1}/{len(n_list)})")

    return out_p25, out_p50, out_p75, out_iqr

def bootstrap_ci_stat(Y_rv, n_list, B=200, seed=0, stat="mean", q=0.95, chunk_rv=512, dtype=np.float32):
    """
    Compute bootstrap CI width for mean or quantile of Y.
    Returns:
      center [T, RV], lo [T, RV], hi [T, RV], width [T, RV]
    """
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
            # bootstrap one-by-one to avoid allocating [B, n, chunk]
            for b in range(B):
                idx_b = rng.integers(0, N, size=n, endpoint=False)
                Xb = Y_rv[idx_b, s:e]  # [n, chunk]
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

def plot_topk_iqr_curves(n_list, iqr, topk_idx, labels, out_png, title):
    """
    iqr: [T, RV]
    topk_idx: list of rv indices
    labels: list of strings per rv
    """
    plt.figure(figsize=(8.8, 4.8))
    x = np.array(n_list, dtype=np.float64)
    for k, rv in enumerate(topk_idx):
        plt.plot(x, iqr[:, rv], lw=2.0, label=labels[rv])
    plt.xscale("log")
    plt.xlabel("Sample size n (log)")
    plt.ylabel("IQR of log(Var-hat) across bootstrap")
    plt.title(title)
    plt.legend(frameon=False, fontsize=9, ncol=1, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_iqr_by_category(n_list, iqr, category_ids, out_png, title):
    """
    Aggregate RV-wise IQR curves into category-wise curves.
    iqr: [T, RV], category_ids: [RV]
    """
    x = np.array(n_list, dtype=np.float64)
    category_ids = np.asarray(category_ids).astype(np.int64)
    uniq = np.unique(category_ids)
    plt.figure(figsize=(8.8, 4.8))
    for cid in uniq:
        idx = np.where(category_ids == cid)[0]
        if idx.size == 0:
            continue
        y = np.mean(iqr[:, idx], axis=1)
        plt.plot(x, y, lw=2.0, label=f"C{int(cid)} (n={idx.size})")
    plt.xscale("log")
    plt.xlabel("Sample size n (log)")
    plt.ylabel("IQR of log(Var-hat) across bootstrap")
    plt.title(title)
    plt.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_ci_band(n_list, center, lo, hi, rv, label, out_png, ylabel, title):
    x = np.array(n_list, dtype=np.float64)
    plt.figure(figsize=(6.8, 4.2))
    plt.plot(x, center[:, rv], lw=2.2)
    plt.fill_between(x, lo[:, rv], hi[:, rv], alpha=0.15)
    plt.xscale("log")
    plt.xlabel("Sample size n (log)")
    plt.ylabel(ylabel)
    plt.title(f"{title}\n{label}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_loglog_slope(n_list, width, rv, label, out_png, title):
    x = np.array(n_list, dtype=np.float64)
    y = np.array(width[:, rv], dtype=np.float64)
    # fit slope in log space
    m = (y > 0) & np.isfinite(y)
    lx = np.log(x[m]); ly = np.log(y[m])
    if len(lx) >= 3:
        a, b = np.polyfit(lx, ly, 1)  # ly ~ a*lx + b
    else:
        a = np.nan

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(x, y, marker="o", lw=2.0)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("n (log)")
    plt.ylabel("CI width (log)")
    plt.title(f"{title}\n{label}\nSlope ≈ {a:.2f} (target -0.5 for SE)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# --------------------------
# Main pipeline
# --------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    default_big_npy = "/lustre/home/2501111516/work/co2-0204-best-for-gen/output/forecast/最终生成_250.npy"
    default_out_dir = os.path.join(project_root, "figs", "figd")

    ap = argparse.ArgumentParser()
    ap.add_argument("--big_npy", type=str, default=default_big_npy,
                    help="Huge npy path (100GB), default: <project_root>/最终生成_250.npy")
    ap.add_argument("--out_dir", type=str, default=default_out_dir,
                    help="Output root, default: <project_root>/figs/figd")
    ap.add_argument("--R", type=int, default=11)
    ap.add_argument("--V", type=int, default=48)
    ap.add_argument("--T_expected", type=int, default=91, help="2010-2100 inclusive => 91")
    ap.add_argument("--chunkN", type=int, default=128, help="streaming chunk on sample axis")
    ap.add_argument("--dtype_y", type=str, default="float32", choices=["float32", "float64"])
    ap.add_argument("--force_recompute_metrics", dest="force_recompute_metrics", action="store_true",
                    help="Ignore cached stats and recompute from raw big_npy (default: on)")
    ap.add_argument("--reuse_cached_stats", dest="force_recompute_metrics", action="store_false",
                    help="Reuse cached stats under out_dir/stats when available")

    # bootstrap configs
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_list", type=str, default="1000,2000,3000,5000,8000,10000,15000,20000,30000,50000,80000,100000,150000,200000,300000,500000,800000,1000000,1500000,2000000,3000000,5000000,8000000,10000000",
                    help="comma list; will be clipped to N")
    ap.add_argument("--topK", type=int, default=12)

    # extra plots (enabled by default)
    ap.add_argument("--no_mean_ci", dest="do_mean_ci", action="store_false")
    ap.add_argument("--no_q95_ci", dest="do_q95_ci", action="store_false")
    ap.set_defaults(force_recompute_metrics=True, do_mean_ci=True, do_q95_ci=True)

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    dtype_y = np.float32 if args.dtype_y == "float32" else np.float64
    stat_dir = os.path.join(args.out_dir, "stats")
    fig_dir = args.out_dir
    ensure_dir(stat_dir)
    ensure_dir(fig_dir)

    n_list_path = os.path.join(stat_dir, "n_list.npy")
    labels_path = os.path.join(stat_dir, "labels.npy")
    topk_path = os.path.join(stat_dir, "topk_iqr_idx.npy")
    cat_ids_path = os.path.join(stat_dir, "category_ids.npy")
    p25_path = os.path.join(stat_dir, "logvar_p25.npy")
    p50_path = os.path.join(stat_dir, "logvar_p50.npy")
    p75_path = os.path.join(stat_dir, "logvar_p75.npy")
    iqr_path = os.path.join(stat_dir, "logvar_iqr.npy")
    mean_center_path = os.path.join(stat_dir, "mean_center.npy")
    mean_lo_path = os.path.join(stat_dir, "mean_lo.npy")
    mean_hi_path = os.path.join(stat_dir, "mean_hi.npy")
    mean_w_path = os.path.join(stat_dir, "mean_ci_width.npy")
    q95_center_path = os.path.join(stat_dir, "q95_center.npy")
    q95_lo_path = os.path.join(stat_dir, "q95_lo.npy")
    q95_hi_path = os.path.join(stat_dir, "q95_hi.npy")
    q95_w_path = os.path.join(stat_dir, "q95_ci_width.npy")

    has_base_stats = all(os.path.isfile(p) for p in [n_list_path, labels_path, topk_path, p25_path, p50_path, p75_path, iqr_path])
    has_mean_stats = all(os.path.isfile(p) for p in [mean_center_path, mean_lo_path, mean_hi_path, mean_w_path])
    has_q95_stats = all(os.path.isfile(p) for p in [q95_center_path, q95_lo_path, q95_hi_path, q95_w_path])
    need_mean_stats = args.do_mean_ci
    need_q95_stats = args.do_q95_ci
    can_plot_from_cache = has_base_stats and (not need_mean_stats or has_mean_stats) and (not need_q95_stats or has_q95_stats)
    must_compute = args.force_recompute_metrics or (not can_plot_from_cache)

    if must_compute:
        if not os.path.isfile(args.big_npy):
            raise FileNotFoundError(
                f"Raw file not found: {args.big_npy}. "
                f"Provide --big_npy or keep existing stats cache under {stat_dir}."
            )
        print(f"[load] mmap big npy: {args.big_npy}")
        big = np.load(args.big_npy, mmap_mode="r")
        shape = big.shape
        print(f"[load] shape={shape}, dtype={big.dtype}")

        # Try canonical format first: [sample, region, variable, time] in arbitrary order.
        try:
            time_axis = infer_time_axis(shape, args.T_expected)
            r_axis, v_axis = infer_rv_axes(shape, args.R, args.V, time_axis=time_axis)
            axes = set(range(len(shape)))
            sample_axis_candidates = list(axes - {time_axis, r_axis, v_axis})
            if len(sample_axis_candidates) != 1:
                raise RuntimeError(f"Cannot infer sample axis, candidates={sample_axis_candidates}, shape={shape}")
            s_axis = sample_axis_candidates[0]
            N = shape[s_axis]
            print(f"[infer] canonical mode: sample_axis={s_axis}, time_axis={time_axis}, r_axis={r_axis}, v_axis={v_axis}, N={N}")

            n_list = parse_n_list(args.n_list, N)
            print(f"[n_list] {n_list}")
            labels = np.array([f"R{r}-V{v}" for r in range(args.R) for v in range(args.V)], dtype=object)
            Y_rv = build_Y_rv_from_big(
                big, shape, s_axis, r_axis, v_axis, time_axis, N, args.R, args.V, args.chunkN, dtype_y
            )
        except RuntimeError as e:
            # Fallback for arrays without explicit time axis.
            print(f"[infer] canonical mode failed: {e}")
            # Prefer GTK5 inference: [N, G(=R*V), T(=10), F(=5)] with feature order
            # [model, category, region, variable, value].
            f_axes = [i for i, s in enumerate(shape) if s == 5]
            g_axes = [i for i, s in enumerate(shape) if s == args.R * args.V]
            t_axes = [i for i, s in enumerate(shape) if s == 10]
            if len(f_axes) == 1 and len(g_axes) == 1 and len(t_axes) == 1:
                f_axis = f_axes[0]
                g_axis = g_axes[0]
                t_axis = t_axes[0]
                axes = set(range(len(shape)))
                sample_axis_candidates = list(axes - {f_axis, g_axis, t_axis})
                if len(sample_axis_candidates) != 1:
                    raise RuntimeError(
                        f"Cannot infer sample axis in gtk5 mode, candidates={sample_axis_candidates}, shape={shape}"
                    )
                s_axis = sample_axis_candidates[0]
                N = shape[s_axis]
                print(
                    f"[infer] gtk5 mode: sample_axis={s_axis}, g_axis={g_axis}, t_axis={t_axis}, "
                    f"f_axis={f_axis}, N={N}"
                )
                n_list = parse_n_list(args.n_list, N)
                print(f"[n_list] {n_list}")
                Y_rv = build_Y_rv_from_gtk5(
                    big, shape, s_axis, g_axis, t_axis, f_axis, N, args.chunkN, dtype_y
                )
                labels = np.array([f"RV{i}" for i in range(Y_rv.shape[1])], dtype=object)
                # category per g (RV): read one tiny slice [1, G, 1, F] then take feature idx=1
                sl = [slice(None)] * len(shape)
                sl[s_axis] = slice(0, 1)
                sl[g_axis] = slice(None)
                sl[t_axis] = slice(0, 1)
                sl[f_axis] = slice(None)
                tiny = big[tuple(sl)]
                tiny = np.moveaxis(tiny, (s_axis, g_axis, t_axis, f_axis), (0, 1, 2, 3))
                cat_ids = tiny[0, :, 0, 1].astype(np.int64)  # [G]
                np.save(cat_ids_path, cat_ids)
            else:
                # Last fallback: flatten all non-sample dims (generic, less semantic).
                s_axis = int(np.argmax(shape))
                N = shape[s_axis]
                print(f"[infer] fallback flat mode: sample_axis={s_axis}, N={N}")
                n_list = parse_n_list(args.n_list, N)
                print(f"[n_list] {n_list}")
                Y_rv, feature_shape = build_flat_features_from_big(
                    big, shape, s_axis, N, args.chunkN, dtype_y
                )
                labels = np.array([f"F{i}" for i in range(Y_rv.shape[1])], dtype=object)
                print(f"[flat] feature_shape={feature_shape}, flattened_features={Y_rv.shape[1]}")

        logvar_p25, logvar_p50, logvar_p75, logvar_iqr = bootstrap_logvar_iqr(
            Y_rv, n_list, B=args.B, seed=args.seed, chunk_rv=512, dtype=np.float32
        )
        np.save(p25_path, logvar_p25)
        np.save(p50_path, logvar_p50)
        np.save(p75_path, logvar_p75)
        np.save(iqr_path, logvar_iqr)

        last = logvar_iqr[-1]
        topk = int(min(args.topK, last.shape[0]))
        top_idx = np.argsort(-last)[:topk].astype(np.int64)
        np.save(topk_path, top_idx)
        np.save(n_list_path, np.array(n_list, dtype=np.int64))
        np.save(labels_path, labels)

        if args.do_mean_ci:
            c, lo, hi, w = bootstrap_ci_stat(
                Y_rv, n_list, B=args.B, seed=args.seed, stat="mean", chunk_rv=512, dtype=np.float32
            )
            np.save(mean_center_path, c)
            np.save(mean_lo_path, lo)
            np.save(mean_hi_path, hi)
            np.save(mean_w_path, w)

        if args.do_q95_ci:
            c, lo, hi, w = bootstrap_ci_stat(
                Y_rv, n_list, B=args.B, seed=args.seed, stat="quantile", q=0.95, chunk_rv=512, dtype=np.float32
            )
            np.save(q95_center_path, c)
            np.save(q95_lo_path, lo)
            np.save(q95_hi_path, hi)
            np.save(q95_w_path, w)
        del Y_rv
    else:
        print("[cache] reuse full cached stats (skip reading raw big_npy)")

    n_list = np.load(n_list_path).astype(np.int64).tolist()
    labels = np.load(labels_path, allow_pickle=True).tolist()
    logvar_p25 = np.load(p25_path, mmap_mode="r")
    logvar_p50 = np.load(p50_path, mmap_mode="r")
    logvar_p75 = np.load(p75_path, mmap_mode="r")
    logvar_iqr = np.load(iqr_path, mmap_mode="r")
    top_idx = np.load(topk_path).astype(np.int64).tolist()
    cat_ids = np.load(cat_ids_path).astype(np.int64) if os.path.isfile(cat_ids_path) else None

    # ---------- Plot 1: IQR curves for topK ----------
    out1 = os.path.join(fig_dir, "fig_sampling_varIQR_topK.png")
    plot_topk_iqr_curves(
        n_list, logvar_iqr, top_idx, labels, out1,
        title=f"Variance estimation uncertainty vs sample size (Top-{len(top_idx)} most unstable at max n)"
    )
    print(f"[fig] saved {out1}")

    # ---------- Plot 1b: IQR curves by category (all categories) ----------
    if cat_ids is not None and len(cat_ids) == logvar_iqr.shape[1]:
        out1b = os.path.join(fig_dir, "fig_sampling_varIQR_by_category.png")
        plot_iqr_by_category(
            n_list, logvar_iqr, cat_ids, out1b,
            title="Variance estimation uncertainty vs sample size (all categories)"
        )
        print(f"[fig] saved {out1b}")
    else:
        print("[fig] skip category plot: category_ids not found or shape mismatch")

    # ---------- Plot 2/3: mean CI ----------
    if args.do_mean_ci:
        if all(os.path.isfile(p) for p in [mean_center_path, mean_lo_path, mean_hi_path, mean_w_path]):
            c = np.load(mean_center_path, mmap_mode="r")
            lo = np.load(mean_lo_path, mmap_mode="r")
            hi = np.load(mean_hi_path, mmap_mode="r")
            w = np.load(mean_w_path, mmap_mode="r")
        else:
            raise RuntimeError("Mean CI stats missing. Please provide raw --big_npy to recompute.")
        rv0 = top_idx[0]
        plot_ci_band(n_list, c, lo, hi, rv0, labels[rv0],
                     os.path.join(fig_dir, "fig_sampling_mean_ci_band.png"),
                     ylabel="Mean(Y) with 95% bootstrap CI",
                     title="Mean estimation convergence (example RV)")
        plot_loglog_slope(n_list, w, rv0, labels[rv0],
                          os.path.join(fig_dir, "fig_sampling_mean_ci_loglog.png"),
                          title="CI width vs n (log-log)")

    # ---------- Plot 4/5: q95 CI ----------
    if args.do_q95_ci:
        if all(os.path.isfile(p) for p in [q95_center_path, q95_lo_path, q95_hi_path, q95_w_path]):
            c = np.load(q95_center_path, mmap_mode="r")
            lo = np.load(q95_lo_path, mmap_mode="r")
            hi = np.load(q95_hi_path, mmap_mode="r")
            w = np.load(q95_w_path, mmap_mode="r")
        else:
            raise RuntimeError("q95 CI stats missing. Please provide raw --big_npy to recompute.")
        rv0 = top_idx[0]
        plot_ci_band(n_list, c, lo, hi, rv0, labels[rv0],
                     os.path.join(fig_dir, "fig_sampling_q95_ci_band.png"),
                     ylabel="q95(Y) with 95% bootstrap CI",
                     title="Tail quantile stabilization (example RV)")
        plot_loglog_slope(n_list, w, rv0, labels[rv0],
                          os.path.join(fig_dir, "fig_sampling_q95_ci_loglog.png"),
                          title="Tail CI width vs n (log-log)")

    print("[done] All requested computations/plots complete.")


if __name__ == "__main__":
    main()