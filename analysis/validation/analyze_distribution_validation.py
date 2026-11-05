#!/usr/bin/env python3
"""Balanced distribution and mode-coverage diagnostics for AR6 pathways."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import ks_2samp, wasserstein_distance


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="Untrimmed fig2a_real_syn.npz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-real", type=int, default=8)
    parser.add_argument("--knn", type=int, default=5)
    return parser.parse_args()


def rbf_kernel(x: np.ndarray, y: np.ndarray, bandwidths: np.ndarray) -> np.ndarray:
    d2 = cdist(x, y, metric="sqeuclidean")
    kernels = [np.exp(-d2 / (2.0 * bw * bw)) for bw in bandwidths]
    return np.mean(kernels, axis=0)


def empirical_bandwidths(x: np.ndarray) -> np.ndarray:
    distances = pdist(x, metric="euclidean")
    positive = distances[np.isfinite(distances) & (distances > 0)]
    median = float(np.median(positive)) if positive.size else 1.0
    return np.asarray([0.5, 1.0, 2.0], dtype=float) * max(median, 1e-6)


def mmd2(x: np.ndarray, y: np.ndarray, bandwidths: np.ndarray) -> tuple[float, float]:
    kxx = rbf_kernel(x, x, bandwidths)
    kyy = rbf_kernel(y, y, bandwidths)
    kxy = rbf_kernel(x, y, bandwidths)
    biased = float(max(0.0, kxx.mean() + kyy.mean() - 2.0 * kxy.mean()))
    if len(x) < 2 or len(y) < 2:
        return float("nan"), biased
    unbiased = (
        (kxx.sum() - np.trace(kxx)) / (len(x) * (len(x) - 1))
        + (kyy.sum() - np.trace(kyy)) / (len(y) * (len(y) - 1))
        - 2.0 * kxy.mean()
    )
    return float(unbiased), biased


def prdc_precision_recall(real: np.ndarray, generated: np.ndarray, k: int) -> tuple[float, float]:
    k_real = min(k, len(real) - 1)
    k_gen = min(k, len(generated) - 1)
    if k_real < 1 or k_gen < 1:
        return float("nan"), float("nan")
    rr = squareform(pdist(real))
    gg = squareform(pdist(generated))
    real_radius = np.partition(rr, kth=k_real, axis=1)[:, k_real]
    gen_radius = np.partition(gg, kth=k_gen, axis=1)[:, k_gen]
    rg = cdist(real, generated)
    precision = np.mean(np.any(rg <= real_radius[:, None], axis=0))
    recall = np.mean(np.any(rg <= gen_radius[None, :], axis=1))
    return float(precision), float(recall)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if np.isfinite(denominator) and denominator > EPS else float("nan")


def q(values: np.ndarray, prob: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, prob)) if values.size else float("nan")


def central_standardized(
    real: np.ndarray, generated: np.ndarray, low: float = 0.05, high: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Return the common central range in training-standardized units."""
    real = np.asarray(real, dtype=float)
    generated = np.asarray(generated, dtype=float)
    lo, hi = np.quantile(np.concatenate([real, generated]), [low, high])
    real_c = real[(real >= lo) & (real <= hi)]
    gen_c = generated[(generated >= lo) & (generated <= hi)]
    return real_c, gen_c


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_median": q(values, 0.5),
        f"{prefix}_q025": q(values, 0.025),
        f"{prefix}_q975": q(values, 0.975),
    }


def load_groups(path: Path) -> dict[tuple[int, int], list[dict[str, np.ndarray]]]:
    data = np.load(path, allow_pickle=False)
    keys = np.asarray(data["keys"], dtype=int)
    means = np.asarray(data["train_means"], dtype=float)
    stds = np.asarray(data["train_stds"], dtype=float)
    rv_groups: dict[tuple[int, int], list[dict[str, np.ndarray]]] = defaultdict(list)
    for i, (category, region, variable) in enumerate(keys):
        real = np.asarray(data[f"real_{i}"], dtype=float)
        generated = np.asarray(data[f"syn_{i}"], dtype=float)
        if real.ndim != 2 or generated.ndim != 2:
            raise ValueError("Use the untrimmed NPZ with arrays shaped [pathway, time].")
        scale = max(float(stds[i]), 1e-8)
        real = (real - float(means[i])) / scale
        generated = (generated - float(means[i])) / scale
        real = real[np.all(np.isfinite(real), axis=1)]
        generated = generated[np.all(np.isfinite(generated), axis=1)]
        if len(real) and len(generated):
            rv_groups[(int(region), int(variable))].append(
                {"category": int(category), "real": real, "generated": generated}
            )
    return rv_groups


def matched_sample(
    blocks: list[dict[str, np.ndarray]], rng: np.random.Generator, bootstrap_real: bool
) -> tuple[np.ndarray, np.ndarray]:
    real_parts, gen_parts = [], []
    for block in blocks:
        real = block["real"]
        generated = block["generated"]
        if bootstrap_real:
            real_idx = rng.choice(len(real), size=len(real), replace=True)
            real_selected = real[real_idx]
        else:
            real_selected = real
        gen_idx = rng.choice(len(generated), size=len(real), replace=len(generated) < len(real))
        real_parts.append(real_selected)
        gen_parts.append(generated[gen_idx])
    return np.concatenate(real_parts), np.concatenate(gen_parts)


def empirical_baseline(
    blocks: list[dict[str, np.ndarray]], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    left, right = [], []
    for block in blocks:
        real = block["real"]
        left.append(real[rng.choice(len(real), size=len(real), replace=True)])
        right.append(real[rng.choice(len(real), size=len(real), replace=True)])
    return np.concatenate(left), np.concatenate(right)


def analyze_rv(
    blocks: list[dict[str, np.ndarray]], bootstrap: int, seed: int, k: int
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    real_fixed, gen_fixed = matched_sample(blocks, rng, bootstrap_real=False)
    bandwidths = empirical_bandwidths(real_fixed)
    metrics = defaultdict(list)

    for _ in range(bootstrap):
        real, generated = matched_sample(blocks, rng, bootstrap_real=True)
        base_left, base_right = empirical_baseline(blocks, rng)
        mmd_u, mmd_b = mmd2(real, generated, bandwidths)
        _, base_b = mmd2(base_left, base_right, bandwidths)
        real_flat, gen_flat = real.ravel(), generated.ravel()
        real_central, gen_central = central_standardized(real_flat, gen_flat)
        metrics["mmd2_unbiased"].append(mmd_u)
        metrics["mmd2_biased"].append(mmd_b)
        metrics["mmd2_empirical_baseline"].append(base_b)
        metrics["wasserstein"].append(wasserstein_distance(real_flat, gen_flat))
        metrics["ks"].append(ks_2samp(real_flat, gen_flat).statistic)
        metrics["central_wasserstein"].append(
            wasserstein_distance(real_central, gen_central)
            if real_central.size and gen_central.size
            else float("nan")
        )
        metrics["central_ks"].append(
            ks_2samp(real_central, gen_central).statistic
            if real_central.size and gen_central.size
            else float("nan")
        )

        real_pd = pdist(real)
        gen_pd = pdist(generated)
        metrics["pairwise_distance_ratio"].append(
            safe_ratio(float(np.median(gen_pd)), float(np.median(real_pd)))
        )
        real_summary = real.mean(axis=1)
        gen_summary = generated.mean(axis=1)
        metrics["variance_ratio"].append(
            safe_ratio(float(np.var(gen_summary, ddof=1)), float(np.var(real_summary, ddof=1)))
        )
        precision, recall = prdc_precision_recall(real, generated, k)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)

    q_grid = np.arange(0.05, 0.951, 0.05)
    real_central, gen_central = central_standardized(real_fixed.ravel(), gen_fixed.ravel())
    qq_real = np.quantile(real_central, q_grid)
    qq_generated = np.quantile(gen_central, q_grid)
    point = {
        "n_real": int(len(real_fixed)),
        "n_generated_available": int(sum(len(b["generated"]) for b in blocks)),
        "n_categories": int(len(blocks)),
        "bandwidth_median": float(bandwidths[1]),
        "qq_mean_absolute_deviation": float(np.mean(np.abs(qq_real - qq_generated))),
    }
    arrays = {name: np.asarray(values, dtype=float) for name, values in metrics.items()}
    for name, values in arrays.items():
        point.update(summarize(values, name))
    point["mmd2_to_empirical_baseline_ratio"] = safe_ratio(
        point["mmd2_biased_median"], point["mmd2_empirical_baseline_median"]
    )
    return point, arrays


def aggregate_summary(frame: pd.DataFrame) -> dict[str, float]:
    summary: dict[str, float] = {
        "region_variable_pairs": int(len(frame)),
        "total_empirical_pair_records": int(frame["n_real"].sum()),
    }
    for metric in [
        "mmd2_unbiased_median",
        "mmd2_biased_median",
        "mmd2_empirical_baseline_median",
        "wasserstein_median",
        "ks_median",
        "central_wasserstein_median",
        "central_ks_median",
        "pairwise_distance_ratio_median",
        "variance_ratio_median",
        "precision_median",
        "recall_median",
        "qq_mean_absolute_deviation",
    ]:
        values = frame[metric].to_numpy(dtype=float)
        summary[f"{metric}_across_rv_median"] = q(values, 0.5)
        summary[f"{metric}_across_rv_q95"] = q(values, 0.95)
        summary[f"{metric}_across_rv_max"] = q(values, 1.0)
    return summary


def aggregate_point_errors(
    groups: dict[tuple[int, int], list[dict[str, np.ndarray]]]
) -> dict[str, float]:
    """Compare each empirical trajectory with its generated ensemble mean."""
    series_mae: list[float] = []
    series_rmse: list[float] = []
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    element_count = 0
    generated_count = 0
    samples_per_parent: list[int] = []

    for blocks in groups.values():
        for block in blocks:
            real = block["real"]
            generated = block["generated"]
            n_parent = len(real)
            if n_parent == 0:
                continue
            if len(generated) % n_parent != 0:
                raise ValueError(
                    "Generated trajectories cannot be assigned evenly to empirical parents: "
                    f"n_generated={len(generated)}, n_parent={n_parent}."
                )
            n_samples = len(generated) // n_parent
            generated_by_parent = generated.reshape(n_parent, n_samples, real.shape[1])
            prediction = generated_by_parent.mean(axis=1)
            difference = prediction - real

            series_mae.extend(np.mean(np.abs(difference), axis=1).tolist())
            series_rmse.extend(np.sqrt(np.mean(np.square(difference), axis=1)).tolist())
            absolute_error_sum += float(np.abs(difference).sum())
            squared_error_sum += float(np.square(difference).sum())
            element_count += int(difference.size)
            generated_count += int(generated.size // generated.shape[1])
            samples_per_parent.append(int(n_samples))

    if not series_mae or element_count == 0:
        raise ValueError("No parent-matched trajectories were available for MAE/RMSE.")

    return {
        "macro_mae": float(np.mean(series_mae)),
        "macro_rmse": float(np.mean(series_rmse)),
        "micro_mae": float(absolute_error_sum / element_count),
        "micro_rmse": float(np.sqrt(squared_error_sum / element_count)),
        "parent_region_variable_series": int(len(series_mae)),
        "generated_region_variable_series": int(generated_count),
        "generated_samples_per_parent_min": int(min(samples_per_parent)),
        "generated_samples_per_parent_max": int(max(samples_per_parent)),
    }


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = load_groups(npz_path)

    rows = []
    bootstrap_store = {}
    for offset, ((region, variable), blocks) in enumerate(sorted(groups.items())):
        n_real = sum(len(block["real"]) for block in blocks)
        if n_real < args.min_real:
            continue
        row, arrays = analyze_rv(
            blocks, bootstrap=args.bootstrap, seed=args.seed + offset * 7919, k=args.knn
        )
        row.update({"region_code": region, "variable_code": variable})
        rows.append(row)
        for name, values in arrays.items():
            bootstrap_store[f"r{region}_v{variable}_{name}"] = values

    frame = pd.DataFrame(rows).sort_values(["region_code", "variable_code"])
    frame.to_csv(output_dir / "region_variable_metrics.csv", index=False)
    np.savez_compressed(output_dir / "region_variable_bootstrap.npz", **bootstrap_store)

    summary = aggregate_summary(frame)
    summary.update(aggregate_point_errors(groups))
    for metric in [
        "mmd2_unbiased",
        "mmd2_biased",
        "mmd2_empirical_baseline",
        "central_wasserstein",
        "central_ks",
        "precision",
        "recall",
    ]:
        matrix = np.vstack(
            [
                bootstrap_store[f"r{int(row.region_code)}_v{int(row.variable_code)}_{metric}"]
                for row in frame.itertuples()
            ]
        )
        macro = np.nanmean(matrix, axis=0)
        summary.update(summarize(macro, f"macro_{metric}"))
    summary.update(
        {
            "input_npz": str(npz_path),
            "bootstrap_replicates": args.bootstrap,
            "seed": args.seed,
            "minimum_empirical_pathways_per_pair": args.min_real,
            "knn": args.knn,
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)

    worst = frame.nlargest(12, "central_wasserstein_median")
    worst.to_csv(output_dir / "worst_region_variable_pairs.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
