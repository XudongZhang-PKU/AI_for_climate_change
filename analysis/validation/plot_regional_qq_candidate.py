#!/usr/bin/env python3
"""Plot aggregate and regional Q-Q diagnostics with pair-level distances."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


TEAL = "#177D89"
TEAL_LIGHT = "#98CAD0"
ORANGE = "#D7793E"
GREY = "#8B8B8B"
LIGHT_GREY = "#D7D7D7"
BLACK = "#222222"


REGION_LABELS = {
    "World": "World",
    "R10AFRICA": "Africa",
    "R10CHINA+": "China+",
    "R10EUROPE": "Europe",
    "R10INDIA+": "India+",
    "R10LATIN_AM": "Latin America",
    "R10MIDDLE_EAST": "Middle East",
    "R10NORTH_AM": "North America",
    "R10PAC_OECD": "Pacific OECD",
    "R10REF_ECON": "Reforming economies",
    "R10REST_ASIA": "Rest Asia",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--region-map", required=True)
    parser.add_argument("--variable-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="Supplementary_regional_QQ_candidate")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def read_map(path: Path) -> dict[int, str]:
    frame = pd.read_csv(path)
    return {int(row.Code): str(row.Value) for row in frame.itertuples()}


def shorten_variable(name: str) -> str:
    parts = name.split("|")
    if len(parts) >= 3:
        return " | ".join(parts[-2:])
    return name


def load_blocks(path: Path) -> dict[tuple[int, int], list[dict[str, np.ndarray]]]:
    data = np.load(path, allow_pickle=False)
    keys = np.asarray(data["keys"], dtype=int)
    means = np.asarray(data["train_means"], dtype=float)
    stds = np.asarray(data["train_stds"], dtype=float)
    blocks = defaultdict(list)
    for index, (category, region, variable) in enumerate(keys):
        scale = max(float(stds[index]), 1e-8)
        real = (np.asarray(data[f"real_{index}"], dtype=float) - float(means[index])) / scale
        generated = (np.asarray(data[f"syn_{index}"], dtype=float) - float(means[index])) / scale
        real = real[np.all(np.isfinite(real), axis=1)]
        generated = generated[np.all(np.isfinite(generated), axis=1)]
        if len(real) and len(generated):
            blocks[(int(region), int(variable))].append(
                {"category": int(category), "real": real, "generated": generated}
            )
    return blocks


def category_matched_sample(
    blocks: list[dict[str, np.ndarray]], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    real_parts, generated_parts = [], []
    for block in blocks:
        real = block["real"]
        generated = block["generated"]
        indices = rng.choice(len(generated), size=len(real), replace=len(generated) < len(real))
        real_parts.append(real)
        generated_parts.append(generated[indices])
    return np.concatenate(real_parts), np.concatenate(generated_parts)


def panel_label(ax: plt.Axes, label: str, x: float = -0.16, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def main() -> None:
    args = parse_args()
    set_style()
    rng = np.random.default_rng(args.seed)
    blocks = load_blocks(Path(args.npz))
    region_names = read_map(Path(args.region_map))
    variable_names = read_map(Path(args.variable_map))
    q_grid = np.arange(0.05, 0.951, 0.05)

    metric_rows = []
    quantile_rows = []
    region_points = defaultdict(lambda: {"real": [], "generated": []})
    all_real_q, all_generated_q = [], []

    for (region, variable), pair_blocks in sorted(blocks.items()):
        real, generated = category_matched_sample(pair_blocks, rng)
        real_flat = real.ravel()
        generated_flat = generated.ravel()
        real_q = np.quantile(real_flat, q_grid)
        generated_q = np.quantile(generated_flat, q_grid)
        quantile_w1 = float(np.mean(np.abs(real_q - generated_q)))
        full_w1 = float(wasserstein_distance(real_flat, generated_flat))
        ks = float(ks_2samp(real_flat, generated_flat).statistic)
        metric_rows.append(
            {
                "region_code": region,
                "region": REGION_LABELS.get(region_names.get(region, str(region)), region_names.get(region, str(region))),
                "variable_code": variable,
                "variable": variable_names.get(variable, str(variable)),
                "n_empirical_pathways": len(real),
                "central_quantile_w1": quantile_w1,
                "wasserstein_normalized": full_w1,
                "ks_statistic": ks,
            }
        )
        for probability, x_value, y_value in zip(q_grid, real_q, generated_q):
            quantile_rows.append(
                {
                    "region_code": region,
                    "variable_code": variable,
                    "quantile": probability,
                    "empirical": x_value,
                    "generated": y_value,
                }
            )
        region_points[region]["real"].extend(real_q)
        region_points[region]["generated"].extend(generated_q)
        all_real_q.extend(real_q)
        all_generated_q.extend(generated_q)

    metrics = pd.DataFrame(metric_rows)
    quantiles = pd.DataFrame(quantile_rows)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "regional_qq_pair_metrics.csv", index=False)
    quantiles.to_csv(output_dir / "regional_qq_quantiles.csv", index=False)

    all_values = np.concatenate([np.asarray(all_real_q), np.asarray(all_generated_q)])
    display_limit = float(max(1.5, np.quantile(np.abs(all_values), 0.975)))
    display_limit = min(display_limit, 4.0)

    fig = plt.figure(figsize=(7.2, 8.15), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[0.95, 2.05])
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.28)
    ax_overall = fig.add_subplot(top[0, 0])
    ax_box = fig.add_subplot(top[0, 1])

    x_all = np.asarray(all_real_q)
    y_all = np.asarray(all_generated_q)
    inside = (np.abs(x_all) <= display_limit) & (np.abs(y_all) <= display_limit)
    ax_overall.scatter(x_all[inside], y_all[inside], s=5, alpha=0.18, color=TEAL, linewidths=0)
    ax_overall.plot(
        [-display_limit, display_limit], [-display_limit, display_limit],
        color=BLACK, lw=0.9, ls="--",
    )
    ax_overall.set(
        xlabel="Empirical quantiles",
        ylabel="Generated quantiles",
        xlim=(-display_limit, display_limit),
        ylim=(-display_limit, display_limit),
    )
    ax_overall.set_aspect("equal", adjustable="box")
    panel_label(ax_overall, "a")
    median_w1 = float(metrics["wasserstein_normalized"].median())
    q95_w1 = float(metrics["wasserstein_normalized"].quantile(0.95))
    ax_overall.text(
        0.04, 0.96,
        f"Median W1 = {median_w1:.3f}\n95th percentile = {q95_w1:.3f}",
        transform=ax_overall.transAxes, ha="left", va="top", fontsize=7.5,
    )

    region_order = sorted(region_points)
    box_data = [
        metrics.loc[metrics["region_code"] == region, "wasserstein_normalized"].to_numpy()
        for region in region_order
    ]
    positions = np.arange(len(region_order))
    bp = ax_box.boxplot(
        box_data,
        positions=positions,
        vert=False,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": TEAL, "linewidth": 1.2},
        boxprops={"facecolor": "#E8F2F3", "edgecolor": TEAL, "linewidth": 0.8},
        whiskerprops={"color": GREY, "linewidth": 0.8},
        capprops={"color": GREY, "linewidth": 0.8},
    )
    del bp
    for pos, values in zip(positions, box_data):
        jitter = rng.normal(0.0, 0.065, size=len(values))
        ax_box.scatter(values, np.full(len(values), pos) + jitter, s=7, color=GREY, alpha=0.42, linewidths=0)

    ax_box.set_xscale("log")
    ax_box.set_yticks(positions)
    ax_box.set_yticklabels(
        [REGION_LABELS.get(region_names.get(region, str(region)), region_names.get(region, str(region))) for region in region_order]
    )
    ax_box.invert_yaxis()
    ax_box.set_xlabel("Wasserstein distance (standardized units)")
    ax_box.grid(axis="x", color=LIGHT_GREY, linewidth=0.45, alpha=0.65)
    panel_label(ax_box, "c", x=-0.18)

    regional = outer[1].subgridspec(3, 4, wspace=0.18, hspace=0.24)
    regional_axes = []
    for index, region in enumerate(region_order):
        ax = fig.add_subplot(regional[index // 4, index % 4])
        regional_axes.append(ax)
        x = np.asarray(region_points[region]["real"])
        y = np.asarray(region_points[region]["generated"])
        inside = (np.abs(x) <= display_limit) & (np.abs(y) <= display_limit)
        ax.scatter(x[inside], y[inside], s=4, alpha=0.20, color=TEAL, linewidths=0)
        ax.plot(
            [-display_limit, display_limit], [-display_limit, display_limit],
            color=BLACK, lw=0.7, ls="--",
        )
        ax.set_xlim(-display_limit, display_limit)
        ax.set_ylim(-display_limit, display_limit)
        ax.set_aspect("equal", adjustable="box")
        label = REGION_LABELS.get(region_names.get(region, str(region)), region_names.get(region, str(region)))
        region_values = metrics.loc[metrics["region_code"] == region, "wasserstein_normalized"]
        ax.set_title(f"{label}\nmedian W1 = {region_values.median():.3f}", pad=2)
        if index % 4 == 0:
            ax.set_ylabel("Generated")
        else:
            ax.set_yticklabels([])
        if index // 4 == 2:
            ax.set_xlabel("Empirical")
        else:
            ax.set_xticklabels([])
    regional_axes[0].text(
        -0.27, 1.20, "b", transform=regional_axes[0].transAxes,
        fontsize=11, fontweight="bold", va="top",
    )
    legend_ax = fig.add_subplot(regional[2, 3])
    legend_ax.axis("off")

    prefix = output_dir / args.output_name
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    series_mae = []
    series_rmse = []
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    element_count = 0
    for pair_blocks in blocks.values():
        for block in pair_blocks:
            real = block["real"]
            generated = block["generated"]
            if len(generated) % len(real) != 0:
                raise ValueError("Generated rows cannot be aligned to empirical parent pathways.")
            samples_per_parent = len(generated) // len(real)
            generated = generated.reshape(len(real), samples_per_parent, generated.shape[1])
            difference = generated.mean(axis=1) - real
            series_mae.extend(np.mean(np.abs(difference), axis=1))
            series_rmse.extend(np.sqrt(np.mean(np.square(difference), axis=1)))
            absolute_error_sum += float(np.abs(difference).sum())
            squared_error_sum += float(np.square(difference).sum())
            element_count += int(difference.size)

    summary = pd.DataFrame(
        [{
            "region_variable_pairs": len(metrics),
            "median_central_quantile_w1": float(metrics["central_quantile_w1"].median()),
            "q95_central_quantile_w1": float(metrics["central_quantile_w1"].quantile(0.95)),
            "median_full_wasserstein": float(metrics["wasserstein_normalized"].median()),
            "q95_full_wasserstein": float(metrics["wasserstein_normalized"].quantile(0.95)),
            "median_ks": float(metrics["ks_statistic"].median()),
            "q95_ks": float(metrics["ks_statistic"].quantile(0.95)),
            "macro_mae": float(np.mean(series_mae)),
            "micro_mae": absolute_error_sum / element_count,
            "macro_rmse": float(np.mean(series_rmse)),
            "micro_rmse": float(np.sqrt(squared_error_sum / element_count)),
            "seed": args.seed,
        }]
    )
    summary.to_csv(output_dir / "regional_qq_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
