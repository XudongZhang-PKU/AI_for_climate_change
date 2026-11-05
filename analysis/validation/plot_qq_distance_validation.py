#!/usr/bin/env python3
"""Plot aggregate Q-Q agreement, pairwise distances and worst-case Q-Q panels."""

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


TEAL = "#137C8B"
TEAL_LIGHT = "#9BCFD3"
ORANGE = "#D97941"
GREY = "#8B8B8B"
BLACK = "#222222"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--region-map", required=True)
    parser.add_argument("--variable-map", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--worst-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def name_map(path: Path) -> dict[int, str]:
    frame = pd.read_csv(path)
    return {int(row.Code): str(row.Value) for row in frame.itertuples()}


def shorten_variable(name: str) -> str:
    parts = name.split("|")
    return " | ".join(parts[-2:]) if len(parts) > 2 else name


def load_normalized_rv(path: Path) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=False)
    keys = np.asarray(data["keys"], dtype=int)
    means = np.asarray(data["train_means"], dtype=float)
    stds = np.asarray(data["train_stds"], dtype=float)
    groups = defaultdict(lambda: {"real": [], "generated": []})
    for i, (_, region, variable) in enumerate(keys):
        scale = max(float(stds[i]), 1e-8)
        real = (np.asarray(data[f"real_{i}"], dtype=float) - float(means[i])) / scale
        generated = (np.asarray(data[f"syn_{i}"], dtype=float) - float(means[i])) / scale
        groups[(int(region), int(variable))]["real"].append(real.ravel())
        groups[(int(region), int(variable))]["generated"].append(generated.ravel())
    return {
        key: {
            "real": np.concatenate(value["real"]),
            "generated": np.concatenate(value["generated"]),
        }
        for key, value in groups.items()
    }


def matched_generated(real: np.ndarray, generated: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = len(real)
    return generated[rng.choice(len(generated), size=size, replace=len(generated) < size)]


def central_dimensionless(real: np.ndarray, generated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.quantile(np.concatenate([real, generated]), [0.05, 0.95])
    real = real[(real >= lo) & (real <= hi)]
    generated = generated[(generated >= lo) & (generated <= hi)]
    scale = max(float(np.max(np.abs(real))) if real.size else 0.0, 1e-12)
    return real / scale, generated / scale


def panel_label(ax, label: str) -> None:
    ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def main() -> None:
    args = parse_args()
    style()
    rng = np.random.default_rng(args.seed)
    rv = load_normalized_rv(Path(args.npz))
    metrics = pd.read_csv(args.metrics)
    regions = name_map(Path(args.region_map))
    variables = name_map(Path(args.variable_map))

    q_grid = np.arange(0.05, 0.951, 0.05)
    aggregate_real, aggregate_generated = [], []
    for key in sorted(rv):
        real = rv[key]["real"]
        generated = matched_generated(real, rv[key]["generated"], rng)
        real, generated = central_dimensionless(real, generated)
        aggregate_real.extend(np.quantile(real, q_grid))
        aggregate_generated.extend(np.quantile(generated, q_grid))
    aggregate_real = np.asarray(aggregate_real)
    aggregate_generated = np.asarray(aggregate_generated)

    worst = metrics.nlargest(args.worst_count, "central_wasserstein_median").copy()
    ranked = metrics.sort_values("central_wasserstein_median").reset_index(drop=True)

    fig = plt.figure(figsize=(7.2, 5.15), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.38, 1.0])
    ax_qq = fig.add_subplot(grid[0, 0])
    ax_rank = fig.add_subplot(grid[0, 1])
    worst_grid = grid[1, :].subgridspec(1, 4, wspace=0.32)
    worst_axes = [fig.add_subplot(worst_grid[0, index]) for index in range(4)]

    display_values = np.concatenate([aggregate_real, aggregate_generated])
    lo, hi = np.quantile(display_values, [0.01, 0.99])
    margin = max(float(hi - lo) * 0.04, 1e-3)
    lo, hi = float(lo - margin), float(hi + margin)
    ax_qq.scatter(aggregate_real, aggregate_generated, s=7, alpha=0.22, color=TEAL, linewidths=0)
    ax_qq.plot([lo, hi], [lo, hi], color=BLACK, lw=1.0, ls="--")
    ax_qq.set(xlabel="Empirical quantiles", ylabel="Generated quantiles", xlim=(lo, hi), ylim=(lo, hi))
    ax_qq.set_aspect("equal", adjustable="box")
    panel_label(ax_qq, "a")
    med_w = metrics["central_wasserstein_median"].median()
    q95_w = metrics["central_wasserstein_median"].quantile(0.95)
    ax_qq.text(
        0.04,
        0.96,
        f"Median normalized W1 = {med_w:.3f}\n95th percentile = {q95_w:.3f}",
        transform=ax_qq.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    x = np.arange(1, len(ranked) + 1)
    ax_rank.scatter(x, ranked["central_wasserstein_median"], s=8, color=GREY, alpha=0.55, linewidths=0)
    worst_mask = ranked.index.isin(ranked.tail(args.worst_count).index)
    ax_rank.scatter(
        x[worst_mask], ranked.loc[worst_mask, "central_wasserstein_median"], s=20, color=ORANGE, zorder=3
    )
    ax_rank.axhline(med_w, color=TEAL, lw=1.0)
    ax_rank.axhline(q95_w, color=TEAL, lw=0.9, ls="--")
    ax_rank.set_yscale("log")
    ax_rank.set(xlabel="Region-variable pair rank", ylabel="Normalized Wasserstein distance")
    panel_label(ax_rank, "b")

    for index, (ax, row) in enumerate(zip(worst_axes, worst.itertuples())):
        key = (int(row.region_code), int(row.variable_code))
        real = rv[key]["real"]
        generated = matched_generated(real, rv[key]["generated"], rng)
        real, generated = central_dimensionless(real, generated)
        xq = np.quantile(real, q_grid)
        yq = np.quantile(generated, q_grid)
        low = float(min(xq.min(), yq.min()))
        high = float(max(xq.max(), yq.max()))
        ax.scatter(xq, yq, s=9, color=ORANGE, alpha=0.8, linewidths=0)
        ax.plot([low, high], [low, high], color=BLACK, lw=0.8, ls="--")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")
        region = regions.get(key[0], f"R{key[0]}").replace("R10", "")
        variable = shorten_variable(variables.get(key[1], f"V{key[1]}"))
        ax.set_title(f"{region}\n{variable}", fontsize=6.5, pad=2)
        ax.set_xlabel("Empirical quantiles", fontsize=6.5, labelpad=1)
        if index == 0:
            ax.set_ylabel("Generated", fontsize=6.5, labelpad=1)
    panel_label(worst_axes[0], "c")

    prefix = Path(args.output_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
