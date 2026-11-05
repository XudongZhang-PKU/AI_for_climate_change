# -*- coding: utf-8 -*-
"""Fig.1 lower-right feasible pathway schematic."""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nature_plot_style import PALETTE, apply_nature_style, save_dual


def smooth_curve(y_anchor, n_dense=400, smooth_window=21):
    x_anchor = np.linspace(0, 1, len(y_anchor))
    x_dense = np.linspace(0, 1, n_dense)
    y_dense = np.interp(x_dense, x_anchor, y_anchor)
    kernel = np.ones(smooth_window) / smooth_window
    y_smooth = np.convolve(y_dense, kernel, mode="same")
    y_smooth[:smooth_window] = y_smooth[smooth_window]
    y_smooth[-smooth_window:] = y_smooth[-smooth_window - 1]
    return x_dense, y_smooth


def build_paths(seed: int = 42):
    rng = np.random.RandomState(seed)
    base_anchor = np.array([0.82, 0.78, 0.64, 0.50, 0.36, 0.24, 0.12, 0.03, -0.02])

    all_paths = []
    for _ in range(12):
        noise = rng.normal(0, 0.075, len(base_anchor))
        tilt = np.linspace(0, rng.normal(0.05, 0.035), len(base_anchor))
        all_paths.append(base_anchor + noise + tilt)

    feasible_paths = []
    for _ in range(4):
        noise = rng.normal(0, 0.03, len(base_anchor))
        tilt = np.linspace(0, rng.normal(-0.015, 0.015), len(base_anchor))
        feasible_paths.append(base_anchor + noise + tilt)

    return np.array(all_paths), np.array(feasible_paths)


def plot_schematic(out_path: str, width_cm: float = 6.0, height_cm: float = 5.0, seed: int = 42):
    apply_nature_style(font_scale=2.0)
    all_paths, feasible_paths = build_paths(seed)

    all_low = np.percentile(all_paths, 4, axis=0)
    all_high = np.percentile(all_paths, 96, axis=0)
    all_mid = all_paths.mean(axis=0)
    feasible_low = np.percentile(feasible_paths, 15, axis=0)
    feasible_high = np.percentile(feasible_paths, 85, axis=0)
    feasible_mid = feasible_paths.mean(axis=0)

    x_all, y_all_low = smooth_curve(all_low)
    _, y_all_high = smooth_curve(all_high)
    _, y_all_mid = smooth_curve(all_mid)
    x_f, y_f_low = smooth_curve(feasible_low)
    _, y_f_high = smooth_curve(feasible_high)
    _, y_f_mid = smooth_curve(feasible_mid)

    all_y_values = np.concatenate([y_all_low, y_all_high, y_all_mid, y_f_low, y_f_high, y_f_mid])
    y_min = all_y_values.min()
    y_max = all_y_values.max()

    def map_x(x):
        return 0.12 + x * (0.94 - 0.12)

    def map_y(y):
        return 0.14 + (y - y_min) / (y_max - y_min) * (0.90 - 0.14)

    cm_to_inch = 1 / 2.54
    fig, ax = plt.subplots(figsize=(width_cm * cm_to_inch, height_cm * cm_to_inch), dpi=300)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    x_all_m = map_x(x_all)
    x_f_m = map_x(x_f)
    y_all_low_m = map_y(y_all_low)
    y_all_high_m = map_y(y_all_high)
    y_all_mid_m = map_y(y_all_mid)
    y_f_low_m = map_y(y_f_low)
    y_f_high_m = map_y(y_f_high)
    y_f_mid_m = map_y(y_f_mid)

    for y in all_paths:
        x, ys = smooth_curve(y, n_dense=350, smooth_window=19)
        ax.plot(
            map_x(x),
            map_y(ys),
            color=PALETTE["grey_light"],
            lw=1.7,
            alpha=0.42,
            solid_capstyle="round",
            zorder=1,
        )

    ax.fill_between(x_all_m, y_all_low_m, y_all_high_m, color=PALETTE["grey_soft"], alpha=0.90, zorder=2)

    for y in feasible_paths:
        x, ys = smooth_curve(y, n_dense=350, smooth_window=19)
        ax.plot(
            map_x(x),
            map_y(ys),
            color=PALETTE["teal"],
            lw=2.0,
            alpha=0.35,
            solid_capstyle="round",
            zorder=3,
        )

    ax.fill_between(x_f_m, y_f_low_m, y_f_high_m, color="#D7EEF2", alpha=0.76, zorder=4)
    ax.plot(x_all_m, y_all_mid_m, color="#B9C3C7", lw=1.7, alpha=0.95, solid_capstyle="round", zorder=5)
    ax.plot(x_f_m, y_f_mid_m, color=PALETTE["white"], lw=4.0, alpha=0.98, solid_capstyle="round", zorder=6)
    ax.plot(x_f_m, y_f_mid_m, color=PALETTE["navy"], lw=2.4, alpha=0.98, solid_capstyle="round", zorder=7)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)

    arrow_style = dict(
        arrowstyle="->",
        color=PALETTE["grey_dark"],
        lw=1.15,
        mutation_scale=10,
        shrinkA=0,
        shrinkB=0,
    )
    ax.annotate(
        "",
        xy=(0.97, 0.12),
        xytext=(0.10, 0.12),
        xycoords="data",
        textcoords="data",
        arrowprops=arrow_style,
        annotation_clip=False,
        zorder=20,
    )
    ax.annotate(
        "",
        xy=(0.10, 0.94),
        xytext=(0.10, 0.12),
        xycoords="data",
        textcoords="data",
        arrowprops=arrow_style,
        annotation_clip=False,
        zorder=20,
    )

    saved = save_dual(fig, out_path, transparent=True, dpi=600, pad_inches=0.03)
    plt.close(fig)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Fig.1 feasible pathway schematic")
    parser.add_argument("--out", type=str, default="schematic_feasible_pathways_6x5cm_transparent.png")
    parser.add_argument("--width_cm", type=float, default=6.0)
    parser.add_argument("--height_cm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    saved = plot_schematic(args.out, width_cm=args.width_cm, height_cm=args.height_cm, seed=args.seed)
    print("[plot_a] saved " + ", ".join(saved))


if __name__ == "__main__":
    main()
