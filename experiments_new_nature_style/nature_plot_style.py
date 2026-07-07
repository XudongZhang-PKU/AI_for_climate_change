# -*- coding: utf-8 -*-
"""Shared Nature-style plotting helpers for the Fig.1/Fig.2 scripts."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Annotation, Text
from matplotlib.ticker import NullFormatter


PALETTE = {
    "navy": "#1B4965",
    "deep_blue": "#3C5488",
    "teal": "#62B6CB",
    "cyan": "#4DBBD5",
    "green": "#00A087",
    "green_soft": "#BFD7B9",
    "coral": "#E64B35",
    "coral_soft": "#F39B7F",
    "grey_dark": "#333333",
    "grey_mid": "#808080",
    "grey_light": "#D3D3D3",
    "grey_soft": "#E8EEF0",
    "white": "#FFFFFF",
}

CATEGORY_COLORS = [
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#8491B4",
    "#F39B7F",
    "#E64B35",
    "#8CCBBE",
    "#A7C957",
]

CORRELATION_CMAP = LinearSegmentedColormap.from_list(
    "nature_correlation", [PALETTE["coral"], "#F8F9FA", PALETTE["navy"]]
)

TEAL_CMAP = LinearSegmentedColormap.from_list(
    "nature_teal", ["#F7FBFC", "#D7EEF2", PALETTE["teal"], PALETTE["navy"]]
)


def category_color(index: int) -> str:
    return CATEGORY_COLORS[index % len(CATEGORY_COLORS)]


def apply_nature_style(font_scale: float = 2.0) -> None:
    base = 10 * font_scale
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": base,
            "axes.titlesize": 11 * font_scale,
            "axes.labelsize": 10 * font_scale,
            "xtick.labelsize": 9 * font_scale,
            "ytick.labelsize": 9 * font_scale,
            "legend.fontsize": 8.5 * font_scale,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.9,
            "axes.edgecolor": PALETTE["grey_dark"],
            "xtick.color": PALETTE["grey_dark"],
            "ytick.color": PALETTE["grey_dark"],
            "text.color": PALETTE["grey_dark"],
            "axes.labelcolor": PALETTE["grey_dark"],
            "legend.frameon": False,
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "axes.grid": False,
            "axes.unicode_minus": False,
        }
    )


def _split_output_paths(out_path: str | os.PathLike[str]) -> tuple[Path, Path, Path, Path, Path | None]:
    out = Path(out_path)
    suffix = out.suffix.lower()
    root = out.with_suffix("") if suffix else out
    png = out if suffix == ".png" else root.with_suffix(".png")
    svg = out if suffix == ".svg" else root.with_suffix(".svg")
    extra = out if suffix and suffix not in {".png", ".svg"} else None
    no_text_png = root.with_name(root.name + "_notext").with_suffix(".png")
    no_text_svg = root.with_name(root.name + "_notext").with_suffix(".svg")
    return png, svg, no_text_png, no_text_svg, extra


def strip_all_text(fig) -> None:
    for legend in fig.findobj(match=lambda artist: artist.__class__.__name__ == "Legend"):
        legend.set_visible(False)

    for ax in fig.axes:
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        if hasattr(ax, "set_zlabel"):
            ax.set_zlabel("")

        ax.tick_params(
            axis="both",
            which="both",
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())

        for axis in (ax.xaxis, ax.yaxis):
            axis.label.set_visible(False)
            axis.get_offset_text().set_visible(False)
        if hasattr(ax, "zaxis"):
            ax.zaxis.set_major_formatter(NullFormatter())
            ax.zaxis.set_minor_formatter(NullFormatter())
            ax.zaxis.label.set_visible(False)
            ax.zaxis.get_offset_text().set_visible(False)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_visible(False)
        if hasattr(ax, "get_zticklabels"):
            for label in ax.get_zticklabels():
                label.set_visible(False)

        legend = ax.get_legend()
        if legend is not None:
            legend.set_visible(False)

    for text in fig.findobj(match=Text):
        if isinstance(text, Annotation):
            if text.get_text():
                text.set_text("")
            continue
        text.set_text("")
        text.set_visible(False)


def save_dual(
    fig,
    out_path: str | os.PathLike[str],
    *,
    transparent: bool = False,
    dpi: int = 600,
    pad_inches: float = 0.04,
) -> list[str]:
    png, svg, no_text_png, no_text_svg, extra = _split_output_paths(out_path)
    for path in (png, svg, no_text_png, no_text_svg, extra):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight_bbox = fig.get_tightbbox(renderer).padded(pad_inches)

    saved = []
    if extra is not None:
        fig.savefig(extra, dpi=dpi, transparent=transparent, bbox_inches=tight_bbox)
        saved.append(str(extra))
    fig.savefig(png, dpi=dpi, transparent=transparent, bbox_inches=tight_bbox)
    fig.savefig(svg, transparent=transparent, bbox_inches=tight_bbox)
    saved.extend([str(png), str(svg)])

    strip_all_text(fig)
    fig.canvas.draw()
    fig.savefig(no_text_png, dpi=dpi, transparent=transparent, bbox_inches=tight_bbox)
    fig.savefig(no_text_svg, transparent=transparent, bbox_inches=tight_bbox)
    saved.extend([str(no_text_png), str(no_text_svg)])
    return saved
