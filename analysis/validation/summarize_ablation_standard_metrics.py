#!/usr/bin/env python3
"""Collect uniform split-2026 fidelity metrics for the manuscript ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--variants", default=str(SCRIPT_DIR / "ablation_variants.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.results_root).resolve()
    variants = json.loads(Path(args.variants).read_text(encoding="utf-8"))
    rows = []
    for variant_id, variant in variants.items():
        summary_path = root / variant_id / "analysis" / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "variant": variant_id,
                "Model": variant["display_name"],
                "MMD2_unbiased": summary["macro_mmd2_unbiased_median"],
                "MMD2_CI_low": summary["macro_mmd2_unbiased_q025"],
                "MMD2_CI_high": summary["macro_mmd2_unbiased_q975"],
                "MMD2_biased": summary["macro_mmd2_biased_median"],
                "MAE": summary["macro_mae"],
                "RMSE": summary["macro_rmse"],
                "micro_MAE": summary["micro_mae"],
                "micro_RMSE": summary["micro_rmse"],
                "region_variable_pairs": summary["region_variable_pairs"],
                "parent_region_variable_series": summary[
                    "parent_region_variable_series"
                ],
                "generated_samples_per_parent": summary[
                    "generated_samples_per_parent_min"
                ],
            }
        )

    frame = pd.DataFrame(rows)
    full = frame.loc[frame["variant"] == "full"].iloc[0]
    for metric in ["MMD2_unbiased", "MAE", "RMSE"]:
        frame[f"{metric}_change_vs_full_pct"] = (
            (frame[metric] - full[metric]) / full[metric] * 100.0
        )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    output.with_suffix(".json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
