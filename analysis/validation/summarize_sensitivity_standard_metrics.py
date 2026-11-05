#!/usr/bin/env python3
"""Collect uniformly recalculated sensitivity metrics for the manuscript."""

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
        "--settings", default=str(SCRIPT_DIR / "sensitivity_configs.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.results_root).resolve()
    settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
    rows = []
    for setting_id, setting in settings.items():
        summary_path = root / setting_id / "analysis" / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "setting_id": setting_id,
                "Parameter": setting["parameter"],
                "Setting": setting["setting"],
                "MMD2_unbiased": summary["macro_mmd2_unbiased_median"],
                "MMD2_CI_low": summary["macro_mmd2_unbiased_q025"],
                "MMD2_CI_high": summary["macro_mmd2_unbiased_q975"],
                "MMD2_biased": summary["macro_mmd2_biased_median"],
                "MAE": summary["macro_mae"],
                "RMSE": summary["macro_rmse"],
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
    reference = frame.loc[frame["setting_id"] == "reference"].iloc[0]
    for metric in ["MMD2_unbiased", "MAE", "RMSE"]:
        frame[f"{metric}_change_vs_reference_pct"] = (
            (frame[metric] - reference[metric]) / reference[metric] * 100.0
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
