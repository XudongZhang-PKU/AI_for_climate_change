#!/usr/bin/env python3
"""Combine distribution and physical outputs into a manuscript-ready row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REFERENCE = {
    "variant": "Full reference model",
    "mmd2": 0.10332240894955894,
    "mae": 0.8752948576546744,
    "rmse": 1.0487320278435792,
    "region_equality_mse": 1.25e-6,
    "variable_equality_mse": 9.03e-7,
    "negative_magnitude": 0.1610,
    "vet_pass_rate": 0.2421,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution-json", required=True)
    parser.add_argument("--physical-json", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def pct_change(value: float, reference: float) -> float:
    return 100.0 * (value - reference) / reference


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    distribution = json.loads(Path(args.distribution_json).read_text(encoding="utf-8"))
    physical = json.loads(Path(args.physical_json).read_text(encoding="utf-8"))

    guidance_off = {
        "variant": "Guidance off (terminal projection retained)",
        "mmd2": distribution["macro_mmd2_unbiased_median"],
        "mmd2_ci_low": distribution["macro_mmd2_unbiased_q025"],
        "mmd2_ci_high": distribution["macro_mmd2_unbiased_q975"],
        "mae": distribution["macro_mae"],
        "rmse": distribution["macro_rmse"],
        "region_equality_mse": physical["region_equality_mse"],
        "variable_equality_mse": physical["variable_equality_mse"],
        "negative_magnitude": physical["negative_magnitude"],
        "vet_pass_rate": physical["vet_pass_rate"],
    }
    comparison = {
        "mmd2_change_pct": pct_change(guidance_off["mmd2"], REFERENCE["mmd2"]),
        "mae_change_pct": pct_change(guidance_off["mae"], REFERENCE["mae"]),
        "rmse_change_pct": pct_change(guidance_off["rmse"], REFERENCE["rmse"]),
        "vet_pass_change_percentage_points": 100.0
        * (guidance_off["vet_pass_rate"] - REFERENCE["vet_pass_rate"]),
    }

    rows = [REFERENCE, guidance_off]
    pd.DataFrame(rows).to_csv(output_dir / "guidance_off_comparison.csv", index=False)
    payload = {
        "reference_source": "current manuscript ablation row and fixed-checkpoint reference analysis",
        "reference": REFERENCE,
        "guidance_off": guidance_off,
        "change_from_reference": comparison,
    }
    (output_dir / "guidance_off_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
