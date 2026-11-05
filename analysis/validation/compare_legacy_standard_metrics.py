#!/usr/bin/env python3
"""Pair legacy singleton-MMD results with the standard two-sample results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SETTING_MAP = {
    "reference": ("beta", "0p1"),
    "beta_0p05": ("beta", "0p05"),
    "beta_0p2": ("beta", "0p2"),
    "softeta_0p0005": ("softeta", "0p0005"),
    "softeta_0p002": ("softeta", "0p002"),
    "proj_1p0": ("proj", "1p0"),
    "proj_3p0": ("proj", "3p0"),
    "steps_75": ("steps", "75"),
    "steps_100": ("steps", "100"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--standard", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legacy = pd.read_csv(args.legacy)
    standard = pd.read_csv(args.standard)
    rows = []
    for row in standard.itertuples(index=False):
        group, label = SETTING_MAP[row.setting_id]
        old = legacy.loc[
            (legacy["sens_group"] == group) & (legacy["sens_label"].astype(str) == label)
        ]
        if len(old) != 1:
            raise ValueError(f"Expected one legacy row for {row.setting_id}, found {len(old)}")
        old = old.iloc[0]
        rows.append(
            {
                "setting_id": row.setting_id,
                "parameter": row.Parameter,
                "setting": row.Setting,
                "legacy_pathway_to_point_MMD2": old["macro_MMD2"],
                "standard_two_sample_MMD2": row.MMD2_unbiased,
                "standard_MMD2_CI_low": row.MMD2_CI_low,
                "standard_MMD2_CI_high": row.MMD2_CI_high,
                "legacy_MAE": old["macro_MAE"],
                "standard_MAE": row.MAE,
                "legacy_RMSE": old["macro_RMSE"],
                "standard_RMSE": row.RMSE,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
