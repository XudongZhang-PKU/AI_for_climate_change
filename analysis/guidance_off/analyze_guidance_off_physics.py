#!/usr/bin/env python3
"""Compute physical diagnostics and Vet Pass from an exported raw test set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


HERE = Path(__file__).resolve()


def find_project_root() -> Path:
    candidates = [
        HERE.parents[2],
        HERE.parent.parent / "co2-0204-best",
        Path.cwd(),
        Path.cwd() / "co2-0204-best",
    ]
    for candidate in candidates:
        if (candidate / "train_val_test.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate the CO2 project root")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.check_cons import NON_NEGATIVE_VARS, VAR_CONSTRAINTS
from utils.inspection import check_one_sample, load_baselines, load_maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--region-csv", default="assets/data/SPM/变量编码对应表/dim_Region_encoding.csv")
    parser.add_argument("--variable-csv", default="assets/data/SPM/变量编码对应表/dim_Variable_encoding.csv")
    parser.add_argument("--variable-baseline-csv", default="assets/data/SPM/variable_baseline.csv")
    parser.add_argument("--region-baseline-csv", default="assets/data/SPM/region_baseline.csv")
    return parser.parse_args()


def build_sample(label: np.ndarray, generated: np.ndarray) -> np.ndarray:
    groups, time = generated.shape
    sample = np.empty((groups, time, 5), dtype=np.float32)
    sample[:, :, 0] = 0.0
    sample[:, :, 1] = label[:, 0, 1][:, None]
    sample[:, :, 2] = label[:, 0, 2][:, None]
    sample[:, :, 3] = label[:, 0, 3][:, None]
    sample[:, :, 4] = generated
    return sample


def physical_diagnostics(sample: np.ndarray) -> dict[str, float]:
    values = sample[:, :, 4]
    variable_codes = sample[:, 0, 3].astype(int)
    region_squared_error = 0.0
    region_count = 0
    variable_squared_error = 0.0
    variable_count = 0

    for time_index in range(sample.shape[1]):
        frame = sample[:, time_index, :]
        table = pd.DataFrame(
            {
                "category": frame[:, 1],
                "region": frame[:, 2],
                "variable": frame[:, 3],
                "value": frame[:, 4],
            }
        )
        by_region = table.pivot_table(
            index=["category", "variable"], columns="region", values="value"
        )
        if 0 in by_region.columns and by_region.shape[1] > 1:
            other_regions = [code for code in by_region.columns if code != 0]
            error = by_region[0].to_numpy() - by_region[other_regions].sum(axis=1).to_numpy()
            region_squared_error += float(np.square(error).sum())
            region_count += int(error.size)

        by_variable = table.pivot_table(
            index=["category", "region"], columns="variable", values="value"
        )
        for total, components in VAR_CONSTRAINTS.items():
            if total not in by_variable.columns or not all(
                component in by_variable.columns for component in components
            ):
                continue
            error = by_variable[total].to_numpy() - by_variable[components].sum(axis=1).to_numpy()
            variable_squared_error += float(np.square(error).sum())
            variable_count += int(error.size)

    nonnegative_mask = np.isin(variable_codes, list(NON_NEGATIVE_VARS))
    nonnegative_values = values[nonnegative_mask, :]
    negative_magnitude = (
        float(np.maximum(-nonnegative_values, 0.0).mean())
        if nonnegative_values.size
        else float("nan")
    )
    return {
        "region_equality_mse": (
            region_squared_error / region_count if region_count else float("nan")
        ),
        "variable_equality_mse": (
            variable_squared_error / variable_count if variable_count else float("nan")
        ),
        "negative_magnitude": negative_magnitude,
    }


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw_npz).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(raw_path, allow_pickle=False)
    labels = np.asarray(raw["test_label"])
    generated = np.asarray(raw["test_syn"])
    if labels.shape[0] != generated.shape[0]:
        raise ValueError("test_label and test_syn have different parent counts")

    _, region_name_to_code, _, variable_name_to_code = load_maps(
        args.region_csv, args.variable_csv
    )
    variable_tolerances, region_tolerances = load_baselines(
        args.variable_baseline_csv, args.region_baseline_csv
    )

    metric_values: dict[str, list[float]] = defaultdict(list)
    condition_pass = Counter()
    diagnostics = Counter()
    passed = 0
    total = 0

    progress = tqdm(total=generated.shape[0] * generated.shape[1], desc="guidance-off vetting")
    for parent_index in range(generated.shape[0]):
        for sample_index in range(generated.shape[1]):
            sample = build_sample(labels[parent_index], generated[parent_index, sample_index])
            metrics = physical_diagnostics(sample)
            for name, value in metrics.items():
                metric_values[name].append(value)

            accepted, diagnostic, conditions = check_one_sample(
                sample,
                region_name_to_code,
                variable_name_to_code,
                variable_tolerances,
                region_tolerances,
            )
            passed += int(bool(accepted))
            total += 1
            diagnostics.update(diagnostic)
            for name, value in conditions.items():
                condition_pass[name] += int(bool(value))
            progress.update(1)
    progress.close()

    summary = {
        "raw_npz": str(raw_path),
        "n_empirical_parents": int(generated.shape[0]),
        "n_generated_per_parent": int(generated.shape[1]),
        "n_generated_total": int(total),
        "region_equality_mse": float(np.nanmean(metric_values["region_equality_mse"])),
        "variable_equality_mse": float(np.nanmean(metric_values["variable_equality_mse"])),
        "negative_magnitude": float(np.nanmean(metric_values["negative_magnitude"])),
        "vet_pass_rate": float(passed / total) if total else float("nan"),
        "vet_pass_count": int(passed),
    }
    (output_dir / "physical_vet_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame([summary]).to_csv(output_dir / "physical_vet_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "condition": name,
                "pass_count": count,
                "total": total,
                "pass_rate": count / total if total else float("nan"),
            }
            for name, count in sorted(condition_pass.items())
        ]
    ).to_csv(output_dir / "vet_condition_pass_rates.csv", index=False)
    (output_dir / "vet_diagnostics.json").write_text(
        json.dumps(dict(diagnostics), indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
