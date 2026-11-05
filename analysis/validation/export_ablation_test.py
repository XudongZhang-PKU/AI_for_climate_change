#!/usr/bin/env python3
"""Export the fixed split-2026 test set for one trained ablation checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from export_full2026_test import build_config


SCRIPT_DIR = Path(__file__).resolve().parent
VARIANT_FILE = SCRIPT_DIR / "ablation_variants.json"


def parse_variant_args() -> tuple[argparse.Namespace, dict, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", required=True)
    known, remaining = parser.parse_known_args()
    variants = json.loads(VARIANT_FILE.read_text(encoding="utf-8"))
    if known.variant not in variants:
        raise SystemExit(
            f"Unknown variant {known.variant!r}; choose from {', '.join(variants)}."
        )
    return known, variants[known.variant], remaining


def apply_variant(config, variant: dict):
    model = config.model
    config.ablation = variant["ablation"]
    model.ablation_nograph = variant["ablation"] == "NoGraph"
    model.ablation_no_constraint = variant["ablation"] == "NoConstraint"
    model.ablation_linear_denoise = False
    model.denoiser = variant["denoiser"]
    model.ablation_no_cat_cond = bool(variant["no_category_condition"])
    model.ablation_no_region_cond = bool(variant["no_region_condition"])
    model.ablation_no_var_cond = bool(variant["no_variable_condition"])
    model.enable_constraint_loss = bool(variant["constraint_loss"])
    model.use_soft_guidance = bool(variant["soft_guidance"])
    model.use_hard_projection = bool(variant["hard_projection"])
    return config


if __name__ == "__main__":
    variant_args, variant_config, remaining_args = parse_variant_args()
    sys.argv = [sys.argv[0]] + remaining_args

    # Reuse the established exporter while injecting the architecture and
    # constraint flags that were used to train this checkpoint.
    original_build_config = build_config

    def variant_build_config(args):
        return apply_variant(original_build_config(args), variant_config)

    import export_full2026_test

    export_full2026_test.build_config = variant_build_config
    export_full2026_test.main()
