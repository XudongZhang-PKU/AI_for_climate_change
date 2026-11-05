#!/usr/bin/env python3
"""Export a held-out sample under one inference-time sensitivity setting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from export_full2026_test import build_config


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / "sensitivity_configs.json"


def parse_setting_args() -> tuple[argparse.Namespace, dict, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--setting-id", required=True)
    known, remaining = parser.parse_known_args()
    settings = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    if known.setting_id not in settings:
        raise SystemExit(
            f"Unknown setting {known.setting_id!r}; choose from {', '.join(settings)}."
        )
    return known, settings[known.setting_id], remaining


def apply_setting(config, setting: dict):
    model = config.model
    model.sample_steps = int(setting["sample_steps"])
    model.beta_end = float(setting["beta_end"])
    model.soft_eta_max = float(setting["soft_eta_max"])
    model.reverse_projection_scale = float(setting["reverse_projection_scale"])
    model.preserve_configured_schedule = True
    return config


if __name__ == "__main__":
    setting_args, setting_config, remaining_args = parse_setting_args()
    sys.argv = [sys.argv[0]] + remaining_args

    original_build_config = build_config

    def sensitivity_build_config(args):
        return apply_setting(original_build_config(args), setting_config)

    import export_full2026_test

    export_full2026_test.build_config = sensitivity_build_config
    export_full2026_test.main()
