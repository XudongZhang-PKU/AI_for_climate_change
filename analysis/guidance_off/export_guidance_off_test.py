#!/usr/bin/env python3
"""Export held-out samples with soft guidance disabled.

The trained reference checkpoint and terminal hard projection are retained.
Only the inference-time physics-gradient update is disabled, which isolates the
contribution of soft guidance from terminal projection.
"""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path


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


def load_reference_export():
    candidates = [
        PROJECT_ROOT / "analysis/validation/export_full2026_test.py",
        HERE.parent.parent / "reviewer_validation_20260904/export_full2026_test.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("reference_export", candidate)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load {candidate}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not locate export_full2026_test.py")


reference_export = load_reference_export()


_reference_build_config = reference_export.build_config


def build_guidance_off_config(args):
    config = _reference_build_config(args)
    config.model.use_soft_guidance = False
    config.model.soft_eta_min = 0.0
    config.model.soft_eta_max = 0.0
    config.model.use_hard_projection = True
    config.model.reverse_projection_scale = 2.0
    config.model.ablation_no_constraint = False
    return config


def main() -> None:
    args = reference_export.parse_args()
    reference_export.build_config = build_guidance_off_config
    reference_export.main()

    manifest_path = Path(args.output_dir).resolve() / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "experiment": "guidance_off",
            "use_soft_guidance": False,
            "use_hard_projection": True,
            "isolated_change": "soft guidance disabled; terminal projection retained",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
