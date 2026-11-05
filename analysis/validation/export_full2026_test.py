#!/usr/bin/env python3
"""Export held-out samples from the fixed split-2026 reference checkpoint.

This is intentionally separate from ``train_val_test.py`` so loading an existing
checkpoint cannot enter the training loop or replace any saved model files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_files.dataset import CleanDataset, SPMDataset
from model_files.spm_diff.model import DiffSPM
from train_val_test import default_config, setup_seed
from utils.export_fig2 import run_export
from scripts.run_model import checkpoint_provenance, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--checkpoint-split-seed", type=int, help="Documented split for legacy weights without metadata")
    parser.add_argument("--normalization-seed", type=int, help="Documented training/resampling seed for legacy weights")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--split-n", type=int, default=4)
    return parser.parse_args()


def build_config(args: argparse.Namespace):
    config = default_config(data_root=args.data_root, device=args.device, output_root=args.output_dir)
    config.is_train = False
    config.is_val = False
    config.is_test = True
    config.is_gen = False
    config.seed = args.seed
    config.split_seed = args.split_seed
    config.train_fraction = 1.0
    config.train_batch_size = args.batch_size
    config.gen_batch_size = args.batch_size
    config.n_samples = args.n_samples
    config.save_inverse_npy = False
    config.resample_before_norm = True

    model = config.model
    model.N = 200
    model.sample_steps = 45
    model.d_h = 64
    model.beta_end = 0.1
    model.n_blocks = 2
    model.lambda_region = 0.002
    model.lambda_var = 0.002
    model.lambda_nn = 0.004
    model.use_soft_guidance = True
    model.soft_guidance_schedule = "exp"
    model.soft_eta_min = 0.0
    model.soft_eta_max = 0.001
    model.constraint_target_ratio = 0.35
    model.constraint_ratio_smooth = 0.70
    model.constraint_gain_min = 0.5
    model.constraint_gain_max = 4.0
    model.constraint_warmup_steps = 50
    model.use_hard_projection = True
    model.reverse_projection_scale = 2.0
    model.ablation_nograph = False
    model.ablation_no_constraint = False
    model.denoiser = "UGnet"
    model.ablation_no_cat_cond = False
    model.ablation_no_region_cond = False
    model.ablation_no_var_cond = False
    model.enable_constraint_loss = False
    model.constraint_weight = 1e-4
    model.tiny_width_mult = 0.5
    model.tiny_n_blocks = 1
    model.tiny_n_resolutions = 2
    model.tiny_num_heads = 1
    return config


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    digest = sha256(checkpoint)
    _, normalization_seed = checkpoint_provenance(state, digest, args)
    if normalization_seed != args.seed:
        raise ValueError("This historical exporter uses one seed for normalization and sampling; --seed must equal the documented --normalization-seed.")
    setup_seed(args.seed)
    torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    config = build_config(args)

    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    clean_data = CleanDataset(config)
    denorm_fn, means_t, stds_t = clean_data.build_denorm_bundle(config.model.device)
    train_dataset = SPMDataset(clean_data, "train", config)
    test_dataset = SPMDataset(clean_data, "test", config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )
    train_mask = torch.as_tensor(clean_data.train_mask, dtype=torch.bool, device=config.device)
    test_mask = torch.as_tensor(clean_data.test_mask, dtype=torch.bool, device=config.device)

    diffusion = DiffSPM(config.model).to(config.device)
    diffusion.attach_denorm(denorm_fn, means_t, stds_t)
    state_dict = state.get("model_state", state)
    schedule_keys = {"beta", "alpha", "alpha_bar", "sigma2"}
    preserve_schedule = bool(config.model.get("preserve_configured_schedule", False))
    if preserve_schedule:
        # Diffusion schedules are persistent buffers in the checkpoint. For an
        # inference-time beta sensitivity run, retain the freshly configured
        # schedule while loading all learned parameters and other buffers.
        state_dict = {
            key: value for key, value in state_dict.items() if key not in schedule_keys
        }
    incompat = diffusion.load_state_dict(state_dict, strict=False)
    missing_keys = [
        key
        for key in incompat.missing_keys
        if not (preserve_schedule and key in schedule_keys)
    ]
    if missing_keys or incompat.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/config mismatch: "
            f"missing={missing_keys}, unexpected={incompat.unexpected_keys}"
        )
    effective_beta_end = float(diffusion.beta[-1].detach().cpu())
    if preserve_schedule and not np.isclose(
        effective_beta_end, float(config.model.beta_end), rtol=1e-6, atol=1e-8
    ):
        raise RuntimeError(
            "Configured beta_end was not preserved after checkpoint loading: "
            f"configured={config.model.beta_end}, effective={effective_beta_end}"
        )
    diffusion.eval()

    # The sampler enables gradients locally to evaluate the physics energy.
    # ``run_export`` and ``sample_predict`` already prevent parameter gradients.
    result_dir = run_export(
        diffusion,
        test_loader,
        train_loader,
        test_mask,
        train_mask,
        config,
        clean_data,
        means_t,
        stds_t,
        out_dir=str(output_dir),
        n_samples_fig2=args.n_samples,
        split_n=args.split_n,
        skip_convergence=True,
        verbose=True,
        use_saved_raw=False,
    )

    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "normalization_seed": normalization_seed,
        "output_dir": str(result_dir),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "n_empirical_test": len(test_dataset),
        "n_generated_per_parent": args.n_samples,
        "sample_steps": config.model.sample_steps,
        "beta_end": config.model.beta_end,
        "effective_beta_end": effective_beta_end,
        "soft_eta_max": config.model.soft_eta_max,
        "reverse_projection_scale": config.model.reverse_projection_scale,
    }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
