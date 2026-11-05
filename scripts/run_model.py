#!/usr/bin/env python3
"""Portable entrypoints for a checkpoint-based demonstration and full training.

The bundled checkpoint is a historical split-1234 model. The demo exercises the
real model and processed data; it does not reproduce the submitted split-2026
results. No training is performed by the ``demo`` command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_SHA256 = "459c0b01e662750a5d25dbc236fd71a53a275436ac67dab36c67c139b35deb7f"


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def checkpoint_provenance(state, digest, args, require_normalization_seed=True):
    """Reject an incorrect split; require explicit provenance for legacy weights."""
    metadata = state.get("parameters", {}) if isinstance(state, dict) else {}
    recorded_split = state.get("split_seed", metadata.get("split_seed"))
    recorded_seed = state.get("seed", metadata.get("seed"))
    if digest == DEMO_SHA256:
        # The original checkpoint does not store its training RNG seed. Only
        # its split is established by its source directory; callers that refit
        # normalization must provide the documented seed explicitly.
        recorded_split = 1234
    if recorded_split is not None and args.checkpoint_split_seed is not None:
        if int(recorded_split) != args.checkpoint_split_seed:
            raise ValueError("--checkpoint-split-seed disagrees with checkpoint metadata")
    if recorded_split is None:
        recorded_split = args.checkpoint_split_seed
    if recorded_split is None:
        raise ValueError("Checkpoint has no split metadata. Supply its documented --checkpoint-split-seed.")
    if int(recorded_split) != args.split_seed:
        raise ValueError(f"Checkpoint split is {recorded_split}, but --split-seed is {args.split_seed}.")
    if args.normalization_seed is not None and recorded_seed is not None:
        if args.normalization_seed != int(recorded_seed):
            raise ValueError("--normalization-seed disagrees with the checkpoint training seed")
    normalization_seed = recorded_seed if args.normalization_seed is None else args.normalization_seed
    if normalization_seed is None and require_normalization_seed:
        raise ValueError("Checkpoint has no training seed. Supply its documented --normalization-seed.")
    return int(recorded_split), None if normalization_seed is None else int(normalization_seed)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "train"):
        sub = commands.add_parser(name)
        sub.add_argument("--data-root", type=Path, default=ROOT / "assets/data/SPM")
        sub.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu" if name == "demo" else "auto")
        sub.add_argument("--output-dir", type=Path, default=ROOT / f"outputs/{name}")
        sub.add_argument("--seed", type=int, default=1234 if name == "demo" else 2026)
        sub.add_argument("--split-seed", type=int, default=1234 if name == "demo" else 2026)
        if name == "demo":
            sub.add_argument("--checkpoint", type=Path,
                             default=ROOT / "assets/checkpoints/demo_split1234_epoch280.ckpt")
            sub.add_argument("--checkpoint-split-seed", type=int,
                             help="Documented split seed for external legacy checkpoints without metadata")
            sub.add_argument("--normalization-seed", type=int,
                             help="Seed for training-category resampling; original training seed when documented")
            sub.add_argument("--normalization-file", type=Path,
                             help="Saved keys/train_means/train_stds; bundled demo uses its matching saved statistics")
            sub.add_argument("--parents", type=positive_int, default=1)
            sub.add_argument("--n-samples", type=positive_int, default=2)
            sub.add_argument("--threads", type=positive_int, default=2)
        else:
            sub.add_argument("--epochs", type=positive_int, default=300)
            sub.add_argument("--batch-size", type=positive_int, default=32)
            sub.add_argument("--no-export", action="store_true", help="Skip optional post-training figure export")
    return parser.parse_args(argv)


def prepare_paths(args):
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    required = [args.data_root / "spm.npz", args.data_root / "变量编码对应表/dim_Region_encoding.csv",
                args.data_root / "变量编码对应表/dim_Variable_encoding.csv"]
    if args.command == "train":
        required += [args.data_root / "region_baseline.csv", args.data_root / "variable_baseline.csv"]
    if args.command == "demo":
        args.checkpoint = args.checkpoint.resolve()
        required.append(args.checkpoint)
        if args.normalization_file:
            args.normalization_file = args.normalization_file.resolve()
            required.append(args.normalization_file)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}. Choose a fresh --output-dir.")


def build_demo_config(args):
    from train_val_test import default_config
    config = default_config(data_root=args.data_root, output_root=args.output_dir, device=args.device)
    config.update(is_train=False, is_val=False, is_test=True, is_gen=False,
                  split_seed=args.split_seed, train_fraction=1.0, save_inverse_npy=False,
                  resample_before_norm=True, train_batch_size=1, gen_batch_size=1,
                  n_samples=args.n_samples)
    # Same architecture and inference settings as the fixed-checkpoint exporter.
    config.model.update(N=200, sample_steps=45, d_h=64, beta_end=0.1, n_blocks=2,
                        lambda_region=0.002, lambda_var=0.002, lambda_nn=0.004,
                        use_soft_guidance=True, soft_guidance_schedule="exp",
                        soft_eta_min=0.0, soft_eta_max=0.001,
                        constraint_target_ratio=0.35, constraint_ratio_smooth=0.70,
                        constraint_gain_min=0.5, constraint_gain_max=4.0,
                        constraint_warmup_steps=50, use_hard_projection=True,
                        reverse_projection_scale=2.0, ablation_nograph=False,
                        ablation_no_constraint=False, denoiser="UGnet",
                        ablation_no_cat_cond=False, ablation_no_region_cond=False,
                        ablation_no_var_cond=False, enable_constraint_loss=False,
                        constraint_weight=1e-4)
    return config


def load_saved_demo_data(data_root, normalization_file, split_seed, parents):
    """Use the checkpoint's exported statistics without fitting on any new data.

    A full category must have statistics for all 11 × 48 model nodes, including
    masked nodes, because constraints and graph operations use the full grid.
    """
    import numpy as np
    from model_files.dataset import AdvancedNormalizer, CleanDataset
    with np.load(normalization_file, allow_pickle=False) as stored:
        keys = stored["keys"].copy()
        means = stored["train_means"].copy()
        stds = stored["train_stds"].copy()
    if keys.shape != (len(means), 3) or stds.shape != means.shape:
        raise ValueError("Invalid saved normalization array shapes")
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError("Saved normalization statistics must be finite, with positive standard deviations")
    if not np.array_equal(keys, keys.astype(int)):
        raise ValueError("Saved normalization keys must be integer category/region/variable codes")
    keys = keys.astype(int)
    if ((keys < 0) | (keys >= np.array([8, 11, 48]))).any():
        raise ValueError("Saved normalization keys are outside the model grid")
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("Duplicate saved normalization keys")
    coverage = np.zeros((8, 11, 48), dtype=bool)
    coverage[tuple(keys.T)] = True
    complete_categories = np.flatnonzero(coverage.all(axis=(1, 2)))
    with np.load(data_root / "spm.npz", allow_pickle=False) as data:
        raw = data["padded"].copy()
        masks = data["mask"].copy()
    n_total = len(raw)
    if raw.shape[1:] != (528, 10, 5) or masks.shape != (n_total, 528):
        raise ValueError("Expected padded=(N,528,10,5) and mask=(N,528)")
    expected_r = np.repeat(np.arange(11), 48)
    expected_v = np.tile(np.arange(48), 11)
    if not (raw[:, :, :, 2] == expected_r[None, :, None]).all():
        raise ValueError("Region codes do not follow the required region-major grid")
    if not (raw[:, :, :, 3] == expected_v[None, :, None]).all():
        raise ValueError("Variable codes do not follow the required region-major grid")
    cats = raw[:, 0, 0, 1]
    if not (raw[:, :, :, 1] == cats[:, None, None]).all():
        raise ValueError("Category code must be constant within each pathway")
    order = np.arange(n_total)
    np.random.RandomState(split_seed).shuffle(order)
    held_out = order[int(n_total * 0.9):]
    available = held_out[np.isin(cats[held_out], complete_categories)]
    if parents > len(available):
        raise ValueError(f"Only {len(available)} held-out parents have complete saved normalization coverage")
    selected = available[:parents]
    # Reuse the original dataset's transformation and differentiable inverse.
    # __init__ is intentionally bypassed: it would fit new training statistics.
    clean = CleanDataset.__new__(CleanDataset)
    clean.normalizer = AdvancedNormalizer()
    clean.cat_codes, clean.region_codes, clean.var_codes = np.arange(8), np.arange(11), np.arange(48)
    clean.normalizer.transform_params = {
        f"{c}_{r}_{v}": {"mean": float(mu), "std": float(sd)}
        for (c, r, v), mu, sd in zip(keys, means, stds)
    }
    clean.test_data = clean.apply_normalize_with_fitted(raw[selected].copy())
    clean.test_mask = masks[selected].copy()
    return clean, n_total, selected, coverage


def run_demo(args):
    import numpy as np
    import torch
    from model_files.dataset import CleanDataset
    from model_files.spm_diff.model import DiffSPM
    from train_val_test import setup_seed
    from utils.eval import sample_predict

    started = time.perf_counter()
    digest = sha256(args.checkpoint)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    normalization_file = args.normalization_file
    if digest == DEMO_SHA256 and normalization_file is None:
        normalization_file = ROOT / "assets/data/SPM/demo_normalization.npz"
    if normalization_file and not normalization_file.is_file():
        raise FileNotFoundError(f"Saved normalization file is missing: {normalization_file}")
    split_seed, normalization_seed = checkpoint_provenance(
        state, digest, args, require_normalization_seed=normalization_file is None)
    setup_seed(args.seed if normalization_seed is None else normalization_seed)
    torch.set_num_threads(args.threads)
    config = build_demo_config(args)
    if normalization_file:
        clean, n_total, parent_indices, coverage = load_saved_demo_data(
            args.data_root, normalization_file, split_seed, args.parents)
        normalization_seed = None
    else:
        clean = CleanDataset(config)
        with np.load(config.data.feature_file, allow_pickle=False) as source:
            n_total = len(source["padded"])
        indices = np.arange(n_total)
        np.random.RandomState(split_seed).shuffle(indices)
        parent_indices = indices[int(n_total * 0.9):][:args.parents]
        coverage = np.ones((8, 11, 48), dtype=bool)
    if args.parents > len(clean.test_data):
        raise ValueError(f"--parents exceeds the {len(clean.test_data)} held-out pathways")
    denorm, means, stds = clean.build_denorm_bundle(config.device)
    model = DiffSPM(config.model).to(config.device)
    model.attach_denorm(denorm, means, stds)
    model.load_state_dict(state.get("model_state", state), strict=True)
    model.eval()
    # Different noise seeds are allowed without changing training normalization.
    setup_seed(args.seed)
    outputs, observations, labels = [], [], []
    for index in range(args.parents):
        label = torch.as_tensor(clean.test_data[index:index + 1], dtype=torch.float32, device=config.device)
        # no_grad is safe; inference_mode would prevent the local guidance gradient.
        generated = sample_predict(model, label, args.n_samples, split=1).numpy()
        if not np.isfinite(generated).all():
            raise RuntimeError("Model generated non-finite values; no successful demo manifest was written")
        outputs.append(generated)
        observations.append(denorm(label).detach().cpu().numpy()[..., 4])
        labels.append(label.detach().cpu().numpy()[..., :4])
    generated = np.concatenate(outputs)
    observed = np.concatenate(observations)
    mask = clean.test_mask[:args.parents].astype(bool)
    np.savez_compressed(args.output_dir / "samples.npz", generated=generated,
                        observed=observed, codes=np.concatenate(labels), mask=mask,
                        years=np.arange(2010, 2101, 10))
    np.savez_compressed(args.output_dir / "normalization.npz",
                        means=np.asarray(means), stds=np.asarray(stds), available=coverage)
    manifest = {
        "purpose": "model demonstration; not reproduction of the submitted split-2026 results",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": digest,
        "checkpoint_epoch": state.get("epoch"), "split_seed": split_seed,
        "original_training_seed": state.get("seed", state.get("parameters", {}).get("seed")),
        "normalization_provenance": (
            "saved training statistics from the checkpoint's historical export; no fitting during demo"
            if normalization_file else "reconstructed using the checkpoint's documented training seed"
        ),
        "normalization_file": str(normalization_file) if normalization_file else None,
        "normalization_sha256": sha256(normalization_file) if normalization_file else None,
        "complete_normalization_categories": np.flatnonzero(coverage.all(axis=(1, 2))).tolist(),
        "normalization_seed": normalization_seed, "sampling_seed": args.seed,
        "data_sha256": sha256(config.data.feature_file), "data_shape": [n_total, 528, 10, 5],
        "held_out_parents": args.parents, "source_row_indices": parent_indices.tolist(),
        "generated_per_parent": args.n_samples, "generated_shape": list(generated.shape),
        "space": "physical values; units depend on variable", "all_generated_values_finite": True,
        "sample_steps": model.sample_steps, "effective_beta_end": float(model.beta[-1].cpu()),
        "soft_eta_max": model.soft_eta_max, "reverse_projection_scale": model.reverse_projection_scale,
        "device": str(config.device), "threads": args.threads,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def run_train(args):
    from train_val_test import get_params, main
    params = vars(get_params([]))
    params.update(data_root=str(args.data_root), output_dir=str(args.output_dir), device=args.device,
                  seed=args.seed, split_seed=args.split_seed, epoch=args.epochs,
                  train_batch_size=args.batch_size, export_fig2=int(not args.no_export))
    main(params)


def main(argv=None):
    args = parse_args(argv)
    prepare_paths(args)
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("MPLBACKEND", "Agg")
    if args.command == "demo":
        run_demo(args)
    else:
        run_train(args)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        raise SystemExit(str(error)) from error
