# Usage

## Input data

Set `--data-root` to a directory containing the following files:

```text
SPM/
├── spm.npz
├── region_baseline.csv
├── variable_baseline.csv
└── 变量编码对应表/
    ├── dim_Category_encoding.csv
    ├── dim_Model_encoding.csv
    ├── dim_Region_encoding.csv
    └── dim_Variable_encoding.csv
```

`spm.npz` contains `padded: (N, 528, 10, 5)` and `mask: (N, 528)`. The 528 groups represent 11 regions × 48 variables. The time axis spans 2010–2100 at ten-year intervals. The final dimension contains the model, category, region and variable codes, followed by the value. The encoding order must match the model.

## Sampling

```bash
python scripts/run_model.py demo --help

python scripts/run_model.py demo \
  --data-root /path/to/SPM \
  --checkpoint /path/to/model.ckpt \
  --checkpoint-split-seed 2026 --normalization-seed 2026 \
  --seed 2026 --split-seed 2026 \
  --parents 2 --n-samples 4 --device cuda \
  --output-dir outputs/samples
```

The checkpoint and seeds must correspond to the same training run. To use saved normalization statistics instead, provide `--normalization-file /path/to/normalization.npz`. This file contains `keys`, `train_means` and `train_stds`, which specify category-region-variable combinations and their means and standard deviations.

Outputs include pathways in physical units, codes, observation masks, years and a configuration manifest. Use a new output directory for each run.

## Training and ablations

```bash
python scripts/run_model.py train \
  --data-root /path/to/SPM --device cuda \
  --epochs 300 --seed 2026 --split-seed 2026 \
  --output-dir outputs/train

python train_val_test.py \
  --data-root /path/to/SPM --output-dir outputs/no_graph \
  --device cuda --ablation NoGraph \
  --seed 2026 --split_seed 2026 --epoch 300
```

For additional options, run `python train_val_test.py --help`.

## Evaluation with a fixed checkpoint

```bash
python analysis/validation/export_full2026_test.py \
  --data-root /path/to/SPM \
  --checkpoint /path/to/model.ckpt \
  --checkpoint-split-seed 2026 --normalization-seed 2026 \
  --seed 2026 --split-seed 2026 --device cuda \
  --output-dir outputs/evaluation

python analysis/validation/analyze_distribution_validation.py \
  --npz outputs/evaluation/fig2a_real_syn.npz \
  --output-dir outputs/distribution --bootstrap 500
```

| Analysis | Entrypoint |
|---|---|
| Ablation evaluation | `analysis/validation/export_ablation_test.py --variant NAME` |
| Inference sensitivity | `analysis/validation/export_sensitivity_test.py --setting-id NAME` |
| Soft guidance disabled | `analysis/guidance_off/export_guidance_off_test.py` |
| Q–Q plots and regional distances | `analysis/validation/plot_regional_qq_candidate.py` |

The first three exporters accept the other arguments shown for fixed-checkpoint evaluation. Variant settings are defined in `ablation_variants.json` and `sensitivity_configs.json`. Use the implementation in `analysis/validation/` for standard two-sample MMD.
