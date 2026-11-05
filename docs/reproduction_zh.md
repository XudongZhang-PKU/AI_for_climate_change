# 使用说明

## 输入数据

`--data-root` 指向包含下列文件的目录：

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

`spm.npz` 包含 `padded: (N, 528, 10, 5)` 和 `mask: (N, 528)`。528 = 11 个区域 × 48 个变量；时间轴为 2010–2100、每十年一个点；最后一维依次为 model、category、region、variable 编码及数值。编码顺序必须与模型匹配。

## 采样

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

权重与种子应来自同一次训练。使用保存的归一化统计时，改为提供 `--normalization-file /path/to/normalization.npz`；该文件包含 `keys`、`train_means`、`train_stds`，分别描述 category-region-variable 组合及其均值、标准差。

输出为物理空间路径、编码、观测掩码、年份和配置清单。每次运行使用新的输出目录。

## 训练与消融

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

更多配置见 `python train_val_test.py --help`。

## 固定权重评估

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

| 分析 | 入口 |
|---|---|
| 消融评估 | `analysis/validation/export_ablation_test.py --variant NAME` |
| 推理敏感性 | `analysis/validation/export_sensitivity_test.py --setting-id NAME` |
| 关闭软引导 | `analysis/guidance_off/export_guidance_off_test.py` |
| Q–Q 与区域距离 | `analysis/validation/plot_regional_qq_candidate.py` |

前三种导出复用固定权重评估的其他参数。变体配置见 `ablation_variants.json` 和 `sensitivity_configs.json`。标准两样本 MMD 使用 `analysis/validation/` 的实现。
