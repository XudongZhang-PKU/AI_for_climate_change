# 运行与结果对应

## 演示与自有数据

`python scripts/run_model.py demo --help` 查看参数。默认 CPU、1条留出路径、2次生成、45步。生成值处于物理空间，单位随变量不同；使用 mask 区分真实观测与未报告组合。

```bash
python scripts/run_model.py demo --parents 2 --n-samples 4 --output-dir outputs/demo_more
```

外部权重必须配对正确数据、划分与归一化统计。脚本拒绝将已知 split1234 权重标为 split2026；缺少元数据的外部权重需显式提供 `--checkpoint-split-seed` 和 `--normalization-seed`，使用者须先核实来源。

自有数据用 `--data-root /path/to/SPM`，提供 spm.npz、同名编码表和 baseline CSV。张量必须保持11区域×48变量×10年份及原编码顺序，最后一维为 [model,category,region,variable,value]；改变变量、地区或时间范围需重新预处理和训练。

## 训练

```bash
python scripts/run_model.py train --device cuda --epochs 300 --seed 2026 --split-seed 2026
```

划分前原始样本数为950/119/119；类别重采样和标准化拟合仅使用训练分区。更完整参数用原入口，例如：

```bash
python train_val_test.py --data-root assets/data/SPM --output-dir outputs/no_graph \
  --device cuda --ablation NoGraph --seed 2026 --split_seed 2026 --epoch 300
```

本次未重跑完整训练；训练耗时及显存应在实际 CUDA 环境测量。

## 固定权重评估（参考权重待补）

将实际参考权重放在下述路径，不能换成历史演示权重：

```bash
python analysis/validation/export_full2026_test.py \
  --checkpoint assets/checkpoints/reference_split2026_epoch290.ckpt \
  --checkpoint-split-seed 2026 --normalization-seed 2026 \
  --seed 2026 --split-seed 2026 --device cuda \
  --output-dir outputs/reference_export

python analysis/validation/analyze_distribution_validation.py \
  --npz outputs/reference_export/fig2a_real_syn.npz \
  --output-dir outputs/reference_analysis --bootstrap 500

python analysis/validation/plot_regional_qq_candidate.py \
  --npz outputs/reference_export/fig2a_real_syn.npz \
  --region-map assets/data/SPM/变量编码对应表/dim_Region_encoding.csv \
  --variable-map assets/data/SPM/变量编码对应表/dim_Variable_encoding.csv \
  --output-dir outputs/regional_qq
```

| 分析 | 入口与配置 | 现状 |
|---|---|---|
| Wasserstein/KS、标准两样本MMD | analysis/validation/analyze_distribution_validation.py | 9月脚本与区域参考表已纳入 |
| 消融 | export_ablation_test.py --variant NAME；ablation_variants.json | 9个变体配置及指标表，权重待补 |
| 固定权重推理敏感性 | export_sensitivity_test.py --setting-id NAME；sensitivity_configs.json | 指标表已纳入；改变beta时保留新调度buffer |
| 关闭软引导 | analysis/guidance_off/export_guidance_off_test.py | 10月脚本；本包不宣称实验已完成 |
| 百万级生产采样 | legacy/generation_cons.py为历史服务器入口 | 全数据权重、归一化与实际配置待核对 |

消融、敏感性和guidance-off导出均复用参考导出的其余参数。`reference_results/` 是既有结果的副本，不是本次演示产物。旧训练期 utils/eval.py 的 MMD 与标准两样本 MMD 口径不同，后者使用 analysis/validation。

## 历史和绘图代码

legacy/ 与 sensitivity/ 保留原服务器路径及配置，当前可移植入口为 scripts/run_model.py。plotting/ 需要对应导出数据；UMAP和凸优化后处理等历史可选模块未纳入本次核心CPU环境验证。正文/SI图号应与送审稿核对。
