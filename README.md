# DeepIAM

约束引导的时空图扩散模型，用于气候减缓路径生成，包含数据预处理、模型训练、采样和评估。

![DeepIAM 模型结构](docs/images/architecture.png)

## 安装与演示（Linux）

```bash
git clone --branch codex/nature-review-package https://github.com/XudongZhang-PKU/AI_for_climate_change.git
cd AI_for_climate_change

conda create -n deepiam python=3.14 -y
conda activate deepiam
pip install -r requirements.txt

# 将数据和权重路径替换为实际位置；种子应与权重对应
python scripts/run_model.py demo \
  --data-root /path/to/SPM \
  --checkpoint /path/to/model.ckpt \
  --checkpoint-split-seed 2026 \
  --normalization-seed 2026 \
  --seed 2026 --split-seed 2026 \
  --device cuda \
  --output-dir outputs/demo
```

## 训练

```bash
python scripts/run_model.py train \
  --data-root /path/to/SPM \
  --device cuda --epochs 300 \
  --seed 2026 --split-seed 2026 \
  --output-dir outputs/train
```

## 目录

| 目录 | 内容 |
|---|---|
| `model_files/` | 模型与数据加载 |
| `scripts/` | 训练、采样入口 |
| `utils/` | 约束、评估与结果导出 |
| `analysis/` | 分布验证、消融与敏感性分析 |
| `preprocessing/`、`plotting/` | 预处理与绘图 |
| `legacy/`、`sensitivity/` | 历史脚本与集群作业 |

数据格式与评估命令见[使用说明](docs/reproduction_zh.md)。

## 许可证

[MIT](LICENSE)
