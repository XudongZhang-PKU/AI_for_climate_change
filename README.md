# DeepIAM

约束引导的时空图扩散模型，用于从 IPCC AR6 情景学习并生成气候减缓路径。本仓库提供数据预处理、训练、生成和模型评估；下游可行性分析由配套仓库维护。

![DeepIAM 模型结构，送审稿 Extended Data Fig. 1](docs/images/architecture.png)

## 安装与演示

已测试：macOS 15.5、Apple M4、24 GB 内存、Python 3.14.2，CPU 模式。演示不需要 GPU。

```bash
git clone --branch codex/nature-review-package https://github.com/XudongZhang-PKU/AI_for_climate_change.git
cd AI_for_climate_change
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/run_model.py demo
```

使用随附数据和已训练权重，执行 45 步采样，为 1 条留出路径生成 2 条路径。结果写入 `outputs/demo/`：

- `samples.npz`：生成与真实路径、编码、掩码、年份；生成数组形状为 `(1, 2, 528, 10)`。
- `normalization.npz`：本次使用的归一化统计。
- `manifest.json`：文件校验值、配置、环境和耗时。

本机演示约 11 秒；新建环境安装约 23 秒（安装包已有缓存，首次下载时间另计）。重复运行请指定新的 `--output-dir`。

**随附权重为历史 split 1234、epoch 280 的演示模型。送审评估使用 split 2026、epoch 290，参考权重尚待从服务器补齐；演示输出不用于核对论文数值。**

## 训练与评估

```bash
python scripts/run_model.py train --device cuda --epochs 300
```

完整训练使用 CUDA GPU。固定权重评估、消融和自有数据格式见[复现说明](docs/reproduction_zh.md)；输入和权重来源见[资产说明](docs/assets_zh.md)；环境及测试记录见[验证说明](docs/environment_zh.md)。

| 目录 | 内容 |
|---|---|
| `scripts/` | 演示、训练入口和环境检查 |
| `model_files/`、`utils/` | 模型、数据加载与评估 |
| `analysis/` | 分布验证、消融、推理敏感性及 guidance-off 脚本 |
| `assets/` | 处理后数据、编码表、演示权重和校验清单 |
| `preprocessing/`、`plotting/` | 原始数据预处理与绘图 |
| `reference_results/` | 既有参考表 |
| `legacy/`、`sensitivity/` | 历史绘图和集群实验脚本 |

本项目代码采用 [MIT 许可证](LICENSE)。AR6 派生数据遵循原数据来源的条款。
