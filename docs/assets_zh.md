# 数据和权重

## 随附资产

`assets/data/SPM/spm.npz` 是现有数据的无损压缩副本，约 19.4 MB（原文件 126.1 MB）。压缩前后数组形状、类型和字节内容完全一致；文件及数组哈希见 `assets/manifest.json`。

| 数组 | 形状 | 内容 |
|---|---|---|
| padded | (1188,528,10,5) | 最后一维：[model, category, region, variable, value] |
| mask | (1188,528) | 区域—变量组合是否有观测 |

528 = 11 个区域 × 48 个变量；2010–2100，每十年一个点。1,188 条路径中，842 条含 R10 信息，346 条仅有全球信息。编码表在 `变量编码对应表/`；两个 baseline CSV 用于模型内部 vetting。

原始来源为 IPCC AR6 Scenario Database v1.1。本包提供已处理输入；原始 World/R10 表和辅助元数据未全部打包。`preprocessing/prepare_spm.ipynb` 保留原代码并清空输出；从原数据库重新构建仍需准备其引用的原始文件。缺失筛选源码为 ≤0.41；本数据固定48变量，≤0.41和≤0.40均保留至多19项缺失，筛选等价；原缺失矩阵未打包，因此本次未重新列出14条排除名称。

`assets/checkpoints/demo_split1234_epoch280.ckpt`（22.2 MB）来源于历史 Full_1234/best.ckpt，仅保存模型状态与 epoch。目录及记录支持 split 1234、epoch 280，不能由此推断原始全局训练种子。

`demo_normalization.npz` 从该历史模型既有 fig2_data/fig2a_real_syn.npz 无数值修改地提取 keys/train_means/train_stds；category 0–6 各覆盖全部528组，category 7不完整。演示使用保存统计并限制在完整覆盖的类别，不从未知训练种子重新拟合。

## 送审参考权重

既有记录指定服务器文件：

```text
co2-0204-best-sensitivity/
  fig2_data_prepare_Full_2026_ab2026_full/report_epoch_290.ckpt
```

参考配置：seed=split_seed=2026，119条留出路径，每条生成64次，45步，beta_end=0.1，soft_eta_max=0.001，projection_scale=2.0。本次没有连接服务器取回该文件。其他消融权重列于 `analysis/validation/ablation_variants.json`。

百万级生产使用的全数据模型属于 Protocol B，需另行对应其权重、归一化和配置，不能用留出评估权重替代。

## 后续补齐

- 参考、消融及生产权重，以及各自训练配置、归一化统计。
- 原训练环境版本；大文件可使用固定版本附件并记录链接及 SHA256。
- 对照原数据来源确认分发条款；本项目代码已采用 MIT 许可证。

本包不含同事负责的下游可行性分析。
