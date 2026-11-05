> 历史工程记录，保留原文供溯源。路径和部分指标口径已过时；当前入口见 [README](../README.md) 和 [usage.md](usage.md)。

# co2-0204-best 完整工程工作流说明书

本文档是工程代码导览，不是论文 Methods。目标读者是新接手项目的工程同学：看完即可按顺序跑通，并理解每一步输入、输出、张量形状、关键超参和常见坑。

---

## 0. 快速总览

主链路：

1. `data/数据预处理0813.ipynb`：原始 AR6 表 -> 固定网格 `spm.npz`
2. `model_files/dataset.py`：读取 `spm.npz`，切分并标准化，构建 `denorm_fn`
3. `model_files/spm_diff/model.py` + `model_files/spm_diff/ugnet.py`：训练扩散模型（DiffSPM + UGnet）
4. 采样（DDIM）时启用全部约束：soft guidance + final projection sandwich（不拆分）
5. `utils/eval.py` + `utils/export_fig2.py`：评估指标并导出 Fig2 数据
6. `experiments/plot_fig2*.py`：读导出文件绘图
7. `generation_cons.py`：大规模采样并落盘 `.npy`

---

## A. Inventory（模块清单 + 依赖）

### A0. 依赖树

- 数据准备
  - `data/数据预处理0813.ipynb`
  - 输出：`data/dataset/SPM/spm.npz`、编码表 CSV
- 数据读取与标准化
  - `model_files/dataset.py`
  - 类：`CleanDataset`、`SPMDataset`
- 模型
  - `model_files/spm_diff/model.py`：`DiffSPM`
  - `model_files/spm_diff/ugnet.py`：`UGnet`、`TinyUGnet`
  - `model_files/spm_diff/ugnet_notcn.py`：`UGnetNoTCN`
- 训练与验证
  - `train_val_test.py`
- 评估与导出
  - `utils/eval.py`
  - `utils/export_fig2.py`
  - `utils/inspection.py`（VET 条件）
- 绘图
  - `experiments/plot_fig2a_my.py`
  - `experiments/plot_fig2b_new.py`
  - `experiments/plot_fig2d.py`
  - `experiments/plot_fig2f.py`
  - `experiments/plot_fig2g.py`
- 大规模生成与后处理
  - `generation_cons.py`
  - `utils/cons.py`
  - `all_gendata_process/use_cons.py`
  - `all_gendata_process/save_to_csv.py`

### A1. 数据预处理阶段（AR6 -> spm.npz）

- 涉及文件
  - `data/数据预处理0813.ipynb`
  - 输入元数据：`data/dataset/SPM/Meta_data/*.csv`
- 关键入口（notebook 关键函数）
  - `fill_missing_zero(...)`
  - `fill_emission_co2_other_by_constraint(...)`
  - `build_spm_fixed_grid(...)`
- 输入输出文件
  - 输入：`AR6_Scenarios_Database_R10_regions_v1.1.csv`、`AR6_Scenarios_Database_World_v1.1.csv`、`Variables48.csv`、`pairs_metadata.csv`
  - 输出：
    - `data/dataset/SPM/spm.npz`
    - `data/dataset/SPM/变量编码对应表/dim_Model_encoding.csv`
    - `.../dim_Category_encoding.csv`
    - `.../dim_Region_encoding.csv`
    - `.../dim_Variable_encoding.csv`
    - 中间文件：`data/dataset/SPM/Meta_data/AR6_merged_world_r10_filtered.csv`
- `spm.npz` 内部数组（实际）
  - `padded`: `(1188, 528, 10, 5)`，`float32`
  - `mask`: `(1188, 528)`，`uint8`
- 维度含义
  - `N=1188`：每个 `(Model,Scenario)` 一个样本
  - `G=528=11*48`：固定 `R*V`
  - `T=10`：`2010..2100`（步长 10 年）
  - `C=5`：`[model, category, region, variable, value]`
- 编码表字段
  - 列名：`Value, Code`
  - 行数：Model=41，Category=8，Region=11，Variable=48

### A2. 数据加载与训练管线

- 涉及文件：`model_files/dataset.py`、`train_val_test.py`
- 入口类/方法
  - `CleanDataset.read_data(config)`
  - `fit_norm_stats_from_raw(raw_data, raw_mask)`
  - `apply_normalize_with_fitted(data)`
  - `build_denorm_bundle(device)`
  - `SPMDataset.__getitem__`
- 标准化
  - 粒度：`(category, region, variable)` 三元组
  - 统计：value 列做 1/99 分位剪裁后求 `mean/std`
  - 反标准化：`y = x * std + mean`
- 切分逻辑
  - `split_seed` 打乱后：train 80%，val 10%，test 10%
  - `is_gen=True`：不切分，全部走生成模式
- mask 用法
  - `mask` shape `(N,G)`，训练中用于 var-balanced loss 的有效组过滤
  - 评估中 broadcast 到 `(B,G,T)` 做有效时间点筛选

### A3. 模型结构（DiffSPM + UGnet）

- Diffusion 主体（`model_files/spm_diff/model.py`）
  - 类：`DiffSPM`
  - 核心方法：
    - `loss(x0, mask_train_data)`
    - `generalized_steps_ddim(x, seq, eta, label)`
    - `p_sample_loop_ddim(label)`
    - `final_projection_sandwich(x_last, label)`
    - `forward(label, n_samples)`
- Denoiser（`model_files/spm_diff/ugnet.py`）
  - 子模块：`TimeEmbedding`、`GATLayer`、`SpatialBlock`、`TcnBlock`、`ResidualBlock`、`Downsample/Upsample`、U-Net down/up
  - 条件嵌入：category/region/variable 三路 embedding
  - 关键变换：
    - 输入 value：`[B,G,T,1] -> Linear -> [B,G,T,d_h]`
    - 拼接条件后 `permute` 为 `[B,4d_h,G,T]`
    - `1x1 Conv(fusion)` 压回 `[B,d_h,G,T]`
    - 输出 `eps_theta`: `[B,G,T,1]`
- 图 supports
  - `GATLayer` 输出 `[B,num_heads,G,G]`
  - `SpatialBlock` 按 `ks=supports_len` 做图卷积 `einsum`

### A4. 采样与约束

- DDIM
  - `N=200`，`sample_steps=45`（默认）
  - `seq = (linspace(0, sqrt(N*0.8), sample_steps)^2).astype(int)`
- soft guidance（每步）
  - 在 `x0_t` 上开梯度，先 `denorm_fn` 到物理空间
  - 计算 `E = compute_constraint_penalty(x_phys)`（region + var + nonneg）
  - 梯度方向更新，并做相对步长 trust 限制
- final projection sandwich（最终收口）
  - 等式投影 -> 非负 pushback -> 等式再投影（代码中末尾重复 eq）
  - 输出强制在物理空间满足最终约束口径

### A5. 评估与绘图

- 评估（`utils/eval.py`）
  - `evaluate_sampling_metrics(...)`
  - 输出：
    - `overall`: MAE/RMSE
    - `overall_macro`: MAE/RMSE/MMD2
    - `groups`: by category/region/variable
    - `physics`: `eq_residual`, `eq_residual_region`, `eq_residual_var`, `neg_ratio`
    - `vet_pass_rate`
- 导出（`utils/export_fig2.py`）
  - 写入目录：`fig2_data_prepare_*/fig2_data/`
  - 核心文件：
    - `fig2_raw_test.npz`, `fig2_raw_train.npz`
    - `fig2a_real_syn.npz`
    - `corr_real.npy`, `corr_syn.npy`, `corr_distance.txt`
    - `embeddings_per_step.npz`, `categories_per_step.npz`, `adjacency_per_step.npz`
    - `X_train.npy`, `y_train.npy`, `X_syn.npy`, `y_syn.npy`
    - `fig2f_metrics.csv`, `fig2f_confusion.png`

### A6. 生成与落盘

- 入口：`generation_cons.py`
  - `main(params)` -> `test_all(...)`
- 采样与打包 shape
  - 模型输出：`[B,S,G,T,1]`
  - 打包后：`[B*S,G,T,5]`（前 4 维复用 code，最后一列是 value）
- 落盘
  - 临时：`output/forecast/temp2/batch_*.npy` + `batch_*.meta.json`
  - 最终：`output/forecast/最终生成_{epoch}.npy`
- 可选后处理
  - 约束修正：`utils/cons.py::process_memmap_in_batches(...)`
  - CSV导出：`utils/save_csv.py::SavePreToCSV`

---

## B. End-to-End 详细流程（时间顺序）

### B1. 数据预处理：原始 AR6 -> `spm.npz`

1. 读取 world 与 R10 原始表，固定年份列 `2010..2100`。
2. 从 `Variables48.csv` 取 48 个变量，附加 `Emissions|CO2|Waste` 用于后续合并。
3. 依据 `pairs_metadata` 约束 `(Model,Scenario)` 的候选集合。
4. 缺失/一致性处理：
   - `Waste` 合并进 `Other`
   - 用 `CO2 = AFOLU + EnergyAndIndustrial + Other` 补缺失 `Other`
   - 对指定地热变量做补 0
5. 统计 pair 缺失率，筛选可用 pair（notebook 中示例阈值 `<=0.41`）。
6. 合并 world + R10，删除 `R10ROWO`，带上 category 标签。
7. 编码四列（Model/Category/Region/Variable）并导出 CSV。
8. `build_spm_fixed_grid` 固定 RV 网格（`k=r*V+v`）：
   - `padded[i,:,:,0]=model_code`
   - `padded[i,:,:,1]=category_code`
   - `padded[i,:,:,2]=region_code`
   - `padded[i,:,:,3]=variable_code`
   - `padded[i,k,:,4]=该 RV 的 T 维序列`
   - `mask[i,k]=1` 表示有效 RV
9. 写 `data/dataset/SPM/spm.npz`。

### B2. 训练：`spm.npz` -> checkpoint

1. 启动：`train_val_test.py` 解析参数并组装 `config`。
2. `CleanDataset` 加载 `spm.npz`，按 `split_seed` 切分，拟合并应用标准化。
3. `build_denorm_bundle` 生成可微反标准化 `denorm_fn`，并 `model.attach_denorm(...)`。
4. 构建 `SPMDataset` + DataLoader；train 集可用 `oversample_by_multiplier` 增强。
5. 每个 batch：
   - `label`: `[B,G,T,5]`
   - `mask_data`: `[B,G]`
   - 扩散随机时间步 `t`，对 value 列加噪得到 `x_t`
   - 预测噪声 `eps_theta`
   - 计算 var-balanced noise loss
   - 可选 constraint loss（`enable_constraint_loss=1`）
   - 反传，梯度裁剪 `max_norm=1.0`，优化器更新
6. 每 epoch 保存 `epoch_*.ckpt`，并按验证指标更新 `best.ckpt`。
7. 训练后（`export_fig2=1`）调用 `run_export` 导出 fig2_data。

### B3. 采样生成：噪声 -> RVT 表

1. 初始化 `x ~ N(0,1)`，shape `[B,G,T,1]`。
2. DDIM 逐步反推：
   - `et = eps_model([code, xt], t)`
   - `x0_t = (xt - et*sqrt(1-at))/sqrt(at)`
3. soft guidance（在 normalized `x0_t` 上）：
   - 拼回完整 `x_full`，经 `denorm_fn` 到物理空间
   - 计算约束能量 `E`
   - `grad = dE/dx0_t`，按 schedule (`soft_eta_min/max`, `schedule`) 更新
4. 计算 `xt_next` 进入下一步。
5. 最后一步执行 `final_projection_sandwich` 保证等式/非负收口。
6. 得到物理空间 value，最终打包为 `[B*S,G,T,5]`。

### B4. 评估：样本 -> Fig2 指标

1. `evaluate_sampling_metrics` 对每个 batch 采样并收集指标。
2. 分布一致性：
   - 逐 `(b,g)` 计算 `MMD2(X_std, y_std)`，然后分组聚合。
3. 依赖结构：
   - 把每个 `(r,v)` 的时间序列压成 summary（sum）
   - 得 `corr_real` / `corr_syn` Spearman 矩阵，计算 `corr_distance`
4. 物理指标：
   - 统计 `eq_residual*` 与 `neg_ratio`
5. category prediction：
   - 用变量汇总特征（48 维）训练分类器，计算 macro-F1
6. embedding / graph：
   - 钩子抓取 `fusion` embedding 与 `gen_graph` adjacency

### B5. 落盘与复现实验

- 单次完整训练（示例）
  - `python train_val_test.py --ablation Full --denoiser UGnet --split_seed 1234 --seed 1234`
- 多组消融（集群脚本）
  - `run.sh`
  - `run_all_exp.sh`
  - `run_train_test.sh`
- 训练后导图数据位置
  - `fig2_data_prepare_*/fig2_data/`
- 绘图脚本
  - `python experiments/plot_fig2a_my.py`
  - `python experiments/plot_fig2b_new.py`
  - `python experiments/plot_fig2d.py`
  - `python experiments/plot_fig2f.py`
  - `python experiments/plot_fig2g.py`
- 大规模生成
  - `python generation_cons.py`
  - 输出：`output/forecast/最终生成_{epoch}.npy`

---

## C. 关键超参数与默认值来源

默认值主要来自：

- `train_val_test.py::get_params()`
- `generation_cons.py::get_params()`

常用项（默认）：

- 扩散：`N=200`，`beta_end=0.1`，`sample_steps=45`
- 网络：`hidden_size(d_h)=64`，`n_blocks=2`
- 训练：`lr=5e-5`，`train_batch_size=32`，`epoch=300`
- 采样：`n_samples=64`（训练评估）/ `6000`（生成脚本）
- 约束损失权重：`lambda_region=0.002`，`lambda_var=0.002`，`lambda_nn=0.004`
- soft guidance：
  - `use_soft_guidance=1`
  - `soft_guidance_schedule=exp`
  - `soft_eta_min=0.0`
  - `soft_eta_max=1e-3`
- projection：
  - `reverse_projection_scale=2`
- 复现实验：
  - `seed=2026`（训练脚本默认）
  - `split_seed=2026`

---

## D. 易错点与排查（工程坑位）

1. `type=bool` 参数解析易误判（建议改 `0/1` 或 `store_true`）。
2. `G` 维顺序必须是 `r` 外、`v` 内（`k=r*V+v`），否则 `view(B,R,V,T)` 全错。
3. `mask` 是 `(N,G)`，训练和评估广播维度要一致。
4. `TinyUGnet` 的 `num_heads` 与 `supports_len` 不一致时，`einsum` 维度会报错。
5. `train_val_test.py` 每轮会删旧 `epoch_*.ckpt`，不会保留完整历史。
6. `generation_cons.py` 当前构造了占位 `test_data`，真实生成请替换为真实条件输入。
7. 大规模生成最后 `np.concatenate` 可能 OOM，建议改流式写目标 memmap。
8. `denorm_fn` 未 attach 时，约束会在错误空间计算或失效。
9. 某些 `(c,r,v)` 统计缺失会回退 `mean=0,std=1`，需关注日志 warning。
10. VET 规则严格，过滤后某批可能 0 样本，后续统计要容错。
11. `NoConstraint` 消融会跳过 soft guidance 与最终投影，物理残差通常显著变差。
12. 读取超大 `.npy` 时应使用 `mmap_mode='r'`，避免一次性内存爆。

---

## E. 最小运行手册（建议）

1. 数据准备：确认 `data/dataset/SPM/spm.npz` 和编码表存在。
2. 单次训练：
   - `python train_val_test.py --ablation Full --denoiser UGnet --split_seed 1234 --seed 1234`
3. 导出数据：训练结束自动写 `fig2_data_prepare_*/fig2_data/`
4. 绘图：运行 `experiments/plot_fig2*.py`
5. 大规模生成：
   - `python generation_cons.py`
   - 结果在 `output/forecast/最终生成_{epoch}.npy`

如需继续做工业级落盘（CSV/zip）与后处理约束修正，再跑：

- `all_gendata_process/use_cons.py`
- `all_gendata_process/save_to_csv.py`

