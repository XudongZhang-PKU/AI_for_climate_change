# Fig.2 绘图

训练时开启 `--export_fig2 1`，会在保存 best ckpt 后把绘图数据写到 `output/fig2_data/`（或 `output/model/<save_dir>/fig2_data/`）。

**导出流程**：先 **Phase 1** 统一生成并落盘（`fig2_raw_test.npz`、`fig2_raw_train.npz`），再 **Phase 2** 从落盘数据计算 (a)(b)(c)(f) 等指标并写出绘图用 CSV/npy。若之后仅需重算指标、不改生成结果，可令 `run_export(..., use_saved_raw=True)` 从已有 npz 加载，只执行 Phase 2（无需再跑模型）。

**Fig.2(b)** 相关矩阵为 **(变量+地区)×(变量+地区)**（即 `num_region*num_variable`，如 11×48=528 维）。

绘图脚本从 fig2_data 目录读取数据并出图（可指定 `--data_dir` 指向实际路径）。

```bash
# 训练时导出（在 train_val_test 中）
python train_val_test.py ... --export_fig2 1

# 绘图（在项目根目录运行，或指定 --data_dir）
python experiments/plot_fig2a.py --data_dir output/model/最后练/fig2_data --out fig2a.png
python experiments/plot_fig2b.py --data_dir output/model/最后练/fig2_data
python experiments/plot_fig2c.py --data_dir output/model/最后练/fig2_data
python experiments/plot_fig2d.py --data_dir output/model/最后练/fig2_data  # 需 export 时生成 convergence.csv
python experiments/plot_fig2e.py --data_dir output/model/最后练/fig2_data
python experiments/plot_fig2f.py --data_dir output/model/最后练/fig2_data
python experiments/plot_fig2g.py --data_dir output/model/最后练/fig2_data  # 需 ablation_metrics.csv（多 run 合并）
```
