# -*- coding: utf-8 -*-
"""
导出 Fig.2 绘图所需数据，供 train_val_test 在训练结束后调用。
先统一生成并落盘（Phase 1），再从落盘数据计算各指标并写出 CSV/npy（Phase 2）。
experiments/plot_fig2a.py ~ plot_fig2g.py 读取本模块写出的文件并绘制 Fig.2。

各子图与输出文件对应关系：
  Fig.2(a) 分布一致性     -> fig2a_real_syn.npz（plot_fig2a 现场按标准化空间算 MMD²/MAE/DTW）
  Fig.2(b) 依赖结构       -> corr_real.npy, corr_syn.npy, corr_distance.txt。corr_* 为 (RV×RV) Spearman 相关矩阵，由 test 真实/合成在「每条样本的每个 (region,variable) 的 summary 标量」上计算。
  Fig.2(c) 物理一致性     -> physics_eq_residual.npy, physics_neg_ratio.npy（已移除 physics_pass）
  Fig.2(d) 采样量敏感性   -> convergence.csv
  Fig.2(e) Embedding/图   -> embeddings_per_step.npz, categories_per_step.npz, adjacency_per_step.npz（按去噪单步）；
    - categories_per_step.npz: 与 embeddings_per_step 一一对应，键 step_K 为第 K 步每 batch 保留的 2 条轨迹对应的原样本 category 标签，形状 (N,)。
    - 兼容：embeddings.npy (N,d_h)、categories.npy (N,) 为最后一步的**全部样本**（非均值）；adjacency.npy 为最后一步**所有 batch 的 supports 均值**，形状 (G,G)。
  Fig.2(f) 分类有效性     -> X_train.npy, y_train.npy, X_syn.npy, y_syn.npy；macro-F1 写 fig2f_metrics.csv；始终写 fig2f_confusion.png。
  Fig.2(g) 消融对比       -> 已移除（不再写 ablation_metrics.csv）
"""
from __future__ import annotations

import csv
import os
import numpy as np
import torch
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from scipy.stats import spearmanr

from .eval import sample_predict, _pack_generated_to_gtk5


FIG2_N_SAMPLES = 200
# 画图用 embedding：每个 batch 只保留前 N 个样本，减少存储与绘图点数量
FIG2_EMB_SAMPLES_PER_BATCH = 2


def _summary_ts(values_bt):
    """整条时间序列累计值（求和）作为标量 summary。服务于 Fig.2(a)(b)(f)。"""
    v = values_bt.cpu().numpy() if torch.is_tensor(values_bt) else np.asarray(values_bt)
    return float(np.sum(v))


def run_export(
    model,
    test_loader,
    train_loader,
    test_mask,
    train_mask,
    config,
    clean_norm_data,
    means_t,
    stds_t,
    out_dir=None,
    n_samples_fig2=FIG2_N_SAMPLES,
    split_n=4,
    skip_convergence=True,
    verbose=True,
    use_saved_raw=False,
    n_physics_samples=50,
    n_syn_fig2f=20,
):
    """主入口：Phase 1 生成并落盘 raw，Phase 2 写出 Fig.2(a)(b)(c)(d)(e)(f) 所需文件。"""
    if out_dir is None:
        work_path = getattr(config, "work_path", None) or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(work_path, "output", "fig2_data")
    os.makedirs(out_dir, exist_ok=True)
    num_v = getattr(config.model, "num_variable", 48)
    num_r = getattr(config.model, "num_region", 11)
    key_var_codes = list(range(num_v))
    n_rv = num_r * num_v
    device = getattr(config, "device", next(model.parameters()).device)
    means_np = means_t.cpu().numpy() if torch.is_tensor(means_t) else means_t
    stds_np = stds_t.cpu().numpy() if torch.is_tensor(stds_t) else stds_t
    test_mask_np = test_mask.cpu().numpy() if torch.is_tensor(test_mask) else test_mask
    train_mask_np = train_mask.cpu().numpy() if torch.is_tensor(train_mask) else train_mask
    path_raw_test = os.path.join(out_dir, "fig2_raw_test.npz")
    path_raw_train = os.path.join(out_dir, "fig2_raw_train.npz")

    if use_saved_raw and os.path.isfile(path_raw_test) and os.path.isfile(path_raw_train):
        if verbose:
            print("[export_fig2] use_saved_raw=True，加载已落盘数据，仅执行 Phase 2")
        test_label = np.load(path_raw_test)["test_label"]
        test_syn = np.load(path_raw_test)["test_syn"]
        test_index = np.load(path_raw_test)["test_index"]
        train_label = np.load(path_raw_train)["train_label"]
        train_index = np.load(path_raw_train)["train_index"]
    else:
        # ---------- Phase 1：生成并落盘 raw，写出 Fig.2(e) 所需 embeddings/adjacency（按去噪单步）----------
        test_label_list, test_syn_list, test_index_list = [], [], []
        eps_model = getattr(model, "eps_model", model)
        # 按步累积：embedding 每步每 batch 只保留 2 条采样轨迹（在 n_samples_fig2 次采样中只取 2 条）；adj 每步对所有 batch 平均
        emb_by_step = defaultdict(list)
        cat_by_step = defaultdict(list)
        adj_by_step = defaultdict(list)
        step_counter = [0]
        current_batch_label = [None]
        n_samp_ref = [n_samples_fig2]

        def _hook_fusion(module, inp, out):
            o = out.detach().cpu()
            emb_b = o.mean(dim=-1).mean(dim=-1) if o.dim() == 4 else o.mean(dim=-1)
            ns = n_samp_ref[0]
            # 在 B*ns 条轨迹中只保留 2 条：取第 0 条与第 ns 条（分别对应原 batch 第 1、第 2 个样本的一条轨迹）
            if emb_b.shape[0] > ns:
                idx = [0, ns]
            else:
                idx = [0, min(1, emb_b.shape[0] - 1)]
            emb_small = emb_b[idx].numpy()
            step_idx = step_counter[0]
            emb_by_step[step_idx].append(emb_small)
            if current_batch_label[0] is not None:
                lab = current_batch_label[0]
                # 与 emb 对应：两条轨迹分别对应原 batch 样本 0 和 1 的 category
                c0 = lab[0, 0, 0, 1].cpu().numpy() if torch.is_tensor(lab) else np.asarray(lab[0, 0, 0, 1])
                c1 = lab[min(1, lab.shape[0] - 1), 0, 0, 1].cpu().numpy() if torch.is_tensor(lab) else np.asarray(lab[min(1, lab.shape[0] - 1), 0, 0, 1])
                cat_by_step[step_idx].append(np.stack([c0, c1]))

        def _hook_supports(module, inp, out):
            o = out.detach().cpu()
            adj_b = o.mean(dim=(0, 1)).numpy()
            step_idx = step_counter[0]
            adj_by_step[step_idx].append(adj_b)
            step_counter[0] += 1

        # 仅 Full 时挂 embedding/graph hook 并写 Fig.2(e)；其他消融不写
        do_embedding_graph = getattr(config, "ablation", None) == "Full"
        h_f = eps_model.fusion.register_forward_hook(_hook_fusion) if do_embedding_graph else None
        h_s = eps_model.gen_graph.register_forward_hook(_hook_supports) if do_embedding_graph else None
        try:
            it = tqdm(test_loader, desc="fig2 Phase1 test") if verbose else test_loader
            for batch_idx, (label, index) in enumerate(it):
                label = label.to(device).float()
                B, G, T, _ = label.shape
                step_counter[0] = 0
                current_batch_label[0] = label
                y_s = sample_predict(model, label, n_samples=n_samples_fig2, split=split_n)
                current_batch_label[0] = None
                test_label_list.append(label.cpu().numpy())
                test_syn_list.append(y_s.numpy())
                test_index_list.append(index.cpu().numpy() if torch.is_tensor(index) else np.asarray(index))
        finally:
            if h_f is not None:
                h_f.remove()
            if h_s is not None:
                h_s.remove()
        test_label = np.concatenate(test_label_list, axis=0)
        test_syn = np.concatenate(test_syn_list, axis=0)
        test_index = np.concatenate(test_index_list, axis=0)
        np.savez_compressed(path_raw_test, test_label=test_label, test_syn=test_syn, test_index=test_index)
        if verbose:
            print(f"[export_fig2] Phase1 已落盘 {path_raw_test}")
        # Fig.2(e) 按步写出：embeddings/categories 每步 (N, d_h)/(N,)；adjacency 每步一张图 (G,G)，为所有样本平均
        steps_sorted = sorted(emb_by_step.keys())
        if steps_sorted:
            emb_dict = {}
            cat_dict = {}
            for s in steps_sorted:
                emb_dict["step_%d" % s] = np.concatenate(emb_by_step[s], axis=0)
                if s in cat_by_step and cat_by_step[s]:
                    cat_dict["step_%d" % s] = np.concatenate(cat_by_step[s], axis=0)
            np.savez_compressed(os.path.join(out_dir, "embeddings_per_step.npz"), **emb_dict)
            if cat_dict:
                np.savez_compressed(os.path.join(out_dir, "categories_per_step.npz"), **cat_dict)
        adj_steps_sorted = sorted(adj_by_step.keys())
        if adj_steps_sorted:
            adj_dict = {}
            for s in adj_steps_sorted:
                adj_dict["step_%d" % s] = np.mean(adj_by_step[s], axis=0)
            np.savez_compressed(os.path.join(out_dir, "adjacency_per_step.npz"), **adj_dict)
        if verbose and (steps_sorted or adj_steps_sorted):
            print("[export_fig2] wrote embeddings_per_step.npz, categories_per_step.npz, adjacency_per_step.npz")
        # 兼容旧绘图脚本：用最后一步作为单文件
        if steps_sorted:
            last_s = steps_sorted[-1]
            np.save(os.path.join(out_dir, "embeddings.npy"), np.concatenate(emb_by_step[last_s], axis=0))
            if last_s in cat_by_step and cat_by_step[last_s]:
                np.save(os.path.join(out_dir, "categories.npy"), np.concatenate(cat_by_step[last_s], axis=0))
        if adj_steps_sorted:
            last_s = adj_steps_sorted[-1]
            np.save(os.path.join(out_dir, "adjacency.npy"), np.mean(adj_by_step[last_s], axis=0))
        train_label_list, train_index_list = [], []
        for label, index in (tqdm(train_loader, desc="fig2 Phase1 train") if verbose else train_loader):
            train_label_list.append(label.numpy() if hasattr(label, "numpy") else np.asarray(label))
            train_index_list.append(index.numpy() if torch.is_tensor(index) else np.asarray(index))
        train_label = np.concatenate(train_label_list, axis=0)
        train_index = np.concatenate(train_index_list, axis=0)
        np.savez_compressed(path_raw_train, train_label=train_label, train_index=train_index)
        if verbose:
            print(f"[export_fig2] Phase1 已落盘 {path_raw_train}")

    N_test, G, T, _ = test_label.shape
    S_saved = test_syn.shape[1]

    # 将 test 真实值反归一化到物理空间，与 test_syn 一致
    lab = test_label
    cat = lab[:, :, 0, 1].astype(np.int64)
    reg = lab[:, :, 0, 2].astype(np.int64)
    var = lab[:, :, 0, 3].astype(np.int64)
    mu = np.expand_dims(means_np[cat, reg, var], axis=-1)
    sd = np.expand_dims(stds_np[cat, reg, var], axis=-1)
    real_phys = lab[:, :, :, 4] * sd + mu

    # ---------- Phase 2：从 raw 计算并写出各子图数据 ----------
    # Fig.2(a) 分布一致性：只写出每个 CRV 的 real/syn 序列（物理空间），指标在 plot_fig2a 中按标准化空间现场计算
    real_ts_bucket = defaultdict(list)
    syn_ts_bucket = defaultdict(list)
    for i in range(N_test):
        mask_b = test_mask_np[test_index[i]]
        lab = test_label[i]
        rp = real_phys[i]
        syn_b = test_syn[i]
        for g in range(G):
            if not mask_b[g]:
                continue
            c, r, v = int(lab[g, 0, 1]), int(lab[g, 0, 2]), int(lab[g, 0, 3])
            key = (c, r, v)
            real_ts_bucket[key].append(rp[g, :])
            for s in range(syn_b.shape[0]):
                syn_ts_bucket[key].append(syn_b[s, g, :])
    keys_with_data = [k for k in real_ts_bucket if k in syn_ts_bucket and len(real_ts_bucket[k]) > 0 and len(syn_ts_bucket[k]) > 0]
    fig2a_npz = os.path.join(out_dir, "fig2a_real_syn.npz")
    save_dict = {"keys": np.array(keys_with_data, dtype=np.int64)}
    # 每个 CRV 的训练集 μ、σ，供 plot_fig2a 用训练数据做标准化
    save_dict["train_means"] = np.array([float(means_np[c, r, v]) for (c, r, v) in keys_with_data], dtype=np.float64)
    save_dict["train_stds"] = np.array([float(stds_np[c, r, v]) for (c, r, v) in keys_with_data], dtype=np.float64)
    for i, key in enumerate(keys_with_data):
        save_dict["real_%d" % i] = np.array(real_ts_bucket[key], dtype=np.float64)
        save_dict["syn_%d" % i] = np.array(syn_ts_bucket[key], dtype=np.float64)
    np.savez_compressed(fig2a_npz, **save_dict)
    if verbose:
        print(f"[export_fig2] wrote {fig2a_npz} ({len(keys_with_data)} CRVs)")

    # Fig.2(b) 依赖结构：(变量+地区)×(变量+地区) Spearman 相关矩阵
    real_mat = np.full((N_test, n_rv), np.nan, dtype=np.float64)
    for i in range(N_test):
        mask_b = test_mask_np[test_index[i]]
        lab = test_label[i]
        rp = real_phys[i]
        for g in range(G):
            if not mask_b[g]:
                continue
            r, v = int(lab[g, 0, 2]), int(lab[g, 0, 3])
            real_mat[i, r * num_v + v] = _summary_ts(rp[g, :])
    n_syn_rows = N_test * S_saved
    syn_mat = np.full((n_syn_rows, n_rv), np.nan, dtype=np.float64)
    row_idx = 0
    for i in range(N_test):
        mask_b = test_mask_np[test_index[i]]
        lab = test_label[i]
        syn_b = test_syn[i]
        for s in range(syn_b.shape[0]):
            for g in range(G):
                if not mask_b[g]:
                    continue
                r, v = int(lab[g, 0, 2]), int(lab[g, 0, 3])
                syn_mat[row_idx, r * num_v + v] = _summary_ts(syn_b[s, g, :])
            row_idx += 1
    syn_mat = syn_mat[:row_idx]

    def _safe_corr(mat):
        mat = np.nan_to_num(mat, nan=0.0)
        n_col = mat.shape[1]
        if mat.shape[0] < 2 or n_col < 2:
            return np.eye(n_col)
        r, _ = spearmanr(mat, nan_policy="omit")
        if np.isscalar(r):
            r = np.array([[1.0]])
        r = np.nan_to_num(r, nan=0.0)
        if r.shape != (n_col, n_col):
            return np.eye(n_col)
        return r
    corr_real = _safe_corr(real_mat)
    corr_syn = _safe_corr(syn_mat)
    corr_distance = float(np.abs(corr_real - corr_syn).mean())
    np.save(os.path.join(out_dir, "corr_real.npy"), corr_real)
    np.save(os.path.join(out_dir, "corr_syn.npy"), corr_syn)
    with open(os.path.join(out_dir, "corr_distance.txt"), "w") as f:
        f.write(str(corr_distance))
    if verbose:
        print(f"[export_fig2] wrote corr_*.npy (RV×RV), corr_distance={corr_distance:.4f}")

    # Fig.2(c) 物理一致性：EqResidual、NegRatio、Pass

    # Fig.2(d) 采样量敏感性：convergence.csv（可选占位）


    # Fig.2(f) 分类有效性：把每条样本压成「每个变量一个 summary 数」的向量，用于 train-on-real / test-on-syn 分类
    # 直观：一条样本有 G 组 (region×variable)，每组一条时间序列；这里按变量编码 vc 取该变量对应组的整条序列做 _summary_ts，
    # 得到 len(key_vars) 个数，组成一个特征向量。这样分类器用「各变量汇总」预测 category（语义有效性）。
    def _sample_to_summary_vec(lab, mask_b, key_vars, values_phys=None):
        lab = np.asarray(lab)
        row = []
        for vc in key_vars:
            for g in range(lab.shape[0]):
                if mask_b[g] and int(lab[g, 0, 3]) == vc:
                    if values_phys is not None:
                        row.append(_summary_ts(values_phys[g, :]))
                    else:
                        row.append(_summary_ts(lab[g, :, 4]))
                    break
            else:
                row.append(np.nan)
        return row

    lab_train = train_label
    cat_t = lab_train[:, :, 0, 1].astype(np.int64)
    reg_t = lab_train[:, :, 0, 2].astype(np.int64)
    var_t = lab_train[:, :, 0, 3].astype(np.int64)
    mu_t = np.expand_dims(means_np[cat_t, reg_t, var_t], axis=-1)
    sd_t = np.expand_dims(stds_np[cat_t, reg_t, var_t], axis=-1)
    train_phys = lab_train[:, :, :, 4] * sd_t + mu_t

    N_train = train_label.shape[0]
    X_train_list = []
    y_train_list = []
    for i in range(N_train):
        mask_b = train_mask_np[train_index[i]]
        X_train_list.append(_sample_to_summary_vec(train_label[i], mask_b, key_var_codes, values_phys=train_phys[i]))
        y_train_list.append(int(train_label[i, 0, 0, 1]))
    X_train = np.array(X_train_list, dtype=np.float64)
    y_train = np.array(y_train_list, dtype=np.int64)
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    S_f = min(n_syn_fig2f, test_syn.shape[1])
    X_syn_list = []
    y_syn_list = []
    for i in range(N_test):
        mask_b = test_mask_np[test_index[i]]
        lab = test_label[i]
        syn_b = test_syn[i]
        for s in range(S_f):
            row = []
            for vc in key_var_codes:
                for g in range(G):
                    if mask_b[g] and int(lab[g, 0, 3]) == vc:
                        row.append(_summary_ts(syn_b[s, g, :]))
                        break
                else:
                    row.append(np.nan)
            X_syn_list.append(row)
            y_syn_list.append(int(lab[0, 0, 1]))
    X_syn = np.array(X_syn_list, dtype=np.float64)
    y_syn = np.array(y_syn_list, dtype=np.int64)
    np.save(os.path.join(out_dir, "X_syn.npy"), X_syn)
    np.save(os.path.join(out_dir, "y_syn.npy"), y_syn)
    if verbose:
        print(f"[export_fig2] wrote X_train.npy, y_train.npy, X_syn.npy, y_syn.npy")

    # Fig.2(f) 在 export 内直接跑简单分类：LogisticRegression train on real / test on syn，写出 macro-F1 与可选混淆矩阵
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score, confusion_matrix
        from sklearn.preprocessing import StandardScaler
        X_tr = np.nan_to_num(X_train, nan=0.0)
        X_sy = np.nan_to_num(X_syn, nan=0.0)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_sy_s = scaler.transform(X_sy)
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X_tr_s, y_train)
        y_pred = clf.predict(X_sy_s)
        macro_f1 = float(f1_score(y_syn, y_pred, average="macro"))
        with open(os.path.join(out_dir, "fig2f_metrics.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["macro_f1"])
            w.writerow([macro_f1])
        if verbose:
            print(f"[export_fig2] Fig.2(f) macro-F1 (train real / test syn) = {macro_f1:.4f}, wrote fig2f_metrics.csv")
        # 始终写出混淆矩阵图（无参数控制）
        cm = confusion_matrix(y_syn, y_pred)
        n_cls = max(len(np.unique(y_syn)), cm.shape[0])
        class_names = [f"C{i}" for i in range(n_cls)]
        cm_path = os.path.join(out_dir, "fig2f_confusion.png")
        try:
            from utils.plot_utils import confusion_matrix_plot
            confusion_matrix_plot(cm, cm_path, class_names=class_names)
        except Exception:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.imshow(np.asarray(cm), cmap="Blues")
            ax.set_xticks(range(n_cls))
            ax.set_yticks(range(n_cls))
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center")
            plt.savefig(cm_path, bbox_inches="tight")
            plt.close()
        if verbose:
            print(f"[export_fig2] wrote fig2f_confusion.png")
    except Exception as e:
        if verbose:
            import warnings
            warnings.warn(f"[export_fig2] Fig.2(f) classifier: {e}")

    return out_dir
