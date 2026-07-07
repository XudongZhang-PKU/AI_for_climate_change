# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from collections import Counter, defaultdict
import math, random
import torch.utils.data as data
import torch


def resample_by_category(data, mask, seed=None):
    """
    Balance by category (label[0,0,1]) before normalization.
    data: (N, G, T, C), mask: (N, G)
    """
    if len(data) == 0:
        return data, mask

    rng = np.random.RandomState(seed) if seed is not None else np.random
    group_indices = defaultdict(list)

    for i in range(len(data)):
        category = int(data[i, 0, 0, 1])
        group_indices[category].append(i)

    new_indices = []
    max_n = max((len(indices) for indices in group_indices.values()), default=0)
    for _, indices in group_indices.items():
        n = len(indices)
        if n == 0:
            continue
        if n < max_n:
            reps = max_n // n
            remainder = max_n % n
            repeated = indices * reps
            if remainder > 0:
                repeated += rng.choice(indices, size=remainder, replace=True).tolist()
            new_indices.extend(repeated)
        else:
            new_indices.extend(indices)

    if not new_indices:
        return data, mask

    new_indices = np.array(new_indices, dtype=int)
    rng.shuffle(new_indices)
    return data[new_indices], mask[new_indices]


def oversample_by_multiplier(dataset, mask, target_min=10, target_max=15):
    """
    按倍数增强：
      - 对于组内样本数 n < target_min，计算 k = ceil(target_min / n)，
        重复整个组 k 次后，再从中随机裁剪到 target_max。
      - 对于 n >= target_min，不做任何操作。
      
    Args:
        dataset: 数据集对象，dataset[i] 返回 [label, index]
        mask: 与数据集大小相同的掩码数组
        target_min: 每组最小目标样本数
        target_max: 每组最大目标样本数
        
    Returns:
        data.Subset 包装的新 dataset 和 mask
    """
    # 收集每个组的样本索引
    group_indices = defaultdict(list)
    
    # 遍历数据集，按组整理索引
    for i in range(len(dataset)):
        label, _ = dataset[i]  # shape:group_len, t, c
        group_key = tuple(label[0, 0, :2].tolist())
        group_indices[group_key].append(i)
    
    # 生成新的索引列表
    new_indices = []
    
    # 对每个组进行处理
    for group, indices in group_indices.items():
        n = len(indices)
        if n < target_min:
            # 计算需要重复的倍数
            k = math.ceil(target_min / n)
            # 重复整个组k次
            repeated_indices = indices * k
            
            # 如果重复后超过target_max，随机裁剪
            if len(repeated_indices) > target_max:
                repeated_indices = random.sample(repeated_indices, target_max)
            
            new_indices.extend(repeated_indices)
        else:
            # 如果样本数已经足够，直接添加
            new_indices.extend(indices)
    
    # 随机打乱索引
    random.shuffle(new_indices)
    # print(new_indices)
    
    # 返回包装后的数据集和掩码
    return data.Subset(dataset, new_indices), mask[new_indices]



def search_recent_data(train, label_start_idx, T_p, T_h):
    """
    T_p: prediction time steps
    T_h: historical time steps
    """
    if label_start_idx + T_p > len(train): return None
    start_idx, end_idx = label_start_idx - T_h, label_start_idx - T_p + T_p
    if start_idx < 0 or end_idx < 0: return None
    return (start_idx, end_idx), (label_start_idx, label_start_idx + T_p)


def search_multihop_neighbor(adj, hops=5):
    node_cnt = adj.shape[0]
    hop_arr = np.zeros((adj.shape[0], adj.shape[0]))
    for h_idx in range(node_cnt):  # refer node idx(n)
        tmp_h_node, tmp_neibor_step = [h_idx], [h_idx]  # save spatial corr node  # 0 step(self) first
        hop_arr[h_idx, :] = -1  # if the value exceed maximum hop, it is set to (hops + 1)
        hop_arr[h_idx, h_idx] = 0  # at begin, the hop of self->self is set to 0
        for hop_idx in range(hops):  # how many spatial steps
            tmp_step_node = []  # neighbor nodes in the previous k step
            tmp_step_node_kth = []  # neighbor nodes in the kth step
            for tmp_nei_node in tmp_neibor_step:
                tmp_neibor_step = list((np.argwhere(adj[tmp_nei_node] == 1).flatten()))  # find the one step neighbor first
                tmp_step_node += tmp_neibor_step
                tmp_step_node_kth += set(tmp_step_node) - set(tmp_h_node)  # the nodes that have appeared in the first k-1 step are no longer needed
                tmp_h_node += tmp_neibor_step
            tmp_neibor_step = tmp_step_node_kth.copy()
            all_spatial_node = list(set(tmp_neibor_step))  # the all spatial node in kth step
            hop_arr[h_idx, all_spatial_node] = hop_idx + 1
    return hop_arr[:, :, np.newaxis]


class AdvancedNormalizer:
    def __init__(self):
        self.transform_params = {}

    def fit_transform(self, data, group_key):
        """
        data: shape (N, T) or (N,), N个样本，T个时间步或特征
        group_key: 地区和变量组合的标识符（通常是一个字符串）
        """
        data = np.asarray(data)
        # 剪裁极端值
        lower_bound = np.percentile(data, 1)
        upper_bound = np.percentile(data, 99)
        clipped = np.clip(data, lower_bound, upper_bound)
        mean = np.mean(clipped)
        std = np.std(clipped)
        if std == 0:
            std = 1.0

        self.transform_params[group_key] = {
            'mean': mean,
            'std': std,
            'lower': lower_bound,
            'upper': upper_bound
        }

        normed = (data - mean) / std
        return normed

    def transform(self, data, group_key):
        data = np.asarray(data)
        params = self.transform_params.get(group_key)
        if params is None:
            raise ValueError(f'Group {group_key} not found in transform_params.')
        mean, std = params['mean'], params['std']
        return (data - mean) / std

    def inverse_transform(self, data, group_key):
        data = np.asarray(data)
        params = self.transform_params.get(group_key)
        if params is None:
            raise ValueError(f'Group {group_key} not found in transform_params.')
        mean, std = params['mean'], params['std']
        return data * std + mean


class CleanDataset():
    def __init__(self, config):

        # self.data_name = config.data.name
        self.feature_file = config.data.feature_file
        # self.idx_path = config.data.idx_path
        self.region_encoding_path = config.data.region_encoding

        self.adj = []
        # for i in range(4):
        #     adj = np.load(config.data.spatial+f"dim{i+1}_adj.npy")
        #     self.adj.append(adj)

        self.normalizer = AdvancedNormalizer()
        self.train_data, self.train_mask, self.val_data, self.val_mask, self.test_data, self.test_mask = self.read_data(config)
        # self.train_data, self.mask = self.read_data()


    def read_data(self, config):
        # 1. 加载数据
        all_data = np.load(self.feature_file, allow_pickle=True)
        data = all_data['padded']   # shape: (N, group, T, C)
        mask = all_data['mask']     # shape: (N, group)

        N, G, T, Cdim = data.shape
        cat_codes    = data[:, :, 0, 1].astype(int)  # shape: (N, G)
        region_codes = data[:, :, 0, 2].astype(int)  # shape: (N, G)
        var_codes    = data[:, :, 0, 3].astype(int)  # shape: (N, G)
        unique_cats    = sorted(np.unique(cat_codes))
        unique_regions = sorted(np.unique(region_codes))
        unique_vars    = sorted(np.unique(var_codes))
        # # ★ 保存真实编码列表（非常重要）
        self.cat_codes    = np.array(unique_cats, dtype=int)
        self.region_codes = np.array(unique_regions, dtype=int)
        self.var_codes    = np.array(unique_vars, dtype=int)

        # 6. 检查顺序是否一致
        reg_sorted = data[:, :, 0, 2]
        var_sorted = data[:, :, 0, 3]
        if np.all(reg_sorted == reg_sorted[0]) and np.all(var_sorted == var_sorted[0]):
            print("全局顺序一致，可以安全 view 为 [B,R,V,T]")
        else:
            print("警告：排序后仍不一致（理论上不会发生）")

        # 7. 标准化（关键：先拟合，再应用；拟合用未标准化的副本）

        N = len(data)  # == len(mask)
        resample_before_norm = getattr(config, "resample_before_norm", True)
        save_inverse_npy = getattr(config, "save_inverse_npy", True)

        if config.is_gen:
            # ---- 生成模式：用“全样本”作为一次性数据集 ----
            # 归一化：如果你希望用训练期统计，可在这里改成 self.normalizer.load(...)
            data_raw = data
            mask_raw = mask
            if resample_before_norm:
                data_raw, mask_raw = resample_by_category(data_raw, mask_raw)
            self.fit_norm_stats_from_raw(data_raw, mask_raw)
            normalized = self.apply_normalize_with_fitted(data_raw)

            train_data = normalized
            train_mask = mask_raw
            # 验证/测试置空（保持下游接口一致）
            val_data   = np.empty((0, G, T, Cdim), dtype=np.float32)
            test_data  = np.empty((0, G, T, Cdim), dtype=np.float32)
            val_mask   = np.empty((0, G), dtype=mask.dtype)
            test_mask  = np.empty((0, G), dtype=mask.dtype)

            print(f"[GEN MODE] N={N} -> {len(train_data)} (resample={resample_before_norm}); val/test 为空。")
            tp = self.normalizer.transform_params
            print(f"[shape] data={normalized.shape}, mask={train_mask.shape}")
            print(f"[fit]   groups={len(tp)}")
            pos_ratio = train_mask.mean()
            print(f"[mask=1 比例] all={pos_ratio:.3f}")
            if save_inverse_npy:
                self._save_inverse_npy(train_data, config, tag="gen_resampled" if resample_before_norm else "gen")
            return train_data, train_mask, val_data, val_mask, test_data, test_mask

        # 划分前打乱，避免顺序偏差（固定 split_seed 可复现，消融时多 run 共用同一 test 集）
        rng = np.random.RandomState(getattr(config, "split_seed", 42))
        idx = np.arange(N)
        rng.shuffle(idx)
        data = data[idx]
        mask = mask[idx]

        i_train = int(N * 0.8)
        i_val   = int(N * 0.9)  # [0, i_train) 训练, [i_train, i_val) 验证, [i_val, N) 测试

        # 1) 用训练集拟合归一化
        train_data_raw = data[:i_train]
        train_mask     = mask[:i_train]
        if resample_before_norm:
            train_data_raw, train_mask = resample_by_category(train_data_raw, train_mask)
        self.fit_norm_stats_from_raw(train_data_raw, train_mask)

        # 2) 归一化：先训练集（已重采样），再验证/测试（原分布）
        train_data = self.apply_normalize_with_fitted(train_data_raw)
        val_data   = self.apply_normalize_with_fitted(data[i_train:i_val])
        test_data  = self.apply_normalize_with_fitted(data[i_val:])

        # 3) 同步切分 mask
        val_mask   = mask[i_train:i_val]
        test_mask  = mask[i_val:]


        # 假设已有：
        # normalized_data, mask
        # i_train, i_val  （若没有就先算：N=len(mask); i_train=int(N*0.7); i_val=int(N*0.9)）
        tp = self.normalizer.transform_params
        print(f"[shape] train={train_data.shape}, val={val_data.shape}, test={test_data.shape}")
        print(f"[fit]   groups={len(tp)}")
        print(f"[split] N={N}, train_raw={i_train}, train_resampled={len(train_data)}, "
              f"val={i_val - i_train}, test={N - i_val}")
        print(f"[mask=1] train={int(train_mask.mean())}, "
              f"val={int(val_mask.mean())}, "
              f"test={int(test_mask.mean())}")
        if save_inverse_npy:
            tag = "train_resampled" if resample_before_norm else "train"
            self._save_inverse_npy(train_data, config, tag=tag)
        return train_data, train_mask, val_data, val_mask, test_data, test_mask

    

    # def _normalize(self, data, mask=None, fit=False):
    #     """
    #     优化的标准化函数
    #     data: shape (N, G, T, C) - N个组, G个样本/组, T个时间步, C个特征
    #     mask: shape (N, G) - 表示哪些样本是有效的
    #     """
    #     mask = mask.astype(bool)
    #     region_encoding = pd.read_csv(self.region_encoding_path)

    #     if mask is None:
    #         mask = np.ones((data.shape[0], data.shape[1]), dtype=bool)

    #     for region_id in region_encoding['Code'].unique():
    #         for variable_id in np.unique(data[:, :, 0, 3]):  # 变量编码位置
    #             # 创建布尔掩码，标识所有属于该 region 和 variable 的数据位置
    #             group_mask = (data[:, :, 0, 2] == region_id) & (data[:, :, 0, 3] == variable_id)

    #             if not np.any(group_mask):
    #                 continue

    #             if fit:
    #                 valid_mask = group_mask & mask
    #                 if not np.any(valid_mask):
    #                     continue
    #                 valid_indices = np.where(valid_mask)
    #                 valid_data = []
    #                 for i, j in zip(*valid_indices):
    #                     valid_data.append(data[i, j, :, 4].flatten())

    #                 valid_data = np.concatenate(valid_data)
    #                 group_key = f"{region_id}_{variable_id}"
    #                 self.normalizer.fit_transform(valid_data, group_key)

    #             for i, j in zip(*np.where(group_mask)):
    #                 group_key = f"{region_id}_{variable_id}"
    #                 data[i, j, :, 4] = self.normalizer.transform(
    #                     data[i, j, :, 4], group_key
    #                 )

    #     return data

    # def reverse_normalization(self, x, mask=None):
    #     region_encoding = pd.read_csv(self.region_encoding_path)

    #     for region_id in region_encoding['Code'].unique():
    #         for variable_id in np.unique(x[:, :, 0, 3]):
    #             group_mask = (x[:, :, 0, 2] == region_id) & (x[:, :, 0, 3] == variable_id)

    #             for i, j in zip(*np.where(group_mask)):
    #                 group_key = f"{region_id}_{variable_id}"
    #                 x[i, j, :, 4] = self.normalizer.inverse_transform(
    #                     x[i, j, :, 4], group_key
    #                 )

    #     return x


    # ====== (A) 先从“未标准化的副本”拟合所有 (category, region, variable) 的 μ/σ ======
    def fit_norm_stats_from_raw(self, raw_data, raw_mask):
        """
        raw_data: 标准化之前的拷贝 (N, G, T, C)
        raw_mask: (N, G) 0/1
        拟合后把统计写入 self.normalizer.transform_params
        - 优先用 valid_mask = group_mask & raw_mask
        - 如果 valid 为空，降级用 group_mask（也就会把补零的那一组也纳入，但胜在“有统计”）
        """
        cat_mat    = raw_data[:, :, 0, 1].astype(int)
        region_mat = raw_data[:, :, 0, 2].astype(int)
        var_mat    = raw_data[:, :, 0, 3].astype(int)
        cat_ids    = np.unique(cat_mat).astype(int)
        region_ids = np.unique(region_mat).astype(int)
        var_ids    = np.unique(var_mat).astype(int)
        fitted_cnt = 0

        for c in cat_ids:
            for r in region_ids:
                for v in var_ids:
                    group_mask = (cat_mat == c) & (region_mat == r) & (var_mat == v)
                    if not np.any(group_mask):
                        continue

                    # 先用带 mask 的有效样本
                    valid_mask = group_mask & raw_mask.astype(bool)
                    use_mask = valid_mask if np.any(valid_mask) else group_mask   # 兜底：全组数据

                    ii, jj = np.where(use_mask)
                    vals = []
                    for i, j in zip(ii, jj):
                        vals.append(raw_data[i, j, :, 4].ravel())
                    vals = np.concatenate(vals) if len(vals) > 0 else None
                    if vals is None or vals.size == 0:
                        # 极少数边界：实在没有值，就给个默认
                        self.normalizer.transform_params[f"{c}_{r}_{v}"] = {
                            "mean": 0.0, "std": 1.0,
                            "lower": 0.0, "upper": 0.0
                        }
                        continue

                    # 1/99 百分位剪裁 + μ/σ
                    lower = np.percentile(vals, 1)
                    upper = np.percentile(vals, 99)
                    clipped = np.clip(vals, lower, upper)
                    mean = float(np.mean(clipped))
                    std  = float(np.std(clipped))
                    if std == 0.0:
                        std = 1.0

                    self.normalizer.transform_params[f"{c}_{r}_{v}"] = {
                        "mean": mean, "std": std,
                        "lower": float(lower), "upper": float(upper)
                    }
                    fitted_cnt += 1

        print(f"[fit_norm_stats_from_raw] fitted groups: {fitted_cnt}")

    # ====== (B) 用已拟合的统计，标准化 data 的 value 列 ======
    def apply_normalize_with_fitted(self, data):
        """
        用 self.normalizer.transform_params 标准化 data[:,:,:,4]（就地修改）
        """
        cat_mat    = data[:, :, 0, 1].astype(int)
        region_mat = data[:, :, 0, 2].astype(int)
        var_mat    = data[:, :, 0, 3].astype(int)
        cat_ids    = np.unique(cat_mat).astype(int)
        region_ids = np.unique(region_mat).astype(int)
        var_ids    = np.unique(var_mat).astype(int)
        miss = 0
        for c in cat_ids:
            for r in region_ids:
                for v in var_ids:
                    key = f"{c}_{r}_{v}"
                    p = self.normalizer.transform_params.get(key)
                    if p is None:
                        miss += 1
                        continue
                    mu, sd = p["mean"], p["std"]
                    # 选出该组全部位置
                    m = (cat_mat == c) & (region_mat == r) & (var_mat == v)
                    ii, jj = np.where(m)
                    for i, j in zip(ii, jj):
                        data[i, j, :, 4] = (data[i, j, :, 4] - mu) / sd
        if miss > 0:
            print(f"[apply_normalize_with_fitted] warning: {miss} groups missing stats; skipped.")

        return data

    def apply_denorm_with_fitted(self, data):
        """
        用 self.normalizer.transform_params 逆标准化 data[:,:,:,4]（就地修改）
        """
        cat_mat    = data[:, :, 0, 1].astype(int)
        region_mat = data[:, :, 0, 2].astype(int)
        var_mat    = data[:, :, 0, 3].astype(int)
        cat_ids    = np.unique(cat_mat).astype(int)
        region_ids = np.unique(region_mat).astype(int)
        var_ids    = np.unique(var_mat).astype(int)
        miss = 0
        for c in cat_ids:
            for r in region_ids:
                for v in var_ids:
                    key = f"{c}_{r}_{v}"
                    p = self.normalizer.transform_params.get(key)
                    if p is None:
                        miss += 1
                        continue
                    mu, sd = p["mean"], p["std"]
                    m = (cat_mat == c) & (region_mat == r) & (var_mat == v)
                    ii, jj = np.where(m)
                    for i, j in zip(ii, jj):
                        data[i, j, :, 4] = data[i, j, :, 4] * sd + mu
        if miss > 0:
            print(f"[apply_denorm_with_fitted] warning: {miss} groups missing stats; skipped.")
        return data

    def _save_inverse_npy(self, normalized_data, config, tag):
        base = os.path.splitext(self.feature_file)[0]
        default_path = f"{base}_inverse_{tag}.npy"
        out_path = getattr(config, "inverse_npy_path", None) or default_path
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        inv = self.apply_denorm_with_fitted(normalized_data.copy())
        np.save(out_path, inv)
        print(f"[inverse_npy] saved {out_path}, shape={inv.shape}")

    # ====== (C) 一次性导出可微逆归一化 bundle ======
    def build_denorm_bundle(self, device):
        cat_codes    = np.asarray(self.cat_codes, dtype=int)
        region_codes = np.asarray(self.region_codes, dtype=int)
        var_codes    = np.asarray(self.var_codes, dtype=int)
        C = len(cat_codes)
        R, V = len(region_codes), len(var_codes)

        # 建在 CPU，更省显存；每次在 denorm_fn 内搬到目标卡
        means = torch.zeros(C, R, V, device=device, dtype=torch.float32)
        stds  = torch.ones( C, R, V, device=device, dtype=torch.float32)

        for ci, cc in enumerate(cat_codes):
            for i, rc in enumerate(region_codes):
                for j, vc in enumerate(var_codes):
                    p = self.normalizer.transform_params.get(f"{cc}_{rc}_{vc}")
                    if p is not None:
                        means[ci, i, j] = float(p["mean"])
                        stds[ci,  i, j] = float(p["std"])

        # 这些断言保留
        assert cat_codes.min()    == 0 and cat_codes.max()    == C-1
        assert region_codes.min() == 0 and region_codes.max() == R-1
        assert var_codes.min()    == 0 and var_codes.max()    == V-1


        def denorm_fn(x):  # x: [B,G,T,5]
            # 将统计量搬到 x 所在设备
            _means = means.to(x.device, non_blocking=True)
            _stds  = stds.to(x.device,  non_blocking=True)

            cat_idx    = x[:, :, 0, 1].long()
            region_idx = x[:, :, 0, 2].long()  # 直接当索引
            var_idx    = x[:, :, 0, 3].long()

            mu = _means[cat_idx, region_idx, var_idx].unsqueeze(-1)  # [B,G,1]
            sd = _stds [cat_idx, region_idx, var_idx].unsqueeze(-1)  # [B,G,1]

            y = x.clone()
            y[:, :, :, 4] = y[:, :, :, 4] * sd + mu
            return y
        print(f"[denorm_bundle] mean[min,max]=({float(means.min())},{float(means.max())}), "
            f"std[min,max]=({float(stds.min())},{float(stds.max())}); "
            f"mean==0 ratio={float((means==0).float().mean()):.3f}, std==1 ratio={float((stds==1).float().mean()):.3f}")
        # 评估阶段还要用 numpy 版统计，这里返回 CPU numpy 即可
        return denorm_fn, means.cpu().numpy(), stds.cpu().numpy()





class SPMDataset(Dataset):
    def __init__(self, clean_data, flag, config):

        if flag=="train":
            self.label = np.array(clean_data.train_data)  # (n_grup, group, T, C)
            self.length = len(self.label)
            print('train sample num:',len(self.label))
        elif flag=="val":
            self.label = np.array(clean_data.val_data)
            self.length = len(self.label)
            print('val sample num:',len(self.label))
        elif flag=="test":
            self.label = np.array(clean_data.test_data)
            self.length = len(self.label)
            print('test sample num:',len(self.label))
        elif flag=="gen":
            self.label = clean_data
            self.length = len(self.label)
            print('test sample num:',len(self.label))

        
    def __getitem__(self, index):
        
        label = self.label[index]
        return [label,index]

    def __len__(self):
        return self.length



if __name__ == '__main__':
    pass
