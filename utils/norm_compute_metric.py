import numpy as np


class AdvancedNormalizer:
    def __init__(self):
        self.transform_params = {}

    def fit_transform(self, data, group_key):
        data = np.asarray(data)
        lower_bound = np.percentile(data, 1)
        upper_bound = np.percentile(data, 99)
        clipped = np.clip(data, lower_bound, upper_bound)
        dmin, dmax = clipped.min(), clipped.max()
        if dmax == dmin:  # 防止除零
            dmax = dmin + 1.0
        self.transform_params[group_key] = {'min': dmin, 'max': dmax}
        return (data - dmin) / (dmax - dmin)

    def transform(self, data, group_key):
        params = self.transform_params.get(group_key)
        if params is None:
            raise ValueError(f'Group {group_key} not found.')
        dmin, dmax = params['min'], params['max']
        return (np.asarray(data) - dmin) / (dmax - dmin)
    



class Norm_data():
    def __init__(self, pre, gt, mask):
        self.pre = pre
        self.gt = gt
        self.mask = mask
        self.norm_data(pre, gt, mask)
    # ===== 标准化流程 =====
    def norm_data(self, pre, gt, mask):
        normalizer = AdvancedNormalizer()
        gt_norm = gt.copy()
        pre_norm = pre.copy()
        mask = mask.astype(bool)

        region_ids = np.unique(gt[:, :, 0, 2])
        var_ids    = np.unique(gt[:, :, 0, 3])

        for region_id in region_ids:
            for variable_id in var_ids:
                key_mask_gt  = (gt[:, :, 0, 2] == region_id) & (gt[:, :, 0, 3] == variable_id)
                key_mask_pre = (pre[:, :, 0, 2] == region_id) & (pre[:, :, 0, 3] == variable_id)

                
                valid_mask = key_mask_gt & mask
                if not np.any(valid_mask):
                    continue

                # 一次性取有效数据
                valid_data = gt[valid_mask, :, 4]
                group_key = f"{region_id}_{variable_id}"
                normalizer.fit_transform(valid_data, group_key)

                # 批量标准化GT
                gt_norm[key_mask_gt, :, 4] = normalizer.transform(gt[key_mask_gt, :, 4], group_key)
                # 批量标准化PRE
                pre_norm[key_mask_pre, :, 4] = normalizer.transform(pre[key_mask_pre, :, 4], group_key)

        print("✅ pre_norm、gt_norm 生成完成")














def jensen_shannon(p, q, n_bins=30, eps=1e-9):
    """计算两个一维数组分箱后的JSD"""
    min_val = min(p.min(), q.min())
    max_val = max(p.max(), q.max())
    
    p_hist, _ = np.histogram(p, bins=n_bins, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(q, bins=n_bins, range=(min_val, max_val), density=True)
    
    p_hist += eps
    q_hist += eps
    
    m = 0.5 * (p_hist + q_hist)
    return 0.5 * (entropy(p_hist, m) + entropy(q_hist, m))

def compute_mmd(X, Y, gamma=0.5):
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        return np.nan
    XX = cdist(X, X, 'sqeuclidean')
    YY = cdist(Y, Y, 'sqeuclidean')
    XY = cdist(X, Y, 'sqeuclidean')
    K_XX = np.exp(-gamma * XX)
    K_YY = np.exp(-gamma * YY)
    K_XY = np.exp(-gamma * XY)
    mean_K_XX = K_XX[~np.eye(n, dtype=bool)].mean()
    mean_K_YY = K_YY[~np.eye(m, dtype=bool)].mean()
    mean_K_XY = K_XY.mean()
    return mean_K_XX + mean_K_YY - 2 * mean_K_XY

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return 100 * np.mean(numerator / (denominator + 1e-3))

def kl_divergence_binned(p, q, n_bins=30, eps=1e-9):
    """计算两个一维数组分箱后的KL散度"""
    min_val = min(p.min(), q.min())
    max_val = max(p.max(), q.max())
    p_hist, _ = np.histogram(p, bins=n_bins, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(q, bins=n_bins, range=(min_val, max_val), density=True)
    p_hist += eps
    q_hist += eps
    return entropy(p_hist, q_hist)


def evaluate_by_merging(pre_norm, gt_norm, mask):
    """
    使用 pandas.merge 高效、安全地对齐和评估 gt 和 pre。
    同时计算平均保真度指标和分布相似性指标。
    """
    B, G, T, Cdim = gt_norm.shape

    # 1. 处理 GT 数据
    valid_idx = mask.reshape(-1) == 1
    gt_flat_data = gt_norm.reshape(-1, T, Cdim)[valid_idx]
    gt_rows = []
    for i in range(gt_flat_data.shape[0]):
        row = gt_flat_data[i, 0, 1:4].tolist()
        row.append(gt_flat_data[i, :, 4])
        gt_rows.append(row)
    df_gt = pd.DataFrame(gt_rows, columns=['category', 'region', 'variable', 'gt_series'])

    # 2. 处理 Pre 数据
    pre_flat_data = pre_norm.reshape(-1, T, Cdim)
    pre_rows = []
    for i in range(pre_flat_data.shape[0]):
        row = pre_flat_data[i, 0, 1:4].tolist()
        row.append(pre_flat_data[i, :, 4])
        pre_rows.append(row)
    df_pre = pd.DataFrame(pre_rows, columns=['category', 'region', 'variable', 'pre_series'])

    # 3. 分组和合并
    df_gt_grouped = df_gt.groupby(['category', 'region', 'variable'])['gt_series'].apply(list).reset_index()
    df_pre_grouped = df_pre.groupby(['category', 'region', 'variable'])['pre_series'].apply(list).reset_index()
    df_merged = pd.merge(df_gt_grouped, df_pre_grouped, on=['category', 'region', 'variable'], how='outer')

    # 4. 计算指标
    results = {}
    metric_cols = ["MAE", "RMSE", "SMAPE", "MMD", "JSD", "KS", "Wasserstein", 
                   "KL_Div", "Var_Diff", "Skew_Diff"]
    metrics_mean = {m: [] for m in metric_cols}

    print("开始计算指标...")
    for _, row in tqdm(df_merged.iterrows(), total=len(df_merged)):
        code = (row['category'], row['region'], row['variable'])
        
        if not isinstance(row['gt_series'], list) or not isinstance(row['pre_series'], list):
            continue

        gt_arr = np.vstack(row['gt_series'])
        pre_arr = np.vstack(row['pre_series'])

        # 用于评估“平均轨迹”
        gt_avg = np.mean(gt_arr, axis=0)
        pre_avg = np.mean(pre_arr, axis=0)
        
        # 用于评估“完整分布”
        gt_flat = gt_arr.flatten()
        pre_flat = pre_arr.flatten()

        res = {}
        # --- 平均保真度指标 (在 avg 上计算) ---
        res['MAE'] = mean_absolute_error(gt_avg, pre_avg)
        res['RMSE'] = mean_squared_error(gt_avg, pre_avg, squared=False)
        res['SMAPE'] = smape(gt_avg, pre_avg)

        # --- 分布相似性指标 (在 flat/arr 上计算) ---
        res['MMD'] = compute_mmd(pre_arr, gt_arr, gamma=0.5)
        res['KS'] = ks_2samp(pre_flat, gt_flat).statistic
        
        try:
            # 这些指标对输入更敏感，放入 try-except
            res['JSD'] = jensen_shannon(pre_flat, gt_flat)
            res['Wasserstein'] = wasserstein_distance(pre_flat, gt_flat)
            res['KL_Div'] = kl_divergence_binned(pre_flat, gt_flat)

            res['Var_Diff'] = np.abs(np.var(pre_flat) - np.var(gt_flat))
            res['Skew_Diff'] = np.abs(skew(pre_flat) - skew(gt_flat))
        except (ValueError, IndexError) as e: # 捕获可能因数据不足或恒定值引发的错误
            for k in ['JSD', 'Wasserstein', 'KL_Div', 'Var_Diff', 'Skew_Diff']:
                res[k] = np.nan
        
        results[code] = res
        for m, v in res.items():
            if not np.isnan(v):
                metrics_mean[m].append(v)

    # ------ 输出总体均值 ------
    print("\n=== Group-Averaged Metrics (valid only) ===")
    for m, vals in metrics_mean.items():
        if vals:
            print(f"{m}: {np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}")
        else:
            print(f"{m}: N/A")
    print(f"有效分组数 = {len(results)}")

    return results

# 调用新的评估函数
results = evaluate_by_merging(pre_norm, gt_norm, mask)










# ========= 问题3：分析和排序各项指标 =========

# 1. 将 results 字典转换为 DataFrame
df_metrics = pd.DataFrame.from_dict(results, orient='index')

# 2. 将多级索引转换为列并重命名
df_metrics.reset_index(inplace=True)
df_metrics.rename(columns={
    'level_0': 'Category_Code',
    'level_1': 'Region_Code',
    'level_2': 'Variable_Code'
}, inplace=True)

# 3. 使用之前加载的 map 进行解码，让结果更具可读性
df_metrics['Category'] = df_metrics['Category_Code'].map(cluster_map)
df_metrics['Region'] = df_metrics['Region_Code'].map(region_map)
df_metrics['Variable'] = df_metrics['Variable_Code'].map(var_map)

# 4. 定义并重排我们关心的列
all_metrics = ['MAE', 'RMSE', 'SMAPE', 'MMD', 'JSD', 'KS', 'Wasserstein', 
               'KL_Div', 'Var_Diff', 'Skew_Diff']
df_metrics = df_metrics[['Category', 'Region', 'Variable'] + all_metrics]

print("\n\n" + "="*20 + " 指标分析报告 " + "="*20)

# 保存到 CSV 文件以便更详细地分析
output_csv_path = "evaluation_metrics_by_group.csv"
df_metrics.to_csv(output_csv_path, index=False)
print(f"\n\n✅ 详细指标已保存到: {output_csv_path}")



# 6. 按地区聚合指标
print("\n\n" + "="*20 + " 按地区聚合指标 (均值) " + "="*20)
df_region_metrics = df_metrics.groupby('Region')[all_metrics].mean()

print("\n--- 按 SMAPE 排序 (从差到好) ---")
print(df_region_metrics.sort_values(by='SMAPE', ascending=False).to_string())

print("\n--- 按 MMD 排序 (从差到好) ---")
print(df_region_metrics.sort_values(by='MMD', ascending=False).to_string())




# 7. 按变量聚合指标
print("\n\n" + "="*20 + " 按变量聚合指标 (均值) " + "="*20)
df_variable_metrics = df_metrics.groupby('Variable')[all_metrics].mean()

print("\n--- 按 SMAPE 排序 (表现最差的 15 个变量) ---")
print(df_variable_metrics.sort_values(by='SMAPE', ascending=False).head(15).to_string())

print("\n--- 按 MMD 排序 (分布差异最大的 15 个变量) ---")
print(df_variable_metrics.sort_values(by='MMD', ascending=False).head(15).to_string())




# 8. 保存聚合结果
region_metrics_csv_path = "evaluation_metrics_by_region.csv"
variable_metrics_csv_path = "evaluation_metrics_by_variable.csv"

df_region_metrics.to_csv(region_metrics_csv_path)
df_variable_metrics.to_csv(variable_metrics_csv_path)

print(f"\n\n✅ 地区聚合指标已保存到: {region_metrics_csv_path}")
print(f"✅ 变量聚合指标已保存到: {variable_metrics_csv_path}")