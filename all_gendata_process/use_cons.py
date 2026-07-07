from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import shutil
import re
import cvxpy as cp
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from collections import defaultdict


from utils.cons import process_memmap_in_batches

# ========= 数据加载 =========
def load_map(csv_path):
    df = pd.read_csv(csv_path)
    df['Code'] = df['Code'].astype(int)
    return dict(zip(df['Code'], df['Value']))
pre_path = "/home/sld/virtual-ZXD/co2-0108/output/forecast/最终生成_250.npy"
print('加载数据...')
gt = np.load("data/dataset/SPM/spm.npz")['padded']
mask = np.load("data/dataset/SPM/spm.npz")['mask']
pre = np.load(pre_path, mmap_mode='r')
shape = pre.shape
print(pre.shape, gt.shape)

model_map  = load_map("data/dataset/SPM/变量编码对应表/dim_Model_encoding.csv")
cluster_map  = load_map("data/dataset/SPM/变量编码对应表/dim_Category_encoding.csv")
region_map = load_map("data/dataset/SPM/变量编码对应表/dim_Region_encoding.csv")
var_map    = load_map("data/dataset/SPM/变量编码对应表/dim_Variable_encoding.csv")






# 自动推断区域与变量数量
num_regions = 11
num_variables = 48
print(f"数据信息: {num_regions} 个地区, {num_variables} 个变量")

# 输出路径（memmap 写出）
out_path = os.path.splitext(pre_path)[0] + ".corrected.npy"
print(f"输出路径: {out_path}")
# 执行后处理：可调整 batch_size 和 parallel_workers
process_memmap_in_batches(
    pre_path=pre_path,
    out_path=out_path,
    num_regions=num_regions,
    num_variables=num_variables,
    batch_size=32,               # 每批 2 个样本（可根据内存/CPU调节）
    parallel_workers=12,         # 并行 worker 数（None 或 1 表示禁用并行）
    solver="OSQP",
    solver_opts={"eps_abs": 1e-6, "eps_rel": 1e-6, "max_iter": 20000},
    show_progress=True,
)

