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
import sys 
sys.path.append("..") 
from utils.save_csv import SavePreToCSV

# ========= 数据加载 =========
def load_map(csv_path):
    df = pd.read_csv(csv_path)
    df['Code'] = df['Code'].astype(int)
    return dict(zip(df['Code'], df['Value']))
pre_path = "/home/sld/virtual-ZXD/co2-0108/output/forecast/最终生成_250.npy"
print('加载数据...')


model_map  = load_map("../data/dataset/SPM/变量编码对应表/dim_Model_encoding.csv")
cluster_map  = load_map("../data/dataset/SPM/变量编码对应表/dim_Category_encoding.csv")
region_map = load_map("../data/dataset/SPM/变量编码对应表/dim_Region_encoding.csv")
var_map    = load_map("../data/dataset/SPM/变量编码对应表/dim_Variable_encoding.csv")


# -*- coding: utf-8 -*-
# 读取 shape 和映射后，调用 SavePreToCSV 进行并行保存与压缩

shape = np.load(pre_path, mmap_mode='r').shape

saver = SavePreToCSV(
    pre_path=pre_path,
    shape=shape,
    cluster_map=cluster_map,
    region_map=region_map,
    var_map=var_map,
    save_dir="../output/forecast/pre_csv",
    n_workers=12,       # 视 CPU/磁盘情况调整
    keep_csv=True,    # 压缩后删除中间 CSV 目录
    base_year=2010,
    year_step=10,
    compresslevel=6
)
saver.save()