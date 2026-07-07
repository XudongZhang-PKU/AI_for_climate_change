# -*- coding: utf-8 -*-
# 并行将 (n_samples, n_grp, seq_len, feat_dim) 的大数组按样本保存为 CSV，并压缩为 zip（带进度条）
# ✅ 支持多个 .npy 文件（虚拟拼接，不占内存）
# ✅ 新增：从 variables48.csv 读取单位映射，在导出的 CSV 中加入 unit 列

import os
import csv
import glob
import zipfile
import shutil
import bisect
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import numpy as np
import pandas as pd
from tqdm import tqdm

# 全局变量（子进程共享只读）
_PRES = None         # list[np.memmap]
_OFFSETS = None      # 样本数前缀和，用于定位文件
_CLUSTER_MAP = None
_REGION_MAP = None
_VAR_MAP = None
_UNIT_MAP = None     # variable(字符串) -> unit(字符串)，已做小写与去空格标准化
_BASE_YEAR = None
_YEAR_STEP = None

# 特征索引（按你当前布局：cat=1, reg=2, var=3, value=4）
_IDX_CAT = 1
_IDX_REG = 2
_IDX_VAR = 3
_IDX_VAL = 4


def _init_worker(pre_path, mmap_mode, cluster_map, region_map, var_map, unit_map, base_year, year_step):
    """每个子进程初始化：以内存映射方式打开所有数组 + 注入各类映射"""
    global _PRES, _OFFSETS
    global _CLUSTER_MAP, _REGION_MAP, _VAR_MAP, _UNIT_MAP
    global _BASE_YEAR, _YEAR_STEP

    # 支持多文件
    if isinstance(pre_path, (list, tuple)):
        _PRES = [np.load(p, mmap_mode=mmap_mode) for p in pre_path]
    else:
        _PRES = [np.load(pre_path, mmap_mode=mmap_mode)]
    _OFFSETS = np.cumsum([0] + [x.shape[0] for x in _PRES])

    _CLUSTER_MAP = cluster_map or {}
    _REGION_MAP  = region_map  or {}
    _VAR_MAP     = var_map     or {}

    # 单位映射：传入时已标准化为小写去空格
    _UNIT_MAP    = unit_map or {}

    _BASE_YEAR = int(base_year)
    _YEAR_STEP = int(year_step)


def _write_one_csv(sid, save_dir):
    """
    将一个样本写为 CSV，返回文件路径。
    支持多文件虚拟拼接；在 variable 后新增 unit 列。
    """
    # 定位该样本属于哪个文件
    idx = bisect.bisect_right(_OFFSETS, sid) - 1
    pre = _PRES[idx]
    local_sid = sid - _OFFSETS[idx]

    n_grp  = pre.shape[1]
    seq_len = pre.shape[2]
    out_path = os.path.join(save_dir, f"sample_{sid}.csv")

    years = [str(_BASE_YEAR + _YEAR_STEP * i) for i in range(seq_len)]
    # 新增 unit 列
    header = ["sample", "category", "region", "variable", "unit"] + years

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for g in range(n_grp):
            category = int(pre[local_sid, g, 0, _IDX_CAT])
            region   = int(pre[local_sid, g, 0, _IDX_REG])
            variable = int(pre[local_sid, g, 0, _IDX_VAR])
            ts       = pre[local_sid, g, :, _IDX_VAL].tolist()

            # 名称映射
            cat_name = _CLUSTER_MAP.get(category, category)
            reg_name = _REGION_MAP.get(region,   region)
            var_name = _VAR_MAP.get(variable,    variable)

            # 依据变量名（标准化为小写+去空格）查单位
            var_key = str(var_name).strip().lower()
            unit    = _UNIT_MAP.get(var_key, "")

            row = [sid, cat_name, reg_name, var_name, unit] + ts
            w.writerow(row)

    return out_path


class SavePreToCSV:
    """
    并行保存预测结果为 CSV，并压缩为 zip。
    - 支持多个 .npy 文件虚拟拼接（memmap，不占内存）
    - 新增 unit 映射：从 variables48.csv 读取 variable→unit，并在导出时写入 unit 列
    """

    def __init__(self,
                 pre_path,            # str 或 list[str]
                 shape,               # (n_samples, n_grp, seq_len, feat_dim)
                 cluster_map,
                 region_map,
                 var_map,
                 save_dir,
                 n_workers=None,
                 keep_csv=True,
                 base_year=2010,
                 year_step=10,
                 compresslevel=6,
                 unit_csv_path=None   # ← 新增：variables48.csv 的路径（包含 variable / unit 列）
                 ):
        self.pre_path = pre_path
        self.shape = tuple(shape)

        self.cluster_map = dict(cluster_map or {})
        self.region_map  = dict(region_map  or {})
        self.var_map     = dict(var_map     or {})

        self.save_dir = str(save_dir)
        self.n_workers = n_workers or min(8, os.cpu_count() or 1)
        self.keep_csv = bool(keep_csv)
        self.base_year = int(base_year)
        self.year_step = int(year_step)
        self.compresslevel = int(compresslevel)

        self.unit_csv_path = unit_csv_path  # 允许为 None

    def _load_unit_map(self):
        """
        读取 variables48.csv，构建 {variable(lower/strip): unit}。
        兼容列名：variable / Value；unit / Unit。
        """
        if not self.unit_csv_path:
            return {}

        df = pd.read_csv(self.unit_csv_path)
        # 容错列名
        cols = {c.lower(): c for c in df.columns}
        var_col  = cols.get("variable") or cols.get("value")
        unit_col = cols.get("unit")

        if not var_col or not unit_col:
            raise ValueError(
                f"[unit_csv_path 列名缺失] 需要包含 'variable'(或'Value') 与 'unit' 列，实际列：{list(df.columns)}"
            )

        unit_map = {}
        for v, u in zip(df[var_col].astype(str), df[unit_col].astype(str)):
            key = v.strip().lower()
            unit_map[key] = u.strip()
        return unit_map

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)

        # 统计总样本数（不加载入内存）
        if isinstance(self.pre_path, (list, tuple)):
            total_samples = sum(np.load(p, mmap_mode="r").shape[0] for p in self.pre_path)
        else:
            total_samples = np.load(self.pre_path, mmap_mode="r").shape[0]

        # 读取单位映射（一次在主进程读好，传给子进程）
        unit_map = self._load_unit_map()

        print(f"开始并行生成 CSV：总样本数={total_samples}，进程数={self.n_workers}")

        with ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=_init_worker,
            initargs=(self.pre_path, 'r',
                      self.cluster_map, self.region_map, self.var_map, unit_map,
                      self.base_year, self.year_step),
        ) as ex:
            for _ in tqdm(
                ex.map(_write_one_csv, range(total_samples), repeat(self.save_dir), chunksize=16),
                total=total_samples, desc="Saving CSVs", smoothing=0
            ):
                pass

        print(f"已保存 {total_samples} 个 CSV 到 {self.save_dir}/")

        zip_path = f"{self.save_dir}.zip"
        self._zip_with_progress(self.save_dir, zip_path, self.compresslevel)
        print(f"压缩完成: {zip_path}")

        if not self.keep_csv:
            shutil.rmtree(self.save_dir, ignore_errors=True)
            print("已删除中间 CSV 目录。")

    @staticmethod
    def _zip_with_progress(src_dir, zip_path, compresslevel=6):
        files = sorted(glob.glob(os.path.join(src_dir, "*.csv")))
        total_bytes = sum(os.path.getsize(p) for p in files)
        with zipfile.ZipFile(zip_path, mode="w",
                             compression=zipfile.ZIP_DEFLATED,
                             compresslevel=int(compresslevel)) as zf, \
             tqdm(total=max(total_bytes, 1), unit="B", unit_scale=True, desc="Zipping", smoothing=0) as pbar:
            progressed = 0
            for f in files:
                arcname = os.path.basename(f)
                zf.write(f, arcname=arcname)
                size = os.path.getsize(f)
                progressed += size
                pbar.update(size)
            if progressed == 0:
                pbar.update(1)