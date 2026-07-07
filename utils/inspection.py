# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, Tuple, List, Optional
from tqdm.auto import tqdm

# ---------- 基础工具 ----------

def load_maps(region_csv: str, variable_csv: str):
    """读取编码表，返回 (reg_code2name, reg_name2code, var_code2name, var_name2code, cat_code2name, cat_name2code)"""
    def _load(path):
        df = pd.read_csv(path)
        df["Code"] = df["Code"].astype(int)
        df["Value"] = df["Value"].astype(str).str.strip()
        c2n = dict(zip(df["Code"], df["Value"]))
        n2c = dict(zip(df["Value"], df["Code"]))
        return c2n, n2c
    reg_c2n, reg_n2c = _load(region_csv)
    var_c2n, var_n2c = _load(variable_csv)

    return reg_c2n, reg_n2c, var_c2n, var_n2c

def load_baselines(variable_baseline_csv: str, region_baseline_csv: str):
    """加载阈值基线，并建查找 dict（与之前 CSV 版本一致）"""
    vb = pd.read_csv(variable_baseline_csv)
    rb = pd.read_csv(region_baseline_csv)
    VAR_REL_ERR = {
        (str(r.variable_name), str(r.region_name), int(r.timestep)): float(r.mean_relative_error)
        for r in vb.itertuples(index=False)
    }
    REG_REL_ERR = {
        (str(r.variable_name), int(r.timestep)): float(r.mean_relative_error)
        for r in rb.itertuples(index=False)
    }
    return VAR_REL_ERR, REG_REL_ERR


# ---------- R 对齐的“软放行”比较与诊断 ----------

def _finite(x) -> bool:
    return isinstance(x, (int, float, np.floating)) and np.isfinite(x)

def between_open(val: float, lo: float, hi: float, diag: Counter) -> bool:
    if not (_finite(val) and _finite(lo) and _finite(hi)):
        diag["nan_in_compare"] += 1
        return True
    return (val > lo) and (val < hi)

def safe_lt(val: float, bound: float, diag: Counter) -> bool:
    if not (_finite(val) and _finite(bound)):
        diag["nan_in_compare"] += 1
        return True
    return val < bound

def safe_gt(val: float, bound: float, diag: Counter) -> bool:
    if not (_finite(val) and _finite(bound)):
        diag["nan_in_compare"] += 1
        return True
    return val > bound

def ratio_between(v2020: float, v2010: float, lo: float, hi: float, diag: Counter) -> bool:
    if not (_finite(v2020) and _finite(v2010) and _finite(lo) and _finite(hi)) or v2010 == 0:
        diag["nan_in_compare"] += 1
        return True
    r = (v2020 - v2010) / v2010
    return (r > lo) and (r < hi)

def diff_gt(a: float, b: float, thr: float, diag: Counter) -> bool:
    if not (_finite(a) and _finite(b) and _finite(thr)):
        diag["nan_in_compare"] += 1
        return True
    return (a - b) > thr

def sum_strict(vals: List[float], diag: Counter) -> float:
    if any((not _finite(v)) for v in vals):
        diag["sumcheck_any_part_nan"] += sum(1 for v in vals if not _finite(v))
        return np.nan
    return float(sum(vals))

# ---------- 单样本访问器 ----------

class SampleAccessor:
    """
    针对单个样本 X: [G, T, 5] 的快速取数器
    约定：feature 维顺序 = [model, category, region, variable, value]
    G=变量*地区组合数, T=10(2010..2100 步长10)
    """
    def __init__(self, sample_gtk5: np.ndarray):
        assert sample_gtk5.ndim == 3 and sample_gtk5.shape[2] >= 5
        self.X = sample_gtk5
        self.G, self.T, _ = self.X.shape
        self.reg_codes = self.X[:, 0, 2].astype(int)  # 每组常数
        self.var_codes = self.X[:, 0, 3].astype(int)
        # 建立 (reg_code, var_code) -> 第一行索引
        pairs = list(zip(self.reg_codes.tolist(), self.var_codes.tolist()))
        self.pair2idx = {}
        for i, p in enumerate(pairs):
            if p not in self.pair2idx:
                self.pair2idx[p] = i
        # 年份索引
        self.year_to_idx = {2010 + 10*k: k for k in range(self.T)}

    def get_ts_by_code(self, reg_code: int, var_code: int) -> Optional[np.ndarray]:
        idx = self.pair2idx.get((reg_code, var_code))
        if idx is None:
            return None
        return self.X[idx, :, 4]  # [T] 值通道

    def get_val_by_code(self, reg_code: int, var_code: int, year: int) -> float:
        ts = self.get_ts_by_code(reg_code, var_code)
        if ts is None:
            return np.nan
        k = self.year_to_idx.get(year, None)
        if k is None or k >= ts.shape[0]:
            return np.nan
        try:
            return float(ts[k])
        except Exception:
            return np.nan


# ---------- 条件检查（与 CSV 版本一致的 1–11,32,33,54,55,76,77,98,99,120,121,122–130） ----------

def check_one_sample(
    sample_gtk5: np.ndarray,
    reg_name2code: Dict[str,int],
    var_name2code: Dict[str,int],
    VAR_REL_ERR: Dict[Tuple[str,str,int], float],
    REG_REL_ERR: Dict[Tuple[str,int], float],
) -> Tuple[bool, Counter, Dict[str, bool]]:
    A = SampleAccessor(sample_gtk5)
    diag = Counter()

    def g(var_name: str, reg_name: str, year: int) -> float:
        vc = var_name2code.get(var_name, None)
        rc = reg_name2code.get(reg_name, None)
        if vc is None or rc is None:
            diag["name_to_code_missing"] += 1
            return np.nan
        return A.get_val_by_code(rc, vc, year)

    def tol_var(var_name: str, reg_name: str, year: int):
        t = VAR_REL_ERR.get((var_name, reg_name, year))
        if t is None:
            diag["missing_threshold"] += 1
        return t

    def tol_world(var_name: str, year: int):
        t = REG_REL_ERR.get((var_name, year))
        if t is None:
            diag["missing_threshold"] += 1
        return t

    # ---- World范围 ----
    c1  = between_open(g("Emissions|CO2", "World", 2020), 26550.6, 61951.5, diag)
    c2  = between_open(g("Emissions|CO2|Energy and Industrial Processes", "World", 2020), 30116.8, 45175.2, diag)

    eip20 = g("Emissions|CO2|Energy and Industrial Processes", "World", 2020)
    eip10 = g("Emissions|CO2|Energy and Industrial Processes", "World", 2010)
    c3  = ratio_between(eip20, eip10, 0.0, 0.5, diag)

    c4  = safe_lt(g("Carbon Sequestration|CCS|Fossil", "World", 2020), 250, diag) and safe_gt(g("Carbon Sequestration|CCS|Fossil", "World", 2020), 0, diag)
    c5  = between_open(g("Primary Energy", "World", 2020), 462.4, 693.6, diag)
    c6  = between_open(g("Secondary Energy|Electricity|Nuclear", "World", 2020), 6.84, 12.70, diag)

    solar20 = g("Secondary Energy|Electricity|Solar", "World", 2020)
    wind20  = g("Secondary Energy|Electricity|Wind",  "World", 2020)
    c7  = between_open(sum_strict([solar20, wind20], diag), 4.26, 12.77, diag)

    c8  = (safe_gt(g("Emissions|CO2", "World", 2010), 0, diag) and
           safe_gt(g("Emissions|CO2", "World", 2020), 0, diag) and
           safe_gt(g("Emissions|CO2", "World", 2030), 0, diag))

    c9  = safe_lt(g("Carbon Sequestration|CCS|Fossil", "World", 2030), 2000, diag)
    c10 = safe_lt(g("Secondary Energy|Electricity|Nuclear", "World", 2030), 20, diag)

    ec2_2020 = g("Emissions|CO2", "World", 2020)
    eip_2020 = g("Emissions|CO2|Energy and Industrial Processes", "World", 2020)
    c11 = diff_gt(ec2_2020, eip_2020, 500, diag)

    c32 = between_open(g("Emissions|CO2|Energy and Industrial Processes", "World", 2010), 24982.81456, 37474.22184, diag)
    c33 = between_open(g("Emissions|CO2|Energy and Industrial Processes", "World", 2020), 26593.35104, 39890.02656, diag)

    c54 = between_open(g("Final Energy", "World", 2010), 293.51208, 440.26812, diag)
    c55 = between_open(g("Final Energy", "World", 2020), 316.97664, 475.46496, diag)

    c76 = between_open(g("Primary Energy|Coal", "World", 2010), 132.02544, 198.03816, diag)
    c77 = between_open(g("Primary Energy|Coal", "World", 2020), 135.26584, 202.89876, diag)

    c98 = between_open(g("Primary Energy|Gas", "World", 2010), 85.58064, 128.37096, diag)
    c99 = between_open(g("Primary Energy|Gas", "World", 2020), 102.66208, 153.99312, diag)

    c120 = between_open(g("Primary Energy|Oil", "World", 2010), 136.81384, 205.22076, diag)
    c121 = between_open(g("Primary Energy|Oil", "World", 2020), 136.59936, 204.89904, diag)

    # ---- 122–128：分解一致性（按各区域、2030–2100）----
    YEARS = list(range(2030, 2101, 10))
    regions = [name for name in reg_name2code.keys()]  # 包含 World 与 R10*
    def sumcheck(var_total: str, parts: List[str]) -> bool:
        for r in regions:
            for y in YEARS:
                total = g(var_total, r, y)
                part_vals = [g(p, r, y) for p in parts]
                part_sum = sum_strict(part_vals, diag)
                tol = tol_var(var_total, r, y)
                if tol is None:
                    continue  # 无阈值 → 放行
                if not (_finite(total) and _finite(part_sum) and _finite(tol)) or total == 0:
                    diag["nan_in_compare"] += 1
                    continue
                if abs(total - part_sum) / abs(total) > tol * 0.8:
                    return False
        return True

    c122 = sumcheck("Emissions|CO2", [
        "Emissions|CO2|Energy and Industrial Processes",
        "Emissions|CO2|Other",
        "Emissions|CO2|AFOLU",
    ])
    c123 = sumcheck("Secondary Energy|Electricity", [
        "Secondary Energy|Electricity|Wind","Secondary Energy|Electricity|Solar","Secondary Energy|Electricity|Oil",
        "Secondary Energy|Electricity|Nuclear","Secondary Energy|Electricity|Hydro","Secondary Energy|Electricity|Geothermal",
        "Secondary Energy|Electricity|Gas","Secondary Energy|Electricity|Coal","Secondary Energy|Electricity|Biomass",
    ])
    c124 = sumcheck("Primary Energy", [
        "Primary Energy|Wind","Primary Energy|Solar","Primary Energy|Oil","Primary Energy|Nuclear",
        "Primary Energy|Hydro","Primary Energy|Geothermal","Primary Energy|Gas","Primary Energy|Coal","Primary Energy|Biomass",
    ])
    c125 = sumcheck("Final Energy|Transportation", [
        "Final Energy|Transportation|Liquids|Oil","Final Energy|Transportation|Hydrogen",
        "Final Energy|Transportation|Gases","Final Energy|Transportation|Electricity",
    ])
    c126 = sumcheck("Final Energy|Residential and Commercial", [
        "Final Energy|Residential and Commercial|Solids","Final Energy|Residential and Commercial|Liquids",
        "Final Energy|Residential and Commercial|Gases","Final Energy|Residential and Commercial|Electricity",
    ])
    c127 = sumcheck("Final Energy|Industry", [
        "Final Energy|Industry|Solids","Final Energy|Industry|Liquids","Final Energy|Industry|Gases",
        "Final Energy|Industry|Electricity","Final Energy|Industry|Hydrogen",
    ])
    c128 = sumcheck("Carbon Sequestration|CCS", [
        "Carbon Sequestration|CCS|Fossil","Carbon Sequestration|CCS|Biomass","Carbon Sequestration|CCS|Industrial Processes",
    ])

    # ---- 129–130：World = Σ Regions（2030–2100）----
    def world_equals_sum_regions(var_list: List[str]) -> bool:
        regs_no_world = [r for r in regions if r != "World"]
        for v in var_list:
            for y in YEARS:
                world = g(v, "World", y)
                region_vals = [g(v, r, y) for r in regs_no_world]
                # 允许缺失区域：按 nansum，加诊断
                nan_cnt = sum(1 for vv in region_vals if not _finite(vv))
                if nan_cnt > 0:
                    diag["world_sum_used_nansum"] += nan_cnt
                region_sum = float(np.nansum(region_vals))
                tol = tol_world(v, y)
                if tol is None:
                    continue
                if not (_finite(world) and _finite(region_sum) and _finite(tol)) or world == 0:
                    diag["nan_in_compare"] += 1
                    continue
                if abs(world - region_sum) / abs(world) > tol * 0.8:
                    return False
        return True

    all_vars = list(var_name2code.keys())
    vars_co2   = sorted([v for v in all_vars if "CO2" in v])
    vars_noco2 = sorted([v for v in all_vars if "CO2" not in v])

    c129 = world_equals_sum_regions(vars_co2)
    c130 = world_equals_sum_regions(vars_noco2)

    condition_results = {
        "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5, "c6": c6, "c7": c7,
        "c8": c8, "c9": c9, "c10": c10, "c11": c11,
        "c32": c32, "c33": c33, "c54": c54, "c55": c55,
        "c76": c76, "c77": c77, "c98": c98, "c99": c99, "c120": c120, "c121": c121,
        "c122": c122, "c123": c123, "c124": c124, "c125": c125, "c126": c126,
        "c127": c127, "c128": c128, "c129": c129, "c130": c130,
    }
    ok = all(condition_results.values())
    return ok, diag, condition_results


def vet_on_tensor(
    samples: np.ndarray,                 # [N, G, T, 5]
    region_csv: str,
    variable_csv: str,
    variable_baseline_csv: str,
    region_baseline_csv: str,
    category_csv: Optional[str] = None,
    show_progress: bool = True,
):
    """
    在原始 4D 张量上直接做校验。返回：
      passed_mask: [N] bool
      diag_total : Counter 诊断汇总
    """
    reg_c2n, reg_n2c, var_c2n, var_n2c = load_maps(region_csv, variable_csv, category_csv)
    VAR_REL_ERR, REG_REL_ERR = load_baselines(variable_baseline_csv, region_baseline_csv)

    N = samples.shape[0]
    passed = np.zeros(N, dtype=bool)
    agg = Counter()

    it = range(N)
    if show_progress:
        it = tqdm(it, desc="Vetting tensor samples", unit="sample", leave=False)

    for i in it:
        ok, diag, _ = check_one_sample(samples[i], reg_n2c, var_n2c, VAR_REL_ERR, REG_REL_ERR)
        passed[i] = bool(ok)
        agg.update(diag)

    return passed, agg
