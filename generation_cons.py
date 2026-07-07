# -*- coding: utf-8 -*-
import os, sys
import torch
import argparse
import numpy as np
import torch.utils.data
import glob
import pandas as pd
import nni
import logging
import shutil
import json

from pprint import pprint
from easydict import EasyDict as edict
from timeit import default_timer as timer
from tqdm import tqdm
from utils.common_utils import dir_check, Logger, get_workspace, assert_rv_layout

from model_files.dataset import CleanDataset, SPMDataset
from model_files.spm_diff.model import DiffSPM

# 速度相关（注意：若禁用 cudnn，则 benchmark 设置不会生效）
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ===== 仅用于通过率统计 & 生成过滤 =====
from utils.inspection import load_maps, load_baselines, check_one_sample


def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_params():
    parser = argparse.ArgumentParser(description='Entry point of the code')

    # model
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--beta_end", type=float, default=0.1)
    parser.add_argument("--sample_steps", type=int, default=45)
    parser.add_argument("--n_blocks", type=int, default=2)

    # gen
    parser.add_argument('--n_samples', type=int, default=6000)   # 每个 batch 的采样数 S
    parser.add_argument("--sample_batch", type=int, default=8)   # DataLoader 的 batch_size（组数）
    parser.add_argument("--epoch", type=int, default=250)
    parser.add_argument("--is_gen", type=bool, default=True)

    # ==== constraints (train-time loss weights) ====
    parser.add_argument("--lambda_region", type=float, default=0.002)
    parser.add_argument("--lambda_var",    type=float, default=0.002)
    parser.add_argument("--lambda_nn",     type=float, default=0.004)

    # ==== soft guidance (sampling-time) ====
    parser.add_argument("--use_soft_guidance", type=int, default=1)
    parser.add_argument("--soft_guidance_schedule", type=str, default="exp",
                        choices=["linear","exp","sigmoid","cosine","step"])
    parser.add_argument("--soft_eta_min", type=float, default=0.0)
    parser.add_argument("--soft_eta_max", type=float, default=1e-3)

    # ==== ratio adaptation (train-time) ====
    parser.add_argument("--constraint_target_ratio", type=float, default=0.35)
    parser.add_argument("--constraint_ratio_smooth", type=float, default=0.70)
    parser.add_argument("--constraint_gain_min", type=float, default=0.5)
    parser.add_argument("--constraint_gain_max", type=float, default=4.0)
    parser.add_argument("--constraint_warmup_steps", type=int, default=50)

    # ==== nonneg & projection (post-process) ====
    parser.add_argument("--reverse_projection_scale", type=float, default=2)

    args, _ = parser.parse_known_args()
    return args


def default_config(data='SPM'):
    work_path = get_workspace()
    device_ids = [0,1,2,3]
    config = edict()
    device = torch.device(f"cuda:{device_ids[0]}")
    config.device = device
    config.device_ids = device_ids

    config.save_dir = "最后练"
    config.PATH_MOD = work_path + '/output/model/'
    config.PATH_LOG = work_path + '/output/log/'
    config.PATH_FORECAST = work_path + '/output/forecast/'

    # Data Config
    config.data = edict()
    config.data.name = data
    config.data.path = work_path + '/data/dataset/'
    config.data.feature_file = config.data.path + config.data.name + '/spm.npz'
    config.data.var_encoding = config.data.path + config.data.name + "/变量编码对应表/dim_Variable_encoding.csv"
    config.data.region_encoding = config.data.path + config.data.name + "/变量编码对应表/dim_Region_encoding.csv"

    config.variable_baseline = config.data.path + config.data.name + "/variable_baseline.csv"
    config.region_baseline   = config.data.path + config.data.name + "/region_baseline.csv"

    config.region_map = pd.read_csv(config.data.region_encoding)
    config.var_map    = pd.read_csv(config.data.var_encoding)

    # model config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.model = edict()
    config.model.T_h          = 10
    config.model.num_c        = 8
    config.model.num_region   = 11
    config.model.num_variable = 48
    config.model.device       = device

    # ---- training-loss weights ----
    config.model.lambda_region = params['lambda_region']
    config.model.lambda_var    = params['lambda_var']
    config.model.lambda_nn     = params['lambda_nn']

    # ---- soft guidance ----
    config.model.use_soft_guidance      = bool(params['use_soft_guidance'])
    config.model.soft_guidance_schedule = params['soft_guidance_schedule']
    config.model.soft_eta_min           = params['soft_eta_min']
    config.model.soft_eta_max           = params['soft_eta_max']

    # ---- ratio adaptation ----
    config.model.constraint_target_ratio = params['constraint_target_ratio']
    config.model.constraint_ratio_smooth = params['constraint_ratio_smooth']
    config.model.constraint_gain_min     = params['constraint_gain_min']
    config.model.constraint_gain_max     = params['constraint_gain_max']
    config.model.constraint_warmup_steps = params['constraint_warmup_steps']

    # ---- post-process ----
    config.model.reverse_projection_scale = params['reverse_projection_scale']

    # UGnet
    config.model.channel_multipliers = [1, 2]
    config.model.supports_len        = 2

    config.logger = Logger()

    os.makedirs(config.PATH_MOD, exist_ok=True)
    os.makedirs(config.PATH_LOG, exist_ok=True)
    os.makedirs(config.PATH_FORECAST, exist_ok=True)
    return config


def compute_vet_pass_rate_total(temp_dir: str) -> float:
    """
    汇总整个 temp 目录的 Vet 通过率（不依赖 GT）:
      - 每个 batch 写入 batch_*.meta.json: {"B": ..., "S": ...}
      - 总候选 = sum_over_batches(B*S)
      - 总通过 = sum_over_batches( np.load(batch_*.npy).shape[0] )
    返回: rate (float)，若无文件则返回 nan
    """
    meta_paths   = sorted(glob.glob(os.path.join(temp_dir, "batch_*.meta.json")))
    sample_paths = sorted(glob.glob(os.path.join(temp_dir, "batch_*.npy")))
    if len(meta_paths) == 0 or len(sample_paths) == 0:
        print("[Vet pass · total] 缺少 meta 或样本文件，无法统计。")
        return float('nan')

    # 将批次索引对齐
    def _idx(p):
        b = os.path.splitext(os.path.basename(p))[0]
        return int(b.split("_")[-1].split(".")[0])
    meta_map   = {_idx(p): p for p in meta_paths}
    sample_map = {_idx(p): p for p in sample_paths}
    keys = sorted(set(meta_map.keys()) & set(sample_map.keys()))
    if not keys:
        print("[Vet pass · total] meta 与样本批次不相交。")
        return float('nan')

    total, kept = 0, 0
    for k in keys:
        with open(meta_map[k], "r", encoding="utf-8") as f:
            meta = json.load(f)
        B = int(meta.get("B", 0))
        S = int(meta.get("S", 0))
        total += B * S

        arr = np.load(sample_map[k])
        kept += int(arr.shape[0])

    rate = (kept / total) if total > 0 else float('nan')
    print(f"[Vet pass · total] kept {kept} / {total} = {rate:.2%}")
    return rate


# ------------------------------------------------
# 仅负责：采样 + inspection 过滤 + 落盘
# ------------------------------------------------
def test_all(model, epoch, data_loader, config, reg_n2c, var_n2c, VAR_REL_ERR, REG_REL_ERR):
    os.makedirs(os.path.join(config.PATH_FORECAST, "temp2"), exist_ok=True)
    setup_seed(1024)
    model.eval()

    n_samples   = config.n_samples
    split       = 64
    num_batches = len(data_loader)
    total_units = num_batches * n_samples  # 进度条总单位 = 批次数 × 每批应采样 S 个
    use_inspection_filter = (
        reg_n2c is not None and var_n2c is not None and
        VAR_REL_ERR is not None and REG_REL_ERR is not None
    )

    with torch.no_grad():
        pbar = tqdm(total=total_units, desc="Generating", ncols=100, unit="smp")
        for i, batch in enumerate(data_loader):
            save_base   = os.path.join(config.PATH_FORECAST, f'temp2/batch_{i}')
            sample_file = save_base + ".npy"        # 仅保存过滤后的样本
            meta_file   = save_base + ".meta.json"  # 记录 B 与 S，便于汇总总通过率

            # 若已有过滤后的样本文件，视为该批已完成（断点续跑）
            if os.path.exists(sample_file) and os.path.exists(meta_file):
                pbar.update(n_samples)
                pbar.set_postfix(batch=f"{i+1}/{num_batches}", samples=f"{n_samples}/{n_samples}", skipped=True)
                continue

            time_start = timer()
            label, index = batch    # [B,G,T,5]
            if i == 0:
                assert_rv_layout(label, config.model.num_region, config.model.num_variable)
            label = label.clone().detach().to(torch.float32).to(config.device)
            B, G, T, _ = label.shape

            # === 采样（分块避免显存爆） ===
            parts = []
            done = 0
            num_splits = (n_samples + split - 1) // split
            for s in range(num_splits):
                ns = min(split, n_samples - s * split)
                xhat = model(label, ns)                  # [B,ns,G,T,1]
                parts.append(xhat.detach().cpu().numpy())
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pbar.update(ns); done += ns
                pbar.set_postfix(batch=f"{i+1}/{num_batches}", samples=f"{done}/{n_samples}")

            x_hat = np.concatenate(parts, axis=1)        # [B,S,G,T,1]

            # === 打包（与原口径一致）：[B*S,G,T,5] ===
            code = label[:, :, :, :4].unsqueeze(1).repeat(1, n_samples, 1, 1, 1)
            code_np = code.detach().cpu().numpy()
            packed = np.concatenate([code_np, x_hat], axis=-1).reshape(-1, x_hat.shape[2], config.model.T_h, 5)

            # === inspection 过滤（已禁用时保留全部样本） ===
            # if use_inspection_filter:
            keep_idx = []
            for k in range(packed.shape[0]):
                ok, _, _ = check_one_sample(packed[k], reg_n2c, var_n2c, VAR_REL_ERR, REG_REL_ERR)
                if ok:
                    keep_idx.append(k)
            packed_keep = packed[keep_idx] if keep_idx else np.empty((0,) + packed.shape[1:], dtype=packed.dtype)

            if packed.shape[0] > 0:
                pass_rate = 100.0 * (len(keep_idx) / packed.shape[0])
                print(f"[Gen-Filter/inspection] keep {len(keep_idx)}/{packed.shape[0]} ({pass_rate:.2f}%) in batch {i+1}")
            # else:
            #     packed_keep = packed

            # === 保存过滤结果（允许为空）===
            np.save(sample_file, packed_keep)

            # === 保存元信息（用于总通过率统计）===
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump({"B": int(B), "S": int(n_samples)}, f)

        pbar.close()

    torch.cuda.empty_cache()
    return


def main(params: dict):
    setup_seed(2022)
    torch.set_num_threads(2)
    config = default_config()
    config.sample_batch = params['sample_batch']
    config.is_gen = params['is_gen']

    # model
    config.model.n_blocks     = params['n_blocks']
    config.model.N            = params['N']
    config.T_h                = config.model.T_h
    config.model.sample_steps = params['sample_steps']
    config.model.d_h          = params['hidden_size']
    config.model.beta_end     = params['beta_end']
    config.n_samples          = params['n_samples']
    config.epoch              = params['epoch']

    config.trial_name = '+'.join([f"{v}" for k, v in params.items()])
    config.log_path   = f"{config.PATH_LOG}/{config.trial_name}.log"

    dir_check(config.log_path)
    config.logger.open(config.log_path, mode="w")
    config.logger.write(config.__str__()+'\n', is_terminal=False)

    # VET 资源（inspection 过滤已禁用）
    # reg_n2c = var_n2c = VAR_REL_ERR = REG_REL_ERR = None
    reg_c2n, reg_n2c, var_c2n, var_n2c = load_maps(
        config.data.region_encoding,
        config.data.var_encoding
    )
    VAR_REL_ERR, REG_REL_ERR = load_baselines(
        config.variable_baseline,
        config.region_baseline
    )

    # 标准化资源（模型采样内部需要）
    clean_norm_data = CleanDataset(config)
    denorm_fn, means_t, stds_t = clean_norm_data.build_denorm_bundle(config.model.device)

    # 加载 ckpt
    save_dir   = config.save_dir
    model_name = f"epoch_{config.epoch}"
    # ckpt_list  = sorted(glob.glob(os.path.join(save_dir, f"{model_name}*.ckpt")))
    # assert len(ckpt_list) > 0, f"未找到 ckpt: {os.path.join(save_dir, model_name+'*.ckpt')}"
    # ckpt_path  = ckpt_list[-1]
    # print("load ckpt:", ckpt_path)
    ckpt_path = "最后练/best.ckpt"
    ckpt = torch.load(ckpt_path, map_location=config.device)

    model = DiffSPM(config.model).to(config.device)
    model.attach_denorm(denorm_fn, means_t, stds_t)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print("[load_state_dict] missing:", missing)
        print("[load_state_dict] unexpected:", unexpected)
    model.eval()
    model = torch.nn.DataParallel(model, device_ids=config.device_ids)

    # —— 构造“gen”数据（占位示例；实际可替换为你的真实 gen loader）——
    num_models   = 128
    num_c        = config.model.num_c
    num_regions  = config.model.num_region
    num_variables= config.model.num_variable
    groups       = num_models * num_c
    group_len    = num_regions * num_variables
    test_data    = np.zeros((groups, group_len, 10, 5), dtype=float)
    for m in range(num_models):
        for c in range(num_c):
            group_index = m * num_c + c
            for r in range(num_regions):
                for v in range(num_variables):
                    vr_idx = r * num_variables + v
                    test_data[group_index, vr_idx, :, 0] = 100   # model
                    test_data[group_index, vr_idx, :, 1] = c     # condition
                    test_data[group_index, vr_idx, :, 2] = r     # region
                    test_data[group_index, vr_idx, :, 3] = v     # variable
                    test_data[group_index, vr_idx, :, 4] = np.random.rand(10)
    print('test data shape:', test_data.shape)
    test_dataset = SPMDataset(test_data, "gen", config)
    test_loader  = torch.utils.data.DataLoader(test_dataset, config.sample_batch, shuffle=False, drop_last=False)

    # ========== 生成（仅采样 + 过滤 + 保存） ==========
    test_all(model, config.epoch, test_loader, config, reg_n2c, var_n2c, VAR_REL_ERR, REG_REL_ERR)

    # ========== 合并保存过滤后的样本 ==========
    temp_dir     = os.path.join(config.PATH_FORECAST, "temp2")
    sample_paths = sorted(glob.glob(os.path.join(temp_dir, "batch_*.npy")))

    all_samples = [np.load(f) for f in sample_paths] if len(sample_paths) > 0 else []
    if len(all_samples) > 0:
        all_samples = np.concatenate(all_samples, axis=0)
        samples_path = os.path.join(config.PATH_FORECAST, f'最终生成_{config.epoch}.npy')
        np.save(samples_path, all_samples)
        print(f"[DONE] saved full samples to {samples_path}")

    # ====== 统计总 Vet 通过率（与 GT 无关） ======
    _ = compute_vet_pass_rate_total(temp_dir)

    torch.cuda.empty_cache()
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    logger = logging.getLogger('training')
    print('GPU:', torch.cuda.current_device())
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
