# -*- coding: utf-8 -*-
import os, sys, csv
import warnings
# 抑制 NNI 在非 HPO 场景下的 "Running trial code without runtime" 刷屏
warnings.filterwarnings("ignore", message=".*Running trial code without runtime.*", category=RuntimeWarning)
import torch
import argparse
import numpy as np
import torch.utils.data
from easydict import EasyDict as edict
from timeit import default_timer as timer
from pprint import  pprint
from utils.eval import evaluate_sampling_metrics, evaluating, append_result_row
from utils.common_utils import dir_check, to_device, get_workspace, unfold_dict, dict_merge, GpuId2CudaId, Logger
from model_files.dataset import CleanDataset, SPMDataset, oversample_by_multiplier
from model_files.spm_diff.model import DiffSPM
import pandas as pd
import shutil
def setup_seed(seed: int):
    """设置随机种子，保证实验可重复。"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LocalIndexDataset(torch.utils.data.Dataset):
    """Dataset wrapper returning local row indices aligned with the supplied mask."""
    def __init__(self, base_dataset, indices=None):
        self.base_dataset = base_dataset
        self.indices = None if indices is None else [int(i) for i in indices]

    def __len__(self):
        return len(self.base_dataset) if self.indices is None else len(self.indices)

    def __getitem__(self, local_index):
        source_index = local_index if self.indices is None else self.indices[local_index]
        label, _ = self.base_dataset[source_index]
        return label, local_index

def get_params():
    parser = argparse.ArgumentParser(description='pass')

    # model
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--beta_end", type=float, default=0.1)  
    parser.add_argument("--sample_steps", type=int, default=45)  # sample_steps
    parser.add_argument("--n_blocks", type=int, default=2)
    #测试config.model.n_blocks

    # eval
    parser.add_argument('--n_samples', type=int, default=64)

    # train
    parser.add_argument("--train_fraction", type=float, default=1.0,
                        help="Fraction of the training split retained before oversampling.")
    parser.add_argument("--is_train", type=bool, default=True) # train or evaluate
    parser.add_argument("--is_val", type=bool, default=True)
    parser.add_argument("--is_test", type=bool, default=False)
    parser.add_argument("--is_gen", type=bool, default=False)

    parser.add_argument("--nni", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--gen_batch_size", type=int, default=32)
    #generation


    # ==== constraints (train-time loss weights) ====
    parser.add_argument("--lambda_region", type=float, default=0.002)
    parser.add_argument("--lambda_var",    type=float, default=0.002)
    parser.add_argument("--lambda_nn",     type=float, default=0.004)

    # ==== soft guidance (sampling-time) ====
    parser.add_argument("--use_soft_guidance", type=int, default=1)  # 1=on, 0=off
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


    # ==== results io ====
    parser.add_argument("--results_csv", type=str, default="最后练.csv")
    parser.add_argument("--exp_tag", type=str, default="default")  # 方便区分一组实验
    parser.add_argument("--save_suffix", type=str, default="")
    parser.add_argument("--epoch", type=int, default=None)
    # 训练结束后是否导出 Fig.2 绘图数据：1=在保存 best 后写入 save_dir/fig2_data/（metrics_dist_full.csv、corr_*.npy 等）
    parser.add_argument("--export_fig2", type=int, default=1)
    # 全局随机种子（torch/numpy/random），保证实验可重复
    parser.add_argument("--seed", type=int, default=2026)
    # 划分 train/val/test 的随机种子，固定后多 run 可复现同一 test 集（消融对比时保持一致）
    parser.add_argument("--split_seed", type=int, default=2026)
    # 消融标识：控制约束/图，不再控制 denoiser 结构
    parser.add_argument("--ablation", type=str, default="Full",
                        choices=["Full", "NoGraph", "NoConstraint"])
    parser.add_argument("--denoiser", type=str, default="UGnet",
                        choices=["UGnet", "TinyUGnet", "UGnetNoTCN"])
    parser.add_argument("--no_cat_cond", type=int, default=0)
    parser.add_argument("--no_region_cond", type=int, default=0)
    parser.add_argument("--no_var_cond", type=int, default=0)
    parser.add_argument("--enable_constraint_loss", type=int, default=0)
    parser.add_argument("--constraint_weight", type=float, default=1e-4)
    parser.add_argument("--tiny_width_mult", type=float, default=0.5)
    parser.add_argument("--tiny_n_blocks", type=int, default=1)
    parser.add_argument("--tiny_n_resolutions", type=int, default=2)
    parser.add_argument("--tiny_num_heads", type=int, default=1)

    args, _ = parser.parse_known_args()
    return args



def default_config(data='SPM'):
    work_path =  get_workspace()
    config = edict()
    config.PATH_MOD = work_path + '/output/model/'
    config.PATH_LOG = work_path + '/output/log/'
    config.PATH_FORECAST = work_path + '/output/forecast/'

    # Data Config
    config.data = edict()
    config.data.name = data
    config.data.path = work_path + '/data/dataset/'
    config.data.feature_file = config.data.path + config.data.name + '/spm.npz'
    # config.data.spatial = config.data.path + config.data.name + '/ADJ/'
    config.data.var_encoding = config.data.path + config.data.name + "/变量编码对应表/dim_Variable_encoding.csv"
    config.data.region_encoding = config.data.path + config.data.name + "/变量编码对应表/dim_Region_encoding.csv"
    
    config.variable_baseline = config.data.path + config.data.name + "/variable_baseline.csv"
    config.region_baseline = config.data.path + config.data.name + "/region_baseline.csv"
    
    config.region_map = pd.read_csv(config.data.region_encoding)
    config.var_map = pd.read_csv(config.data.var_encoding)
    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    config.device = device

    # model config
    config.model = edict()
    config.model.T_h = 10
    config.model.num_c = 8
    config.model.num_region = 11
    config.model.num_variable = 48
    config.model.device = device

    # config for UGnet
    config.model.channel_multipliers = [1, 2]  # The list of channel numbers at each resolution.
    config.model.supports_len = 2

    # training config
    config.epoch = 300  # Number of max training epoch
    config.save_dir = "fig2_data_prepare"
    
    config.logger = Logger()


    # exist_ok=True 避免多任务并行时竞态导致 FileExistsError
    os.makedirs(config.PATH_MOD, exist_ok=True)
    os.makedirs(config.PATH_LOG, exist_ok=True)
    os.makedirs(config.PATH_FORECAST, exist_ok=True)
    return config



def main(params: dict):
    setup_seed(params.get("seed", 42))
    torch.set_num_threads(2)
    config = default_config()

    config.is_test = params['is_test']
    config.is_val = params['is_val']
    config.is_train = params['is_train']
    config.is_gen = params['is_gen']

    config.nni = params['nni']
    config.lr = params['lr']
    config.train_batch_size = params['train_batch_size']
    config.gen_batch_size = params['gen_batch_size']


    # model
    config.model.N = params['N']
    config.T_h = config.model.T_h
    config.model.sample_steps = params['sample_steps']
    config.model.d_h = params['hidden_size']
    config.model.beta_end = params['beta_end']
    config.model.n_blocks = params['n_blocks']
    config.n_samples = params['n_samples']
    config.train_fraction = float(params.get('train_fraction', 1.0))


        # ---- training-loss weights ----
    config.model.lambda_region = params['lambda_region']
    config.model.lambda_var    = params['lambda_var']
    config.model.lambda_nn     = params['lambda_nn']

    # ---- soft guidance ----
    config.model.use_soft_guidance       = bool(params['use_soft_guidance'])
    config.model.soft_guidance_schedule  = params['soft_guidance_schedule']
    config.model.soft_eta_min            = params['soft_eta_min']
    config.model.soft_eta_max            = params['soft_eta_max']

    # ---- ratio adaptation ----
    config.model.constraint_target_ratio  = params['constraint_target_ratio']
    config.model.constraint_ratio_smooth  = params['constraint_ratio_smooth']
    config.model.constraint_gain_min      = params['constraint_gain_min']
    config.model.constraint_gain_max      = params['constraint_gain_max']
    config.model.constraint_warmup_steps  = params['constraint_warmup_steps']

    # ---- post-process ----
    config.model.reverse_projection_scale = params['reverse_projection_scale']

    # ---- results io ----
    config.exp_tag     = params['exp_tag']
    config.export_fig2 = params['export_fig2']
    config.seed       = params.get('seed', 42)
    config.split_seed = params['split_seed']
    base_ablation = params['ablation']
    config.model.ablation_nograph = (base_ablation == "NoGraph")
    config.model.ablation_no_constraint = (base_ablation == "NoConstraint")
    config.model.denoiser = params["denoiser"]
    config.model.ablation_no_cat_cond = bool(params["no_cat_cond"])
    config.model.ablation_no_region_cond = bool(params["no_region_cond"])
    config.model.ablation_no_var_cond = bool(params["no_var_cond"])
    config.model.enable_constraint_loss = bool(params["enable_constraint_loss"])
    config.model.constraint_weight = float(params["constraint_weight"])
    config.model.tiny_width_mult = float(params["tiny_width_mult"])
    config.model.tiny_n_blocks = int(params["tiny_n_blocks"])
    config.model.tiny_n_resolutions = int(params["tiny_n_resolutions"])
    config.model.tiny_num_heads = int(params["tiny_num_heads"])
    config.save_suffix = str(params.get("save_suffix", "") or "")
    if params.get("epoch") is not None:
        config.epoch = int(params["epoch"])

    # 统一构造 ablation 标签，避免 Full 下不同开关落到同一目录
    ablation_parts = [base_ablation]
    if config.model.denoiser == "TinyUGnet":
        ablation_parts.append("TinyUGnet")
    elif config.model.denoiser == "UGnetNoTCN":
        ablation_parts.append("UGnetNoTCN")
    if config.model.ablation_no_cat_cond:
        ablation_parts.append("NoCatCond")
    if config.model.ablation_no_region_cond:
        ablation_parts.append("NoRegionCond")
    if config.model.ablation_no_var_cond:
        ablation_parts.append("NoVarCond")
    if config.model.enable_constraint_loss:
        ablation_parts.append("ConstraintLoss")
    config.ablation = "_".join(ablation_parts)

    # 按 ablation 标签、split_seed 与实验后缀区分目录；训练记录 CSV 放入该目录并改名
    suffix = f"_{config.save_suffix}" if config.save_suffix else ""
    config.save_dir = f"fig2_data_prepare_{config.ablation}_{config.split_seed}{suffix}"
    config.results_csv = os.path.join(config.save_dir, "训练记录.csv")

    config.trial_name = '+'.join([f"{v}" for k, v in params.items()])
    config.log_path = f"{config.PATH_LOG}/{config.trial_name}.log"

    # pprint(config)
    dir_check(config.log_path)
    config.logger.open(config.log_path, mode="w")

    config.logger.write(config.__str__()+'\n', is_terminal=False)


    clean_norm_data = CleanDataset(config)
    denorm_fn, means_t, stds_t = clean_norm_data.build_denorm_bundle(config.model.device)

    train_mask = torch.tensor(clean_norm_data.train_mask, dtype=torch.bool).to(config.device)
    val_mask = torch.tensor(clean_norm_data.val_mask, dtype=torch.bool).to(config.device)
    test_mask = torch.tensor(clean_norm_data.test_mask, dtype=torch.bool).to(config.device)

    model = DiffSPM(config.model).to(config.device)
    # 你之前的 attach_denorm 签名如果只接收一个函数，就这样传；如果你想把均值/方差也塞进去方便 debug，也可以：
    model.attach_denorm(denorm_fn, means_t, stds_t)
    config.logger.write(
        f"[denoiser] type={getattr(model, 'denoiser_type', 'UGnet')} "
        f"ablation={config.ablation} split_seed={config.split_seed}\n"
        f"[flags] denoiser={config.model.denoiser} "
        f"no_cat={int(config.model.ablation_no_cat_cond)} "
        f"no_region={int(config.model.ablation_no_region_cond)} "
        f"no_var={int(config.model.ablation_no_var_cond)} "
        f"enable_constraint_loss={int(config.model.enable_constraint_loss)} "
        f"constraint_weight={config.model.constraint_weight}\n",
        is_terminal=True,
    )





    # Load training dataset
    orig_train_ds = SPMDataset(clean_norm_data, "train", config)
    val_dataset = SPMDataset(clean_norm_data, "val", config)
    test_dataset = SPMDataset(clean_norm_data, "test", config)
    print(f"增强前训练集样本数: {len(orig_train_ds)}")

    if not (0 < config.train_fraction <= 1.0):
        raise ValueError(f"train_fraction must be in (0, 1], got {config.train_fraction}")
    if config.train_fraction < 1.0:
        rng = np.random.default_rng(int(config.split_seed) + int(config.seed) * 1000003)
        n_keep = max(1, int(np.ceil(len(orig_train_ds) * config.train_fraction)))
        keep_idx = np.sort(rng.choice(len(orig_train_ds), size=n_keep, replace=False))
        orig_train_ds = LocalIndexDataset(orig_train_ds, keep_idx)
        train_mask = train_mask[torch.as_tensor(keep_idx, dtype=torch.long, device=train_mask.device)]
        msg = f"[data_scaling] train_fraction={config.train_fraction:.3f}, retained={n_keep}/{len(clean_norm_data.train_data)} before oversampling\n"
        config.logger.write(msg, is_terminal=True)


    # --- （A）GEN 评估用的 loader（最小侵入：只在 is_gen 时启用，其他不受影响） ---
    gen_eval_loader = None
    if config.is_gen:
        # 用“未增强”的原训练集做评估（也可换成增强后的 train_dataset，二者都可；这里更客观一点，用未增强）
        gen_eval_loader = torch.utils.data.DataLoader(
            orig_train_ds,  # 注意：就是 SPMDataset(clean_norm_data, "train", config)
            config.train_batch_size,
            shuffle=False,
            pin_memory=True
        )
        config.logger.write("[GEN MODE] eval will run on the full-train split (same as training data).\n", is_terminal=True)
    train_dataset, mask_train_data = oversample_by_multiplier(
        orig_train_ds,
        train_mask,
        target_min=6,
        target_max=12
    )
    train_dataset = LocalIndexDataset(train_dataset)
    print(f"增强后训练集样本数: {len(train_dataset)}")
    train_loader = torch.utils.data.DataLoader(train_dataset, config.train_batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, config.train_batch_size, shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, config.train_batch_size, shuffle=False, pin_memory=True)
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0)
    # log model architecture
    # print(model)
    # config.logger.write(model.__str__())

    # log training process
    config.logger.write(f'Num_of_parameters:{sum([p.numel() for p in model.parameters()])}\n', is_terminal=True)
    message = "      |---Train--- |------------- Val ------------|\n"
    config.logger.write(message, is_terminal=True)

    # 原有（包含 CRPS/MMD 等）——替换掉
    message = "Epoch | Loss  Time | (macro) MAE   RMSE  |  MMD2   Pass   VetPass\n"
    config.logger.write(message, is_terminal=True)

    def _best_score(metrics, alpha=0.3):
        mae = metrics['overall_macro']['MAE']
        vet = metrics.get('vet_pass_rate', 0.0)
        vet = 0.0 if (vet is None or (isinstance(vet, float) and np.isnan(vet))) else float(vet)
        return vet - alpha * mae

    train_start_t = timer()

    LOSS = []
    best_val = None
    best_ckpt = None

    for epoch in range(config.epoch):
        n, avg_loss, time_lst = 0, 0, []
        avg_constraint = 0.0
        avg_constraint_region = 0.0
        avg_constraint_var = 0.0
        avg_constraint_nn = 0.0
        for i, batch in enumerate(train_loader):
            time_start = timer()

            label, index = batch
            mask_data = mask_train_data[index].to(config.device)
            label = label.clone().detach().to(torch.float32).to(config.device)

            loss = 10 * model.loss(label, mask_data)
            if not torch.isfinite(loss):
                warn = f"\n[warn] non-finite loss at epoch={epoch+1}, iter={i+1}; skip optimizer step.\n"
                config.logger.write(warn, is_terminal=True)
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            n += 1
            avg_loss = avg_loss * (n - 1) / n + loss.item() / n
            stats = getattr(model, "last_loss_stats", None)
            if stats is not None:
                avg_constraint = avg_constraint * (n - 1) / n + stats.get("loss_constraint", 0.0) / n
                avg_constraint_region = avg_constraint_region * (n - 1) / n + stats.get("loss_constraint_region", 0.0) / n
                avg_constraint_var = avg_constraint_var * (n - 1) / n + stats.get("loss_constraint_var", 0.0) / n
                avg_constraint_nn = avg_constraint_nn * (n - 1) / n + stats.get("loss_constraint_nn", 0.0) / n
            time_lst.append((timer() - time_start))
            message = (
                f"{i / len(train_loader) + epoch:6.1f}| {avg_loss:0.3f} {np.sum(time_lst):.1f}s "
                f"| lc={avg_constraint:.5f} "
                f"(r={avg_constraint_region:.5f},v={avg_constraint_var:.5f},nn={avg_constraint_nn:.5f})"
            )
            print('\r' + message, end='', flush=True)
            LOSS.append(avg_loss)

        config.logger.message_buffer += message

        # ====== 存 ckpt：只保留当前轮，旧的 epoch_*.ckpt 删掉 ======
        save_dir = config.save_dir
        os.makedirs(save_dir, exist_ok=True)
        import glob
        for old in glob.glob(os.path.join(save_dir, "epoch_*.ckpt")):
            try:
                os.remove(old)
            except OSError:
                pass
        ckpt_path = os.path.join(save_dir, f"epoch_{epoch}_loss{avg_loss:.6f}.ckpt")
        torch.save({"model_state": model.state_dict(), "epoch": epoch}, ckpt_path)


        # ====== GEN（全样本）评估：与 val/test 并列的独立分支 ======
        # if config.is_gen and epoch > 220 and (epoch) % 10 == 0:
        if config.is_gen and epoch>100 and (epoch) % 10 == 0:
            # 评估载具与掩码：在 GEN 模式下用全量训练数据做评估
            eval_loader = gen_eval_loader  # 若未定义，可用 train_loader 兜底
            eval_mask   = train_mask
            desc_label  = "gen(full-train)"

            with evaluating(model):  # eval + inference_mode，退出后自动恢复
                eval_metrics = evaluate_sampling_metrics(
                    model, eval_loader, eval_mask, config, means_t, stds_t,
                    n_samples=config.n_samples, split_n=4, desc=desc_label,
                    region_map=config.region_map, var_map=config.var_map,
                    vet_region_csv=config.data.region_encoding,
                    vet_variable_csv=config.data.var_encoding,
                    vet_variable_baseline_csv=config.variable_baseline,
                    vet_region_baseline_csv=config.region_baseline,
                    vet_default_model_code=0,
                )

            mm = eval_metrics['overall_macro']
            grp = eval_metrics['groups']
            vet = eval_metrics.get('vet_pass_rate', float('nan'))
            cat_mean = grp['by_category']['mean']
            reg_mean = grp['by_region']['mean']
            var_mean = grp['by_variable']['mean']

            line = (
                f"\n[{desc_label}@{epoch+1}] "
                f"(macro) MAE={mm['MAE']:.4f} RMSE={mm['RMSE']:.4f}  "
                f"MMD2={mm.get('MMD2', float('nan')):.4f}  VetPass={vet:.2%}\n"
                f"[groups.mean]  cat: MMD2={cat_mean['MMD2']:.4f} MAE={cat_mean['MAE']:.4f}  "
                f"reg: MMD2={reg_mean['MMD2']:.4f} MAE={reg_mean['MAE']:.4f}  "
                f"var: MMD2={var_mean['MMD2']:.4f} MAE={var_mean['MAE']:.4f}  (ckpt: {os.path.basename(ckpt_path)})\n"
            )
            config.logger.write(line, is_terminal=True)

            if (best_val is None) or (mm['MAE'] < best_val['overall_macro']['MAE']):
                best_val = eval_metrics
                best_ckpt = ckpt_path
                # 保持原有 best.ckpt，同时额外给 GEN 留一份 best_gen.ckpt
                shutil.copyfile(best_ckpt, os.path.join(save_dir, "best.ckpt"))
                shutil.copyfile(best_ckpt, os.path.join(save_dir, "best_gen.ckpt"))

            # 记录到 CSV，split 标为 "gen"
            append_result_row(
                config, params, split="gen", epoch=epoch+1,
                ckpt_path=ckpt_path, metrics=eval_metrics, avg_train_loss=None
            )

        # ====== 验证（每 20 轮），用采样+反标准化+指标 ======
        if config.is_val and epoch > 200 and (epoch) % 10 == 0:
        # if config.is_val and (epoch) % 10 == 0:
            with evaluating(model):  # eval + inference_mode，退出后自动恢复
                val_metrics = evaluate_sampling_metrics(
                    model, val_loader, val_mask, config, means_t, stds_t,
                    n_samples=config.n_samples, split_n=4, desc="test",
                    region_map=config.region_map, var_map=config.var_map,
                    vet_region_csv=config.data.region_encoding,
                    vet_variable_csv=config.data.var_encoding,
                    vet_variable_baseline_csv=config.variable_baseline,
                    vet_region_baseline_csv=config.region_baseline,
                    vet_default_model_code=0,
                )
            mm = val_metrics['overall_macro']
            grp = val_metrics['groups']
            vet = val_metrics.get('vet_pass_rate', float('nan'))
            cat_mean = grp['by_category']['mean']
            reg_mean = grp['by_region']['mean']
            var_mean = grp['by_variable']['mean']

            line = (
                f"\n[val@{epoch+1}] "
                f"(macro) MAE={mm['MAE']:.4f} RMSE={mm['RMSE']:.4f}  "
                f"MMD2={mm.get('MMD2', float('nan')):.4f}  VetPass={vet:.2%}\n"
                f"[groups.mean]  cat: MMD2={cat_mean['MMD2']:.4f} MAE={cat_mean['MAE']:.4f}  "
                f"reg: MMD2={reg_mean['MMD2']:.4f} MAE={reg_mean['MAE']:.4f}  "
                f"var: MMD2={var_mean['MMD2']:.4f} MAE={var_mean['MAE']:.4f}  (ckpt: {os.path.basename(ckpt_path)})\n"
            )
            config.logger.write(line, is_terminal=True)

            # 最优：融合 macro MAE 与 vet_pass_rate（score = vet - alpha*MAE 取最大，alpha=0.5）
            if (best_val is None) or (_best_score(val_metrics) > _best_score(best_val)):
                best_val = val_metrics
                best_ckpt = ckpt_path
                shutil.copyfile(best_ckpt, os.path.join(save_dir, "best.ckpt"))
            append_result_row(
                config, params, split="val", epoch=epoch+1,
                ckpt_path=ckpt_path, metrics=val_metrics, avg_train_loss=None  # 可填None
            )
        # ====== 测试（每 20 轮），同样口径；最终汇报可用 best_ckpt 再跑一遍 ======
        # if config.is_test and epoch > 100 and (epoch) % 10 == 0:
        #     with evaluating(model):
        
        #         eval_metrics = evaluate_sampling_metrics(
        #             model, test_loader, test_mask, config, means_t, stds_t,
        #             n_samples=config.n_samples, split_n=10, desc="test",
        #             region_map=config.region_map, var_map=config.var_map,
        #             vet_region_csv=config.data.region_encoding,
        #             vet_variable_csv=config.data.var_encoding,
        #             vet_variable_baseline_csv=config.variable_baseline,
        #             vet_region_baseline_csv=config.region_baseline,
        #             vet_default_model_code=0,
        #         )
        #     m  = eval_metrics['overall']        # micro
        #     mm = eval_metrics['overall_macro']  # macro
        #     grp = eval_metrics['groups']        # 分组分布一致性
        #     vet = eval_metrics.get('vet_pass_rate', float('nan'))

        #     # 三类分组 mean（分布性口径）
        #     cat_mean = grp['by_category']['mean']
        #     reg_mean = grp['by_region']['mean']
        #     var_mean = grp['by_variable']['mean']

        #     line = (
        #         f"\n[{desc_label}@{epoch+1}] "
        #         f"(micro)  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  MAPE_std={m['MAPE_std']:.4f}  R2={m['R2_micro']:.4f}\n"
        #         f"(macro)  MAE={mm['MAE']:.4f} RMSE={mm['RMSE']:.4f} MAPE_std={mm['MAPE_std']:.4f} R2={mm['R2_macro']:.4f}\n"
        #         f"[groups.mean]  "
        #         f"cat: MMD2={cat_mean['MMD2']:.4f}  MMD={cat_mean['MMD']:.4f}  W={cat_mean['W']:.4f}  R2={cat_mean['R2']:.4f}  |  "
        #         f"reg: MMD2={reg_mean['MMD2']:.4f}  MMD={reg_mean['MMD']:.4f}  W={reg_mean['W']:.4f}  R2={reg_mean['R2']:.4f}  |  "
        #         f"var: MMD2={var_mean['MMD2']:.4f}  MMD={var_mean['MMD']:.4f}  W={var_mean['W']:.4f}  R2={var_mean['R2']:.4f}\n"
        #         f"VetPass={vet:.2%}  (ckpt: {os.path.basename(ckpt_path)})\n"
        #     )
        #     config.logger.write(line, is_terminal=True)

        #     append_result_row(
        #         config, params, split="test", epoch=epoch+1,
        #         ckpt_path=ckpt_path, metrics=eval_metrics, avg_train_loss=None
        #     )
# ====== 训练结束后：可选导出 Fig.2 绘图数据（--export_fig2 1 时执行） ======
    if getattr(config, 'export_fig2', 0) and not config.is_gen and len(test_dataset) > 0:
        sd = torch.load(os.path.join(save_dir, "best.ckpt"), map_location=config.device)
        model.load_state_dict(sd.get('model_state', sd), strict=False)
        from utils.export_fig2 import run_export
        fig2_dir = run_export(
            model, test_loader, train_loader, test_mask, train_mask,
            config, clean_norm_data, means_t, stds_t,
            out_dir=os.path.join(save_dir, "fig2_data"),
            n_samples_fig2=config.n_samples, verbose=True,
        )
        config.logger.write(f"[export_fig2] wrote data to {fig2_dir}\n", is_terminal=True)

# ====== 训练结束后的最终测试 ======

    # final_ckpt = best_ckpt if (config.is_test and ('best_ckpt' in locals()) and (best_ckpt is not None)) else ckpt_path
    # sd = torch.load(final_ckpt, map_location=config.device)
    # model.load_state_dict(sd['model_state'], strict=False)

    # with evaluating(model):
    #     test_metrics = evaluate_sampling_metrics(
    #         model, test_loader, test_mask, config, means_t, stds_t,
    #                 n_samples=config.n_samples, split_n=10, desc="test",
    #                 region_map=config.region_map, var_map=config.var_map,
    #                 vet_region_csv=config.data.region_encoding,
    #                 vet_variable_csv=config.data.var_encoding,
    #                 vet_variable_baseline_csv=config.variable_baseline,
    #                 vet_region_baseline_csv=config.region_baseline,
    #                 vet_default_model_code=0,
    #             )

    # m  = test_metrics['overall']        # micro（逐点）
    # mm = test_metrics['overall_macro']  # macro（先 g 内均值，再 across g）
    # test_vet = test_metrics.get('vet_pass_rate', float('nan'))

    # line = (f"\n[Final Test @ {os.path.basename(final_ckpt)}] "
    #         f"(micro) MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} MAPE_std={m['MAPE_std']:.4f} "
    #         f"CRPS={m['CRPS']:.4f} MMD={m['MMD']:.4f} MMD2={m['MMD2']:.4f} W={m['Wasserstein']:.4f}\n"
    #         f"                      "
    #         f"TestPass={test_vet:.2%}\n"
            
    #         f"(macro) MAE={mm['MAE']:.4f} RMSE={mm['RMSE']:.4f} MAPE_std={mm['MAPE_std']:.4f} "
    #         f"CRPS={mm['CRPS']:.4f} MMD={mm['MMD']:.4f} MMD2={mm['MMD2']:.4f} W={mm['Wasserstein']:.4f}\n")
    # config.logger.write(line, is_terminal=True)


    # if 'by_g_readable' in test_metrics and len(test_metrics['by_g_readable']) > 0:
    #     worst = sorted(
    #         test_metrics['by_g_readable'].items(),
    #         key=lambda kv: kv[1]['MAE'], reverse=True
    #     )[:5]
    #     config.logger.write("[Final Test] Worst 5 g by MAE (standardized):\n", is_terminal=True)
    #     for (r_name, v_name), metr in worst:
    #         config.logger.write(
    #             f"  - ({r_name} | {v_name})  MAE={metr['MAE']:.4f}  RMSE={metr['RMSE']:.4f}  "
    #             f"MAPE_std={metr['MAPE_std']:.4f}  CRPS={metr['CRPS']:.4f}\n",
    #             is_terminal=True
    #         )



if __name__ == '__main__':
    print('GPU:', torch.cuda.current_device())
    params = vars(get_params())
    try:
        import nni
        import logging
        logger = logging.getLogger('training')
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params.update(tuner_params)
    except Exception:
        pass  # 非 NNI 环境：直接用命令行参数
    main(params)
