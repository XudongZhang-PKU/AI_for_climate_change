# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn

from .ugnet import UGnet, TinyUGnet
from .ugnet_notcn import UGnetNoTCN
from utils.common_utils import gather


class DiffSPM(nn.Module):
    """
    - Training loss: noise loss + averaged constraint penalty
    - Sampling (DDIM): early/mid soft guidance (light projection), last-K steps hard projection (POCS)
    Tensor layout:
        x_full: [B, G, T, C], where G = R * V, and the value column is at index 4 (C>=5)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device

        # --------- core diffusion params ----------
        self.N = config.N
        self.sample_steps = config.sample_steps
        
        self.beta_start = config.get('beta_start', 1e-4)
        self.beta_end = config.get('beta_end', 2e-2)

        # Regions / Variables
        self.num_regions = config.num_region
        self.num_variables = config.num_variable

        # Ablation: no constraint (skip soft guidance + final hard projection)
        self.ablation_no_constraint = getattr(config, "ablation_no_constraint", False)
        self.denoiser = getattr(config, "denoiser", "UGnet")
        self.enable_constraint_loss = bool(getattr(config, "enable_constraint_loss", False))
        self.constraint_weight = float(getattr(config, "constraint_weight", 1e-4))

        if self.denoiser == "TinyUGnet":
            self.eps_model = TinyUGnet(config).to(self.device)
            self.denoiser_type = "TinyUGnet"
        elif self.denoiser == "UGnetNoTCN":
            self.eps_model = UGnetNoTCN(config).to(self.device)
            self.denoiser_type = "UGnetNoTCN"
        else:
            self.eps_model = UGnet(config).to(self.device)
            self.denoiser_type = "UGnet"

        # Build beta  —— 全都注册为 buffer
        beta = torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.N, device="cpu") ** 2
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        sigma2 = beta

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sigma2", sigma2)
        

        # training loss weights
        self.lambda_region = config.get('lambda_region', 2e-3)
        self.lambda_var = config.get('lambda_var', 2e-3)
        self.lambda_nn = config.get('lambda_nn', 4e-3)

        # guidance during sampling
        self.use_soft_guidance = config.get('use_soft_guidance', True)
        self.soft_guidance_schedule = config.get('soft_guidance_schedule', 'exp')
        #软约束引导的上下限
        self.soft_eta_min = config.get('soft_eta_min', 0)
        self.soft_eta_max = config.get('soft_eta_max', 1e-3)
        #软约束的幅值

        # ==== 比例自适应（方案B）配置 ====
        self.constraint_target_ratio = getattr(config, "constraint_target_ratio", 0.35)  # 约束/噪声 目标占比
        self.constraint_ratio_smooth = getattr(config, "constraint_ratio_smooth", 0.70)  # EMA 平滑
        self.constraint_gain_min = getattr(config, "constraint_gain_min", 0.5)          # 每步最小缩放
        self.constraint_gain_max = getattr(config, "constraint_gain_max", 4.00)          # 每步最大缩放
        self.constraint_warmup_steps = getattr(config, "constraint_warmup_steps", 50)   # 自适应热身步数

        # 注册缓冲区（随 state_dict 保存/加载；会自动搬到 device 上）
        self.register_buffer("_ratio_ma", torch.tensor(self.constraint_target_ratio, dtype=torch.float32))
        self.register_buffer("_g_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("_num_updates", torch.tensor(0, dtype=torch.long))

        # indices in C-dim (metadata layout)
        self.region_index = 2
        self.var_index = 3
        self.value_index = 4

        # “最后K步硬约束”相关
        # __init__ 里
        self.reverse_projection_scale = config.get("reverse_projection_scale", 1.5)

        # Keep compatibility for existing external checks
        self.enable_constraints = self.enable_constraint_loss

        # constraints
        self.setup_constraints()
    def attach_denorm(self, denorm_fn, means=None, stds=None):
        """
        仅注册可微反标准化函数；不再把 means/stds 挂成 buffer，
        以免出现 DataParallel 对 buffer 设备的一致性要求。
        """
        self.denorm_fn = denorm_fn

        # if (means is not None) and (stds is not None):
        #     # 兼容 numpy
        #     if isinstance(means, np.ndarray):
        #         means = torch.from_numpy(means).to(dtype=torch.float32)
        #     if isinstance(stds, np.ndarray):
        #         stds  = torch.from_numpy(stds).to(dtype=torch.float32)

        #     # 保持在 CPU，避免 DataParallel 不同卡上索引冲突
        #     means = means.to('cpu', non_blocking=True)
        #     stds  = stds.to('cpu',  non_blocking=True)

        #     # 如果你不想把它们写进 ckpt，可以 persistent=False
        #     self.register_buffer("denorm_means", means, persistent=False)
        #     self.register_buffer("denorm_stds",  stds,  persistent=False)


    # -------------------- constraints setup --------------------
    def setup_constraints(self):
        """
        Build:
        - var_constraints mapping
        - non_negative mask
        - region_sum_vector (for energy)
        - projection matrices P_var, P_reg (for hard projection)
        """
        # --- variable sum constraints (new code table) ---
        # total -> list of component variables
        dev_build = "cpu"
        self.var_constraints = {
            # CCS 总量 = 生物质 + 化石 + 工业过程
            0:  [1, 2, 45],

            # CO2 总排放 = AFOLU + 能源与工业过程 + 其他
            # （保持原口径，不纳入供需明细/分部门拆分）
            3:  [4, 5, 33],

            # 居民/商业终端能源 = 电 + 气 + 液体 + 固体
            8:  [40, 41, 42, 44],

            # 交通终端能源 = 电 + 液体(油) + 气体 + 氢
            9:  [10, 39, 43, 47],

            # 工业终端能源 = 电 + 气 + 液体 + 固体 + 氢
            28: [35, 36, 37, 38, 46],

            # 一次能源总量 = 生物质 + 煤 + 气 + 核 + 油 + 地热 + 水电 + 太阳能 + 风能
            11: [12, 13, 14, 15, 16, 29, 30, 31, 32],

            # 二次能源（电力）= 各电源分项
            17: [18, 19, 20, 21, 22, 23, 24, 25, 26],
        }


        # --- non-negative variables (set) ---
        self.non_negative_vars = {
            0, 1, 2, 45,
            # 二次能源（电力）及子项
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26,

            # 一次能源及子项
            11, 12, 13, 14, 15, 16, 29, 30, 31, 32,

            # 终端能源：居民/商业、交通、工业 及其子项
            8, 40, 41, 42, 44,
            9, 10, 39, 43, 47,
            28, 35, 36, 37, 38, 46,
        }

        # region_sum_vector
        region_sum_vector = torch.zeros((self.num_regions, 1), device=dev_build)
        region_sum_vector[0, 0] = 1.0
        if self.num_regions > 1:
            region_sum_vector[1:, 0] = -1.0
        self.register_buffer("region_sum_vector", region_sum_vector)

        # var_sum_matrix
        Cnum = len(self.var_constraints)
        var_sum_matrix = torch.zeros((self.num_variables, Cnum), device=dev_build)
        for c_idx, (total_var, comp_vars) in enumerate(self.var_constraints.items()):
            if total_var < self.num_variables:
                var_sum_matrix[total_var, c_idx] = 1.0
            for v in comp_vars:
                if v < self.num_variables:
                    var_sum_matrix[v, c_idx] -= 1.0
        self.register_buffer("var_sum_matrix", var_sum_matrix)

        # non_negative_mask
        non_negative_mask = torch.zeros((1, 1, self.num_variables, 1), device=dev_build)
        for vid in self.non_negative_vars:
            if vid < self.num_variables:
                non_negative_mask[0, 0, vid, 0] = 1.0
        self.register_buffer("non_negative_mask", non_negative_mask)

        # P_var
        A = torch.zeros((Cnum, self.num_variables), device=dev_build)
        for row, (tot, comps) in enumerate(self.var_constraints.items()):
            if tot < self.num_variables: A[row, tot] = 1.0
            for c in comps:
                if c < self.num_variables: A[row, c] -= 1.0
        AA = A @ A.T
        AA_inv = torch.linalg.pinv(AA)
        P_var = torch.eye(self.num_variables, device=dev_build) - A.T @ AA_inv @ A
        self.register_buffer("P_var", P_var)

        # P_reg
        a = torch.zeros((self.num_regions, 1), device=dev_build)
        a[0, 0] = 1.0
        if self.num_regions > 1: a[1:, 0] = -1.0
        denom = float((a.T @ a).item()) or 1.0
        P_reg = torch.eye(self.num_regions, device=dev_build) - (a @ a.T) / denom
        self.register_buffer("P_reg", P_reg)

        # P_eq_joint
        I_R = torch.eye(self.num_regions, device=dev_build)
        I_V = torch.eye(self.num_variables, device=dev_build)

        A_var = torch.zeros((Cnum, self.num_variables), device=dev_build)
        for row, (tot, comps) in enumerate(self.var_constraints.items()):
            if tot < self.num_variables: A_var[row, tot] = 1.0
            for c in comps:
                if c < self.num_variables: A_var[row, c] -= 1.0

        a_reg = torch.zeros((self.num_regions, 1), device=dev_build); a_reg[0, 0] = 1.0
        if self.num_regions > 1: a_reg[1:, 0] = -1.0
        a_reg_T = a_reg.T

        A_eq_top = torch.kron(I_R, A_var)
        A_eq_bot = torch.kron(a_reg_T, I_V)
        A_eq = torch.cat([A_eq_top, A_eq_bot], dim=0)

        n = self.num_regions * self.num_variables
        I_n = torch.eye(n, device=dev_build)
        eps = 1e-8
        AA = A_eq @ A_eq.T
        AA_inv = torch.linalg.pinv(AA + eps * torch.eye(AA.shape[0], device=dev_build))
        P_eq_joint = I_n - A_eq.T @ AA_inv @ A_eq
        self.register_buffer("P_eq_joint", P_eq_joint)



    # -------------------- energy (averaged) --------------------
# -------------------- energy (averaged, balanced) --------------------
    def compute_constraint_penalty(self, x_pred):
        """
        Averaged penalty per sample with per-group balancing.
        x_pred: [B, G, T, C] (value at C=4), 已在物理空间
        return: [B]
        """
        B, G, T, _ = x_pred.shape
        values = x_pred[:, :, :, self.value_index].view(
            B, self.num_regions, self.num_variables, T
        )  # [B,R,V,T]
        total = torch.zeros(B, device=x_pred.device)

        eps = 1e-8
        # 可选：鲁棒化阈值（Huber转折点）

        # 1) 地区和：region0 == sum(others)
        if self.lambda_region > 0:
            region_diff = torch.einsum('brvt,rk->bvtk', values, self.region_sum_vector)  # [B,V,T,1]
            # 以该等式参与项的“L1幅值和”为尺度进行相对化
            # denom_region ≈ |world| + Σ|others|
            # 用同一 einsum 在绝对值上做：把 region_sum_vector 换成它的绝对值（等价于把系数都当 +1）
            reg_abs = values.abs()
            ones_vec = self.region_sum_vector.abs()  # 实际就是 [1,1,1,...]
            denom_region = torch.einsum('brvt,rk->bvtk', reg_abs, ones_vec).clamp_min(eps)  # [B,V,T,1]

            r = region_diff / denom_region

            pen_region = (r ** 2).mean(dim=(1, 2, 3))

            total += self.lambda_region * pen_region

        # 2) 变量和：total_var == sum(comp_vars)
        if self.lambda_var > 0:
            var_diff = torch.einsum('brvt,vc->brct', values, self.var_sum_matrix)         # [B,R,C,T]
            # 对每个约束列 c：用 |coeff| 做权重，把所有参与项的 |value| 相加作为尺度
            M = self.var_sum_matrix.abs()                                                  # [V,C]
            denom_var = torch.einsum('brvt,vc->brct', values.abs(), M).clamp_min(eps)      # [B,R,C,T]

            r = var_diff / denom_var
            pen_var = (r ** 2).mean(dim=(1, 2, 3))

            total += self.lambda_var * pen_var

        # 3) 非负约束：按“参与元素数目”做平均，避免 R/V/T 变化膨胀
        if self.lambda_nn > 0:
            mask = (self.non_negative_mask > 0).to(values.dtype)          # [1,1,V,1]
            neg = torch.relu(-values) * mask                               # [B,R,V,T]
            denom = mask.expand(B, self.num_regions, self.num_variables, T) \
                        .sum(dim=(1, 2, 3)).clamp_min(1.0)
            pen_nn = (neg ** 2).sum(dim=(1, 2, 3)) / denom
            total += self.lambda_nn * pen_nn

        return total  # [B]

    # -------------------- helper views & projections --------------------
    def _values_view(self, x_full):
        # x_full: [B,G,T,C] -> values: [B,R,V,T]
        B, G, T, _ = x_full.shape
        return x_full[:, :, :, self.value_index].view(B, self.num_regions, self.num_variables, T)


    # -------------------- diffusion utilities --------------------
    @staticmethod
    def compute_alpha(beta_vec: torch.Tensor, t: torch.Tensor):
        """
        beta_vec: [N], t: [B] long
        return a(t) = prod_{s<=t} (1 - beta_s) as [B,1,1,1]
        """
        t = t.to(beta_vec.device)
        beta = torch.cat([torch.zeros(1, device=beta_vec.device), beta_vec], dim=0)  # pad
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def calculate_constraint_lambda(self, t, min_lambda=0.01, max_lambda=1.0, schedule='linear'):
        """time-dependent scalar (per-sample) for soft guidance"""
        dev = t.device if isinstance(t, torch.Tensor) else self.alpha_bar.device
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=dev, dtype=torch.long)
        t = t.to(dev).float()
        N = float(self.N)
        u = t / N  # 1 -> early, 0 -> late

        if schedule == 'linear':
            lam = min_lambda + (max_lambda - min_lambda) * (1.0 - u)
        elif schedule == 'exp':
            decay = 3.0
            lam = min_lambda + (max_lambda - min_lambda) * torch.exp(-decay * u)
        elif schedule == 'sigmoid':
            beta = 10.0; midpoint = 0.2
            lam = min_lambda + (max_lambda - min_lambda) * (1.0 / (1.0 + torch.exp((u - midpoint) * beta)))
        elif schedule == 'cosine':
            lam = min_lambda + (max_lambda - min_lambda) * 0.5 * (1.0 + torch.cos(math.pi * u))
        elif schedule == 'step':
            threshold = 0.3
            lam = torch.where(u < threshold, torch.tensor(max_lambda, device=t.device), torch.tensor(min_lambda, device=t.device))
        else:
            lam = torch.ones_like(t) * max_lambda
        return lam

    # -------------------- training loss --------------------
    def loss(self, x0, mask_train_data):
        B, G, T, Cdim = x0.shape
        dev = x0.device
        # 1) 随机 t，加噪
        t = torch.randint(0, self.N, (B,), device=dev, dtype=torch.long)
        group_series = x0[:, :, :, self.value_index].unsqueeze(-1)  # [B,G,T,1]
        eps = torch.randn_like(group_series)
        alpha_bar_t = gather(self.alpha_bar.to(dev), t)                      # [B]
        mean = alpha_bar_t.sqrt().view(-1,1,1,1) * group_series
        var  = (1 - alpha_bar_t).view(-1,1,1,1)
        group_series_t = mean + eps * var.sqrt()

        xt = x0.clone()
        xt[:, :, :, self.value_index] = group_series_t.squeeze(-1)

        # 2) 预测噪声
        eps_theta = self.eps_model(xt, t)  # [B,G,T,1]

        # 3) var-balanced 噪声 MSE
        var_codes = x0[:, :, 0, self.var_index]
        se = (eps_theta - eps) ** 2
        variable_losses = []
        unique_vars_in_batch = torch.unique(var_codes[mask_train_data])
        for var_id in unique_vars_in_batch:
            vm = (var_codes == var_id) & mask_train_data
            m = vm.unsqueeze(-1).unsqueeze(-1).expand_as(se)
            num = m.sum()
            if num > 0:
                variable_losses.append((se * m).sum() / num)
        loss_noise = torch.stack(variable_losses).mean() if variable_losses else torch.tensor(0.0, device=xt.device)

        loss_constraint = torch.tensor(0.0, device=xt.device)
        loss_constraint_region = torch.tensor(0.0, device=xt.device)
        loss_constraint_var = torch.tensor(0.0, device=xt.device)
        loss_constraint_nn = torch.tensor(0.0, device=xt.device)

        if self.enable_constraint_loss:
            x0_hat = (group_series_t - eps_theta * (1 - alpha_bar_t).sqrt().view(-1, 1, 1, 1)) / alpha_bar_t.sqrt().view(-1, 1, 1, 1)
            x0_pred_full = xt.clone()
            x0_pred_full[:, :, :, self.value_index] = x0_hat.squeeze(-1)
            x0_phys = self.denorm_fn(x0_pred_full)

            values = x0_phys[:, :, :, self.value_index].view(B, self.num_regions, self.num_variables, T)
            eps_c = 1e-8
            if self.lambda_region > 0:
                region_diff = torch.einsum("brvt,rk->bvtk", values, self.region_sum_vector)
                denom_region = torch.einsum("brvt,rk->bvtk", values.abs(), self.region_sum_vector.abs()).clamp_min(eps_c)
                loss_constraint_region = ((region_diff / denom_region) ** 2).mean()
            if self.lambda_var > 0:
                var_diff = torch.einsum("brvt,vc->brct", values, self.var_sum_matrix)
                denom_var = torch.einsum("brvt,vc->brct", values.abs(), self.var_sum_matrix.abs()).clamp_min(eps_c)
                loss_constraint_var = ((var_diff / denom_var) ** 2).mean()
            if self.lambda_nn > 0:
                neg = torch.relu(-values) * (self.non_negative_mask > 0).to(values.dtype)
                loss_constraint_nn = (neg ** 2).mean()

            residual = (
                self.lambda_region * loss_constraint_region
                + self.lambda_var * loss_constraint_var
                + self.lambda_nn * loss_constraint_nn
            )
            loss_constraint = nn.functional.mse_loss(residual, torch.zeros_like(residual))

        total_loss = loss_noise + self.constraint_weight * loss_constraint
        self.last_loss_stats = {
            "loss_noise": float(loss_noise.detach().item()),
            "loss_constraint": float(loss_constraint.detach().item()),
            "loss_constraint_region": float(loss_constraint_region.detach().item()),
            "loss_constraint_var": float(loss_constraint_var.detach().item()),
            "loss_constraint_nn": float(loss_constraint_nn.detach().item()),
            "constraint_weight": float(self.constraint_weight),
            "enable_constraint_loss": bool(self.enable_constraint_loss),
        }
        return total_loss



    @torch.no_grad()
    def final_projection_sandwich(self, x_last, label):
        assert hasattr(self, "denorm_fn") and (self.denorm_fn is not None)
        B, G, T, _ = x_last.shape
        R, V = self.num_regions, self.num_variables

        x_full = label.clone()
        x_full[:, :, :, self.value_index] = x_last.squeeze(-1)
        x_phys = self.denorm_fn(x_full)

        # # denorm only when constraints are disabled
        # if not self.enable_constraints:
        #     return x_phys[:, :, :, self.value_index].unsqueeze(-1)

        vals = self._values_view(x_phys)

        # eq 一次
        def proj_eq_once(z):
            if hasattr(self, "P_eq_joint"):
                B_,R_,V_,T_ = z.shape
                X = z.permute(0,3,1,2).contiguous().view(B_*T_, R_*V_)
                X = X @ self.P_eq_joint.T
                return X.view(B_,T_,R_,V_).permute(0,2,3,1).contiguous()
            out = z
            x = out.permute(0,1,3,2).contiguous(); x = torch.matmul(x, self.P_var.T)
            out = x.permute(0,1,3,2).contiguous()
            x = out.permute(0,2,3,1).contiguous(); x = torch.matmul(x, self.P_reg.T)
            return x.permute(0,3,1,2).contiguous()

        vals = proj_eq_once(vals)

        # NN 只作用受约束变量
        mask_v = None
        if len(self.non_negative_vars) > 0:
            mask_v = torch.zeros((1,1,V,1), device=vals.device, dtype=torch.bool)
            for vid in self.non_negative_vars:
                if vid < V:
                    mask_v[0,0,vid,0] = True
        if mask_v is not None:
            neg = torch.relu(-vals)
            push = self.reverse_projection_scale * neg
            vals = torch.where(mask_v, vals + push, vals)


        # 再 eq 一次，确保等式最后成立
        vals = proj_eq_once(vals)
        vals = proj_eq_once(vals)

        x_phys[:, :, :, self.value_index] = vals.view(B, -1, T)
        return x_phys[:, :, :, self.value_index].unsqueeze(-1)

    # -------------------- DDIM only --------------------
    @torch.no_grad()
    def generalized_steps_ddim(self, x, seq, eta, label):
        """
        DDIM with:
        - SOFT guidance in physical space (single grad step on E)
        - HARD constraint in the last-K steps (project in physical space, then renorm back)
        Args:
            x   : [B, G, T, 1]  (init noise, normalized space)
            seq : list[int] timesteps (0..N-1), usually sparse
            eta : float, DDIM eta
            label: [B, G, T, C] (meta in [:,:,:, :4], value col at self.value_index)
        Returns:
            xs, x0_preds
        """
        code = label[:, :, :, :4]
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds, xs = [], [x]
        b = self.beta.to(x.device)

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n, device=x.device) * i).long()
            next_t = (torch.ones(n, device=x.device) * j).long()
            at      = self.compute_alpha(b, t)        # [B,1,1,1]
            at_next = self.compute_alpha(b, next_t)   # [B,1,1,1]

            xt  = xs[-1].to(x.device)
            inp = torch.cat((code, xt), dim=-1)

            # 1) predict eps and x0_t (normalized space)
            et   = self.eps_model(inp, t)                           # [B,G,T,1]
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()          # [B,G,T,1]

            # 2a) SOFT guidance: single grad step in PHYSICAL space (skip when ablation_no_constraint)
            if getattr(self, "use_soft_guidance", True) and not getattr(self, "ablation_no_constraint", False):
            # if self.enable_constraints and getattr(self, "use_soft_guidance", True):
                with torch.enable_grad():
                    x0_t_req = x0_t.detach().clone().requires_grad_(True)
                    x0_full_grad = label.clone()
                    x0_full_grad[:, :, :, self.value_index] = x0_t_req.squeeze(-1)
                    x0_phys = self.denorm_fn(x0_full_grad)              # [B,G,T,5] (physical)

                    E_batch = self.compute_constraint_penalty(x0_phys)  # [B]
                    E = E_batch.mean()

                    grad_x0 = torch.autograd.grad(E, x0_t_req, retain_graph=False, create_graph=False)[0]

                # step size schedule (weak early, strong late)
                eta_t = self.calculate_constraint_lambda(
                    t, min_lambda=self.soft_eta_min, max_lambda=self.soft_eta_max,
                    schedule=self.soft_guidance_schedule
                ).view(-1, 1, 1, 1)

                # normalize grad + trust region (<=20% relative step)
                g_norm  = grad_x0.view(grad_x0.size(0), -1).norm(dim=1, keepdim=True).view(-1,1,1,1).clamp_min(1e-8)
                grad_dir = grad_x0 / g_norm
                delta    = -eta_t * grad_dir

                x0_norm = x0_t.view(x0_t.size(0), -1).norm(dim=1, keepdim=True).view(-1,1,1,1).clamp_min(1e-8)
                rel     = (delta.view(delta.size(0), -1).norm(dim=1, keepdim=True).view(-1,1,1,1) / x0_norm).clamp_min(1e-8)
                scale   = torch.clamp(0.35 / rel, max=1.0)

                x0_t = (x0_t + scale * delta).detach()

            # 2b) HARD constraint in the last-K steps (project in PHYSICAL -> renorm back)
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et

            xs.append(xt_next)

        return xs, x0_preds



    def p_sample_loop_ddim(self, label):
        """
        label: [B, G, T, C]
        returns xs, x0_preds
        注意：xs[-1] 为物理空间 [B,G,T,1]
        """
        B, G, T, Cdim = label.shape
        N = self.N
        timesteps = self.sample_steps

        seq = [int(s) for s in (np.linspace(0, np.sqrt(N * 0.8), timesteps) ** 2)]


        x = torch.randn([B, G, T, 1], device=label.device)
        xs, x0_preds = self.generalized_steps_ddim(x, seq, eta=1.0, label=label)

        # === 最终收口：输出物理空间 ===
        if getattr(self, "ablation_no_constraint", False):
            x_full = label.clone()
            x_full[:, :, :, self.value_index] = xs[-1].squeeze(-1)
            if hasattr(self, "denorm_fn") and self.denorm_fn is not None:
                xs[-1] = self.denorm_fn(x_full)[:, :, :, self.value_index : self.value_index + 1]
            else:
                xs[-1] = xs[-1]
        else:
            xs[-1] = self.final_projection_sandwich(xs[-1], label)
        return xs, x0_preds



    def evaluate(self, label, n_samples=2):
        """
        Only supports 'ddim_multi'
        label: [B, G, T, C]
        return: [B, n_samples, G, T, 1]
        """
        B, G, T, Cdim = label.shape
        label_rep = label.unsqueeze(1).repeat(1, n_samples, 1, 1, 1).reshape(-1, G, T, Cdim)
        xs, _ = self.p_sample_loop_ddim(label_rep)
        x = xs[-1].reshape(B, n_samples, G, T, 1)
        return x

    def forward(self, label, n_samples=1):
        return self.evaluate(label, n_samples)
