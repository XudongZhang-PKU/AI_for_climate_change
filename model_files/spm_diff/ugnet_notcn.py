# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from .ugnet import TimeEmbedding, SpatialBlock, Downsample, Upsample, GATLayer


class ResidualBlockNoTCN(nn.Module):
    """
    Keep spatial path + residual/norm, remove temporal convolutions.
    """

    def __init__(self, c_in, c_out, config, T, kernel_size=3):
        _ = kernel_size
        super().__init__()
        self.tcn1 = nn.Identity()
        self.tcn2 = nn.Identity()
        self.tcn3 = nn.Identity()
        self.shortcut = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, (1, 1))
        self.pre_proj = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, (1, 1))
        self.t_conv = nn.Linear(config.d_h, c_out)
        self.spatial = SpatialBlock(config.supports_len, c_out, c_out, T)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x, t, A_hat):
        h = self.pre_proj(x)
        t = self.t_conv(t).unsqueeze(-1).unsqueeze(-1)
        h = h + t
        h = self.bn(h)
        h = h.transpose(2, 3)
        h = self.spatial(h, A_hat)
        return h + self.shortcut(x)


class DownBlockNoTCN(nn.Module):
    def __init__(self, c_in, c_out, config, T):
        super().__init__()
        self.res = ResidualBlockNoTCN(c_in, c_out, config, T, kernel_size=3)

    def forward(self, x, t, supports):
        return self.res(x, t, supports)


class UpBlockNoTCN(nn.Module):
    def __init__(self, c_in, c_out, config, T):
        super().__init__()
        self.res = ResidualBlockNoTCN(c_in + c_out, c_out, config, T, kernel_size=3)

    def forward(self, x, t, supports):
        return self.res(x, t, supports)


class MiddleBlockNoTCN(nn.Module):
    def __init__(self, c_in, config, T):
        super().__init__()
        self.res1 = ResidualBlockNoTCN(c_in, c_in, config, T, kernel_size=3)

    def forward(self, x, t, supports):
        return self.res1(x, t, supports)


class _UGnetNoTCNBase(nn.Module):
    def __init__(self, config, d_h, n_blocks, channel_multipliers, num_heads):
        super().__init__()
        self.config = config
        self.d_h = d_h
        self.supports_len = getattr(config, "supports_len", 2)
        self.T_h = config.T_h
        T = self.T_h
        self.n_blocks = n_blocks
        n_resolutions = len(channel_multipliers)
        self.T_levels = [T]

        down = []
        out_channels = in_channels = self.d_h
        for i in range(n_resolutions):
            out_channels = in_channels * channel_multipliers[i]
            for _ in range(self.n_blocks):
                down.append(DownBlockNoTCN(in_channels, out_channels, self, T))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
                T = T // 2
                self.T_levels.append(T)
        self.down = nn.ModuleList(down)
        self.middle = MiddleBlockNoTCN(out_channels, self, T)

        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(self.n_blocks):
                up.append(UpBlockNoTCN(in_channels, out_channels, self, T))
            out_channels = in_channels // channel_multipliers[i]
            up.append(UpBlockNoTCN(in_channels, out_channels, self, T))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))
                T = T * 2
                self.T_levels.append(T)
        self.up = nn.ModuleList(up)

        self.x_proj = nn.Linear(1, self.d_h)
        self.out = nn.Sequential(
            nn.Conv2d(self.d_h, 1, (1, 1)),
            nn.Linear(T, T),
        )

        self.senior_emb = nn.Embedding(config.num_c, self.d_h).to(config.device)
        self.region_emb = nn.Embedding(config.num_region, self.d_h).to(config.device)
        self.variable_emb = nn.Embedding(config.num_variable, self.d_h).to(config.device)
        nn.init.xavier_uniform_(self.senior_emb.weight)
        nn.init.xavier_uniform_(self.region_emb.weight)
        nn.init.xavier_uniform_(self.variable_emb.weight)

        emb_scale_cat = getattr(config, "emb_scale_cat", 2.0)
        emb_scale_reg = getattr(config, "emb_scale_reg", 0.7)
        emb_scale_var = getattr(config, "emb_scale_var", 0.7)
        self.emb_scale = nn.Parameter(
            torch.tensor([emb_scale_cat, emb_scale_reg, emb_scale_var], dtype=torch.float32)
        )
        rv_drop = getattr(config, "rv_drop", 0.2)
        self.rv_drop = nn.Dropout(rv_drop)
        self.fusion = nn.Conv2d(in_channels=self.d_h * 4, out_channels=self.d_h, kernel_size=(1, 1))
        self.gen_graph = GATLayer(self.d_h, self.d_h, config.T_h, num_heads=num_heads)

    def _apply_conditioning_ablation(self, category_emb=None, region_emb=None, variable_emb=None):
        cat_scale = 0.0 if getattr(self.config, "ablation_no_cat_cond", False) else 1.0
        reg_scale = 0.0 if getattr(self.config, "ablation_no_region_cond", False) else 1.0
        var_scale = 0.0 if getattr(self.config, "ablation_no_var_cond", False) else 1.0
        if category_emb is not None:
            category_emb = category_emb * cat_scale
        if region_emb is not None:
            region_emb = region_emb * reg_scale
        if variable_emb is not None:
            variable_emb = variable_emb * var_scale
        return category_emb, region_emb, variable_emb

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        category_emb = self.senior_emb(x[:, :, :, 1].long())
        region_emb = self.region_emb(x[:, :, :, 2].long())
        variable_emb = self.variable_emb(x[:, :, :, 3].long())
        category_emb = category_emb * self.emb_scale[0]
        region_emb = self.rv_drop(region_emb * self.emb_scale[1])
        variable_emb = self.rv_drop(variable_emb * self.emb_scale[2])
        category_emb, region_emb, variable_emb = self._apply_conditioning_ablation(category_emb, region_emb, variable_emb)

        input_emb = torch.cat((category_emb, region_emb, variable_emb), dim=-1)
        x = x[:, :, :, -1].unsqueeze(-1)
        x = self.x_proj(x)
        x = torch.cat((x, input_emb), dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = self.fusion(x)

        t = TimeEmbedding(t, self.d_h)

        if getattr(self.config, "ablation_nograph", False):
            B, _, group_len, _ = x.shape
            num_heads = getattr(self.gen_graph, "num_heads", 1)
            supports = torch.ones(B, num_heads, group_len, group_len, device=x.device, dtype=x.dtype) / group_len
        else:
            supports = self.gen_graph(x)

        h = [x]
        for m in self.down:
            x = m(x, t, supports)
            h.append(x)
        x = self.middle(x, t, supports)
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t, supports)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t, supports)

        e = self.out(x)
        e = e.squeeze(1).unsqueeze(-1)
        return e


class UGnetNoTCN(_UGnetNoTCNBase):
    def __init__(self, config):
        d_h = int(getattr(config, "d_h", 64))
        n_blocks = int(getattr(config, "n_blocks", 2))
        channel_multipliers = list(getattr(config, "channel_multipliers", [1, 2]))
        num_heads = int(getattr(config, "num_heads", 2))
        super().__init__(config, d_h, n_blocks, channel_multipliers, num_heads)


class TinyUGnetNoTCN(_UGnetNoTCNBase):
    def __init__(self, config):
        d_h = int(getattr(config, "d_h", 64))
        tiny_width_mult = float(getattr(config, "tiny_width_mult", 0.5))
        tiny_d_h = max(16, int(d_h * tiny_width_mult))
        tiny_n_blocks = int(getattr(config, "tiny_n_blocks", 1))
        tiny_n_resolutions = int(getattr(config, "tiny_n_resolutions", 2))
        tiny_num_heads = int(getattr(config, "tiny_num_heads", 1))
        channel_multipliers = list(getattr(config, "channel_multipliers", [1, 2]))
        tiny_n_resolutions = max(1, min(len(channel_multipliers), tiny_n_resolutions))
        tiny_channel_multipliers = channel_multipliers[:tiny_n_resolutions]
        super().__init__(config, tiny_d_h, tiny_n_blocks, tiny_channel_multipliers, tiny_num_heads)
