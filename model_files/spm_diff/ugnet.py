# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import math
import torch.nn.init as init
import torch.nn.functional as F

from .graph_algo import *

"""
Implementation of UGnet
Tcnblock: extract time feature
SpatialBlock: extract the spatial feature
"""

def TimeEmbedding(timesteps: torch.Tensor, embedding_dim: int):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb




class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, T, num_heads=2, dropout=0.6, alpha=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.out_features_per_head = out_features // num_heads
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        self.T = T

        # 查询、键的线性变换
        self.W_x = nn.Linear(in_features*T, self.out_features_per_head * num_heads, bias=False)
        self.K_x = nn.Linear(in_features*T, self.out_features_per_head * num_heads, bias=False)
        
        # 激活函数和正则化
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.out_features_per_head * num_heads)
        
        # 存储维度信息
        self.xx = in_features*T
        
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.xavier_uniform_(self.K_x.weight)

    def forward(self, x: torch.Tensor):
        '''
        x: (batch, dh, T, group_len)
        返回: attention (batch, num_heads, group_len, group_len)
        '''
        x = x.transpose(-1,-2)
        B, c_in, t, group_len = x.shape
        
        # 重塑输入以便于线性变换
        x = x.transpose(1, 3).reshape(B, group_len, -1)  # x: (B, group_len, t*d)

        # 计算查询和键
        q_x = self.layer_norm(self.W_x(x))
        q_x = q_x.view(B, group_len, self.num_heads, self.out_features_per_head)
        
        k_x = self.layer_norm(self.K_x(x))
        k_x = k_x.view(B, group_len, self.num_heads, self.out_features_per_head)
        
        # 转置以便矩阵乘法: (B, num_heads, group_len, out_features_per_head)
        q_x = q_x.transpose(1, 2)
        k_x = k_x.transpose(1, 2)
        
        # 计算注意力得分 - 使用矩阵乘法
        attention = torch.matmul(q_x, k_x.transpose(-2, -1))  # (B, num_heads, group_len, group_len)
        attention = attention / (self.out_features_per_head ** 0.5)  # 缩放
        
        # 应用LeakyReLU和softmax
        attention = self.leakyrelu(attention)
        attention = F.softmax(attention, dim=-1)  # (B, num_heads, group_len, group_len)
        attention = self.dropout(attention)
        
        return attention






class SpatialBlock(nn.Module):
    def __init__(self, ks, c_in, c_out, T):
        super(SpatialBlock, self).__init__()

        self.c_conv = nn.Conv2d(in_channels=c_in, out_channels=c_in * 2, kernel_size=(1, 1))
        self.o_conv = nn.Conv2d(in_channels=c_in * 2, out_channels=c_out, kernel_size=(1, 1))
        self.x_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))
        self.gate = nn.Sequential(
            nn.Conv2d(c_out * 2, c_out, kernel_size=1),
            nn.Sigmoid()
        )
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1))

        self.T = T
        self.reset_parameters()
        self.bn = nn.BatchNorm2d(c_out)

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, mode='fan_in', nonlinearity='relu')
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

# 正确的实现
    def forward(self, x, Lk):
        '''
        x: (B, c_in, T, group_len)
        Lk: (B, ks, group_len, group_len) - 图拉普拉斯或邻接矩阵
        
        return: (B, c_out, group_len, T)
        '''
        B, C, T, G = x.shape
        ks = Lk.shape[1]
        
        # 1. 重塑输入
        x_reshaped = x.permute(0, 2, 3, 1)  # [B, T, G, C]
        
        # 2. 图卷积 + 特征变换 (单步)
        x_gc = torch.einsum("btgc,bkgn,cok->btno", 
                            x_reshaped,  # [B, T, G, C]
                            Lk,          # [B, ks, G, G]
                            self.theta)  # [C, c_out, ks]
        
        # 3. 添加偏置

        # print(x_gc.shape, self.b.shape)
        x_gc = x_gc.permute(0, 3, 1, 2)  # [B, c_out, T, G]
        # print(x_gc.shape)
        x_gc = x_gc + self.b # 添加偏置
        
        # 4. 重塑为卷积格式
        
        
        # 5. 残差连接
        x_shortcut = self.x_conv(x)  # [B, c_out, T, G]
        
        # 6. 应用门控机制
        gate_input = torch.cat([x_gc, x_shortcut], dim=1)  # [B, c_out*2, T, G]
        gate = self.gate(gate_input)  # [B, c_out, T, G]
        
        # 7. 应用门控与残差连接
        out = gate * x_gc + (1 - gate) * x_shortcut
        
        # 8. 应用批归一化
        out = self.bn(out)
        
        # 9. 转置以匹配期望的输出形状
        out = out.permute(0, 1, 3, 2)  # [B, c_out, G, T]
        
        return out


class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :, : -self.chomp_size]


class TcnBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation_size=1, droupout=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.padding = (self.kernel_size - 1) * self.dilation_size

        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(3, self.kernel_size), padding=(1, self.padding), dilation=(1, self.dilation_size))

        self.chomp = Chomp(self.padding)
        self.drop =  nn.Dropout(droupout)

        self.net = nn.Sequential(self.conv, self.chomp, self.drop)

        self.shortcut = nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in != c_out else None


    def forward(self, x):
        # x: (B, C_in, group_len, T) -> (B, C_out, group_len, T)
        out = self.net(x)
        x_skip = x if self.shortcut is None else self.shortcut(x)

        return out + x_skip

class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, config, T, kernel_size=3):
        """
        :param c_in: in channels
        :param c_out: out channels
        :param kernel_size:
        TCN convolution
            input: (B, c_in, V, T)
            output:(B, c_out, V, T)
        """
        super().__init__()
        self.tcn1 = TcnBlock(c_in, c_out, kernel_size=kernel_size)
        self.tcn2 = TcnBlock(c_out, c_out, kernel_size=kernel_size)
        self.tcn3 = TcnBlock(c_in, c_out, kernel_size=kernel_size)
        self.shortcut = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, (1,1))
        self.t_conv = nn.Linear(config.d_h, c_out)
        self.spatial = SpatialBlock(config.supports_len, c_out, c_out, T)
        self.bn = nn.BatchNorm2d(c_out)
    def forward(self, x, t, A_hat):

        # x: (batch, c_in, group_len, T), return (B, c_out, group_len, T)
        h = self.tcn1(x)    #(B, C_out, group_len, T)

        t = self.t_conv(t)  #(B, C_out)
        t = t.unsqueeze(-1).unsqueeze(-1)  #(B, C_out, 1, 1)

        h += t  # (B, C_out, group_len, T)
        h = self.tcn2(h)    #(B, C_out, group_len, T)
        h = self.bn(h)  #(B, C_out, group_len, T)

        h = h.transpose(2,3) #(B, c_out, T, group_len)
        
        h = self.spatial(h, A_hat) # (B, c_out, group_len, T)
        return h + self.shortcut(x)

class DownBlock(nn.Module):
    def __init__(self, c_in, c_out, config, T):
        """
        :param c_in: in channels, out channels
        :param c_out:
        """
        super().__init__()
        self.res = ResidualBlock(c_in, c_out, config, T, kernel_size=3)

    def forward(self, x, t, supports):
        #x: (batch, c_in, group_len, T), return (B, c_out, group_len, T)

        x = self.res(x, t, supports)

        return x


class Downsample(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_in,  kernel_size= (1,3), stride=(1,2), padding=(0,1))
        self.conv2 = nn.Conv2d(c_in, c_in,  kernel_size= (1,3), stride=(1,2), padding=(0,1))
    def forward(self, x: torch.Tensor, t: torch.Tensor, supports):
        _ = t
        _ = supports
        x = self.conv(x)
        return x


class  UpBlock(nn.Module):
    def __init__(self, c_in, c_out, config, T):
        super().__init__()
        self.res = ResidualBlock(c_in + c_out, c_out, config, T, kernel_size=3)

    def forward(self, x, t, supports):

        x = self.res(x, t, supports)
        return x

class Upsample(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.ConvTranspose2d(c_in, c_in, (1, 4), (1, 2), (0, 1))
        self.conv2 = nn.ConvTranspose2d(c_in, c_in, (1, 4), (1, 2), (0, 1))
    def forward(self, x, t, supports):
        _ = t
        _ = supports
        x = self.conv(x)
        return x

class MiddleBlock(nn.Module):
    def __init__(self, c_in, config, T):
        super().__init__()
        self.res1 = ResidualBlock(c_in, c_in, config, T, kernel_size=3)
        # self.res2 = ResidualBlock(c_in, c_in, config, T, kernel_size=3)

    def forward(self, x, t, supports):
        x = self.res1(x, t, supports)

        # x,condition = self.res2(x, condition, t, supports)

        return x


class UGnet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.d_h = config.d_h
        self.T_h = config.T_h
        T = self.T_h
        # F = self.F = config.F
        
        self.n_blocks = config.n_blocks
        n_resolutions = len(config.channel_multipliers)
        self.T_levels = [T]   
        
        # 构建下采样部分
        down = []
        out_channels = in_channels = self.d_h
        for i in range(n_resolutions):
            out_channels = in_channels * config.channel_multipliers[i]
            for _ in range(self.n_blocks):
                down.append(DownBlock(in_channels, out_channels, config, T))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
                T = T // 2
                self.T_levels.append(T)
        self.down = nn.ModuleList(down)
        self.middle = MiddleBlock(out_channels, config, T)
        
        # 构建上采样部分
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(self.n_blocks):
                up.append(UpBlock(in_channels, out_channels, config, T))
            out_channels = in_channels // config.channel_multipliers[i]
            up.append(UpBlock(in_channels, out_channels, config, T))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))
                T = T * 2
                self.T_levels.append(T)
        self.up = nn.ModuleList(up)
        
        self.x_proj = nn.Linear(1, self.d_h)
        self.con_proj = nn.Linear(1, self.d_h)
        self.out = nn.Sequential(
            nn.Conv2d(self.d_h, 1, (1,1)),
            nn.Linear(T, T)
        )
        self.condition_out = nn.Sequential(
            nn.Conv2d(self.d_h, 1, (1,1)),
            nn.Linear(T, T)
        )
        # 使用 nn.Embedding 替换原来的 nn.Parameter
        self.senior_emb = nn.Embedding(config.num_c, self.d_h).to(config.device)
        self.region_emb = nn.Embedding(config.num_region, self.d_h).to(config.device)
        self.variable_emb = nn.Embedding(config.num_variable, self.d_h).to(config.device)
        # nn.init.xavier_uniform_(self.model_emb.weight)
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
        
        self.fusion = nn.Conv2d(in_channels=self.d_h*4, out_channels=self.d_h, kernel_size=(1,1))

        self.gen_graph = GATLayer(self.d_h, self.d_h, config.T_h)

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
        # x: batch, group_len, T, C

        # 利用 nn.Embedding 动态索引输入的 embedding
        # model_emb = self.model_emb(x[:, :, :, 0].long())
        category_emb = self.senior_emb(x[:, :, :, 1].long())
        region_emb = self.region_emb(x[:, :, :, 2].long())
        variable_emb = self.variable_emb(x[:, :, :, 3].long())
        category_emb = category_emb * self.emb_scale[0]
        region_emb = region_emb * self.emb_scale[1]
        variable_emb = variable_emb * self.emb_scale[2]
        region_emb = self.rv_drop(region_emb)
        variable_emb = self.rv_drop(variable_emb)
        category_emb, region_emb, variable_emb = self._apply_conditioning_ablation(category_emb, region_emb, variable_emb)
        input_emb = torch.cat((category_emb, region_emb, variable_emb), dim=-1)

        
        x = x[:, :, :, -1].unsqueeze(-1)     #x: (batch, group_len, T, 1)
        x = self.x_proj(x)
        x = torch.cat((x, input_emb), dim=-1)   #x: (batch, group_len, T, dh*5)

        x = x.permute(0, 3, 1, 2)   #x: (batch, dh*5, group_len, T)

        x = self.fusion(x)  #x: (batch, dh, group_len, T)

        t = TimeEmbedding(t, self.d_h)  #t: batch, dh

        if getattr(self.config, "ablation_nograph", False):
            B, _, group_len, _ = x.shape
            num_heads = getattr(self.gen_graph, "num_heads", 2)
            supports = torch.ones(B, num_heads, group_len, group_len, device=x.device, dtype=x.dtype) / group_len
        else:
            supports = self.gen_graph(x)  # supports: batch, num_head, group_len, group_len

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

        # (B, c_out, group_len, T)
        e = self.out(x) #(B,1,group_len,T)
        e = e.squeeze(1).unsqueeze(-1)  # (B, group_len, T, 1)
        return e


class TinyUGnet(UGnet):
    """Tiny variant of UGnet with the same forward interface and output shape."""

    def __init__(self, config) -> None:
        tiny_cfg = config
        d_h = int(getattr(config, "d_h", 64))
        tiny_width_mult = float(getattr(config, "tiny_width_mult", 0.5))
        tiny_d_h = max(16, int(d_h * tiny_width_mult))
        tiny_n_blocks = int(getattr(config, "tiny_n_blocks", 1))
        tiny_n_resolutions = int(getattr(config, "tiny_n_resolutions", 2))
        tiny_num_heads = int(getattr(config, "tiny_num_heads", 1))

        channel_multipliers = list(getattr(config, "channel_multipliers", [1, 2]))
        tiny_n_resolutions = max(1, min(len(channel_multipliers), tiny_n_resolutions))
        tiny_channel_multipliers = channel_multipliers[:tiny_n_resolutions]

        class _TinyProxy:
            pass

        proxy = _TinyProxy()
        proxy.__dict__.update(tiny_cfg.__dict__)
        proxy.d_h = tiny_d_h
        proxy.n_blocks = tiny_n_blocks
        proxy.channel_multipliers = tiny_channel_multipliers

        super().__init__(proxy)
        self.config = config
        self.gen_graph = GATLayer(self.d_h, self.d_h, config.T_h, num_heads=tiny_num_heads)
