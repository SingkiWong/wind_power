"""
Lightweight Models Module for Wind Power Prediction
轻量化模型模块

实现：
1. Tiny Time Mixers (TTM) - IBM的百万参数奇迹
2. PatchTSMixer - 抗分布偏移的鲁棒设计
3. TSMixer - 全MLP架构

核心思想：特定领域的"小模型"可以战胜通用的"大模型"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Optional, Dict


class PatchEmbedding(nn.Module):
    """Patch嵌入模块"""
    
    def __init__(
        self,
        input_len: int,
        patch_size: int,
        num_features: int,
        d_model: int,
        stride: Optional[int] = None
    ):
        super().__init__()
        self.input_len = input_len
        self.patch_size = patch_size
        self.num_features = num_features
        self.stride = stride or patch_size
        
        self.num_patches = (input_len - patch_size) // self.stride + 1
        self.patch_proj = nn.Linear(patch_size * num_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            end = start + self.patch_size
            patch = x[:, start:end, :]
            patches.append(patch.flatten(1))
        patches = torch.stack(patches, dim=1)
        patches = self.patch_proj(patches)
        return patches + self.pos_embed


class Transpose(nn.Module):
    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.dim1, self.dim2 = dim1, dim2
    
    def forward(self, x): return x.transpose(self.dim1, self.dim2)


class MLPMixer(nn.Module):
    """MLP-Mixer层"""
    
    def __init__(self, num_patches: int, d_model: int, token_dim: int, channel_dim: int, dropout: float = 0.1):
        super().__init__()
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(d_model), Transpose(1, 2),
            nn.Linear(num_patches, token_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(token_dim, num_patches), nn.Dropout(dropout), Transpose(1, 2)
        )
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, channel_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(channel_dim, d_model), nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.token_mixer(x)
        return x + self.channel_mixer(x)


class TSMixer(nn.Module):
    """TSMixer: 轻量级时间序列混合器"""
    
    def __init__(self, input_len: int, output_len: int, num_features: int,
                 patch_size: int = 8, d_model: int = 64, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.output_len, self.num_features = output_len, num_features
        self.patch_embed = PatchEmbedding(input_len, patch_size, num_features, d_model)
        num_patches = self.patch_embed.num_patches
        self.mixers = nn.ModuleList([MLPMixer(num_patches, d_model, num_patches*2, d_model*4, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(num_patches * d_model, output_len * num_features)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        for mixer in self.mixers: x = mixer(x)
        x = self.head(self.norm(x).flatten(1))
        return x.view(batch_size, self.output_len, self.num_features)


class TinyTimeMixer(nn.Module):
    """TTM: IBM的轻量化时序预测模型"""
    
    def __init__(self, input_len: int = 512, output_len: int = 96, num_features: int = 1,
                 d_model: int = 64, n_layers: int = 3, patch_sizes: List[int] = [8, 16, 32],
                 dropout: float = 0.1, channel_independence: bool = True):
        super().__init__()
        self.output_len, self.num_features = output_len, num_features
        self.channel_independence = channel_independence
        
        self.patch_embeds = nn.ModuleList()
        self.total_patches = 0
        for ps in patch_sizes:
            embed = PatchEmbedding(input_len, ps, 1 if channel_independence else num_features, d_model)
            self.patch_embeds.append(embed)
            self.total_patches += embed.num_patches
        
        self.mixers = nn.ModuleList([MLPMixer(self.total_patches, d_model, self.total_patches*2, d_model*4, dropout) for _ in range(n_layers)])
        if channel_independence: self.channel_fuse = nn.Linear(num_features, 1)
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(self.total_patches * d_model, output_len)
        if num_features > 1: self.feature_expand = nn.Linear(output_len, output_len * num_features)
        
    def forward(self, x):
        batch_size = x.shape[0]
        if self.channel_independence:
            channel_outputs = []
            for c in range(self.num_features):
                patches = torch.cat([embed(x[:,:,c:c+1]) for embed in self.patch_embeds], dim=1)
                for mixer in self.mixers: patches = mixer(patches)
                channel_outputs.append(patches)
            x = self.channel_fuse(torch.stack(channel_outputs, dim=-1)).squeeze(-1)
        else:
            x = torch.cat([embed(x) for embed in self.patch_embeds], dim=1)
            for mixer in self.mixers: x = mixer(x)
        
        x = self.output_proj(self.norm(x).flatten(1))
        if self.num_features > 1:
            x = self.feature_expand(x).view(batch_size, self.output_len, self.num_features)
        else: x = x.unsqueeze(-1)
        return x


class RevIN(nn.Module):
    """可逆实例归一化"""
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps, self.affine = eps, affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x, mode):
        if mode == 'norm':
            self._mean = x.mean(1, keepdim=True)
            self._std = x.std(1, keepdim=True) + self.eps
            x = (x - self._mean) / self._std
            if self.affine: x = x * self.weight + self.bias
        else:
            if self.affine: x = (x - self.bias) / (self.weight + self.eps)
            x = x * self._std + self._mean
        return x


class PatchTSMixer(nn.Module):
    """PatchTSMixer: 抗分布偏移"""
    
    def __init__(self, input_len: int, output_len: int, num_features: int,
                 patch_size: int = 16, d_model: int = 64, n_layers: int = 4,
                 dropout: float = 0.1, use_revin: bool = True):
        super().__init__()
        self.output_len, self.num_features, self.use_revin = output_len, num_features, use_revin
        if use_revin: self.revin = RevIN(num_features)
        self.patch_embeds = nn.ModuleList([PatchEmbedding(input_len, patch_size, 1, d_model) for _ in range(num_features)])
        num_patches = self.patch_embeds[0].num_patches
        self.mixers = nn.ModuleList([MLPMixer(num_patches, d_model, num_patches*2, d_model*4, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleList([nn.Linear(num_patches * d_model, output_len) for _ in range(num_features)])
    
    def forward(self, x):
        if self.use_revin: x = self.revin(x, 'norm')
        outputs = []
        for c in range(self.num_features):
            patches = self.patch_embeds[c](x[:,:,c:c+1])
            for mixer in self.mixers: patches = mixer(patches)
            outputs.append(self.heads[c](self.norm(patches).flatten(1)))
        x = torch.stack(outputs, dim=-1)
        if self.use_revin: x = self.revin(x, 'denorm')
        return x


if __name__ == "__main__":
    print("=" * 60)
    print("Lightweight Models Module Test")
    print("=" * 60)
    
    models = {
        'TSMixer': TSMixer(96, 24, 5, 8, 64, 4),
        'TinyTimeMixer': TinyTimeMixer(512, 96, 5, 64, 3, [8, 16, 32]),
        'PatchTSMixer': PatchTSMixer(96, 24, 5, 16, 64, 4)
    }
    
    for name, model in models.items():
        if name == 'TinyTimeMixer':
            x = torch.randn(8, 512, 5)
        else:
            x = torch.randn(8, 96, 5)
        y = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {x.shape} -> {y.shape}, Params: {params:,}")
    
    print("=" * 60)
