"""
Mamba Module for Wind Power Prediction
选择性状态空间模型 (Selective State Space Model)

核心优势：
1. O(L)线性复杂度，突破Transformer的O(L²)瓶颈
2. 输入依赖的参数化，动态选择"记忆"和"遗忘"
3. 硬件感知优化，极低推理延迟
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat


class SelectiveSSM(nn.Module):
    """
    选择性状态空间模型核心
    
    状态空间方程：
        h'(t) = Ah(t) + Bx(t)
        y(t) = Ch(t) + Dx(t)
    
    Mamba的关键创新：A, B, C参数是输入依赖的
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # 输入投影: x -> (z, x, B, C, dt)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # 1D卷积用于局部特征提取
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        
        # SSM参数投影
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        
        # dt投影
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # 初始化dt偏置以获得良好的初始时间步长
        dt_init_std = dt_scale / math.sqrt(self.d_inner)
        if dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # A矩阵（状态转移）- 使用对数参数化保证稳定性
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = repeat(A, 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D矩阵（直接跳跃连接）
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            y: [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape
        
        # 输入投影
        xz = self.in_proj(x)  # [batch, seq_len, d_inner * 2]
        x_proj, z = xz.chunk(2, dim=-1)  # 各 [batch, seq_len, d_inner]
        
        # 1D卷积
        x_conv = rearrange(x_proj, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = F.silu(x_conv)
        
        # SSM参数（输入依赖）
        ssm_params = self.x_proj(x_conv)  # [batch, seq_len, d_state*2 + 1]
        B = ssm_params[..., :self.d_state]
        C = ssm_params[..., self.d_state:2*self.d_state]
        dt = F.softplus(ssm_params[..., -1:])  # 确保dt > 0
        
        # 扩展dt到d_inner维度
        dt = self.dt_proj(dt)  # [batch, seq_len, d_inner]
        
        # 获取A矩阵
        A = -torch.exp(self.A_log)  # [d_inner, d_state]
        
        # 离散化SSM（使用零阶保持）
        # dA = exp(A * dt)
        # dB = B * dt
        
        # 选择性扫描
        y = self._selective_scan(x_conv, dt, A, B, C)
        
        # 跳跃连接
        y = y + self.D * x_conv
        
        # 门控
        y = y * F.silu(z)
        
        # 输出投影
        return self.out_proj(y)
    
    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        选择性扫描算法
        
        这是Mamba的核心 - 使用递归方式高效处理序列
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # 初始化隐藏状态
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        for t in range(seq_len):
            # 当前输入
            x_t = x[:, t, :]  # [batch, d_inner]
            dt_t = dt[:, t, :]  # [batch, d_inner]
            B_t = B[:, t, :]  # [batch, d_state]
            C_t = C[:, t, :]  # [batch, d_state]
            
            # 离散化
            dA = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))  # [batch, d_inner, d_state]
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # [batch, d_inner, d_state]
            
            # 状态更新: h = dA * h + dB * x
            h = dA * h + dB * x_t.unsqueeze(-1)
            
            # 输出: y = C * h
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)  # [batch, d_inner]
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)  # [batch, seq_len, d_inner]


class MambaBlock(nn.Module):
    """
    Mamba块：包含SSM层和残差连接
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """残差连接"""
        return x + self.dropout(self.ssm(self.norm(x)))


class BiMamba(nn.Module):
    """
    双向Mamba：同时利用过去和未来信息
    
    在离线训练和数据补全任务中特别有用
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2
    ):
        super().__init__()
        self.forward_ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.backward_ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.merge = nn.Linear(d_model * 2, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        双向扫描
        """
        # 前向
        y_forward = self.forward_ssm(x)
        
        # 后向（翻转序列）
        x_backward = torch.flip(x, dims=[1])
        y_backward = self.backward_ssm(x_backward)
        y_backward = torch.flip(y_backward, dims=[1])
        
        # 合并
        y = torch.cat([y_forward, y_backward], dim=-1)
        return self.merge(y)


class WindMambaformer(nn.Module):
    """
    Wind-Mambaformer: 专为风电预测设计的混合架构
    
    结合了：
    1. Mamba的线性复杂度序列建模
    2. Flow-Attention的局部特征增强
    3. 多尺度时间分解
    
    参考论文: Wind-Mambaformer (2025)
    """
    
    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int,
        d_model: int = 64,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_flow_attention: bool = True
    ):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.d_model = d_model
        
        # 输入嵌入
        self.input_embedding = nn.Linear(num_features, d_model)
        
        # 位置编码
        self.pos_encoding = self._create_pos_encoding(input_len, d_model)
        
        # Mamba编码器层
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        
        # 可选的Flow-Attention层用于局部特征增强
        self.use_flow_attention = use_flow_attention
        if use_flow_attention:
            self.flow_attention = FlowAttention(d_model, n_heads=4)
        
        # 输出投影
        self.output_proj = nn.Linear(d_model, num_features)
        
        # 序列长度适配
        self.seq_adapter = nn.Linear(input_len, output_len)
    
    def _create_pos_encoding(self, max_len: int, d_model: int) -> nn.Parameter:
        """创建正弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_len, num_features]
            
        Returns:
            [batch, output_len, num_features]
        """
        batch_size = x.shape[0]
        
        # 输入嵌入
        x = self.input_embedding(x)  # [batch, input_len, d_model]
        
        # 添加位置编码
        x = x + self.pos_encoding[:, :self.input_len, :]
        
        # Mamba层
        for mamba in self.mamba_layers:
            x = mamba(x)
        
        # Flow-Attention增强（可选）
        if self.use_flow_attention:
            x = x + self.flow_attention(x)
        
        # 序列长度适配: [batch, input_len, d_model] -> [batch, output_len, d_model]
        x = rearrange(x, 'b l d -> b d l')
        x = self.seq_adapter(x)
        x = rearrange(x, 'b d l -> b l d')
        
        # 输出投影
        return self.output_proj(x)


class FlowAttention(nn.Module):
    """
    Flow-Attention: 降低复杂度的注意力变体
    
    通过流机制（flow mechanism）减少计算量，
    同时保持对局部模式的敏感性
    """
    
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape
        
        # 投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 多头切分
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.n_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.n_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.n_heads)
        
        # Flow-Attention: 使用softmax归一化的竞争机制
        # 源流 (Source flow)
        src_flow = F.softmax(q, dim=-1)
        # 汇流 (Sink flow)
        sink_flow = F.softmax(k, dim=-2)
        
        # 归一化竞争
        flow = src_flow * sink_flow
        
        # 加权聚合
        attn_output = torch.einsum('bhld,bhle->bhde', flow, v)
        attn_output = torch.einsum('bhld,bhde->bhle', q, attn_output)
        
        # 合并头
        attn_output = rearrange(attn_output, 'b h l d -> b l (h d)')
        
        return self.dropout(self.out_proj(attn_output))


class SiMBA(nn.Module):
    """
    SiMBA: Simplified Mamba-based Architecture
    
    特点：
    1. 使用Mamba处理时间依赖
    2. 使用EinFFT处理通道间依赖
    3. 特别适合多变量风电预测（风速、风向、温度、气压等）
    """
    
    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int,
        d_model: int = 64,
        n_layers: int = 3,
        d_state: int = 16
    ):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        
        # 输入嵌入
        self.input_embed = nn.Linear(num_features, d_model)
        
        # Mamba层（处理时间依赖）
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state)
            for _ in range(n_layers)
        ])
        
        # EinFFT层（处理通道依赖）
        self.einfft = EinFFT(d_model)
        
        # 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_features)
        )
        
        # 长度适配
        self.len_adapter = nn.Linear(input_len, output_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_len, num_features]
        """
        # 嵌入
        x = self.input_embed(x)
        
        # Mamba处理时间依赖
        for mamba in self.mamba_layers:
            x = mamba(x)
        
        # EinFFT处理通道依赖
        x = self.einfft(x)
        
        # 长度适配
        x = rearrange(x, 'b l d -> b d l')
        x = self.len_adapter(x)
        x = rearrange(x, 'b d l -> b l d')
        
        # 输出
        return self.output_proj(x)


class EinFFT(nn.Module):
    """
    Einstein FFT: 高效的通道混合模块
    
    利用傅里叶变换在频域进行高效的特征混合
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # 频域权重
        self.freq_weight = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        residual = x
        
        # FFT变换到频域
        x_freq = torch.fft.rfft(x, dim=1)
        
        # 频域加权（Einstein求和）
        x_freq = torch.einsum('bfd,de->bfe', x_freq, self.freq_weight.to(x_freq.dtype))
        
        # iFFT回到时域
        x = torch.fft.irfft(x_freq, n=x.shape[1], dim=1)
        
        # 残差连接和归一化
        return self.norm(x + residual)


class MambaKANHybrid(nn.Module):
    """
    Mamba-KAN混合架构
    
    结合两者优势：
    - Mamba: 高效长序列建模
    - KAN: 非线性表达和可解释性
    """
    
    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int,
        d_model: int = 64,
        n_mamba_layers: int = 3,
        kan_hidden: int = 32,
        num_knots: int = 8
    ):
        super().__init__()
        
        # 导入KAN模块
        from kan_module import KAN
        
        # 输入嵌入
        self.input_embed = nn.Linear(num_features, d_model)
        
        # Mamba骨干（负责长序列特征提取）
        self.mamba_backbone = nn.Sequential(*[
            MambaBlock(d_model)
            for _ in range(n_mamba_layers)
        ])
        
        # KAN头（负责非线性输出和可解释性）
        self.kan_head = KAN(
            [d_model, kan_hidden, num_features],
            num_knots=num_knots
        )
        
        # 长度适配
        self.len_adapter = nn.Linear(input_len, output_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_len, num_features]
        """
        batch_size, seq_len, _ = x.shape
        
        # 嵌入
        x = self.input_embed(x)
        
        # Mamba处理
        x = self.mamba_backbone(x)
        
        # 长度适配
        x = rearrange(x, 'b l d -> b d l')
        x = self.len_adapter(x)
        x = rearrange(x, 'b d l -> b l d')
        
        # KAN输出（逐时间步）
        batch_size, out_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)
        out_flat = self.kan_head(x_flat)
        
        return out_flat.reshape(batch_size, out_len, -1)


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("Mamba Module for Wind Power Prediction")
    print("=" * 60)
    
    # 测试SelectiveSSM
    print("\n1. Testing Selective SSM:")
    ssm = SelectiveSSM(d_model=64, d_state=16)
    x = torch.randn(8, 96, 64)
    y = ssm(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    
    # 测试Wind-Mambaformer
    print("\n2. Testing Wind-Mambaformer:")
    model = WindMambaformer(
        input_len=96,
        output_len=24,
        num_features=5,
        d_model=64,
        n_layers=4
    )
    x = torch.randn(16, 96, 5)
    y = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,}")
    
    # 测试SiMBA
    print("\n3. Testing SiMBA:")
    simba = SiMBA(
        input_len=96,
        output_len=24,
        num_features=5,
        d_model=64
    )
    y = simba(x)
    print(f"   Output shape: {y.shape}")
    
    # 测试BiMamba
    print("\n4. Testing Bi-Mamba:")
    bi_mamba = BiMamba(d_model=64)
    x = torch.randn(8, 96, 64)
    y = bi_mamba(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    
    print("\n" + "=" * 60)
    print("Mamba Module Test Complete!")
    print("=" * 60)
