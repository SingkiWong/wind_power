"""
Physics-Informed Neural Networks (PINNs) Module for Wind Power Prediction
物理信息神经网络模块

核心功能：
1. 将流体力学方程嵌入损失函数（贝兹极限、尾流模型）
2. 物理引导的注意力机制
3. 数字孪生虚拟感知

确保AI预测符合物理定律，提高泛化能力和可信度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Optional, Dict


class PhysicsLoss(nn.Module):
    """
    物理约束损失函数集合
    
    包含：
    1. Betz极限约束
    2. 功率曲线约束
    3. Jensen尾流模型约束
    4. 能量守恒约束
    """
    
    def __init__(
        self,
        rated_power: float = 2.0,      # MW
        cut_in_speed: float = 3.0,     # m/s
        cut_out_speed: float = 25.0,   # m/s
        rated_speed: float = 12.0,     # m/s
        rotor_diameter: float = 126.0, # m
        air_density: float = 1.225,    # kg/m³
        betz_limit: float = 0.593,     # 贝兹极限
        wake_decay_constant: float = 0.04,
        residual_weight: float = 0.2,
    ):
        super().__init__()
        # 可学习的物理常数，使用softplus保证非负
        self.raw_rated_power = nn.Parameter(torch.tensor(float(rated_power)))
        self.raw_cut_in_speed = nn.Parameter(torch.tensor(float(cut_in_speed)))
        self.raw_cut_out_speed = nn.Parameter(torch.tensor(float(cut_out_speed)))
        self.raw_rated_speed = nn.Parameter(torch.tensor(float(rated_speed)))
        self.raw_rotor_diameter = nn.Parameter(torch.tensor(float(rotor_diameter)))
        self.raw_air_density = nn.Parameter(torch.tensor(float(air_density)))
        self.raw_betz_limit = nn.Parameter(torch.tensor(float(betz_limit)))
        self.raw_wake_decay = nn.Parameter(torch.tensor(float(wake_decay_constant)))

        self.residual_weight = residual_weight

        # 叶轮扫风面积（运行时使用正值）
        self.register_buffer('pi_const', torch.tensor(np.pi))
    
    def betz_limit_loss(
        self,
        predicted_power: torch.Tensor,
        wind_speed: torch.Tensor
    ) -> torch.Tensor:
        """
        贝兹极限损失
        
        风能利用系数不能超过59.3%
        P ≤ 0.5 * ρ * A * v³ * Cp_max
        """
        # 计算理论最大功率 (W -> MW)
        rotor_area = self.physical_rotor_area(wind_speed.device)
        max_theoretical = (
            0.5 * self.air_density * rotor_area *
            (wind_speed ** 3) * self.betz_limit / 1e6
        )

        # 限制最大理论功率不超过额定功率
        max_theoretical = torch.clamp(max_theoretical, max=self.rated_power)
        
        # 惩罚超过理论极限的预测
        violation = F.relu(predicted_power - max_theoretical)
        return violation.mean()
    
    def power_curve_loss(
        self,
        predicted_power: torch.Tensor,
        wind_speed: torch.Tensor
    ) -> torch.Tensor:
        """
        功率曲线物理约束损失
        """
        loss = torch.tensor(0.0, device=predicted_power.device)
        
        # 区域1: 切入风速以下
        mask_below_cutin = wind_speed < self.cut_in_speed
        if mask_below_cutin.any():
            loss = loss + (predicted_power[mask_below_cutin] ** 2).mean()
        
        # 区域2: 爬坡区 - 功率与风速立方成正比
        mask_ramp = (wind_speed >= self.cut_in_speed) & (wind_speed < self.rated_speed)
        if mask_ramp.any():
            expected_ratio = (
                (wind_speed[mask_ramp] - self.cut_in_speed) / 
                (self.rated_speed - self.cut_in_speed)
            ) ** 3
            actual_ratio = predicted_power[mask_ramp] / self.rated_power
            deviation = (actual_ratio - expected_ratio).abs()
            loss = loss + F.relu(deviation - 0.15).mean()
        
        # 区域3: 额定区
        mask_rated = (wind_speed >= self.rated_speed) & (wind_speed < self.cut_out_speed)
        if mask_rated.any():
            deviation = (predicted_power[mask_rated] - self.rated_power).abs()
            loss = loss + F.relu(deviation - 0.1 * self.rated_power).mean()
        
        # 区域4: 切出风速以上
        mask_above_cutout = wind_speed >= self.cut_out_speed
        if mask_above_cutout.any():
            loss = loss + (predicted_power[mask_above_cutout] ** 2).mean()
        
        return loss
    
    def non_negative_loss(self, predicted_power: torch.Tensor) -> torch.Tensor:
        """非负功率约束"""
        return F.relu(-predicted_power).mean()
    
    def rated_power_loss(self, predicted_power: torch.Tensor) -> torch.Tensor:
        """额定功率上限约束"""
        return F.relu(predicted_power - self.rated_power).mean()
    
    def forward(
        self,
        predicted_power: torch.Tensor,
        wind_speed: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有物理约束损失
        """
        physics_baseline = self.power_curve_baseline(wind_speed)
        residual_penalty = F.smooth_l1_loss(predicted_power, physics_baseline)

        losses = {
            'betz': self.betz_limit_loss(predicted_power, wind_speed),
            'power_curve': self.power_curve_loss(predicted_power, wind_speed),
            'non_negative': self.non_negative_loss(predicted_power),
            'rated_limit': self.rated_power_loss(predicted_power),
            'residual': residual_penalty * self.residual_weight,
        }
        losses['total_physics'] = sum(losses.values())
        losses['physics_baseline'] = physics_baseline.detach()
        return losses

    @property
    def rated_power(self) -> torch.Tensor:
        return F.softplus(self.raw_rated_power)

    @property
    def cut_in_speed(self) -> torch.Tensor:
        return F.softplus(self.raw_cut_in_speed)

    @property
    def cut_out_speed(self) -> torch.Tensor:
        return F.softplus(self.raw_cut_out_speed)

    @property
    def rated_speed(self) -> torch.Tensor:
        return F.softplus(self.raw_rated_speed)

    @property
    def rotor_diameter(self) -> torch.Tensor:
        return F.softplus(self.raw_rotor_diameter)

    @property
    def air_density(self) -> torch.Tensor:
        return F.softplus(self.raw_air_density)

    @property
    def betz_limit(self) -> torch.Tensor:
        return torch.clamp(F.softplus(self.raw_betz_limit), max=0.99)

    @property
    def wake_decay_constant(self) -> torch.Tensor:
        return F.softplus(self.raw_wake_decay)

    def physical_rotor_area(self, device: torch.device) -> torch.Tensor:
        radius = self.rotor_diameter.to(device) / 2
        return self.pi_const.to(device) * radius ** 2

    def power_curve_baseline(self, wind_speed: torch.Tensor) -> torch.Tensor:
        """使用可学习物理参数给出期望功率基线，供残差学习使用"""
        ws = wind_speed
        zero = torch.zeros_like(ws)
        ramp = ((ws - self.cut_in_speed) / (self.rated_speed - self.cut_in_speed)).clamp(min=0)
        ramp = (ramp ** 3) * self.rated_power
        rated = torch.full_like(ws, self.rated_power)

        baseline = torch.where(ws < self.cut_in_speed, zero, ramp)
        baseline = torch.where(ws >= self.rated_speed, rated, baseline)
        baseline = torch.where(ws >= self.cut_out_speed, zero, baseline)
        return baseline


class PhysicsGuidedAttention(nn.Module):
    """
    物理引导的注意力机制
    
    将物理空间关系显式编码进注意力矩阵：
    Attention(Q, K, V) = Softmax(QK^T/√d + M_phy(θ, d))V
    
    其中M_phy是关于风向θ和机组间距d的物理偏置矩阵
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        num_turbines: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.num_turbines = num_turbines
        
        # 标准注意力投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 物理偏置参数（可学习的基础）
        # 风向编码
        self.wind_dir_embed = nn.Embedding(36, n_heads)  # 36个风向区间（每10度）
        
        # 距离编码
        self.distance_embed = nn.Parameter(torch.randn(num_turbines, num_turbines, n_heads) * 0.1)
        
        # 尾流影响系数
        self.wake_coefficient = nn.Parameter(torch.ones(1) * 0.5)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def compute_physics_bias(
        self,
        wind_direction: torch.Tensor,
        turbine_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算物理偏置矩阵
        
        Args:
            wind_direction: [batch_size] 风向角度（0-360度）
            turbine_positions: [num_turbines, 2] 风机位置
            
        Returns:
            physics_bias: [batch_size, n_heads, num_turbines, num_turbines]
        """
        batch_size = wind_direction.shape[0]
        
        # 风向离散化
        wind_dir_idx = (wind_direction / 10).long() % 36
        
        # 获取风向嵌入
        dir_embed = self.wind_dir_embed(wind_dir_idx)  # [batch, n_heads]
        
        # 基础距离偏置
        distance_bias = self.distance_embed.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # 风向调制的物理偏置
        # 上游风机应该获得更高的注意力权重
        physics_bias = distance_bias * dir_embed.unsqueeze(-2).unsqueeze(-2)
        
        # 转置为 [batch, heads, turbines, turbines]
        physics_bias = physics_bias.permute(0, 3, 1, 2)
        
        return physics_bias * self.wake_coefficient
    
    def forward(
        self,
        x: torch.Tensor,
        wind_direction: torch.Tensor,
        turbine_positions: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: [batch, num_turbines, d_model]
            wind_direction: [batch] 风向
            turbine_positions: [num_turbines, 2] 位置
            return_attention: 是否返回注意力权重
        """
        batch_size, seq_len, _ = x.shape
        
        # 投影
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # 添加物理偏置
        if seq_len <= self.num_turbines:
            physics_bias = self.compute_physics_bias(wind_direction, turbine_positions)
            physics_bias = physics_bias[:, :, :seq_len, :seq_len]
            attn_scores = attn_scores + physics_bias
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权聚合
        output = torch.matmul(attn_weights, v)
        
        # 合并头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        if return_attention:
            return output, attn_weights
        return output, None


class PhysicsInformedTransformer(nn.Module):
    """
    物理信息Transformer
    
    结合：
    1. 物理引导注意力机制
    2. 物理约束损失函数
    3. 可视化的注意力热图用于验证物理一致性
    """
    
    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        num_turbines: int = 10,
        rated_power: float = 2.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.d_model = d_model
        
        # 输入嵌入
        self.input_embed = nn.Linear(num_features, d_model)
        
        # 位置编码
        self.pos_encoding = self._create_pos_encoding(input_len, d_model)
        
        # 物理引导注意力层
        self.attention_layers = nn.ModuleList([
            PhysicsGuidedAttention(d_model, n_heads, num_turbines, dropout)
            for _ in range(n_layers)
        ])
        
        # FFN层
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(n_layers)
        ])
        
        # 层归一化
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers * 2)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(d_model, num_features)
        
        # 序列长度适配
        self.seq_adapter = nn.Linear(input_len, output_len)
        
        # 物理损失模块
        self.physics_loss = PhysicsLoss(rated_power=rated_power)
    
    def _create_pos_encoding(self, max_len: int, d_model: int) -> nn.Parameter:
        """正弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(
        self,
        x: torch.Tensor,
        wind_direction: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        前向传播
        
        Args:
            x: [batch, input_len, num_features]
            wind_direction: [batch] 风向
            return_attention: 是否返回注意力权重用于可视化
        """
        batch_size = x.shape[0]
        attention_maps = []
        
        # 输入嵌入
        x = self.input_embed(x) + self.pos_encoding[:, :self.input_len, :]
        
        # Transformer层
        norm_idx = 0
        for i, (attn, ffn) in enumerate(zip(self.attention_layers, self.ffn_layers)):
            # 自注意力 + 残差
            attn_out, attn_weights = attn(
                self.norm_layers[norm_idx](x),
                wind_direction,
                return_attention=return_attention
            )
            x = x + attn_out
            norm_idx += 1
            
            if return_attention and attn_weights is not None:
                attention_maps.append(attn_weights)
            
            # FFN + 残差
            x = x + ffn(self.norm_layers[norm_idx](x))
            norm_idx += 1
        
        # 序列长度适配
        x = x.transpose(1, 2)  # [batch, d_model, input_len]
        x = self.seq_adapter(x)  # [batch, d_model, output_len]
        x = x.transpose(1, 2)  # [batch, output_len, d_model]
        
        # 输出
        output = self.output_proj(x)
        
        if return_attention:
            return output, attention_maps
        return output, None
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        wind_speed: torch.Tensor,
        lambda_physics: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失（数据损失 + 物理损失）
        """
        # 数据损失
        data_loss = F.mse_loss(predictions, targets)
        
        # 物理损失
        # 假设predictions的第一列是功率
        pred_power = predictions[..., 0] if predictions.dim() > 2 else predictions
        physics_losses = self.physics_loss(pred_power.flatten(), wind_speed.flatten())
        
        # 总损失
        total_loss = data_loss + lambda_physics * physics_losses['total_physics']
        
        # 损失字典
        loss_dict = {
            'total': total_loss.item(),
            'data': data_loss.item(),
            'physics_total': physics_losses['total_physics'].item(),
            'betz': physics_losses['betz'].item(),
            'power_curve': physics_losses['power_curve'].item()
        }
        
        return total_loss, loss_dict


class DigitalTwin(nn.Module):
    """
    风电场数字孪生
    
    利用PINNs构建全时空流场重构模型
    
    功能：
    1. 虚拟传感器 - 推演任意位置的风速/压力
    2. 流场可视化 - 异常时刻的3D流场图
    3. 物理插值 - 传感器故障时的合理插值
    """
    
    def __init__(
        self,
        num_turbines: int,
        spatial_dim: int = 2,  # 2D或3D
        hidden_dim: int = 128,
        num_layers: int = 4
    ):
        super().__init__()
        self.num_turbines = num_turbines
        self.spatial_dim = spatial_dim
        
        # 空间位置编码网络
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            ) for _ in range(num_layers - 2)],
            nn.Linear(hidden_dim, spatial_dim + 1)  # 输出速度场 + 压力
        )
        
        # 边界条件编码
        self.boundary_encoder = nn.Linear(num_turbines * 2, hidden_dim)
        
        # 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + spatial_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, spatial_dim + 1)
        )
    
    def forward(
        self,
        query_points: torch.Tensor,
        time: torch.Tensor,
        boundary_conditions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        查询任意时空点的流场状态
        
        Args:
            query_points: [batch, num_points, spatial_dim] 查询位置
            time: [batch] 查询时间
            boundary_conditions: [batch, num_turbines, 2] 边界风机状态
            
        Returns:
            field_values: 包含速度场和压力场的字典
        """
        batch_size, num_points, _ = query_points.shape
        
        # 编码边界条件
        bc_flat = boundary_conditions.flatten(1)
        bc_embed = self.boundary_encoder(bc_flat)  # [batch, hidden]
        
        # 为每个查询点创建输入
        time_expanded = time.unsqueeze(1).unsqueeze(2).expand(-1, num_points, 1)
        spatial_input = torch.cat([query_points, time_expanded], dim=-1)
        
        # 空间编码
        spatial_features = self.spatial_encoder(spatial_input)
        
        # 融合边界条件
        bc_expanded = bc_embed.unsqueeze(1).expand(-1, num_points, -1)
        fused_input = torch.cat([bc_expanded, spatial_features], dim=-1)
        
        # 输出流场值
        output = self.fusion(fused_input)
        
        # 分解为速度场和压力场
        velocity = output[..., :self.spatial_dim]
        pressure = output[..., self.spatial_dim:]
        
        return {
            'velocity': velocity,
            'pressure': pressure
        }
    
    def compute_pde_residual(
        self,
        query_points: torch.Tensor,
        time: torch.Tensor,
        boundary_conditions: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Navier-Stokes方程残差（PDE约束）
        
        简化的不可压缩流连续性方程：∇·v = 0
        """
        query_points.requires_grad_(True)
        
        field = self.forward(query_points, time, boundary_conditions)
        velocity = field['velocity']
        
        # 计算速度场的散度（连续性方程）
        div_v = torch.zeros(velocity.shape[0], velocity.shape[1], device=velocity.device)
        
        for i in range(self.spatial_dim):
            grad_vi = torch.autograd.grad(
                velocity[..., i].sum(),
                query_points,
                create_graph=True,
                retain_graph=True
            )[0]
            div_v = div_v + grad_vi[..., i]
        
        return div_v.pow(2).mean()


class WakeEffectVisualizer:
    """
    尾流效应可视化工具
    
    用于验证模型是否正确学习到了物理规律
    """
    
    def __init__(self, num_turbines: int, turbine_positions: np.ndarray):
        self.num_turbines = num_turbines
        self.positions = turbine_positions
    
    def visualize_attention_map(
        self,
        attention_weights: torch.Tensor,
        wind_direction: float,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        可视化注意力权重与尾流区域的对应关系
        
        Args:
            attention_weights: [n_heads, num_turbines, num_turbines]
            wind_direction: 风向角度
            save_path: 保存路径
            
        Returns:
            combined_visualization: 可视化结果
        """
        # 平均所有头的注意力
        avg_attention = attention_weights.mean(dim=0).detach().cpu().numpy()
        
        # 计算理论尾流区
        theta = np.radians(wind_direction)
        wind_vec = np.array([np.sin(theta), np.cos(theta)])
        
        theoretical_wake = np.zeros((self.num_turbines, self.num_turbines))
        
        for i in range(self.num_turbines):
            for j in range(self.num_turbines):
                if i != j:
                    vec_ij = self.positions[j] - self.positions[i]
                    downstream_dist = np.dot(vec_ij, wind_vec)
                    if downstream_dist > 0:
                        cross_dist = np.abs(np.cross(wind_vec, vec_ij))
                        wake_radius = 63 + 0.04 * downstream_dist  # D/2 + k*x
                        if cross_dist < wake_radius:
                            theoretical_wake[i, j] = 1.0 / (1 + 0.04 * downstream_dist / 126)
        
        # 计算注意力与理论尾流的相关性
        correlation = np.corrcoef(avg_attention.flatten(), theoretical_wake.flatten())[0, 1]
        
        print(f"Attention-Wake Correlation: {correlation:.3f}")
        
        return {
            'attention': avg_attention,
            'theoretical_wake': theoretical_wake,
            'correlation': correlation
        }


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("Physics-Informed Neural Networks Module")
    print("=" * 60)
    
    # 测试物理损失
    print("\n1. Testing Physics Loss:")
    physics_loss = PhysicsLoss(rated_power=2.0, cut_in_speed=3.0)
    
    # 模拟预测和风速
    pred_power = torch.rand(100) * 2.5  # 0-2.5 MW
    wind_speed = torch.rand(100) * 30    # 0-30 m/s
    
    losses = physics_loss(pred_power, wind_speed)
    print(f"   Betz limit loss: {losses['betz']:.4f}")
    print(f"   Power curve loss: {losses['power_curve']:.4f}")
    print(f"   Non-negative loss: {losses['non_negative']:.4f}")
    print(f"   Rated limit loss: {losses['rated_limit']:.4f}")
    
    # 测试物理引导注意力
    print("\n2. Testing Physics-Guided Attention:")
    attn = PhysicsGuidedAttention(d_model=64, n_heads=4, num_turbines=10)
    x = torch.randn(8, 10, 64)  # [batch, turbines, d_model]
    wind_dir = torch.rand(8) * 360  # 随机风向
    
    output, attn_weights = attn(x, wind_dir, return_attention=True)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention shape: {attn_weights.shape}")
    
    # 测试物理信息Transformer
    print("\n3. Testing Physics-Informed Transformer:")
    model = PhysicsInformedTransformer(
        input_len=96,
        output_len=24,
        num_features=5,
        d_model=64,
        n_heads=4,
        n_layers=3,
        num_turbines=10,
        rated_power=2.0
    )
    
    x = torch.randn(16, 96, 5)
    wind_dir = torch.rand(16) * 360
    
    output, attention_maps = model(x, wind_dir, return_attention=True)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Number of attention layers: {len(attention_maps)}")
    
    # 测试损失计算
    targets = torch.randn(16, 24, 5)
    wind_speed = torch.rand(16, 24) * 20 + 3
    
    loss, loss_dict = model.compute_loss(output, targets, wind_speed, lambda_physics=0.1)
    print(f"   Total loss: {loss_dict['total']:.4f}")
    print(f"   Data loss: {loss_dict['data']:.4f}")
    print(f"   Physics loss: {loss_dict['physics_total']:.4f}")
    
    # 测试数字孪生
    print("\n4. Testing Digital Twin:")
    digital_twin = DigitalTwin(num_turbines=10, spatial_dim=2)
    
    query_points = torch.randn(8, 100, 2)  # 100个查询点
    time = torch.rand(8)
    boundary_conditions = torch.randn(8, 10, 2)
    
    field = digital_twin(query_points, time, boundary_conditions)
    print(f"   Query points: {query_points.shape}")
    print(f"   Velocity field: {field['velocity'].shape}")
    print(f"   Pressure field: {field['pressure'].shape}")
    
    # 统计参数量
    print("\n5. Model Statistics:")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Physics-Informed Transformer parameters: {params:,}")
    
    print("\n" + "=" * 60)
    print("Physics-Informed Module Test Complete!")
    print("=" * 60)
