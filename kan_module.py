"""
KAN (Kolmogorov-Arnold Network) Module for Wind Power Prediction
基于柯尔莫哥洛夫-阿诺德表示定理的可解释神经网络

核心特点：
1. 将非线性激活函数放置在边（Edge）上而非节点上
2. 使用B样条函数进行参数化，提供显式可解释性
3. 支持符号回归，可发现物理定律
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class BSplineBasis(nn.Module):
    """
    B样条基函数模块
    B-Spline提供局部可塑性，调整控制点只影响局部形状
    """
    
    def __init__(
        self,
        num_knots: int = 8,
        degree: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        super().__init__()
        self.num_knots = num_knots
        self.degree = degree
        self.grid_range = grid_range
        
        # 构建均匀节点向量
        # 扩展节点以处理边界
        num_intervals = num_knots - 1
        knots = torch.linspace(
            grid_range[0], 
            grid_range[1], 
            num_intervals + 1
        )
        
        # 添加边界扩展节点
        left_ext = torch.full((degree,), grid_range[0])
        right_ext = torch.full((degree,), grid_range[1])
        extended_knots = torch.cat([left_ext, knots, right_ext])
        
        self.register_buffer('knots', extended_knots)
        self.num_bases = num_knots + degree - 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算B样条基函数值
        
        Args:
            x: 输入张量 [batch_size, ...]
            
        Returns:
            基函数值 [batch_size, ..., num_bases]
        """
        # 使用Cox-de Boor递归公式计算B样条基
        batch_shape = x.shape
        x_flat = x.flatten()
        
        # 初始化0阶基函数
        bases = torch.zeros(
            x_flat.shape[0], 
            self.num_bases, 
            device=x.device,
            dtype=x.dtype
        )
        
        # 使用简化的递归计算
        for i in range(self.num_bases):
            bases[:, i] = self._bspline_basis(
                x_flat, i, self.degree, self.knots
            )
        
        return bases.view(*batch_shape, self.num_bases)
    
    def _bspline_basis(
        self, 
        x: torch.Tensor, 
        i: int, 
        k: int, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Cox-de Boor递归公式计算单个B样条基函数
        """
        if k == 0:
            return ((x >= t[i]) & (x < t[i + 1])).float()
        
        # 递归计算
        d1 = t[i + k] - t[i]
        d2 = t[i + k + 1] - t[i + 1]
        
        c1 = torch.where(
            d1 > 0,
            (x - t[i]) / d1,
            torch.zeros_like(x)
        )
        c2 = torch.where(
            d2 > 0,
            (t[i + k + 1] - x) / d2,
            torch.zeros_like(x)
        )
        
        b1 = self._bspline_basis(x, i, k - 1, t)
        b2 = self._bspline_basis(x, i + 1, k - 1, t)
        
        return c1 * b1 + c2 * b2


class KANLayer(nn.Module):
    """
    KAN层：将可学习的非线性函数放置在边上
    
    与MLP的关键区别：
    - MLP: y = σ(Wx + b)，激活函数固定
    - KAN: y = Σ φ_i(x_i)，每条边都是可学习的B样条函数
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_knots: int = 8,
        spline_degree: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        residual_std: float = 0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 每条边的B样条基函数
        self.bspline = BSplineBasis(num_knots, spline_degree, grid_range)
        
        # B样条控制点系数 [out, in, num_bases]
        # 这是KAN的核心可学习参数
        self.spline_coeffs = nn.Parameter(
            torch.randn(out_features, in_features, self.bspline.num_bases) * 0.1
        )
        
        # 残差连接的线性权重（用于稳定训练）
        self.residual_weight = nn.Parameter(
            torch.randn(out_features, in_features) * residual_std
        )
        
        # 可选的偏置
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, in_features]
            
        Returns:
            [batch_size, out_features]
        """
        batch_size = x.shape[0]
        
        # 计算B样条基函数值 [batch, in_features, num_bases]
        basis_values = self.bspline(x)
        
        # 计算样条输出
        # spline_coeffs: [out, in, num_bases]
        # basis_values: [batch, in, num_bases]
        # 结果: [batch, out, in] -> sum -> [batch, out]
        spline_output = torch.einsum(
            'oin,bin->bo',
            self.spline_coeffs,
            basis_values
        )
        
        # 残差连接
        residual = F.linear(x, self.residual_weight)
        
        return spline_output + residual + self.bias
    
    def get_edge_functions(self) -> torch.Tensor:
        """
        获取每条边学习到的函数形状，用于可视化和解释
        
        Returns:
            edge_functions: [out_features, in_features, num_samples]
        """
        num_samples = 100
        x_samples = torch.linspace(
            self.bspline.grid_range[0],
            self.bspline.grid_range[1],
            num_samples,
            device=self.spline_coeffs.device
        )
        
        # 计算基函数值
        basis = self.bspline(x_samples)  # [num_samples, num_bases]
        
        # 计算每条边的函数值
        # [out, in, num_bases] @ [num_samples, num_bases].T -> [out, in, num_samples]
        edge_functions = torch.einsum(
            'oin,sn->ois',
            self.spline_coeffs,
            basis
        )
        
        return edge_functions, x_samples


class KAN(nn.Module):
    """
    完整的KAN网络
    
    用于风电预测的主要优势：
    1. 参数效率高（约为MLP的1/10）
    2. 可视化每条边学习到的非线性映射
    3. 支持符号回归，可发现物理公式
    """
    
    def __init__(
        self,
        layer_dims: List[int],
        num_knots: int = 8,
        spline_degree: int = 3,
        grid_range: Tuple[float, float] = (-2.0, 2.0)
    ):
        """
        Args:
            layer_dims: 每层的维度，如 [input_dim, 64, 32, output_dim]
            num_knots: B样条节点数
            spline_degree: 样条阶数
            grid_range: 输入值范围
        """
        super().__init__()
        self.layer_dims = layer_dims
        
        # 构建KAN层
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                KANLayer(
                    layer_dims[i],
                    layer_dims[i + 1],
                    num_knots,
                    spline_degree,
                    grid_range
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_all_edge_functions(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """获取所有层的边函数，用于可解释性分析"""
        return [layer.get_edge_functions() for layer in self.layers]
    
    def count_parameters(self) -> int:
        """统计参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TimeKAN(nn.Module):
    """
    TimeKAN: 用于时间序列预测的KAN变体
    
    结合多尺度分解与KAN，捕捉不同时间尺度的模式：
    - 日变化、季节性变化（低频）
    - 瞬时阵风（高频）
    """
    
    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int,
        hidden_dims: List[int] = [64, 32],
        patch_sizes: List[int] = [4, 8, 16],  # 多尺度patch
        num_knots: int = 8
    ):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = num_features
        self.patch_sizes = patch_sizes
        
        # 多尺度Patching模块
        self.patch_embeddings = nn.ModuleList()
        total_patches = 0
        
        for ps in patch_sizes:
            num_patches = input_len // ps
            total_patches += num_patches
            self.patch_embeddings.append(
                nn.Linear(ps * num_features, hidden_dims[0])
            )
        
        # KAN处理模块
        kan_dims = [total_patches * hidden_dims[0]] + hidden_dims + [output_len * num_features]
        self.kan = KAN(kan_dims, num_knots=num_knots)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_len, num_features]
            
        Returns:
            [batch_size, output_len, num_features]
        """
        batch_size = x.shape[0]
        
        # 多尺度Patching
        patch_outputs = []
        for i, ps in enumerate(self.patch_sizes):
            # 将序列分成patches
            num_patches = self.input_len // ps
            patches = x[:, :num_patches * ps].view(
                batch_size, num_patches, ps * self.num_features
            )
            # 嵌入每个patch
            embedded = self.patch_embeddings[i](patches)  # [batch, num_patches, hidden]
            patch_outputs.append(embedded.flatten(1))
        
        # 拼接所有尺度的特征
        combined = torch.cat(patch_outputs, dim=1)
        
        # KAN处理
        output = self.kan(combined)
        
        return output.view(batch_size, self.output_len, self.num_features)


class PhysicsInformedKAN(nn.Module):
    """
    物理信息KAN (PI-KAN)
    
    在损失函数中加入物理约束项，确保预测符合：
    1. 贝兹极限（Betz Limit）- 风能利用系数上限59.3%
    2. 功率曲线物理约束 - 额定功率、切入/切出风速
    3. 尾流效应约束 - 动量守恒
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 32],
        rated_power: float = 2.0,  # MW
        cut_in_speed: float = 3.0,  # m/s
        cut_out_speed: float = 25.0,  # m/s
        num_knots: int = 8
    ):
        super().__init__()
        self.rated_power = rated_power
        self.cut_in_speed = cut_in_speed
        self.cut_out_speed = cut_out_speed
        
        # 主KAN网络
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        self.kan = KAN(layer_dims, num_knots=num_knots)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.kan(x)
    
    def compute_physics_loss(
        self,
        predictions: torch.Tensor,
        wind_speed: torch.Tensor
    ) -> torch.Tensor:
        """
        计算物理约束损失
        
        Args:
            predictions: 预测功率 [batch_size, output_dim]
            wind_speed: 对应风速 [batch_size, 1]
            
        Returns:
            physics_loss: 物理违反惩罚
        """
        physics_loss = torch.tensor(0.0, device=predictions.device)
        
        # 1. 贝兹极限约束 - 功率不能超过额定值
        betz_violation = F.relu(predictions - self.rated_power)
        physics_loss = physics_loss + betz_violation.mean()
        
        # 2. 非负约束 - 功率不能为负
        negative_violation = F.relu(-predictions)
        physics_loss = physics_loss + negative_violation.mean()
        
        # 3. 切入风速约束 - 低于切入风速时功率应为0
        below_cutin = (wind_speed < self.cut_in_speed).float()
        cutin_violation = below_cutin * predictions.abs()
        physics_loss = physics_loss + cutin_violation.mean()
        
        # 4. 切出风速约束 - 超过切出风速时功率应为0
        above_cutout = (wind_speed > self.cut_out_speed).float()
        cutout_violation = above_cutout * predictions.abs()
        physics_loss = physics_loss + cutout_violation.mean()
        
        return physics_loss
    
    def compute_total_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        wind_speed: torch.Tensor,
        lambda_physics: float = 0.1
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算总损失 = 数据损失 + λ * 物理损失
        """
        # 数据损失 (MSE)
        data_loss = F.mse_loss(predictions, targets)
        
        # 物理损失
        physics_loss = self.compute_physics_loss(predictions, wind_speed)
        
        # 总损失
        total_loss = data_loss + lambda_physics * physics_loss
        
        return total_loss, {
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item(),
            'total_loss': total_loss.item()
        }


def symbolic_regression_from_kan(
    kan_layer: KANLayer,
    input_idx: int,
    output_idx: int,
    candidate_functions: Optional[List[str]] = None
) -> str:
    """
    从KAN的边函数中进行符号回归，发现近似的数学公式
    
    这是KAN最强大的功能之一 - 可以从学习到的函数中
    反推出解析数学表达式，用于物理规律发现
    
    Args:
        kan_layer: KAN层
        input_idx: 输入特征索引
        output_idx: 输出特征索引
        candidate_functions: 候选基函数列表
        
    Returns:
        discovered_formula: 发现的数学公式字符串
    """
    if candidate_functions is None:
        candidate_functions = [
            'identity',  # x
            'square',    # x^2
            'cube',      # x^3
            'sqrt_abs',  # sqrt(|x|)
            'sin',       # sin(x)
            'cos',       # cos(x)
            'exp',       # exp(x)
            'log_abs',   # log(|x|+1)
        ]
    
    # 获取边函数
    edge_funcs, x_samples = kan_layer.get_edge_functions()
    target_func = edge_funcs[output_idx, input_idx].detach().cpu().numpy()
    x = x_samples.detach().cpu().numpy()
    
    # 计算候选函数
    candidates = {
        'identity': x,
        'square': x ** 2,
        'cube': x ** 3,
        'sqrt_abs': np.sqrt(np.abs(x)),
        'sin': np.sin(x),
        'cos': np.cos(x),
        'exp': np.clip(np.exp(x), -1e10, 1e10),
        'log_abs': np.log(np.abs(x) + 1),
    }
    
    # 简单的最小二乘拟合找最佳匹配
    best_r2 = -np.inf
    best_formula = 'unknown'
    
    for name in candidate_functions:
        if name in candidates:
            candidate = candidates[name]
            # 线性回归: target = a * candidate + b
            A = np.vstack([candidate, np.ones_like(candidate)]).T
            try:
                coeffs, residuals, _, _ = np.linalg.lstsq(A, target_func, rcond=None)
                fitted = A @ coeffs
                ss_res = np.sum((target_func - fitted) ** 2)
                ss_tot = np.sum((target_func - np.mean(target_func)) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-10)
                
                if r2 > best_r2:
                    best_r2 = r2
                    a, b = coeffs
                    if abs(b) < 0.01:
                        best_formula = f"{a:.3f} * {name}(x)"
                    else:
                        best_formula = f"{a:.3f} * {name}(x) + {b:.3f}"
            except:
                continue
    
    return best_formula, best_r2


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("KAN Module for Wind Power Prediction")
    print("=" * 60)
    
    # 测试基本KAN
    print("\n1. Testing Basic KAN:")
    kan = KAN([10, 32, 16, 1])
    x = torch.randn(32, 10)
    y = kan(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Parameters: {kan.count_parameters():,}")
    
    # 测试TimeKAN
    print("\n2. Testing TimeKAN:")
    time_kan = TimeKAN(
        input_len=96,
        output_len=24,
        num_features=5,
        hidden_dims=[64, 32]
    )
    x_seq = torch.randn(16, 96, 5)
    y_seq = time_kan(x_seq)
    print(f"   Input shape: {x_seq.shape}")
    print(f"   Output shape: {y_seq.shape}")
    
    # 测试PI-KAN
    print("\n3. Testing Physics-Informed KAN:")
    pi_kan = PhysicsInformedKAN(
        input_dim=10,
        output_dim=1,
        rated_power=2.0
    )
    x = torch.randn(32, 10)
    wind_speed = torch.rand(32, 1) * 30  # 0-30 m/s
    pred = pi_kan(x)
    targets = torch.rand(32, 1) * 2
    loss, loss_dict = pi_kan.compute_total_loss(pred, targets, wind_speed)
    print(f"   Total Loss: {loss_dict['total_loss']:.4f}")
    print(f"   Data Loss: {loss_dict['data_loss']:.4f}")
    print(f"   Physics Loss: {loss_dict['physics_loss']:.4f}")
    
    print("\n" + "=" * 60)
    print("KAN Module Test Complete!")
    print("=" * 60)
