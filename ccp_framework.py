"""
CCP Framework: Causal-Cognitive-Physical Wind Power Prediction System
CCP架构：因果-认知-物理 风电预测系统

整合创新框架，融合：
1. 物理感知层 - Physics-Informed Neural Networks + 物理引导注意力
2. 因果推理层 - PCMCI + 动态尾流图谱
3. 认知交互层 - Wind-Agent + RAG + CoT

架构设计参考：
- 第一篇文章：风电预测可解释性的多维重构
- 第二篇文章：轻量级模型替代LLM思路（KAN、Mamba、TTM）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass

# 导入自定义模块
from kan_module import KAN, TimeKAN, PhysicsInformedKAN
from mamba_module import WindMambaformer, SiMBA, MambaBlock
from physics_informed import PhysicsLoss, PhysicsGuidedAttention, PhysicsInformedTransformer
from causal_discovery import PCMCI, DynamicWakeGraph, PhysicsConstrainedCausalDiscovery
from lightweight_models import TinyTimeMixer, PatchTSMixer, TSMixer, RevIN
from wind_agent import WindAgent, PredictionContext, ReportGenerator


@dataclass
class CCPConfig:
    """CCP系统配置"""
    # 数据配置
    input_len: int = 96           # 输入序列长度
    output_len: int = 24          # 预测长度
    num_features: int = 5         # 特征数量
    num_turbines: int = 10        # 风机数量
    
    # 模型配置
    d_model: int = 64             # 模型维度
    n_heads: int = 4              # 注意力头数
    n_layers: int = 4             # 层数
    dropout: float = 0.1          # Dropout率
    
    # 物理约束配置
    rated_power: float = 2.0      # 额定功率 (MW)
    cut_in_speed: float = 3.0     # 切入风速 (m/s)
    cut_out_speed: float = 25.0   # 切出风速 (m/s)
    physics_mask_threshold: float = 0.6  # 偏离基线超过该比例视为限电/异常
    physics_mask_floor: float = 0.05     # 基线归一化分母平滑项（额定功率比例）
    
    # 训练配置
    lambda_physics: float = 0.1   # 物理损失权重
    lambda_causal: float = 0.05   # 因果一致性损失权重
    # 动态权重与课程学习配置
    loss_weighting: str = "static"   # "static" 或 "uncertainty"
    curriculum_start_physics: int = 3  # 物理损失开始参与的epoch
    curriculum_start_causal: int = 5   # 因果损失开始参与的epoch
    curriculum_warmup_epochs: int = 3  # 从起始到完全权重的线性升温周期
    residual_weight: float = 0.2       # 物理残差正则权重

    # 架构选择
    backbone: str = "mamba"       # 骨干网络: "mamba", "mamba_kan", "transformer", "ttm"
    use_kan: bool = True          # 是否使用KAN增强
    kan_usage: str = "selective"  # "full" 对所有时间步用KAN, "selective" 只对关键步
    kan_focus_steps: int = 4       # 使用KAN精修的末尾时间步数量
    use_physics_attention: bool = True  # 是否使用物理引导注意力
    use_revin: bool = True        # 是否使用可逆实例归一化稳定分布

    # 因果图更新节奏
    causal_refresh_minutes: int = 15   # 离线尾流图更新间隔


class PhysicalPerceptionLayer(nn.Module):
    """
    物理感知层
    
    功能：
    1. 提取时空特征
    2. 物理引导的注意力约束
    3. 确保特征提取符合流体力学原理
    
    输出：高精度功率预测 + 物理一致的注意力权重图
    """
    
    def __init__(self, config: CCPConfig):
        super().__init__()
        self.config = config

        self.revin = RevIN(config.num_features) if config.use_revin else None
        
        # 输入嵌入
        self.input_embed = nn.Linear(config.num_features, config.d_model)
        
        # 选择骨干网络
        if config.backbone == "mamba":
            self.backbone = WindMambaformer(
                input_len=config.input_len,
                output_len=config.output_len,
                num_features=config.d_model,  # 嵌入后的维度
                d_model=config.d_model,
                n_layers=config.n_layers,
                use_flow_attention=config.use_physics_attention
            )
        elif config.backbone == "mamba_kan":
            # 混合架构：长序列特征由Mamba完成，KAN仅在输出头精修，降低总体FLOPs
            self.backbone = MambaKANHybrid(
                input_len=config.input_len,
                output_len=config.output_len,
                num_features=config.d_model,
                d_model=config.d_model,
                n_mamba_layers=max(1, config.n_layers - 1),
                kan_hidden=32,
                num_knots=6
            )
        elif config.backbone == "transformer":
            self.backbone = PhysicsInformedTransformer(
                input_len=config.input_len,
                output_len=config.output_len,
                num_features=config.d_model,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                num_turbines=config.num_turbines,
                rated_power=config.rated_power
            )
        else:  # ttm
            self.backbone = TinyTimeMixer(
                input_len=config.input_len,
                output_len=config.output_len,
                num_features=config.d_model,
                d_model=config.d_model,
                n_layers=config.n_layers
            )
        
        # 物理引导注意力（可选）
        if config.use_physics_attention and config.backbone != "transformer":
            self.physics_attention = PhysicsGuidedAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                num_turbines=config.num_turbines
            )
        else:
            self.physics_attention = None
        
        # KAN增强层（可选）
        self.linear_head = nn.Linear(config.d_model, config.num_features)
        if config.use_kan:
            self.kan_head = KAN(
                [config.d_model, 32, config.num_features],
                num_knots=6 if config.kan_usage == "selective" else 8
            )
        else:
            self.kan_head = self.linear_head

        # 异方差噪声建模（仅对功率通道提供 log-variance）
        self.power_uncertainty_head = nn.Linear(config.d_model, 1)
        
        # 物理损失计算
        self.physics_loss = PhysicsLoss(
            rated_power=config.rated_power,
            cut_in_speed=config.cut_in_speed,
            cut_out_speed=config.cut_out_speed,
            residual_weight=config.residual_weight
        )
    
    def forward(
        self,
        x: torch.Tensor,
        wind_direction: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            x: [batch, input_len, num_features]
            wind_direction: [batch] 风向（可选）
            return_attention: 是否返回注意力权重
        Returns:
            predictions, attention, power_log_var
        """
        batch_size = x.shape[0]

        # 输入归一化，防止不同物理量级带来的梯度病态
        if self.revin is not None:
            x = self.revin(x, 'norm')

        # 输入嵌入
        x_embed = self.input_embed(x)  # [batch, input_len, d_model]
        
        attention_weights = None
        
        # 骨干网络处理
        if self.config.backbone == "transformer" and wind_direction is not None:
            backbone_out, attention_weights = self.backbone(
                x_embed, wind_direction, return_attention=return_attention
            )
        else:
            backbone_out, _ = self.backbone(x_embed), None
            
            # 物理引导注意力增强
            if self.physics_attention is not None and wind_direction is not None:
                attn_out, attention_weights = self.physics_attention(
                    backbone_out, wind_direction, return_attention=return_attention
                )
                backbone_out = backbone_out + attn_out
        
        # KAN输出头（可选局部精修以降低开销）
        if isinstance(self.kan_head, KAN) and self.config.kan_usage == "selective":
            b, t, d = backbone_out.shape
            focus = min(self.config.kan_focus_steps, t)
            fast_path = self.linear_head(backbone_out)
            flat = backbone_out[:, -focus:, :].reshape(-1, d)
            refined_flat = self.kan_head(flat)
            fast_path[:, -focus:, :] = refined_flat.reshape(b, focus, -1)
            output = fast_path
        elif isinstance(self.kan_head, KAN):
            b, t, d = backbone_out.shape
            flat = backbone_out.reshape(-1, d)
            out_flat = self.kan_head(flat)
            output = out_flat.reshape(b, t, -1)
        else:
            output = self.linear_head(backbone_out)

        power_log_var = torch.tanh(self.power_uncertainty_head(backbone_out))

        # 反归一化恢复物理量纲
        if self.revin is not None:
            output = self.revin(output, 'denorm')

        return output, attention_weights, power_log_var


class CausalReasoningLayer(nn.Module):
    """
    因果推理层
    
    功能：
    1. 动态构建风机因果拓扑图
    2. 识别功率波动根因
    3. 反事实推理支持
    
    输出：动态因果图谱 + 故障传播路径
    """
    
    def __init__(self, config: CCPConfig):
        super().__init__()
        self.config = config
        
        # PCMCI因果发现
        self.pcmci = PCMCI(max_lag=10, significance_level=0.05)
        
        # 动态尾流图
        self.wake_graph = DynamicWakeGraph(
            num_turbines=config.num_turbines,
            max_lag=10
        )
        
        # 因果嵌入网络（将因果图编码为向量）
        self.causal_encoder = nn.Sequential(
            nn.Linear(config.num_turbines * config.num_turbines, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # 因果注意力偏置
        self.causal_bias = nn.Parameter(
            torch.zeros(config.num_turbines, config.num_turbines)
        )

        # 离线-在线分离：缓存因果图，按时间窗口刷新
        self.cached_causal_graph: Optional[torch.Tensor] = None
        self.last_graph_refresh: Optional[datetime] = None
        self.refresh_interval = config.causal_refresh_minutes
    
    def discover_causal_graph(
        self,
        power_data: np.ndarray,
        wind_direction: float,
        force: bool = False
    ) -> Dict:
        """
        发现因果图谱

        Args:
            power_data: [time_steps, num_turbines] 功率数据
            wind_direction: 当前风向
            force: 是否忽略刷新间隔强制更新
        """
        should_refresh = force or self.last_graph_refresh is None
        if not should_refresh:
            delta = datetime.now() - self.last_graph_refresh
            should_refresh = delta.total_seconds() > self.refresh_interval * 60

        if should_refresh:
            graph = self.wake_graph.build_graph(power_data, wind_direction)
            self.cached_causal_graph = torch.tensor(graph, dtype=torch.float32)
            self.last_graph_refresh = datetime.now()
        return self.cached_causal_graph

    def get_cached_graph(self) -> Optional[torch.Tensor]:
        """获取最近一次离线更新的因果图"""
        return self.cached_causal_graph
    
    def get_causal_embedding(
        self,
        causal_strength: torch.Tensor
    ) -> torch.Tensor:
        """
        将因果强度矩阵编码为向量
        
        Args:
            causal_strength: [num_turbines, num_turbines]
        """
        flat = causal_strength.flatten()
        return self.causal_encoder(flat)
    
    def compute_causal_consistency_loss(
        self,
        attention_weights: torch.Tensor,
        causal_graph: torch.Tensor
    ) -> torch.Tensor:
        """
        计算注意力权重与因果图的一致性损失
        
        鼓励模型的注意力模式与发现的因果关系一致
        """
        if attention_weights is None:
            return torch.tensor(0.0, device=causal_graph.device)
        
        # 平均注意力头
        avg_attn = attention_weights.mean(dim=1)  # [batch, seq, seq]
        
        # 扩展因果图到batch维度
        causal_expanded = causal_graph.unsqueeze(0).expand(avg_attn.shape[0], -1, -1)
        
        # 计算KL散度作为一致性度量
        # 注意力应该在因果连接强的地方更大
        attn_flat = avg_attn.flatten(1)
        causal_flat = F.softmax(causal_expanded.flatten(1), dim=-1)
        
        consistency_loss = F.kl_div(
            F.log_softmax(attn_flat, dim=-1),
            causal_flat,
            reduction='batchmean'
        )

        return consistency_loss

    def prepare_causal_graph(
        self,
        causal_graph: Any,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Ensure causal graphs are tensors on the right device/dtype.

        Handles numpy arrays and CPU tensors by moving them to the target device
        before any normalization or loss computation, preventing device mismatch
        errors during training and evaluation.
        """
        if isinstance(causal_graph, np.ndarray):
            causal_graph = torch.from_numpy(causal_graph)

        if not isinstance(causal_graph, torch.Tensor):
            raise TypeError("causal_graph must be a torch.Tensor or numpy.ndarray")

        if causal_graph.dtype != dtype:
            causal_graph = causal_graph.to(dtype)

        if causal_graph.device != device:
            causal_graph = causal_graph.to(device)

        return causal_graph

    def normalize_causal_graph(self, causal_graph: torch.Tensor) -> torch.Tensor:
        """
        标准化因果图以便与注意力矩阵对齐。

        通过最小-最大归一化让边权落在[0,1]区间，避免梯度爆炸，
        同时保持结构稀疏性。
        """
        if causal_graph.numel() == 0:
            return causal_graph

        min_val = causal_graph.min()
        max_val = causal_graph.max()
        if (max_val - min_val) < 1e-8:
            return torch.zeros_like(causal_graph)

        normed = (causal_graph - min_val) / (max_val - min_val)
        return normed


class CognitiveInterfaceLayer:
    """
    认知交互层
    
    功能：
    1. 将数值预测转化为自然语言描述
    2. RAG检索历史案例和技术文档
    3. 多轮对话回应操作员质询
    
    输出：综合预测报告（预测曲线、置信区间、自然语言归因解释、操作建议）
    """
    
    def __init__(self, config: CCPConfig):
        self.config = config
        self.agent = WindAgent(rated_power=config.rated_power)
        self.reporter = ReportGenerator(self.agent)
    
    def create_context(
        self,
        turbine_id: str,
        predicted_power: float,
        confidence: float,
        wind_speed: float,
        wind_direction: float,
        temperature: float = 15.0,
        pressure: float = 1013.25,
        causal_info: Optional[Dict] = None
    ) -> PredictionContext:
        """创建预测上下文"""
        return PredictionContext(
            timestamp=datetime.now(),
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            temperature=temperature,
            pressure=pressure,
            turbine_id=turbine_id,
            predicted_power=predicted_power,
            confidence=confidence,
            causal_graph=causal_info
        )
    
    def explain(self, context: PredictionContext) -> str:
        """生成解释"""
        explanation = self.agent.explain_prediction(context)
        return explanation.summary
    
    def query(self, question: str, context: PredictionContext) -> str:
        """交互式问答"""
        return self.agent.interactive_query(question, context)
    
    def generate_report(
        self,
        contexts: List[PredictionContext],
        report_type: str = "detailed"
    ) -> str:
        """生成报告"""
        return self.reporter.generate_report(contexts, report_type)


class CCPSystem(nn.Module):
    """
    CCP系统：完整的因果-认知-物理风电预测框架
    
    整合三层架构：
    1. 物理感知层 - 数据驱动 + 物理约束
    2. 因果推理层 - 拓扑发现 + 归因分析
    3. 认知交互层 - 自然语言解释 + 智能问答
    """
    
    def __init__(self, config: Optional[CCPConfig] = None):
        super().__init__()
        self.config = config or CCPConfig()
        
        # 三层架构
        self.physical_layer = PhysicalPerceptionLayer(self.config)
        self.causal_layer = CausalReasoningLayer(self.config)
        self.cognitive_layer = CognitiveInterfaceLayer(self.config)

        # 动态损失权重参数（不含梯度裁剪的开销）
        if self.config.loss_weighting == "uncertainty":
            self.loss_log_vars = nn.ParameterDict({
                'data': nn.Parameter(torch.zeros(1)),
                'physics': nn.Parameter(torch.zeros(1)),
                'causal': nn.Parameter(torch.zeros(1)),
            })
        else:
            self.loss_log_vars = None
        
        # 置信度估计网络
        self.confidence_net = nn.Sequential(
            nn.Linear(self.config.d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        wind_direction: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [batch, input_len, num_features]
            wind_direction: [batch] 风向
            return_attention: 是否返回注意力
            
        Returns:
            outputs: 包含预测、置信度、注意力等的字典
        """
        # 物理感知层
        predictions, attention, power_log_var = self.physical_layer(
            x, wind_direction, return_attention
        )

        # 基于预测方差估计置信度（方差越大置信度越低）
        if power_log_var is not None:
            avg_log_var = power_log_var.mean(dim=[1, 2])
            confidence = torch.sigmoid(-avg_log_var)
        else:
            x_embed = self.physical_layer.input_embed(x)
            x_mean = x_embed.mean(dim=1)  # [batch, d_model]
            confidence = self.confidence_net(x_mean)  # [batch, 1]

        outputs = {
            'predictions': predictions,
            'confidence': confidence.squeeze(-1),
            'power_log_var': power_log_var,
        }
        
        if return_attention and attention is not None:
            outputs['attention'] = attention
        
        return outputs
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        wind_speed: torch.Tensor,
        causal_graph: Optional[torch.Tensor] = None,
        current_epoch: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失

        L_total = L_data + λ_phy * L_physics + λ_causal * L_causal
        """
        predictions = outputs['predictions']
        power_log_var = outputs.get('power_log_var')

        physics_weight = self._compute_curriculum_weight(
            self.config.lambda_physics,
            current_epoch,
            self.config.curriculum_start_physics
        )
        causal_weight = self._compute_curriculum_weight(
            self.config.lambda_causal,
            current_epoch,
            self.config.curriculum_start_causal
        )

        # 1. 数据损失（功率采用高斯NLL，其余特征使用MSE）
        target_power = targets[..., 0]
        pred_power = predictions[..., 0]
        feature_loss = F.mse_loss(predictions[..., 1:], targets[..., 1:]) if predictions.shape[-1] > 1 else 0.0

        if power_log_var is not None:
            log_var = power_log_var.squeeze(-1)
            log_var = torch.clamp(log_var, min=-10.0, max=6.0)
            nll = 0.5 * (log_var + (pred_power - target_power) ** 2 / torch.exp(log_var))
            power_loss = nll.mean()
        else:
            power_loss = F.mse_loss(pred_power, target_power)

        data_loss = power_loss + (feature_loss if isinstance(feature_loss, torch.Tensor) else torch.tensor(feature_loss, device=predictions.device))
        
        # 2. 物理损失
        pred_power = predictions[..., 0] if predictions.dim() > 2 else predictions

        # 基于真实功率与物理基线偏差的动态掩码，避免限电/脏数据误触发物理惩罚
        with torch.no_grad():
            baseline = self.physical_layer.physics_loss.power_curve_baseline(wind_speed.flatten())
            floor = self.config.physics_mask_floor * self.config.rated_power
            deviation = (target_power.flatten() - baseline).abs()
            ratio = deviation / (baseline.abs() + floor)
            physics_mask = (ratio < self.config.physics_mask_threshold).float()

        physics_losses = self.physical_layer.physics_loss(
            pred_power.flatten(), wind_speed.flatten(), mask=physics_mask
        )
        physics_loss = physics_losses['total_physics']

        # 3. 因果一致性损失
        causal_loss = torch.tensor(0.0, device=predictions.device)
        candidate_graph = causal_graph
        if candidate_graph is None:
            candidate_graph = self.causal_layer.get_cached_graph()

        if candidate_graph is not None and 'attention' in outputs:
            prepared_graph = self.causal_layer.prepare_causal_graph(
                candidate_graph, device=predictions.device
            )
            normalized_graph = self.causal_layer.normalize_causal_graph(prepared_graph)
            causal_loss = self.causal_layer.compute_causal_consistency_loss(
                outputs['attention'], normalized_graph
            )

        # 总损失（支持静态/不确定性动态权重）
        if self.loss_log_vars is None:
            total_loss = (
                data_loss + physics_weight * physics_loss + causal_weight * causal_loss
            )
            effective_weights = {
                'data': 1.0,
                'physics': physics_weight,
                'causal': causal_weight,
            }
        else:
            weighted_data = torch.exp(-self.loss_log_vars['data']) * data_loss + self.loss_log_vars['data']
            weighted_physics = torch.exp(-self.loss_log_vars['physics']) * (physics_weight * physics_loss) + self.loss_log_vars['physics']
            weighted_causal = torch.exp(-self.loss_log_vars['causal']) * (causal_weight * causal_loss) + self.loss_log_vars['causal']
            total_loss = 0.5 * (weighted_data + weighted_physics + weighted_causal)
            effective_weights = {
                'data': float(torch.exp(-self.loss_log_vars['data']).detach().cpu()),
                'physics': float(torch.exp(-self.loss_log_vars['physics']).detach().cpu() * physics_weight),
                'causal': float(torch.exp(-self.loss_log_vars['causal']).detach().cpu() * causal_weight),
            }

        # 物理一致性分数（越接近1越好）
        physics_consistency = torch.exp(-physics_loss.detach())

        loss_dict = {
            'total': total_loss.item(),
            'data': data_loss.item(),
            'physics': physics_loss.item(),
            'physics_consistency': physics_consistency.item(),
            'causal': causal_loss.item() if isinstance(causal_loss, torch.Tensor) else causal_loss,
            'w_data': effective_weights['data'],
            'w_physics': effective_weights['physics'],
            'w_causal': effective_weights['causal'],
        }

        return total_loss, loss_dict

    def _compute_curriculum_weight(
        self,
        base_weight: float,
        current_epoch: Optional[int],
        start_epoch: int
    ) -> float:
        """线性课程学习权重，在冷启动阶段抑制物理/因果损失。

        Args:
            base_weight: 静态基准权重（如 lambda_physics）
            current_epoch: 当前epoch（从0开始）
            start_epoch: 该损失开始介入的epoch
        """
        if current_epoch is None:
            return base_weight
        if current_epoch < start_epoch:
            return 0.0

        warmup = max(1, self.config.curriculum_warmup_epochs)
        progress = min(1.0, (current_epoch - start_epoch + 1) / warmup)
        return float(base_weight * progress)
    
    def predict_with_explanation(
        self,
        x: torch.Tensor,
        wind_speed: torch.Tensor,
        wind_direction: torch.Tensor,
        turbine_ids: List[str],
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        预测并生成解释
        
        完整的CCP流水线
        """
        self.eval()
        with torch.no_grad():
            # 1. 物理感知层预测
            outputs = self.forward(x, wind_direction, return_attention=True)
            predictions = outputs['predictions']
            confidence = outputs['confidence']
            attention = outputs.get('attention')

            # 计算物理一致性分数
            pred_power = predictions[..., 0] if predictions.dim() > 2 else predictions
            physics_losses = self.physical_layer.physics_loss(
                pred_power.flatten(), wind_speed.flatten()
            )
            physics_consistency = torch.exp(-physics_losses['total_physics'])

        # 2. 创建上下文
        contexts = []
        batch_size = x.shape[0]

        for i in range(min(batch_size, len(turbine_ids))):
            # 取预测的最后一个时间步的功率
            pred_power = predictions[i, -1, 0].item() if predictions.dim() > 2 else predictions[i, -1].item()

            causal_info = None
            if attention is not None:
                # 使用平均注意力作为因果强度近似
                avg_attn = attention[i].mean(dim=0)
                causal_info = {
                    'attention_graph': self.causal_layer.normalize_causal_graph(avg_attn).cpu()
                }

            ctx = self.cognitive_layer.create_context(
                turbine_id=turbine_ids[i],
                predicted_power=max(0, pred_power),  # 确保非负
                confidence=confidence[i].item(),
                wind_speed=wind_speed[i].mean().item(),
                wind_direction=wind_direction[i].item() if wind_direction.dim() > 0 else wind_direction.item(),
                causal_info=causal_info
            )
            contexts.append(ctx)

        # 3. 生成解释和报告
        result = {
            'predictions': predictions,
            'confidence': confidence,
            'contexts': contexts,
            'physics_consistency': physics_consistency.cpu()
        }
        
        if generate_report:
            result['summary_report'] = self.cognitive_layer.generate_report(contexts, 'summary')
            result['detailed_report'] = self.cognitive_layer.generate_report(contexts, 'detailed')
        
        return result


def create_ccp_system(
    backbone: str = "mamba",
    use_kan: bool = True,
    **kwargs
) -> CCPSystem:
    """
    工厂函数：创建CCP系统
    
    Args:
        backbone: 骨干网络类型 ("mamba", "transformer", "ttm")
        use_kan: 是否使用KAN增强
        **kwargs: 其他配置参数
    """
    config = CCPConfig(
        backbone=backbone,
        use_kan=use_kan,
        **kwargs
    )
    return CCPSystem(config)


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("CCP Framework: Causal-Cognitive-Physical Wind Power Prediction System")
    print("=" * 70)
    
    # 创建系统
    print("\n1. Creating CCP System with Mamba backbone + KAN enhancement...")
    system = create_ccp_system(
        backbone="mamba",
        use_kan=True,
        input_len=96,
        output_len=24,
        num_features=5,
        num_turbines=10
    )
    
    # 统计参数
    params = sum(p.numel() for p in system.parameters() if p.requires_grad)
    print(f"   Total parameters: {params:,}")
    
    # 测试前向传播
    print("\n2. Testing Forward Pass...")
    batch_size = 8
    x = torch.randn(batch_size, 96, 5)
    wind_dir = torch.rand(batch_size) * 360
    wind_speed = torch.rand(batch_size, 96) * 20 + 3
    
    outputs = system(x, wind_dir, return_attention=True)
    print(f"   Input shape: {x.shape}")
    print(f"   Prediction shape: {outputs['predictions'].shape}")
    print(f"   Confidence shape: {outputs['confidence'].shape}")
    
    # 测试损失计算
    print("\n3. Testing Loss Computation...")
    targets = torch.randn(batch_size, 24, 5)
    loss, loss_dict = system.compute_loss(outputs, targets, wind_speed)
    print(f"   Total loss: {loss_dict['total']:.4f}")
    print(f"   Data loss: {loss_dict['data']:.4f}")
    print(f"   Physics loss: {loss_dict['physics']:.4f}")
    
    # 测试完整预测流水线
    print("\n4. Testing Full Prediction Pipeline with Explanation...")
    turbine_ids = [f"T{i+1}" for i in range(batch_size)]
    
    result = system.predict_with_explanation(
        x[:2],  # 只取2个样本测试
        wind_speed[:2],
        wind_dir[:2],
        turbine_ids[:2],
        generate_report=True
    )
    
    print("\n   Summary Report:")
    print(result['summary_report'])
    
    # 测试不同骨干网络
    print("\n5. Testing Different Backbones...")
    for backbone in ["mamba", "ttm"]:
        sys = create_ccp_system(backbone=backbone, use_kan=True)
        params = sum(p.numel() for p in sys.parameters() if p.requires_grad)
        out = sys(x, wind_dir)
        print(f"   {backbone.upper()}: {params:,} params, output shape {out['predictions'].shape}")
    
    print("\n" + "=" * 70)
    print("CCP Framework Test Complete!")
    print("=" * 70)
    print("\n实现的核心功能：")
    print("  ✓ 物理感知层 - Mamba/Transformer/TTM骨干 + 物理引导注意力 + KAN增强")
    print("  ✓ 因果推理层 - PCMCI因果发现 + 动态尾流图谱 + 因果一致性约束")
    print("  ✓ 认知交互层 - Wind-Agent智能体 + RAG检索 + 自然语言报告生成")
    print("  ✓ 多模态损失函数 - 数据损失 + 物理约束损失 + 因果一致性损失")
