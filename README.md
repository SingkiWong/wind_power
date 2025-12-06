# CCP Framework: 因果-认知-物理 风电预测可解释性系统

## 项目概述

本项目实现了一个融合**认知可解释性**、**结构可解释性**与**物理可解释性**的风电功率预测系统，基于两篇核心文章提出的创新框架：

1. **《风电预测可解释性的多维重构》** - CCP架构（因果-认知-物理）
2. **《后LLM时代的工业级时间序列预测范式》** - 轻量级模型替代方案（KAN、Mamba、TTM）

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    CCP Framework 整体架构                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │  物理感知层     │  │  因果推理层     │  │  认知交互层    │    │
│  │  Physical      │  │  Causal        │  │  Cognitive     │    │
│  │  Perception    │  │  Reasoning     │  │  Interface     │    │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘    │
│          │                   │                   │              │
│          ▼                   ▼                   ▼              │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                      核心技术栈                          │    │
│  ├────────────────────────────────────────────────────────┤    │
│  │  • Mamba/Transformer (骨干网络)                         │    │
│  │  • KAN (可解释非线性建模)                               │    │
│  │  • PINNs (物理约束)                                     │    │
│  │  • PCMCI (因果发现)                                     │    │
│  │  • Wind-Agent (智能问答)                                │    │
│  │  • RAG (检索增强)                                       │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 核心模块

### 1. KAN模块 (`kan_module.py`)
- **KAN**: 基础Kolmogorov-Arnold网络，使用B样条实现可学习边函数
- **TimeKAN**: 多尺度时间序列KAN变体
- **PhysicsInformedKAN**: 物理约束KAN（贝兹极限、功率曲线约束）
- **符号回归**: 从学习到的函数发现数学公式

### 2. Mamba模块 (`mamba_module.py`)
- **SelectiveSSM**: 选择性状态空间模型核心
- **WindMambaformer**: 风电专用Mamba-Transformer混合架构
- **SiMBA**: 简化Mamba架构，支持多变量
- **BiMamba**: 双向Mamba，用于数据补全

### 3. 物理信息模块 (`physics_informed.py`)
- **PhysicsLoss**: 物理约束损失函数集合
  - 贝兹极限约束
  - 功率曲线约束
  - 非负约束
- **PhysicsGuidedAttention**: 物理引导注意力机制
- **PhysicsInformedTransformer**: 完整的物理信息Transformer
- **DigitalTwin**: 数字孪生虚拟感知

### 4. 因果发现模块 (`causal_discovery.py`)
- **PCMCI**: Peter-Clark Momentary Conditional Independence算法
- **LiNGAM**: 线性非高斯无环模型
- **DynamicWakeGraph**: 动态尾流因果图谱
- **PhysicsConstrainedCausalDiscovery**: 物理约束因果发现

### 5. 轻量化模型模块 (`lightweight_models.py`)
- **TSMixer**: 全MLP架构
- **TinyTimeMixer (TTM)**: IBM百万参数模型
- **PatchTSMixer**: 抗分布偏移设计
- **RevIN**: 可逆实例归一化

### 6. Wind-Agent模块 (`wind_agent.py`)
- **KnowledgeBase**: 风电领域知识库
- **ReasoningChain**: 思维链推理引擎
- **RAGEngine**: 检索增强生成引擎
- **ReflectionAgent**: 反思智能体（预测-反思-修正闭环）
- **WindAgent**: 完整智能体
- **ReportGenerator**: 报告生成器

### 7. CCP框架 (`ccp_framework.py`)
- **CCPConfig**: 系统配置
- **PhysicalPerceptionLayer**: 物理感知层
- **CausalReasoningLayer**: 因果推理层
- **CognitiveInterfaceLayer**: 认知交互层
- **CCPSystem**: 完整系统

## 快速开始

### 安装依赖

```bash
pip install torch numpy scipy einops
```

### 基础使用

```python
from ccp_framework import create_ccp_system

# 创建CCP系统
system = create_ccp_system(
    backbone="mamba",      # 可选: "mamba", "transformer", "ttm"
    use_kan=True,          # 是否使用KAN增强
    input_len=96,          # 输入序列长度
    output_len=24,         # 预测长度
    num_features=5         # 特征数量
)

# 预测
import torch
x = torch.randn(8, 96, 5)  # [batch, seq_len, features]
wind_direction = torch.rand(8) * 360

outputs = system(x, wind_direction)
predictions = outputs['predictions']  # [8, 24, 5]
confidence = outputs['confidence']    # [8]
```

### 生成解释报告

```python
# 带解释的预测
result = system.predict_with_explanation(
    x, wind_speed, wind_direction,
    turbine_ids=["T1", "T2", "T3", ...],
    generate_report=True
)

# 查看报告
print(result['summary_report'])
print(result['detailed_report'])
```

### 训练模型

```python
from train_eval import run_experiment

# 运行实验
result = run_experiment(
    backbone="mamba",
    use_kan=True,
    num_epochs=50,
    batch_size=32
)

print(f"Test RMSE: {result['metrics']['rmse']:.4f}")
```

## 核心创新点

### 1. 物理可解释性
- 将贝兹极限、功率曲线等物理约束嵌入损失函数
- 物理引导的注意力机制，确保模型关注正确的上游风机
- 注意力热图与CFD模拟尾流区域的对比验证

### 2. 结构可解释性
- PCMCI算法从数据中发现真实因果关系
- 动态尾流拓扑图谱随风向变化
- 故障传播路径追踪

### 3. 认知可解释性
- 自然语言形式的预测解释
- 思维链(CoT)推理过程
- RAG检索历史案例和技术文档
- 预测-反思-修正闭环

### 4. 轻量化设计
- KAN以1/10参数量达到MLP精度
- Mamba实现O(L)线性复杂度
- TTM百万参数击败数十亿大模型

## 模型参数量对比

| 模型配置 | 参数量 | 特点 |
|---------|--------|------|
| CCP-Mamba-KAN | ~500K | 推荐配置，平衡精度与效率 |
| CCP-Transformer | ~800K | 最高精度，计算量较大 |
| CCP-TTM | ~300K | 最轻量，适合边缘部署 |

## 评估指标

- **MSE**: 均方误差
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **MAPE**: 平均绝对百分比误差
- **R²**: 决定系数
- **物理一致性分数**: 预测与物理约束的符合程度

## 文件结构

```
wind_power_prediction/
├── kan_module.py           # KAN核心模块
├── mamba_module.py         # Mamba核心模块
├── physics_informed.py     # 物理信息模块
├── causal_discovery.py     # 因果发现模块
├── lightweight_models.py   # 轻量化模型
├── wind_agent.py          # 智能体模块
├── ccp_framework.py       # CCP主框架
├── train_eval.py          # 训练评估脚本
└── README.md              # 本文档
```

## 参考文献

### 核心算法
- Time-LLM (ICLR 2024)
- Mamba: Linear-Time Sequence Modeling
- KAN: Kolmogorov-Arnold Networks
- PCMCI for Causal Discovery
- Physics-Informed Neural Networks

### 风电领域
- Wind-Mambaformer
- WindFM Foundation Model
- Jensen Wake Model

## 许可证

MIT License

## 作者

基于文献综合实现，融合最新的LLM、因果发现、物理信息学习技术。
