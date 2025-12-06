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
- **SelectiveKAN / Mamba-KAN Hybrid**: 关键层使用KAN或与Mamba混合，降低计算时延

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
- **Learnable Physics Parameters**: 尾流衰减、切入/切出风速等可设为可学习参数，随数据自适应
- **Residual Physics Modeling**: 预测 = 物理基线 + 神经网络残差，提升现实场景泛化

### 4. 因果发现模块 (`causal_discovery.py`)
- **PCMCI**: Peter-Clark Momentary Conditional Independence算法
- **LiNGAM**: 线性非高斯无环模型
- **DynamicWakeGraph**: 动态尾流因果图谱
- **PhysicsConstrainedCausalDiscovery**: 物理约束因果发现
- **Graph Refresh Scheduling**: 支持离线周期刷新图谱并在推理时复用缓存，降低在线计算

### 5. 轻量化模型模块 (`lightweight_models.py`)
- **TSMixer**: 全MLP架构
- **TinyTimeMixer (TTM)**: IBM百万参数模型
- **PatchTSMixer**: 抗分布偏移设计
- **RevIN**: 可逆实例归一化

### 6. Wind-Agent模块 (`wind_agent.py`)
- **KnowledgeBase**: 风电领域知识库，可重置与JSONL批量摄取
- **ReasoningChain**: 思维链推理引擎，由真实LLM驱动（OpenAI / HuggingFace），默认拒绝使用TemplateLLM假回显
- **RAGEngine**: 语义检索增强生成引擎（sentence-transformer优先，TF-IDF三元组回退且无哈希碰撞）
- **ReflectionAgent**: LLM+物理联合反思（数值/语义双通道批注与修正建议）
- **WindAgent**: 完整智能体
- **ReportGenerator**: 报告生成器
- **Vector RAG**: 语义向量检索与动态文档摄取，替代静态关键词匹配；TemplateLLM仅在显式允许时作占位警示

### 7. CCP框架 (`ccp_framework.py`)
- **CCPConfig**: 系统配置
- **PhysicalPerceptionLayer**: 物理感知层
- **CausalReasoningLayer**: 因果推理层
- **CognitiveInterfaceLayer**: 认知交互层
- **CCPSystem**: 完整系统
- **Dynamic Loss Weighting**: 支持静态/不确定度自适应损失平衡与课程学习分阶段开启
- **Teacher-Student Distillation**: 可选TinyTimeMixer蒸馏路径，满足边缘部署时延

### 8. 训练与部署 (`train_eval.py`)
- **Curriculum Learning**: 先优化数据损失，逐步引入物理/因果约束，避免梯度冲突
- **Causal Graph Gating**: 训练/验证均可按epoch调度因果图使用并缓存刷新
- **Student Export**: 同步导出教师与TinyTimeMixer学生模型以满足低算力场景

## 快速开始

### 安装依赖

```bash
pip install torch numpy scipy einops
# LLM / RAG 可选依赖
# pip install openai transformers sentence-transformers
```

### 基础使用

```python
from ccp_framework import create_ccp_system

# 创建CCP系统
system = create_ccp_system(
    backbone="mamba",      # 可选: "mamba", "transformer", "ttm"
    use_kan=True,          # 是否使用KAN增强（支持SelectiveKAN或Hybrid模式）
    kan_mode="hybrid",    # "full" | "selective" | "hybrid"，兼顾精度与效率
    input_len=96,          # 输入序列长度
    output_len=24,         # 预测长度
    num_features=5         # 特征数量
)

# 预测（自动复用缓存的尾流因果图，降低推理开销）
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
    kan_mode="hybrid",           # 计算友好模式
    num_epochs=50,
    batch_size=32,
    loss_weighting="uncertainty",# 动态损失平衡: static / uncertainty
    curriculum_warmup=5,          # 前5个epoch仅优化数据损失
    distill_student=True          # 同步蒸馏TinyTimeMixer学生模型
)

print(f"Test RMSE: {result['metrics']['rmse']:.4f}")
```

### Wind-Agent: LLM与RAG配置

```bash
# 选择LLM后端（必须提供真实LLM，默认会拒绝TemplateLLM回显）
export WIND_AGENT_LLM=openai
export OPENAI_API_KEY=sk-...
# 或使用本地Transformers模型
# export WIND_AGENT_LLM=transformers
# export WIND_AGENT_LLM_MODEL=Qwen/Qwen2.5-0.5B
```

```python
from wind_agent import WindAgent, PredictionContext

# 默认强制真实LLM；若要在无LLM环境下调试，可显式 allow_template_llm=True
agent = WindAgent(allow_template_llm=False, conversation_max_turns=12)

# 动态摄取知识库（外部JSONL日志或工单）
agent.knowledge_base.load_jsonl("om_logs.jsonl", namespace="om")

context = PredictionContext(
    wind_speed=9.5,
    wind_direction=240,
    temperature=12.0,
    pressure=101.2,
    historical_power=[0.8, 0.9, 1.0, 1.1, 1.0, 0.95],
)
result = agent.predict_with_explanation(context)
print(agent.generate_report(context, result))

# 面向对话的问答（滑动窗口保留最近多轮对话历史与检索证据）
print(agent.answer_question("为什么这次预测置信度较低？", context, result))
```

## 核心创新点

### 1. 物理可解释性
- 将贝兹极限、功率曲线等物理约束嵌入损失函数
- 物理引导的注意力机制，确保模型关注正确的上游风机
- 注意力热图与CFD模拟尾流区域的对比验证
- 物理参数可学习并支持残差建模，适应设备老化和场景漂移

### 2. 结构可解释性
- PCMCI算法从数据中发现真实因果关系
- 动态尾流拓扑图谱随风向变化
- 故障传播路径追踪
- 因果图可缓存并按调度刷新，降低实时推理开销

### 3. 认知可解释性
- 自然语言形式的预测解释
- 思维链(CoT)推理过程
- RAG检索历史案例和技术文档
- 预测-反思-修正闭环
- 支持向量语义检索的RAG引擎，动态摄取运维/日志文档

### 4. 轻量化设计
- KAN以1/10参数量达到MLP精度
- Mamba实现O(L)线性复杂度
- TTM百万参数击败数十亿大模型
- 支持SelectiveKAN/Hybrid与TinyTimeMixer蒸馏，兼顾精度与推理时延

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
