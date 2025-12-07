# CCP Framework：因果-认知-物理一体化风电预测

本仓库实现了一个将 **因果发现、认知型智能体、物理约束** 与 **高效时序骨干（Mamba / Transformer / KAN / TinyTimeMixer）** 融合的风电功率预测体系。系统支持真实 LLM 与向量检索、可学习物理参数、因果一致性约束，以及将智能体检索到的上下文嵌入直接注入预测网络，实现 **神经-符号融合（Neuro-Symbolic AI）** 的端到端链路。

## 设计脉络与运行流程
1. **数据进入物理感知层**：
   - RevIN 对多特征（功率/风速/温度/气压等）进行可逆归一化，避免量纲差异导致梯度失衡。
   - Input embedding / 可选上下文 token 拼接后，交给 Mamba / Transformer / KAN 骨干提取时序表示。
   - Heteroscedastic 头输出功率均值与 log-variance，用高斯 NLL 训练并由方差反推置信度。
2. **因果层并行运行**：
   - Dataset 周期生成尾流因果图（PCMCI / LiNGAM），批次随数据送入模型。
   - compute_loss 会将因果图缩放到注意力尺寸，计算因果一致性损失，与数据/物理损失动态加权求和。
3. **认知层驱动与回灌**：
   - Wind-Agent 使用真实 LLM + 向量检索回答问答、生成解释/报告，并在预测前构造“今日台风”等知识向量。
   - 该上下文向量作为额外 token 注入物理层，让预测直接感知认知信息，实现神经-符号融合。
4. **训练/评估/蒸馏闭环**：
   - 课程学习、动态/不确定度权重平衡、可选 TinyTimeMixer 学生蒸馏，确保上线可用的轻量模型。

## 模块与关键类详解
### `wind_agent.py` — 认知智能体与 RAG
- **LLM 接入**：优先本地 Transformers / HuggingFace 模型，OpenAI 为显式 opt-in；TemplateLLM 默认禁用（需允许才启用）。
- **知识检索**：优先使用 sentence-transformer 语义向量；若依赖缺失则回退到 sklearn TF-IDF 三元组，不再使用哈希取模。支持 JSONL 动态摄取和索引重建。
- **SafetyGuardrails + 语义反思**：硬护栏执行非负/切出等数值检查；LLM 语义批注提供“软”反思，二者合并输出修正与解释。
- **ReportGenerator / explain_prediction**：生成摘要+详情报告，并把 session directives（如“只用中文”）持续注入，避免历史裁剪丢失指令。
- **Context Embedding**：`build_context_embedding` 返回池化向量（默认 384 维），可直接传入物理层做上下文 token。

### `ccp_framework.py` — 主干框架
- **PhysicalPerceptionLayer**：
  - RevIN 可选；输入投影 + 可选 context token 拼接；Mamba / Transformer / KAN / 混合骨干；异方差功率头输出 μ 和 log σ。
  - 通过 `context_embedding` 参数把智能体知识注入序列，自动调整位置编码与长度适配。
- **Physics-aware/Confidence Pipeline**：
  - 高斯 NLL 训练，`confidence` 由预测方差转换；支持功率通道残差输出与非负截断。
- **CausalReasoningLayer**：缓存/更新尾流因果图，提供 `consistency_loss`，支持从 batch 读取或在线发现。
- **CognitiveInterfaceLayer**：封装预测 + 解释 + 报告生成，直接调用 Wind-Agent 的 explain_prediction / generate_report。
- **create_ccp_system**：统一构建入口，配置骨干、KAN 模式、RevIN、物理/因果/认知组件与上下文注入。

### `physics_informed.py` — 物理损失与可学习参数
- **可学习物理常数 + 先验正则**：额定功率、尾流衰减等以可训练参数存在，使用正则化约束偏离初值，防止“学坏”。
- **多重物理约束**：贝兹极限、功率曲线、非负/切出/切入检查；返回 per-sample 损失以便掩码。
- **限电影响屏蔽**：对偏离物理基线过大的样本自动掩码，避免限电/异常数据触发错误物理惩罚。

### `mamba_module.py` — 时序骨干
- **SelectiveSSM / WindMambaformer / SiMBA**：核心序列建模模块，可选混合 KAN 或纯 Mamba 路径。
- **BiMamba 标注**：仅限编码器或补全场景，避免未来信息泄露到预测端。
- **上下文 token 自适配**：当物理层前置上下文 token 时，自动插值/扩展位置编码与长度适配器，保持输出步长与预测 horizon 对齐。

### 其他文件
- **`physics_informed.py`**：物理损失与掩码逻辑、物理参数先验正则。
- **`causal_discovery.py`**：PCMCI / LiNGAM 尾流因果发现，支持定期刷新缓存。
- **`kan_module.py`**：KAN 族组件与时间序列扩展，建议与 Mamba/Transformer 混合使用。
- **`lightweight_models.py`**：RevIN、TinyTimeMixer 等轻量模块，可用于蒸馏或边缘部署。
- **`train_eval.py`**：训练/验证脚本，含因果图批次接入、动态/不确定度加权、课程学习、蒸馏与评估。

## 使用指南
### 安装依赖
```bash
pip install torch numpy scipy einops scikit-learn
# 可选：LLM / RAG
pip install transformers sentence-transformers openai
```

### 构建预测系统并注入上下文
```python
from ccp_framework import create_ccp_system
from wind_agent import WindAgent
import torch

system = create_ccp_system(
    backbone="mamba", input_len=96, output_len=24, num_features=5,
    use_kan=True, kan_mode="hybrid", use_revin=True,
)

agent = WindAgent(conversation_max_turns=12, allow_physics_fallback=False)
context_vec = agent.build_context_embedding(query="今日天气风险", top_k=3)

x = torch.randn(4, 96, 5)
wind_dir = torch.rand(4) * 360
outputs = system(x, wind_dir, context_embedding=context_vec)
pred, conf = outputs["predictions"], outputs["confidence"]
```

### 带解释的智能体预测与报告
```python
from wind_agent import PredictionContext
ctx = PredictionContext(
    wind_speed=9.5, wind_direction=240, temperature=12.0,
    pressure=101.2, historical_power=[0.8, 0.9, 1.0, 1.1, 1.0, 0.95],
)
result = agent.predict_with_explanation(ctx, system=system, context_embedding=context_vec)
print(result["summary_report"])
```

### 训练示例
```python
from train_eval import run_experiment
run_experiment(
    backbone="mamba", use_kan=True, kan_mode="hybrid",
    loss_weighting="uncertainty", curriculum_warmup=5,
    distill_student=True, use_revin=True,
)
```

## 注意事项
- **LLM 必须真实可用**：未配置本地模型或 OpenAI Key 时，TemplateLLM 默认抛错；可显式允许占位模式才会降级。
- **物理参数正则**：通过 `physics_prior_strength` 调节先验约束，防止额定功率等被学偏。
- **因果图刷新**：数据集周期生成/缓存尾流因果图，并随 batch 返回 `causal_graph` 参与因果一致性损失。
- **上下文长度与指令保持**：对话历史有轮数/字符双重截断，可用 `register_session_directive` 固定全局指令避免被裁剪。
- **KAN 归一化要求**：开启 RevIN 或自行确保输入在合理范围，避免 B 样条激活/梯度为零。

## 目录导航
- `ccp_framework.py`：主框架、物理层/因果层/认知层定义
- `mamba_module.py`：Mamba 变体与位置适配
- `kan_module.py`：KAN 族与时间序列扩展
- `physics_informed.py`：物理损失、可学习参数与先验正则
- `causal_discovery.py`：PCMCI/LiNGAM 动态尾流图
- `lightweight_models.py`：RevIN、TinyTimeMixer 等轻量模型
- `wind_agent.py`：LLM + RAG 智能体与报告生成
- `train_eval.py`：训练/验证脚本，含因果图与蒸馏支持

## 许可证
MIT License
