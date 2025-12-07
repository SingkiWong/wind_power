# CCP Framework：因果-认知-物理一体化风电预测

本仓库实现了一个将 **因果发现、认知型智能体、物理约束** 与 **高效时序骨干（Mamba / Transformer / TinyTimeMixer）** 融合的风电功率预测体系。系统支持真实 LLM 与向量检索、可学习物理参数、因果一致性约束，以及将智能体检索到的上下文嵌入直接注入预测网络，实现 **神经-符号融合（Neuro-Symbolic AI）** 的端到端链路。

## 近期更新（重点）
- **智能体上下文注入**：Wind-Agent 检索到的文本会编码为向量，作为额外 token 送入 PhysicalPerceptionLayer，让预测直接感知“台风预警”等知识。
- **可学习物理参数 + 先验正则**：物理常数（额定功率、尾流衰减等）可微调，并通过先验正则抑制漂移，防止物理常量被模型“学坏”。
- **限电影响屏蔽**：PhysicsLoss 支持动态掩码，自动跳过疑似限电或异常样本，避免物理惩罚误伤真实标签。
- **因果图贯通训练**：数据集会生成/缓存尾流因果图并随批次下发，compute_loss 会将其缩放到注意力形状，使因果一致性损失真正参与训练。
- **真实 LLM 约束与报告生成**：默认要求本地 Transformers 或 OpenAI LLM，TemplateLLM 仅作为显式允许的占位；新增 ReportGenerator / explain_prediction 用于生成可追溯的报告。
- **RevIN + 异方差置信度**：入口可选 RevIN 稳定量纲；功率通道输出均值与 log-variance，使用高斯 NLL 训练，置信度由方差反推而非拍脑袋。

## 架构总览
```
物理感知层 (RevIN → Mamba/Transformer/KAN → 异方差头)
   ↑          ↑
   │          └─ 神经-符号融合：Agent 检索到的上下文向量作为额外 token 拼接
   │
因果推理层：PCMCI / LiNGAM / 动态尾流图（可缓存刷新，参与因果一致性损失）
   │
认知交互层：Wind-Agent (RAG + LLM 推理 + 反思) → 报告 / 对话 / 指令回灌
```

## 主要模块
- **`wind_agent.py`**
  - 真实 LLM（Transformers / OpenAI）驱动的推理与对话；TemplateLLM 默认禁用。
  - 向量检索：sentence-transformer 优先，TF-IDF 回退，无哈希碰撞；支持 JSONL 动态摄取。
  - Reflection：硬护栏安全检查 + LLM 语义批注；ReportGenerator 输出摘要/详情报告。
  - Context Embedding：提供知识库池化向量，供物理层作为上下文 token 使用。

- **`ccp_framework.py`**
  - PhysicalPerceptionLayer 支持 RevIN、上下文 token 拼接、KAN/Mamba/Transformer 骨干。
  - Heteroscedastic 头输出功率均值与方差，高斯 NLL + 方差派生置信度。
  - CognitiveInterfaceLayer 将 Agent 解释/反思与预测结果组合，支持预测与报告一键返回。

- **`physics_informed.py`**
  - PhysicsLoss 含贝兹极限、功率曲线、非负约束；可学习物理参数 + 先验正则。
  - Curtailment-aware 掩码：偏离物理基线过大的样本可排除物理惩罚。

- **`mamba_module.py`**
  - SelectiveSSM / WindMambaformer / SiMBA；BiMamba 标注为仅限编码器使用（防信息泄露）。
  - 在上下文 token 预置时自动插值位置编码与长度自适配，保持输出与预测步长对齐。

- **`train_eval.py`**
  - 课程学习 + 动态/不确定度权重平衡；TinyTimeMixer 蒸馏可选。
  - 批量可带因果图进入 compute_loss，因果一致性损失实时生效。

## 快速上手
```bash
pip install torch numpy scipy einops scikit-learn
# 可选：LLM / RAG
# pip install transformers sentence-transformers openai
```

### 构建系统并预测
```python
from ccp_framework import create_ccp_system
import torch

system = create_ccp_system(
    backbone="mamba", input_len=96, output_len=24, num_features=5,
    use_kan=True, kan_mode="hybrid", use_revin=True
)

# agent 先检索/编码上下文（如天气预警），返回 embedding
from wind_agent import WindAgent
agent = WindAgent(conversation_max_turns=12, allow_physics_fallback=False)
context_vec = agent.build_context_embedding(query="今日天气风险", top_k=3)

x = torch.randn(4, 96, 5)
wind_dir = torch.rand(4) * 360
outputs = system(x, wind_dir, context_embedding=context_vec)
pred, conf = outputs["predictions"], outputs["confidence"]
```

### 带解释的智能体预测
```python
from wind_agent import PredictionContext
ctx = PredictionContext(wind_speed=9.5, wind_direction=240, temperature=12.0,
                        pressure=101.2, historical_power=[0.8, 0.9, 1.0, 1.1, 1.0, 0.95])
result = agent.predict_with_explanation(ctx, system=system, context_embedding=context_vec)
print(result["summary_report"])
```

### 训练
```python
from train_eval import run_experiment
run_experiment(
    backbone="mamba", use_kan=True, kan_mode="hybrid",
    loss_weighting="uncertainty", curriculum_warmup=5,
    distill_student=True, use_revin=True
)
```

## 注意事项
- **LLM 必须真实可用**：未配置本地模型或 OpenAI Key 时，TemplateLLM 默认抛错，需显式允许占位模式才会降级。
- **物理参数正则**：可通过 `physics_prior_strength` 调整正则强度，防止额定功率等被学偏。
- **因果图刷新**：数据集会周期生成/缓存尾流因果图，需确保 `batch['causal_graph']` 随数据加载返回。
- **上下文长度**：对话历史有轮数和字符双重截断；可通过 `register_session_directive` 固定全局指令避免被裁剪。

## 目录
- `ccp_framework.py`：主框架、物理层/因果层/认知层定义
- `mamba_module.py`：Mamba 变体与位置适配
- `kan_module.py`：KAN 族与时间序列扩展
- `physics_informed.py`：物理损失与可学习物理参数
- `causal_discovery.py`：PCMCI/LiNGAM 动态尾流图
- `lightweight_models.py`：RevIN、TinyTimeMixer 等轻量模型
- `wind_agent.py`：LLM + RAG 智能体与报告生成
- `train_eval.py`：训练/验证脚本，含因果图与蒸馏支持

## 许可证
MIT License
