"""
Wind-Agent: LLM-Driven Intelligent Agent for Wind Power Prediction
风电智能体模块

核心功能：
1. 思维链（CoT）推理 - 解释预测背后的逻辑
2. 检索增强生成（RAG）- 基于知识库的证据溯源
3. 多轮对话交互 - 回应操作员质询
4. 预测-反思-修正闭环

将"黑盒预测"转化为"可解释的智能专家"
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np
from datetime import datetime
import torch.nn.functional as F


class AgentRole(Enum):
    """智能体角色"""
    PREDICTOR = "predictor"      # 预测智能体
    REFLECTOR = "reflector"      # 反思智能体
    REFINER = "refiner"          # 修正智能体
    EXPLAINER = "explainer"      # 解释智能体


@dataclass
class PredictionContext:
    """预测上下文"""
    timestamp: datetime
    wind_speed: float
    wind_direction: float
    temperature: float
    pressure: float
    turbine_id: int
    historical_power: List[float]
    nwp_forecast: Optional[Dict] = None
    
    def to_prompt(self) -> str:
        """转换为提示文本"""
        return f"""
当前时间: {self.timestamp}
风机ID: T{self.turbine_id}
当前风速: {self.wind_speed:.1f} m/s
当前风向: {self.wind_direction:.0f}°
温度: {self.temperature:.1f}°C
气压: {self.pressure:.0f} hPa
过去6小时功率: {[f'{p:.2f}' for p in self.historical_power[-6:]]} MW
"""


@dataclass
class PredictionResult:
    """预测结果"""
    predicted_power: float
    confidence: float
    reasoning_chain: List[str]
    evidence: List[Dict]
    warnings: List[str]


class VectorIndex:
    """轻量级向量索引，模拟向量数据库的语义检索能力"""

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vectors: List[torch.Tensor] = []
        self.metadata: List[Dict] = []

    def _encode(self, text: str) -> torch.Tensor:
        tokens = text.lower().split()
        vec = torch.zeros(self.dim)
        for tok in tokens:
            idx = hash(tok) % self.dim
            vec[idx] += 1.0
        return F.normalize(vec, dim=0) if vec.norm() > 0 else vec

    def add(self, doc: Dict):
        content = doc.get('content', '') + ' ' + doc.get('title', '')
        self.vectors.append(self._encode(content))
        self.metadata.append(doc)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        query_vec = self._encode(query)
        if len(self.vectors) == 0:
            return []
        matrix = torch.stack(self.vectors)
        scores = torch.mv(matrix, query_vec)
        topk = torch.topk(scores, k=min(top_k, scores.numel())).indices.tolist()
        return [self.metadata[i] for i in topk]


class KnowledgeBase:
    """
    风电知识库
    
    存储：
    - 风机技术手册
    - 历史故障记录
    - 气象灾害事件
    - 运维日志
    """
    
    def __init__(self):
        # 模拟知识库条目
        self.documents = {
            'technical_manual': [
                {
                    'id': 'TM001',
                    'title': '风机功率曲线规范',
                    'content': '额定功率2MW，切入风速3m/s，额定风速12m/s，切出风速25m/s。',
                    'embedding': None  # 实际应用中会有向量嵌入
                },
                {
                    'id': 'TM002',
                    'title': '偏航系统说明',
                    'content': '偏航角偏差超过15度时，发电效率下降约10-20%。',
                    'embedding': None
                },
                {
                    'id': 'TM003',
                    'title': '尾流效应说明',
                    'content': '下游风机在尾流区内功率损失可达40%，取决于间距和风向。',
                    'embedding': None
                }
            ],
            'fault_records': [
                {
                    'id': 'FR001',
                    'date': '2023-01-15',
                    'turbine': 'T5',
                    'fault_type': '齿轮箱油温过高',
                    'symptom': '功率曲线偏离，伴随振动异常',
                    'solution': '更换润滑油，检查冷却系统'
                },
                {
                    'id': 'FR002',
                    'date': '2023-06-20',
                    'turbine': 'T3',
                    'fault_type': '变桨系统故障',
                    'symptom': '功率波动剧烈，限功率运行',
                    'solution': '更换变桨电机'
                }
            ],
            'weather_events': [
                {
                    'id': 'WE001',
                    'date': '2023-08-10',
                    'event': '台风天鹅',
                    'impact': '全场停机48小时',
                    'wind_speed_peak': '35 m/s'
                }
            ]
        }

        # 向量索引支持语义检索与动态扩充
        self.vector_index = VectorIndex(dim=128)
        for category_docs in self.documents.values():
            for doc in category_docs:
                self.vector_index.add(doc)

    def ingest_documents(self, records: List[Dict]):
        """动态接入SCADA日志或运维工单，保持知识库实时性"""
        for rec in records:
            rec.setdefault('id', f"EXT-{len(self.vector_index.metadata)+1:04d}")
            self.vector_index.add(rec)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回文档数量
            
        Returns:
            相关文档列表
        """
        return self.vector_index.search(query, top_k=top_k)


class ChainOfThought:
    """
    思维链推理模块
    
    实现显式的逻辑推理过程，使预测可解释
    """
    
    def __init__(self):
        self.reasoning_templates = {
            'wind_speed_check': """
步骤1: 风速分析
- 当前风速: {wind_speed} m/s
- 判断: {wind_speed_judgment}
- 预期影响: {wind_speed_impact}
""",
            'wind_direction_check': """
步骤2: 风向分析
- 当前风向: {wind_direction}°
- 上游风机: {upstream_turbines}
- 尾流影响: {wake_impact}
""",
            'historical_pattern': """
步骤3: 历史模式匹配
- 近期趋势: {trend}
- 类似工况记录: {similar_cases}
- 预期变化: {expected_change}
""",
            'physics_constraint': """
步骤4: 物理约束检查
- 贝兹极限检查: {betz_check}
- 功率曲线一致性: {power_curve_check}
- 额定功率约束: {rated_power_check}
""",
            'final_prediction': """
步骤5: 综合预测
- 预测功率: {predicted_power} MW
- 置信度: {confidence}%
- 主要依据: {main_evidence}
"""
        }
    
    def generate_reasoning(
        self,
        context: PredictionContext,
        prediction: float,
        wake_info: Optional[Dict] = None,
        similar_cases: Optional[List] = None
    ) -> List[str]:
        """
        生成思维链推理过程
        """
        reasoning_chain = []
        
        # 步骤1: 风速分析
        if context.wind_speed < 3:
            wind_judgment = "低于切入风速"
            wind_impact = "预期功率为0"
        elif context.wind_speed < 12:
            wind_judgment = "处于爬坡区"
            wind_impact = "功率随风速立方增长"
        elif context.wind_speed < 25:
            wind_judgment = "处于额定区"
            wind_impact = "预期满发或接近满发"
        else:
            wind_judgment = "超过切出风速"
            wind_impact = "预期停机保护"
        
        reasoning_chain.append(self.reasoning_templates['wind_speed_check'].format(
            wind_speed=context.wind_speed,
            wind_speed_judgment=wind_judgment,
            wind_speed_impact=wind_impact
        ))
        
        # 步骤2: 风向分析
        upstream = wake_info.get('upstream', []) if wake_info else []
        wake_impact = wake_info.get('wake_loss', 0) if wake_info else 0
        
        reasoning_chain.append(self.reasoning_templates['wind_direction_check'].format(
            wind_direction=context.wind_direction,
            upstream_turbines=upstream if upstream else "无",
            wake_impact=f"约{wake_impact*100:.0f}%功率损失" if wake_impact > 0 else "无明显影响"
        ))
        
        # 步骤3: 历史模式
        if context.historical_power:
            recent = context.historical_power[-3:]
            if len(recent) >= 2:
                if recent[-1] > recent[0] * 1.1:
                    trend = "上升趋势"
                elif recent[-1] < recent[0] * 0.9:
                    trend = "下降趋势"
                else:
                    trend = "平稳"
            else:
                trend = "数据不足"
        else:
            trend = "无历史数据"
        
        reasoning_chain.append(self.reasoning_templates['historical_pattern'].format(
            trend=trend,
            similar_cases=len(similar_cases) if similar_cases else 0,
            expected_change="延续当前趋势" if trend != "无历史数据" else "不确定"
        ))
        
        # 步骤4: 物理约束
        rated_power = 2.0  # MW
        betz_check = "通过" if prediction <= rated_power else "警告：超过额定功率"
        power_curve_check = "一致" if (
            (context.wind_speed < 3 and prediction < 0.1) or
            (context.wind_speed >= 3 and prediction > 0)
        ) else "存在偏差"
        
        reasoning_chain.append(self.reasoning_templates['physics_constraint'].format(
            betz_check=betz_check,
            power_curve_check=power_curve_check,
            rated_power_check="通过" if prediction <= rated_power else "需修正"
        ))
        
        # 步骤5: 最终预测
        confidence = 85 if power_curve_check == "一致" else 70
        main_evidence = f"风速{context.wind_speed}m/s，{trend}"
        
        reasoning_chain.append(self.reasoning_templates['final_prediction'].format(
            predicted_power=f"{prediction:.2f}",
            confidence=confidence,
            main_evidence=main_evidence
        ))
        
        return reasoning_chain


class ReflectionAgent:
    """
    反思智能体
    
    实现预测-反思-修正闭环
    扮演"批评家"角色，检查预测的物理一致性
    """
    
    def __init__(self, rated_power: float = 2.0):
        self.rated_power = rated_power
        self.cut_in = 3.0
        self.cut_out = 25.0
        self.rated_speed = 12.0
    
    def reflect(self, context: PredictionContext) -> Dict[str, Any]:
        """
        对预测结果进行反思
        
        Returns:
            reflection: 包含批评意见和建议修正的字典
        """
        criticisms = []
        corrections = []
        score = 100  # 满分100
        
        # 检查1: 物理边界
        if context.predicted_power < 0:
            criticisms.append("预测功率为负值，违反物理约束")
            corrections.append({"type": "clamp", "min": 0})
            score -= 30
        
        if context.predicted_power > self.rated_power:
            criticisms.append(f"预测功率{context.predicted_power:.2f}MW超过额定功率{self.rated_power}MW")
            corrections.append({"type": "clamp", "max": self.rated_power})
            score -= 20
        
        # 检查2: 功率曲线一致性
        ws = context.wind_speed
        if ws < self.cut_in and context.predicted_power > 0.05:
            criticisms.append(f"风速{ws:.1f}m/s低于切入风速，但预测功率非零")
            corrections.append({"type": "set", "value": 0})
            score -= 25
        
        if ws > self.cut_out and context.predicted_power > 0.05:
            criticisms.append(f"风速{ws:.1f}m/s超过切出风速，但预测功率非零")
            corrections.append({"type": "set", "value": 0})
            score -= 25
        
        # 检查3: 合理性范围
        if self.cut_in <= ws < self.rated_speed:
            expected_max = self.rated_power * ((ws - self.cut_in) / (self.rated_speed - self.cut_in)) ** 3
            if context.predicted_power > expected_max * 1.3:
                criticisms.append(f"爬坡区功率偏高，超过理论值30%以上")
                corrections.append({"type": "scale", "factor": 0.85})
                score -= 15
        
        # 检查4: 置信度
        if context.confidence < 0.5:
            criticisms.append(f"模型置信度过低({context.confidence:.0%})，预测可靠性存疑")
            score -= 10
        
        return {
            "criticisms": criticisms,
            "corrections": corrections,
            "physical_consistency_score": score,
            "needs_refinement": len(corrections) > 0,
            "recommendation": "建议修正后重新评估" if corrections else "预测结果通过物理一致性检查"
        }
    
    def refine(
        self,
        context: PredictionContext,
        reflection: Dict
    ) -> float:
        """
        根据反思结果修正预测
        
        Returns:
            refined_power: 修正后的功率预测
        """
        power = context.predicted_power
        
        for correction in reflection['corrections']:
            if correction['type'] == 'clamp':
                if 'min' in correction:
                    power = max(power, correction['min'])
                if 'max' in correction:
                    power = min(power, correction['max'])
            elif correction['type'] == 'set':
                power = correction['value']
            elif correction['type'] == 'scale':
                power = power * correction['factor']
        
        return power


class WindAgent:
    """
    Wind-Agent: 风电预测智能体
    
    整合所有组件，提供完整的认知交互能力
    """
    
    def __init__(self, rated_power: float = 2.0):
        self.kb = KnowledgeBase()
        self.reasoning = ReasoningChain(self.kb)
        self.rag = RAGEngine(self.kb)
        self.reflection = ReflectionAgent(rated_power)
        self.rated_power = rated_power
    
    def explain_prediction(
        self,
        context: PredictionContext
    ) -> ExplanationResult:
        """
        生成完整的预测解释
        
        Args:
            context: 预测上下文
            
        Returns:
            ExplanationResult: 完整的解释结果
        """
        # 1. 思维链推理
        reasoning_chain = self.reasoning.analyze_prediction(context)
        
        # 2. 检索增强
        evidence, augmented_text = self.rag.retrieve_and_augment(context)
        
        # 3. 反思检查
        reflection = self.reflection.reflect(context)
        
        # 4. 如果需要修正
        if reflection['needs_refinement']:
            refined_power = self.reflection.refine(context, reflection)
            reasoning_chain.append(
                f"[修正] 基于物理一致性检查，功率预测从{context.predicted_power:.2f}MW"
                f"修正为{refined_power:.2f}MW。"
            )
        
        # 5. 生成摘要
        summary = self._generate_summary(context, reasoning_chain, reflection)
        
        # 6. 生成建议
        recommendations = self._generate_recommendations(context, reflection)
        
        # 7. 确定置信度等级
        confidence_level = self._get_confidence_level(context.confidence, reflection)
        
        return ExplanationResult(
            summary=summary,
            reasoning_chain=reasoning_chain,
            evidence=evidence,
            confidence_level=confidence_level,
            recommendations=recommendations
        )
    
    def _generate_summary(
        self,
        context: PredictionContext,
        reasoning_chain: List[str],
        reflection: Dict
    ) -> str:
        """生成简洁摘要"""
        ws = context.wind_speed
        power = context.predicted_power
        
        # 确定主要影响因素
        if ws < 3:
            main_factor = "低风速待机"
        elif ws > 25:
            main_factor = "高风速保护停机"
        elif context.causal_graph and context.causal_graph.get('upstream'):
            main_factor = "受尾流效应影响"
        else:
            main_factor = "正常运行"
        
        consistency = "通过" if reflection['physical_consistency_score'] >= 80 else "需关注"
        
        return (
            f"{context.turbine_id}在{context.timestamp}时刻预测功率为{power:.2f}MW，"
            f"主要因素：{main_factor}，物理一致性检查{consistency}，"
            f"置信度{context.confidence:.0%}。"
        )
    
    def _generate_recommendations(
        self,
        context: PredictionContext,
        reflection: Dict
    ) -> List[str]:
        """生成操作建议"""
        recommendations = []
        
        # 基于风速的建议
        if context.wind_speed > 20:
            recommendations.append("注意监控风速变化，准备应对可能的切出停机")
        
        if context.wind_speed < 5:
            recommendations.append("低风速期间关注启停频繁带来的设备磨损")
        
        # 基于置信度的建议
        if context.confidence < 0.7:
            recommendations.append("预测不确定性较高，建议增加备用容量")
        
        # 基于反思结果的建议
        if reflection['physical_consistency_score'] < 70:
            recommendations.append("预测结果与物理模型偏差较大，建议人工复核")
        
        # 基于尾流的建议
        if context.causal_graph and context.causal_graph.get('upstream'):
            recommendations.append("考虑实施偏航优化策略减少尾流损失")
        
        if not recommendations:
            recommendations.append("当前运行状态正常，维持常规监控")
        
        return recommendations
    
    def _get_confidence_level(
        self,
        confidence: float,
        reflection: Dict
    ) -> str:
        """确定置信度等级"""
        adjusted = confidence * (reflection['physical_consistency_score'] / 100)
        
        if adjusted >= 0.85:
            return "高"
        elif adjusted >= 0.65:
            return "中"
        else:
            return "低"
    
    def interactive_query(self, question: str, context: PredictionContext) -> str:
        """
        交互式问答
        
        Args:
            question: 用户问题
            context: 当前上下文
            
        Returns:
            answer: 回答
        """
        question_lower = question.lower()
        
        # 关于功率预测
        if any(kw in question_lower for kw in ['为什么', '功率', '预测']):
            explanation = self.explain_prediction(context)
            return explanation.summary + "\n\n详细推理：\n" + "\n".join(explanation.reasoning_chain)
        
        # 关于尾流
        if any(kw in question_lower for kw in ['尾流', '上游', '影响']):
            if context.causal_graph and context.causal_graph.get('upstream'):
                upstream = context.causal_graph['upstream']
                return f"当前风向下，{context.turbine_id}受到以下机组尾流影响：" + \
                       "、".join([f"{u['id']}(影响{u.get('wake_loss',0)*100:.1f}%)" for u in upstream])
            return f"当前风向下，{context.turbine_id}不受明显尾流影响。"
        
        # 关于建议
        if any(kw in question_lower for kw in ['建议', '怎么办', '操作']):
            explanation = self.explain_prediction(context)
            return "操作建议：\n" + "\n".join(f"• {r}" for r in explanation.recommendations)
        
        # 默认回答
        return f"当前{context.turbine_id}预测功率{context.predicted_power:.2f}MW，" \
               f"风速{context.wind_speed:.1f}m/s，置信度{context.confidence:.0%}。" \
               f"如需详细解释，请询问'为什么预测这个功率？'"


class ReportGenerator:
    """
    报告生成器
    
    生成综合预测报告
    """
    
    def __init__(self, agent: WindAgent):
        self.agent = agent
    
    def generate_report(
        self,
        contexts: List[PredictionContext],
        report_type: str = "summary"
    ) -> str:
        """
        生成预测报告
        
        Args:
            contexts: 多个预测上下文
            report_type: 报告类型 ("summary", "detailed", "technical")
        """
        if report_type == "summary":
            return self._generate_summary_report(contexts)
        elif report_type == "detailed":
            return self._generate_detailed_report(contexts)
        else:
            return self._generate_technical_report(contexts)
    
    def _generate_summary_report(self, contexts: List[PredictionContext]) -> str:
        """生成摘要报告"""
        total_power = sum(c.predicted_power for c in contexts)
        avg_confidence = np.mean([c.confidence for c in contexts])
        
        lines = [
            "=" * 50,
            "风电场功率预测摘要报告",
            "=" * 50,
            f"报告时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"预测机组数: {len(contexts)}",
            f"总预测功率: {total_power:.2f} MW",
            f"平均置信度: {avg_confidence:.0%}",
            "-" * 50,
            "各机组预测:",
        ]
        
        for ctx in contexts:
            status = "正常" if ctx.confidence > 0.7 else "需关注"
            lines.append(f"  {ctx.turbine_id}: {ctx.predicted_power:.2f}MW [{status}]")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def _generate_detailed_report(self, contexts: List[PredictionContext]) -> str:
        """生成详细报告"""
        lines = [
            "=" * 60,
            "风电场功率预测详细报告",
            "=" * 60,
        ]
        
        for ctx in contexts:
            explanation = self.agent.explain_prediction(ctx)
            lines.extend([
                f"\n【{ctx.turbine_id}】",
                f"预测功率: {ctx.predicted_power:.2f} MW",
                f"置信度: {ctx.confidence:.0%} ({explanation.confidence_level})",
                f"风速: {ctx.wind_speed:.1f} m/s",
                f"风向: {ctx.wind_direction:.0f}°",
                "\n推理过程:",
            ])
            for step in explanation.reasoning_chain:
                lines.append(f"  {step}")
            
            lines.append("\n建议:")
            for rec in explanation.recommendations:
                lines.append(f"  • {rec}")
            
            lines.append("-" * 60)
        
        return "\n".join(lines)
    
    def _generate_technical_report(self, contexts: List[PredictionContext]) -> str:
        """生成技术报告（JSON格式）"""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "turbines": []
        }
        
        for ctx in contexts:
            reflection = self.agent.reflection.reflect(ctx)
            report_data["turbines"].append({
                "id": ctx.turbine_id,
                "predicted_power_mw": ctx.predicted_power,
                "confidence": ctx.confidence,
                "wind_speed_ms": ctx.wind_speed,
                "wind_direction_deg": ctx.wind_direction,
                "physical_consistency_score": reflection["physical_consistency_score"],
                "criticisms": reflection["criticisms"]
            })
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("Wind-Agent Module Test")
    print("=" * 60)
    
    # 创建智能体
    agent = WindAgent(rated_power=2.0)
    
    # 创建测试上下文
    test_context = PredictionContext(
        timestamp=datetime.now(),
        wind_speed=10.5,
        wind_direction=270,
        temperature=15.0,
        pressure=1013.25,
        turbine_id="T5",
        predicted_power=1.35,
        confidence=0.82,
        causal_graph={
            'upstream': [
                {'id': 'T1', 'wake_loss': 0.15},
                {'id': 'T2', 'wake_loss': 0.08}
            ]
        }
    )
    
    # 测试解释生成
    print("\n1. Testing Explanation Generation:")
    explanation = agent.explain_prediction(test_context)
    print(f"   Summary: {explanation.summary}")
    print(f"   Confidence Level: {explanation.confidence_level}")
    print(f"   Reasoning Steps: {len(explanation.reasoning_chain)}")
    
    # 测试交互式问答
    print("\n2. Testing Interactive Query:")
    questions = [
        "为什么预测这个功率？",
        "有尾流影响吗？",
        "有什么操作建议？"
    ]
    for q in questions:
        print(f"\n   Q: {q}")
        print(f"   A: {agent.interactive_query(q, test_context)[:200]}...")
    
    # 测试报告生成
    print("\n3. Testing Report Generation:")
    contexts = [
        test_context,
        PredictionContext(
            timestamp=datetime.now(),
            wind_speed=12.0,
            wind_direction=270,
            temperature=15.0,
            pressure=1013.25,
            turbine_id="T6",
            predicted_power=1.85,
            confidence=0.91,
            causal_graph=None
        )
    ]
    
    reporter = ReportGenerator(agent)
    summary = reporter.generate_report(contexts, "summary")
    print(summary)
    
    print("\n" + "=" * 60)
    print("Wind-Agent Module Test Complete!")
    print("=" * 60)
    """
    反思智能体
    
    扮演"批评家"角色，检查预测的物理一致性和逻辑合理性
    """
    
    def __init__(self, rated_power: float = 2.0):
        self.rated_power = rated_power
        self.cut_in_speed = 3.0
        self.cut_out_speed = 25.0
    
    def reflect(
        self,
        prediction: float,
        context: PredictionContext,
        reasoning_chain: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        反思预测结果
        
        Returns:
            is_valid: 是否通过检查
            criticisms: 批评意见列表
        """
        criticisms = []
        is_valid = True
        
        # 检查1: 额定功率约束
        if prediction > self.rated_power:
            criticisms.append(
                f"批评：预测功率{prediction:.2f}MW超过额定功率{self.rated_power}MW"
            )
            is_valid = False
        
        # 检查2: 非负约束
        if prediction < 0:
            criticisms.append(
                f"批评：预测功率{prediction:.2f}MW为负值，违反物理定律"
            )
            is_valid = False
        
        # 检查3: 切入风速约束
        if context.wind_speed < self.cut_in_speed and prediction > 0.1:
            criticisms.append(
                f"批评：风速{context.wind_speed}m/s低于切入风速，"
                f"但预测功率{prediction:.2f}MW不为零"
            )
            is_valid = False
        
        # 检查4: 切出风速约束
        if context.wind_speed > self.cut_out_speed and prediction > 0.1:
            criticisms.append(
                f"批评：风速{context.wind_speed}m/s超过切出风速，"
                f"应停机但预测功率{prediction:.2f}MW"
            )
            is_valid = False
        
        # 检查5: 功率曲线合理性
        if self.cut_in_speed <= context.wind_speed <= 12:
            # 爬坡区：功率应与风速立方相关
            expected_ratio = ((context.wind_speed - self.cut_in_speed) / 
                            (12 - self.cut_in_speed)) ** 3
            actual_ratio = prediction / self.rated_power
            
            if abs(actual_ratio - expected_ratio) > 0.3:
                criticisms.append(
                    f"批评：爬坡区功率比{actual_ratio:.2f}与预期{expected_ratio:.2f}偏差较大"
                )
        
        # 检查6: 历史一致性
        if context.historical_power:
            recent_avg = np.mean(context.historical_power[-3:])
            if abs(prediction - recent_avg) > 0.5 * self.rated_power:
                criticisms.append(
                    f"提醒：预测值{prediction:.2f}MW与近期均值{recent_avg:.2f}MW差异较大，"
                    f"请确认是否有工况突变"
                )
        
        return is_valid, criticisms


class RefinementAgent:
    """
    修正智能体
    
    根据反思意见修正预测结果
    """
    
    def __init__(self, rated_power: float = 2.0):
        self.rated_power = rated_power
        self.cut_in_speed = 3.0
        self.cut_out_speed = 25.0
    
    def refine(
        self,
        original_prediction: float,
        context: PredictionContext,
        criticisms: List[str]
    ) -> Tuple[float, str]:
        """
        修正预测
        
        Returns:
            refined_prediction: 修正后的预测
            refinement_note: 修正说明
        """
        refined = original_prediction
        notes = []
        
        # 修正1: 限制在额定功率以内
        if refined > self.rated_power:
            refined = self.rated_power
            notes.append(f"功率限制在额定值{self.rated_power}MW")
        
        # 修正2: 非负
        if refined < 0:
            refined = 0
            notes.append("功率修正为非负值")
        
        # 修正3: 切入风速以下归零
        if context.wind_speed < self.cut_in_speed:
            refined = 0
            notes.append(f"风速{context.wind_speed}m/s低于切入，功率归零")
        
        # 修正4: 切出风速以上归零
        if context.wind_speed > self.cut_out_speed:
            refined = 0
            notes.append(f"风速{context.wind_speed}m/s超切出，停机保护")
        
        # 修正5: 功率曲线平滑
        if self.cut_in_speed <= context.wind_speed <= 12:
            expected_ratio = ((context.wind_speed - self.cut_in_speed) / 
                            (12 - self.cut_in_speed)) ** 3
            expected = expected_ratio * self.rated_power
            
            # 取原预测和期望值的加权平均
            refined = 0.7 * refined + 0.3 * expected
            notes.append(f"功率曲线平滑修正至{refined:.2f}MW")
        
        refinement_note = "; ".join(notes) if notes else "无需修正"
        
        return refined, refinement_note


class WindAgent:
    """
    风电智能体主类
    
    集成：
    - 预测模型
    - 知识库检索（RAG）
    - 思维链推理（CoT）
    - 反思-修正闭环
    """
    
    def __init__(
        self,
        prediction_model: Optional[nn.Module] = None,
        rated_power: float = 2.0
    ):
        self.prediction_model = prediction_model
        self.knowledge_base = KnowledgeBase()
        self.cot = ChainOfThought()
        self.reflector = ReflectionAgent(rated_power)
        self.refiner = RefinementAgent(rated_power)
        
        self.conversation_history = []
    
    def predict_with_explanation(
        self,
        context: PredictionContext,
        wake_info: Optional[Dict] = None
    ) -> PredictionResult:
        """
        带解释的预测
        
        实现完整的预测-反思-修正闭环
        """
        # 1. 检索相关知识
        query = f"风速{context.wind_speed} 风向{context.wind_direction} 功率预测"
        evidence = self.knowledge_base.search(query)
        
        # 2. 初始预测（简化：基于物理模型）
        if context.wind_speed < 3:
            initial_prediction = 0.0
        elif context.wind_speed < 12:
            ratio = ((context.wind_speed - 3) / 9) ** 3
            initial_prediction = ratio * 2.0
        elif context.wind_speed < 25:
            initial_prediction = 2.0
        else:
            initial_prediction = 0.0
        
        # 考虑尾流
        if wake_info and wake_info.get('wake_loss', 0) > 0:
            initial_prediction *= (1 - wake_info['wake_loss'])
        
        # 3. 生成推理链
        reasoning_chain = self.cot.generate_reasoning(
            context, initial_prediction, wake_info
        )
        
        # 4. 反思检查
        is_valid, criticisms = self.reflector.reflect(
            initial_prediction, context, reasoning_chain
        )
        
        # 5. 必要时修正
        if not is_valid:
            final_prediction, refinement_note = self.refiner.refine(
                initial_prediction, context, criticisms
            )
            reasoning_chain.append(f"\n修正过程: {refinement_note}")
            reasoning_chain.extend([f"反思意见: {c}" for c in criticisms])
        else:
            final_prediction = initial_prediction
        
        # 6. 计算置信度
        confidence = 0.9 if is_valid else 0.7
        
        return PredictionResult(
            predicted_power=final_prediction,
            confidence=confidence,
            reasoning_chain=reasoning_chain,
            evidence=evidence,
            warnings=criticisms if criticisms else []
        )
    
    def answer_question(
        self,
        question: str,
        context: Optional[PredictionContext] = None,
        prediction_result: Optional[PredictionResult] = None
    ) -> str:
        """
        回答操作员问题
        
        支持的问题类型：
        - 为什么预测这个值？
        - 尾流影响多大？
        - 类似情况历史上怎样？
        """
        # 记录对话
        self.conversation_history.append({
            'role': 'user',
            'content': question
        })
        
        # 检索相关知识
        evidence = self.knowledge_base.search(question)
        
        # 生成回答
        answer = self._generate_answer(question, context, prediction_result, evidence)
        
        self.conversation_history.append({
            'role': 'assistant',
            'content': answer
        })
        
        return answer
    
    def _generate_answer(
        self,
        question: str,
        context: Optional[PredictionContext],
        result: Optional[PredictionResult],
        evidence: List[Dict]
    ) -> str:
        """生成回答"""
        
        # 问题类型判断和回答
        if "为什么" in question or "原因" in question:
            if result and result.reasoning_chain:
                return "基于以下推理过程：\n" + "\n".join(result.reasoning_chain)
            else:
                return "抱歉，没有可用的推理过程。请先进行预测。"
        
        elif "尾流" in question:
            if context:
                return f"""
关于尾流效应的分析：
- 当前风向: {context.wind_direction}°
- 根据知识库记录: {evidence[0]['content'] if evidence else '下游风机在尾流区内功率损失可达40%'}
- 建议: 检查上游风机运行状态，必要时调整偏航角
"""
            else:
                return "请提供当前的风向和风机布局信息。"
        
        elif "历史" in question or "类似" in question:
            fault_records = self.knowledge_base.documents.get('fault_records', [])
            if fault_records:
                recent = fault_records[0]
                return f"""
历史相似案例：
- 日期: {recent['date']}
- 风机: {recent['turbine']}
- 故障类型: {recent['fault_type']}
- 症状: {recent['symptom']}
- 解决方案: {recent['solution']}
"""
            else:
                return "未找到相似的历史案例。"
        
        elif "置信度" in question or "可靠" in question:
            if result:
                return f"""
预测置信度分析：
- 当前置信度: {result.confidence * 100:.0f}%
- 影响因素:
  - 数据质量: {'良好' if context and context.historical_power else '数据不足'}
  - 物理一致性: {'通过' if not result.warnings else '存在警告'}
  - 历史验证: 基于过去类似工况的准确率
- 警告信息: {result.warnings if result.warnings else '无'}
"""
            else:
                return "请先进行预测以获取置信度信息。"
        
        else:
            # 通用回答
            if evidence:
                return f"根据知识库：\n{evidence[0].get('content', '无相关信息')}"
            else:
                return "抱歉，我无法理解您的问题。请尝试询问关于预测原因、尾流影响或历史案例的问题。"
    
    def generate_report(
        self,
        context: PredictionContext,
        result: PredictionResult
    ) -> str:
        """
        生成综合预测报告
        """
        report = f"""
========================================
        风电功率预测报告
========================================

【基本信息】
- 时间: {context.timestamp}
- 风机: T{context.turbine_id}
- 风速: {context.wind_speed} m/s
- 风向: {context.wind_direction}°

【预测结果】
- 预测功率: {result.predicted_power:.2f} MW
- 置信度: {result.confidence * 100:.0f}%

【推理过程】
{''.join(result.reasoning_chain)}

【参考依据】
"""
        for i, ev in enumerate(result.evidence, 1):
            report += f"{i}. [{ev.get('id', 'N/A')}] {ev.get('title', ev.get('fault_type', 'N/A'))}\n"
        
        if result.warnings:
            report += "\n【警告信息】\n"
            for w in result.warnings:
                report += f"⚠️ {w}\n"
        
        report += """
【操作建议】
- 持续监控风向变化
- 关注上游风机运行状态
- 如预测偏差过大，请检查传感器状态

========================================
"""
        return report


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("Wind-Agent: LLM-Driven Intelligent Agent")
    print("=" * 60)
    
    # 创建智能体
    agent = WindAgent(rated_power=2.0)
    
    # 创建测试上下文
    context = PredictionContext(
        timestamp=datetime.now(),
        wind_speed=10.5,
        wind_direction=270,
        temperature=15.0,
        pressure=1013,
        turbine_id=5,
        historical_power=[1.2, 1.3, 1.4, 1.5, 1.6, 1.5]
    )
    
    # 测试预测
    print("\n1. Testing Prediction with Explanation:")
    wake_info = {'upstream': ['T1', 'T3'], 'wake_loss': 0.15}
    result = agent.predict_with_explanation(context, wake_info)
    
    print(f"   Predicted Power: {result.predicted_power:.2f} MW")
    print(f"   Confidence: {result.confidence * 100:.0f}%")
    print(f"   Reasoning Steps: {len(result.reasoning_chain)}")
    print(f"   Evidence Count: {len(result.evidence)}")
    print(f"   Warnings: {len(result.warnings)}")
    
    # 测试问答
    print("\n2. Testing Q&A:")
    questions = [
        "为什么预测这个功率值？",
        "尾流影响有多大？",
        "历史上有类似情况吗？",
        "这个预测可靠吗？"
    ]
    
    for q in questions:
        print(f"\n   Q: {q}")
        answer = agent.answer_question(q, context, result)
        print(f"   A: {answer[:100]}..." if len(answer) > 100 else f"   A: {answer}")
    
    # 测试报告生成
    print("\n3. Testing Report Generation:")
    report = agent.generate_report(context, result)
    print(report[:500] + "..." if len(report) > 500 else report)
    
    # 测试反思-修正闭环
    print("\n4. Testing Reflection-Refinement Loop:")
    
    # 创建一个违反物理约束的场景
    bad_context = PredictionContext(
        timestamp=datetime.now(),
        wind_speed=2.0,  # 低于切入风速
        wind_direction=180,
        temperature=20.0,
        pressure=1015,
        turbine_id=3,
        historical_power=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    
    result_bad = agent.predict_with_explanation(bad_context)
    print(f"   Wind Speed: {bad_context.wind_speed} m/s (below cut-in)")
    print(f"   Predicted Power: {result_bad.predicted_power:.2f} MW")
    print(f"   Warnings: {result_bad.warnings}")
    
    print("\n" + "=" * 60)
    print("Wind-Agent Test Complete!")
    print("=" * 60)
