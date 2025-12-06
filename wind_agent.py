"""
Wind-Agent: LLM-Driven Intelligent Agent for Wind Power Prediction

This module provides a cognitive agent with:
- Chain-of-thought style reasoning backed by an injectable LLM client
- Retrieval-augmented generation (RAG) over a lightweight vector index
- Reflection and refinement loops with both LLM critique and physics guardrails
- Multi-turn Q&A that tracks dialogue state instead of rigid keyword trees
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class PredictionContext:
    timestamp: datetime
    wind_speed: float
    wind_direction: float
    temperature: float
    pressure: float
    turbine_id: int
    historical_power: List[float]
    nwp_forecast: Optional[Dict] = None
    causal_graph: Optional[Dict] = None
    predicted_power: float = 0.0
    confidence: float = 0.5
    session_directives: List[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        history = ", ".join(f"{p:.2f}MW" for p in self.historical_power[-6:]) or "无"
        return (
            f"当前时间: {self.timestamp}\n"
            f"风机ID: T{self.turbine_id}\n"
            f"风速: {self.wind_speed:.1f} m/s\n"
            f"风向: {self.wind_direction:.0f}°\n"
            f"温度: {self.temperature:.1f}°C\n"
            f"气压: {self.pressure:.0f} hPa\n"
            f"近6小时功率: [{history}]\n"
        )


@dataclass
class PredictionResult:
    predicted_power: float
    confidence: float
    reasoning_chain: List[str]
    evidence: List[Dict]
    warnings: List[str]
    narrative: str = ""


@dataclass
class ExplanationResult:
    summary: str
    reasoning_chain: List[str]
    evidence: List[Dict]
    confidence_level: str
    recommendations: List[str]


@dataclass
class ChatTurn:
    role: str
    content: str


# ---------------------------------------------------------------------------
# LLM plumbing
# ---------------------------------------------------------------------------


class LLMClient:
    """Interface for large language model backends."""

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        messages: Optional[Sequence[ChatTurn]] = None,
        max_tokens: int = 512,
    ) -> str:
        raise NotImplementedError


class TemplateLLM(LLMClient):
    """Hard-stop stub to prevent fake reasoning.

    The previous echo-style implementation is removed entirely so that missing
    LLM wiring fails fast instead of silently polluting downstream logic.
    """

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        messages: Optional[Sequence[ChatTurn]] = None,
        max_tokens: int = 512,
    ) -> str:
        raise RuntimeError(
            "TemplateLLM is disabled. Configure a real LLM backend (Transformers or OpenAI-compatible) before use."
        )


class OpenAIChatLLM(LLMClient):
    """OpenAI-compatible chat completion backend."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        try:
            import openai  # type: ignore

            self._client = openai.Client(api_key=api_key, base_url=base_url)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - runtime wiring
            raise ImportError(
                "openai python package is required for OpenAIChatLLM; install via `pip install openai`."
            ) from exc

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        messages: Optional[Sequence[ChatTurn]] = None,
        max_tokens: int = 512,
    ) -> str:
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        for msg in messages or []:
            chat_messages.append({"role": msg.role, "content": msg.content})
        chat_messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content  # type: ignore[no-any-return]


class TransformersLLM(LLMClient):
    """HuggingFace transformers text-generation backend."""

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B", **pipeline_kwargs: Any):
        try:
            from transformers import pipeline  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime wiring
            raise ImportError(
                "transformers is required for TransformersLLM; install via `pip install transformers`."
            ) from exc

        self.generator = pipeline("text-generation", model=model, **pipeline_kwargs)

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        messages: Optional[Sequence[ChatTurn]] = None,
        max_tokens: int = 512,
    ) -> str:
        system_prefix = f"[SYSTEM]{system}\n" if system else ""
        history = "\n".join(f"[{m.role}]{m.content}" for m in messages or [])
        full_prompt = f"{system_prefix}{history}\n{prompt}\n回答:"
        output = self.generator(full_prompt, max_new_tokens=max_tokens, num_return_sequences=1)
        return output[0]["generated_text"][len(full_prompt) :].strip()


def resolve_llm_from_env() -> LLMClient:
    """Select a production-ready LLM backend using environment hints.

    默认优先使用本地Transformers模型，避免依赖远程API。如需调用OpenAI兼容
    服务需显式设置 WIND_AGENT_LLM=openai 并提供 OPENAI_API_KEY。
    """

    import os

    provider = os.getenv("WIND_AGENT_LLM", "transformers").lower()
    model_name = os.getenv("WIND_AGENT_LLM_MODEL", "./models/local-llm")

    if provider in {"transformers", "local"}:
        pipeline_kwargs: Dict[str, Any] = {}
        device = os.getenv("WIND_AGENT_LLM_DEVICE")
        if device:
            pipeline_kwargs["device"] = device
        trust_remote_code = os.getenv("WIND_AGENT_TRUST_REMOTE_CODE")
        if trust_remote_code:
            pipeline_kwargs["trust_remote_code"] = trust_remote_code.lower() == "true"
        return TransformersLLM(model=model_name, **pipeline_kwargs)

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when WIND_AGENT_LLM=openai")
        return OpenAIChatLLM(model=model_name, api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))

    raise ValueError(
        "No valid LLM configured. Set WIND_AGENT_LLM=transformers and WIND_AGENT_LLM_MODEL to your local checkpoint, or WIND_AGENT_LLM=openai with a valid key."
    )


# ---------------------------------------------------------------------------
# Retrieval: lightweight semantic vector index
# ---------------------------------------------------------------------------


class VectorIndex:
    """Lightweight semantic index with pluggable encoder.

    If `sentence_transformers` is available, a multilingual embedding model is
    used automatically. Otherwise, a TF-IDF trigram encoder (no hashing/modulo)
    is used as a deterministic fallback to preserve basic semantic grouping and
    avoid random collisions.
    """

    def __init__(self, dim: int = 384, embedding_fn: Optional[Callable[[str], torch.Tensor]] = None):
        self.dim = dim
        self.vectors: List[torch.Tensor] = []
        self.metadata: List[Dict] = []
        self.vocab: Dict[str, int] = {}
        self.doc_freq: Dict[str, int] = {}
        self.num_docs = 0
        self._sklearn_vectorizer = None
        self._sklearn_corpus: List[str] = []
        self._encoder_type = "custom"
        self.embedding_fn = embedding_fn or self._build_encoder()

    def _build_encoder(self) -> Callable[[str], torch.Tensor]:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model = SentenceTransformer("BAAI/bge-m3")
            self._encoder_type = "sentence_transformers"

            def encode(text: str) -> torch.Tensor:
                with torch.inference_mode():
                    emb = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
                return emb.to(dtype=torch.float)

            return encode
        except Exception:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

                self._encoder_type = "sklearn_tfidf"
                self._sklearn_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1)

                def encode(text: str) -> torch.Tensor:
                    if self._sklearn_vectorizer is None:
                        return torch.zeros(1)
                    matrix = self._sklearn_vectorizer.transform([text]) if self._sklearn_vectorizer.vocabulary_ else self._sklearn_vectorizer.fit_transform([text])
                    dense = torch.tensor(matrix.toarray()[0], dtype=torch.float)
                    return F.normalize(dense, dim=0) if dense.norm() > 0 else dense

                return encode
            except Exception:
                # TF-IDF trigram encoder without hashing collisions.
                def encode(text: str, *, update_vocab: bool = False) -> torch.Tensor:
                    tokens: List[str] = []
                    lowered = text.lower().replace(" ", "")
                    for i in range(len(lowered) - 2):
                        tokens.append(lowered[i : i + 3])
                    if update_vocab:
                        for tok in tokens:
                            if tok not in self.vocab:
                                self.vocab[tok] = len(self.vocab)
                    indices = [self.vocab.get(tok) for tok in tokens if tok in self.vocab]
                    if not indices:
                        return torch.zeros(len(self.vocab) or 1)
                    vec = torch.zeros(len(self.vocab))
                    counts: Dict[int, int] = {}
                    for idx in indices:
                        if idx is None:
                            continue
                        counts[idx] = counts.get(idx, 0) + 1
                    for idx, cnt in counts.items():
                        term = list(self.vocab.keys())[idx]
                        df = self.doc_freq.get(term, 0) if self.num_docs > 0 else 0
                        idf = torch.log(torch.tensor((self.num_docs + 1) / (df + 1))) + 1.0
                        vec[idx] = (cnt / len(tokens)) * idf
                    return F.normalize(vec, dim=0) if vec.norm() > 0 else vec

                self._tfidf_encode = encode  # type: ignore[attr-defined]
                return encode

    def add(self, doc: Dict):
        content = doc.get("content", "") + " " + doc.get("title", "")
        if self._encoder_type == "sklearn_tfidf" and self._sklearn_vectorizer is not None:
            self._sklearn_corpus.append(content)
            matrix = self._sklearn_vectorizer.fit_transform(self._sklearn_corpus)
            dense = torch.tensor(matrix.toarray(), dtype=torch.float)
            self.vectors = [F.normalize(row, dim=0) if row.norm() > 0 else row for row in dense]
            self.metadata.append(doc)
            return

        if hasattr(self, "_tfidf_encode"):
            tokens = []
            lowered = content.lower().replace(" ", "")
            for i in range(len(lowered) - 2):
                tokens.append(lowered[i : i + 3])
            unique_tokens = set(tokens)
            for tok in unique_tokens:
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)
            for tok in unique_tokens:
                self.doc_freq[tok] = self.doc_freq.get(tok, 0) + 1
            self.num_docs += 1
            embedding = self._tfidf_encode(content, update_vocab=False)
            if embedding.size(0) > 0 and self.vectors:
                pad = embedding.size(0) - self.vectors[0].size(0)
                if pad > 0:
                    self.vectors = [F.pad(v, (0, pad)) for v in self.vectors]
            if embedding.numel() == 0:
                embedding = torch.zeros(len(self.vocab) or 1)
        else:
            embedding = self.embedding_fn(content)
        self.vectors.append(F.normalize(embedding, dim=0) if embedding.norm() > 0 else embedding)
        self.metadata.append(doc)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.metadata:
            return []
        if self._encoder_type == "sklearn_tfidf" and self._sklearn_vectorizer is not None:
            query_vec = torch.tensor(
                self._sklearn_vectorizer.transform([query]).toarray()[0], dtype=torch.float
            )
            query_vec = F.normalize(query_vec, dim=0) if query_vec.norm() > 0 else query_vec
        elif hasattr(self, "_tfidf_encode"):
            query_vec = self._tfidf_encode(query, update_vocab=False)
            if query_vec.numel() == 0:
                return []
            if query_vec.size(0) != self.vectors[0].size(0):
                pad = query_vec.size(0) - self.vectors[0].size(0)
                if pad > 0:
                    self.vectors = [F.pad(v, (0, pad)) for v in self.vectors]
                elif pad < 0:
                    query_vec = F.pad(query_vec, (0, -pad))
        else:
            query_vec = self.embedding_fn(query)
        matrix = torch.stack(self.vectors)
        scores = torch.mv(matrix, query_vec)
        topk = torch.topk(scores, k=min(top_k, scores.numel())).indices.tolist()
        return [self.metadata[i] for i in topk]


class KnowledgeBase:
    def __init__(self, embedding_fn: Optional[Callable[[str], torch.Tensor]] = None):
        self.vector_index = VectorIndex(dim=384, embedding_fn=embedding_fn)
        self.documents: Dict[str, List[Dict]] = {
            "technical_manual": [
                {
                    "id": "TM001",
                    "title": "风机功率曲线规范",
                    "content": "额定功率2MW，切入风速3m/s，额定风速12m/s，切出风速25m/s。",
                },
                {
                    "id": "TM002",
                    "title": "偏航系统说明",
                    "content": "偏航角偏差超过15度时，发电效率下降约10-20%。",
                },
                {
                    "id": "TM003",
                    "title": "尾流效应说明",
                    "content": "下游风机在尾流区内功率损失可达40%，取决于间距和风向。",
                },
            ],
            "fault_records": [
                {
                    "id": "FR001",
                    "date": "2023-01-15",
                    "turbine": "T5",
                    "fault_type": "齿轮箱油温过高",
                    "symptom": "功率曲线偏离，伴随振动异常",
                    "solution": "更换润滑油，检查冷却系统",
                },
                {
                    "id": "FR002",
                    "date": "2023-06-20",
                    "turbine": "T3",
                    "fault_type": "变桨系统故障",
                    "symptom": "功率波动剧烈，限功率运行",
                    "solution": "更换变桨电机",
                },
            ],
            "weather_events": [
                {
                    "id": "WE001",
                    "date": "2023-08-10",
                    "event": "台风天鹅",
                    "impact": "全场停机48小时",
                    "wind_speed_peak": "35 m/s",
                }
            ],
        }
        for docs in self.documents.values():
            for doc in docs:
                self.vector_index.add(doc)

    def reset(self):
        self.vector_index = VectorIndex(dim=self.vector_index.dim, embedding_fn=self.vector_index.embedding_fn)
        for docs in self.documents.values():
            for doc in docs:
                self.vector_index.add(doc)

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode arbitrary text using the active vector encoder."""

        return self.vector_index.embedding_fn(text)

    def embed_documents(self, docs: List[Dict]) -> Optional[torch.Tensor]:
        """Aggregate embeddings for a list of documents (mean pooled)."""

        if not docs:
            return None
        vectors = []
        for doc in docs:
            content = doc.get("content", "") + " " + doc.get("title", "")
            vec = self.encode_text(content)
            vectors.append(vec)
        stacked = torch.stack(vectors)
        mean_vec = stacked.mean(dim=0)
        return F.normalize(mean_vec, dim=0) if mean_vec.norm() > 0 else mean_vec

    def retrieve_with_embedding(self, query: str, top_k: int = 3) -> Tuple[Optional[torch.Tensor], List[Dict]]:
        """Search and return both evidence and a pooled embedding for context injection."""

        docs = self.search(query, top_k=top_k)
        embedding = self.embed_documents(docs)
        return embedding, docs

    def load_jsonl(self, path: str, namespace: str = "external"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                record.setdefault("namespace", namespace)
                self.ingest_documents([record])

    def ingest_documents(self, records: List[Dict]):
        for rec in records:
            rec.setdefault("id", f"EXT-{len(self.vector_index.metadata)+1:04d}")
            self.vector_index.add(rec)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        return self.vector_index.search(query, top_k=top_k)


# ---------------------------------------------------------------------------
# Reasoning and reflection
# ---------------------------------------------------------------------------


class ChainOfThought:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(
        self,
        context: PredictionContext,
        prediction: float,
        evidence: List[Dict],
    ) -> List[str]:
        evidence_snippets = "\n".join(
            f"- {doc.get('title', doc.get('id', 'doc'))}: {doc.get('content', '')[:160]}"
            for doc in evidence
        ) or "- 无检索到的相关证据"
        prompt = (
            "请基于以下风电场上下文，给出逐步的因果-物理推理链。"
            "每一步说明假设、观察和结论，避免模板化回答。\n"
            f"预测功率: {prediction:.2f}MW\n"
            f"上下文:\n{context.to_prompt()}\n"
            f"检索证据:\n{evidence_snippets}\n"
        )
        reasoning_text = self.llm.generate(prompt, system="风电预测专家，使用简洁分点形式。", max_tokens=640)
        return [step.strip() for step in reasoning_text.split("\n") if step.strip()]


class SafetyGuardrails:
    """Hard-coded numeric guardrails kept separate from LLM critique.

    This keeps deterministic protections explicit while allowing the reflection
    stage to focus on softer, semantic feedback.
    """

    def __init__(self, rated_power: float = 2.0, cut_in: float = 3.0, cut_out: float = 25.0, rated_speed: float = 12.0):
        self.rated_power = rated_power
        self.cut_in = cut_in
        self.cut_out = cut_out
        self.rated_speed = rated_speed

    def evaluate(self, context: PredictionContext) -> Tuple[List[str], List[Dict]]:
        criticisms: List[str] = []
        corrections: List[Dict] = []
        ws = context.wind_speed
        if context.predicted_power < 0:
            criticisms.append("预测功率为负，违反物理约束")
            corrections.append({"type": "clamp", "min": 0})
        if context.predicted_power > self.rated_power:
            criticisms.append("预测功率超过额定值")
            corrections.append({"type": "clamp", "max": self.rated_power})
        if ws < self.cut_in and context.predicted_power > 0.05:
            criticisms.append("切入风速以下功率应接近0")
            corrections.append({"type": "set", "value": 0})
        if ws > self.cut_out and context.predicted_power > 0.05:
            criticisms.append("切出风速以上应停机保护")
            corrections.append({"type": "set", "value": 0})
        if self.cut_in <= ws < self.rated_speed:
            expected_max = self.rated_power * ((ws - self.cut_in) / (self.rated_speed - self.cut_in)) ** 3
            if context.predicted_power > expected_max * 1.3:
                criticisms.append("爬坡区预测偏高，超过理论值30%")
                corrections.append({"type": "scale", "factor": 0.85})
        return criticisms, corrections


class ReflectionAgent:
    def __init__(self, llm: LLMClient, rated_power: float = 2.0):
        self.llm = llm
        self.guardrails = SafetyGuardrails(rated_power=rated_power)
        self.rated_power = rated_power

    def _numeric_checks(self, context: PredictionContext) -> Tuple[List[str], List[Dict]]:
        """Backward-compatible shim around SafetyGuardrails."""

        return self.guardrails.evaluate(context)

    def _semantic_checks(self, context: PredictionContext, evidence: List[Dict]) -> Tuple[List[str], List[Dict[str, Any]], str]:
        trend = ", ".join(f"{p:.2f}" for p in context.historical_power[-6:]) or "无"
        evidence_snippets = "\n".join(
            f"- {doc.get('title', doc.get('id', 'doc'))}: {doc.get('content', '')[:160]}"
            for doc in evidence
        ) or "- 无"
        prompt = (
            "请作为专家审查以下预测是否符合趋势和物理直觉，输出JSON: "
            "{\"criticisms\":[], \"corrections\":[], \"confidence_penalty\":0, \"commentary\":\"\"}.\n"
            f"历史功率序列(近6条): {trend}\n"
            f"当前预测功率: {context.predicted_power:.2f}MW, 风速 {context.wind_speed:.1f}m/s\n"
            f"检索证据:\n{evidence_snippets}"
        )
        raw = self.llm.generate(prompt, system="语义一致性审查", max_tokens=320)
        try:
            parsed = json.loads(raw)
            criticisms = [c for c in parsed.get("criticisms", []) if isinstance(c, str)]
            corrections = [c for c in parsed.get("corrections", []) if isinstance(c, dict)]
            commentary = parsed.get("commentary", raw)
        except Exception:
            criticisms, corrections, commentary = ["语义审查解析失败，使用原始文本"], [], raw
        return criticisms, corrections, commentary

    def reflect(self, context: PredictionContext, evidence: List[Dict]) -> Dict[str, Any]:
        numeric_criticisms, corrections = self._numeric_checks(context)
        semantic_criticisms, semantic_corrections, semantic_commentary = self._semantic_checks(context, evidence)
        evidence_snippets = "\n".join(
            f"- {doc.get('title', doc.get('id', 'doc'))}: {doc.get('content', '')[:160]}"
            for doc in evidence
        ) or "- 无"
        prompt = (
            "请作为审查员，评估下面的风电功率预测是否合理，给出原因和改进建议。\n"
            "以JSON输出: {\"criticisms\":[], \"corrections\":[], \"confidence_penalty\":0, \"explanation\": \"\"}\\n"
            "可用修正规范: {type: clamp|min|max|set|scale, value/max/min/factor: number}.\\n"
            f"预测功率: {context.predicted_power:.2f}MW\n"
            f"上下文:\n{context.to_prompt()}\n"
            f"检索证据:\n{evidence_snippets}\n"
            f"已发现的数值问题: {json.dumps(numeric_criticisms, ensure_ascii=False)}"
        )
        critique_raw = self.llm.generate(prompt, system="物理一致性审查员", max_tokens=480)
        llm_criticisms: List[str] = []
        llm_corrections: List[Dict[str, Any]] = []
        confidence_penalty = 0.0
        try:
            parsed = json.loads(critique_raw)
            llm_criticisms = parsed.get("criticisms", [])
            llm_corrections = [c for c in parsed.get("corrections", []) if isinstance(c, dict)]
            confidence_penalty = float(parsed.get("confidence_penalty", 0))
            critique_text = parsed.get("explanation", critique_raw)
        except Exception:
            critique_text = critique_raw
        hard_guardrails = list(dict.fromkeys(numeric_criticisms))
        soft_criticisms = list(dict.fromkeys(llm_criticisms + semantic_criticisms))
        merged_corrections = (
            corrections
            + self._sanitize_corrections(llm_corrections)
            + self._sanitize_corrections(semantic_corrections)
        )
        merged_criticisms = hard_guardrails + [c for c in soft_criticisms if c not in hard_guardrails]
        score = max(0, 100 - 10 * len(numeric_criticisms) - int(confidence_penalty))
        critique_text = f"{critique_text}\n[语义审查]\n{semantic_commentary}" if semantic_commentary else critique_text
        return {
            "criticisms": merged_criticisms,
            "hard_guardrails": hard_guardrails,
            "soft_criticisms": soft_criticisms,
            "llm_feedback": critique_text,
            "corrections": merged_corrections,
            "physical_consistency_score": score,
            "needs_refinement": len(merged_corrections) > 0,
        }

    def _sanitize_corrections(self, corrections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        allowed_types = {"clamp", "set", "scale"}
        sanitized: List[Dict[str, Any]] = []
        for corr in corrections:
            if corr.get("type") not in allowed_types:
                continue
            clean = {"type": corr["type"]}
            for key in ("min", "max", "value", "factor"):
                if key in corr and isinstance(corr[key], (int, float)):
                    clean[key] = float(corr[key])
            sanitized.append(clean)
        return sanitized

    def apply_corrections(self, prediction: float, corrections: List[Dict]) -> float:
        power = prediction
        for correction in corrections:
            if correction["type"] == "clamp":
                if "min" in correction:
                    power = max(power, correction["min"])
                if "max" in correction:
                    power = min(power, correction["max"])
            elif correction["type"] == "set":
                power = correction["value"]
            elif correction["type"] == "scale":
                power = power * correction["factor"]
        return power


# ---------------------------------------------------------------------------
# WindAgent
# ---------------------------------------------------------------------------


class WindAgent:
    def __init__(
        self,
        prediction_model: Optional[Any] = None,
        rated_power: float = 2.0,
        llm: Optional[LLMClient] = None,
        embedding_fn: Optional[Callable[[str], torch.Tensor]] = None,
        *,
        allow_physics_fallback: bool = False,
        conversation_max_turns: int = 12,
        conversation_max_chars: int = 6000,
        feature_keys: Optional[Sequence[str]] = None,
        feature_extractor: Optional[Callable[[PredictionContext], torch.Tensor]] = None,
    ):
        self.prediction_model = prediction_model
        if llm is not None:
            self.llm = llm
        else:
            try:
                self.llm = resolve_llm_from_env()
            except Exception as exc:
                raise RuntimeError(
                    "未配置真实LLM，请设置 OPENAI_API_KEY 或提供本地 Transformers 模型 (WIND_AGENT_LLM=transformers)。"
                ) from exc
        self.knowledge_base = KnowledgeBase(embedding_fn=embedding_fn)
        self.cot = ChainOfThought(self.llm)
        self.reflector = ReflectionAgent(self.llm, rated_power)
        self.rated_power = rated_power
        self.allow_physics_fallback = allow_physics_fallback
        self.feature_keys = list(feature_keys) if feature_keys else [
            "wind_speed",
            "wind_direction",
            "temperature",
            "pressure",
        ]
        self.feature_extractor = feature_extractor
        self.conversation_history: List[ChatTurn] = []
        self.session_directives: List[ChatTurn] = []
        self.conversation_max_turns = max(4, conversation_max_turns)
        self.conversation_max_chars = max(2000, conversation_max_chars)

    @property
    def context_dim(self) -> int:
        return self.knowledge_base.vector_index.dim

    def build_context_embedding_from_conditions(
        self,
        *,
        wind_speed: float,
        wind_direction: float,
        turbine_id: Optional[str] = None,
        note: str = "",
        top_k: int = 3,
    ) -> Tuple[Optional[torch.Tensor], List[Dict]]:
        """Retrieve RAG evidence and return a pooled embedding for model fusion."""

        query = (
            f"风机{turbine_id or ''} 风速{wind_speed:.1f} 风向{wind_direction:.0f} "
            f"特殊情况 {note}".strip()
        )
        embedding, docs = self.knowledge_base.retrieve_with_embedding(query, top_k=top_k)
        if embedding is not None and embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        return embedding, docs

    def register_session_directive(self, content: str, *, role: str = "system") -> None:
        """Register a sticky directive that is always injected into prompts.

        Use this to keep初始安全/风场限定条件在长对话中不被滑动窗口丢弃。
        """

        self.session_directives.append(ChatTurn(role=role, content=content))

    def _model_predict(self, context: PredictionContext, wake_info: Optional[Dict]) -> Tuple[float, List[str]]:
        warnings: List[str] = []
        if self.prediction_model is None:
            if not self.allow_physics_fallback:
                raise RuntimeError(
                    "prediction_model 未注入且未允许物理回退；请提供模型或设置 allow_physics_fallback=True"
                )
            warnings.append("未注入预测模型，使用物理功率曲线回退计算结果")
            if context.wind_speed < 3 or context.wind_speed >= 25:
                base = 0.0
            elif context.wind_speed < 12:
                ratio = ((context.wind_speed - 3) / 9) ** 3
                base = ratio * self.rated_power
            else:
                base = self.rated_power
        else:
            if self.feature_extractor is not None:
                features = self.feature_extractor(context)
                if features.dim() == 1:
                    features = features.unsqueeze(0)
                inputs = features.to(dtype=torch.float)
            else:
                try:
                    feature_values = [float(getattr(context, key)) for key in self.feature_keys]
                except AttributeError as exc:
                    raise ValueError(
                        f"预测模型特征缺失，请确保上下文包含 {self.feature_keys}"
                    ) from exc
                inputs = torch.tensor(feature_values, dtype=torch.float).unsqueeze(0)
            base = float(self.prediction_model(inputs).squeeze().item())
        if wake_info and wake_info.get("wake_loss", 0) > 0:
            base *= 1 - wake_info["wake_loss"]
        return base, warnings

    def predict_with_explanation(
        self, context: PredictionContext, wake_info: Optional[Dict] = None
    ) -> PredictionResult:
        evidence = self.knowledge_base.search(
            f"风速{context.wind_speed} 风向{context.wind_direction} 功率预测"
        )
        raw_prediction, predict_warnings = self._model_predict(context, wake_info)
        context.predicted_power = raw_prediction
        reasoning_chain = self.cot.generate(context, raw_prediction, evidence)
        reflection = self.reflector.reflect(context, evidence)
        final_prediction = (
            self.reflector.apply_corrections(raw_prediction, reflection["corrections"])
            if reflection["needs_refinement"]
            else raw_prediction
        )
        warnings = list(dict.fromkeys(predict_warnings + reflection["criticisms"]))
        narrative = reflection["llm_feedback"]
        confidence = 0.9 if not warnings else 0.7
        context.confidence = confidence
        context.predicted_power = final_prediction
        return PredictionResult(
            predicted_power=final_prediction,
            confidence=confidence,
            reasoning_chain=reasoning_chain,
            evidence=evidence,
            warnings=warnings,
            narrative=narrative,
        )

    def explain_prediction(self, context: PredictionContext) -> ExplanationResult:
        """Provide a structured explanation without re-running the predictor."""

        evidence = self.knowledge_base.search(
            f"风速{context.wind_speed} 风向{context.wind_direction} 功率解释"
        )
        reasoning_chain = self.cot.generate(
            context, context.predicted_power, evidence
        )
        reflection = self.reflector.reflect(context, evidence)

        confidence_level = (
            "高" if context.confidence >= 0.8 else "中" if context.confidence >= 0.4 else "低"
        )
        recommendations = []
        if reflection["criticisms"]:
            recommendations.append("请复核传感器数据或限电状态，因反思提示存在异常。")
        if reflection["needs_refinement"]:
            recommendations.append("已执行安全护栏校正，建议结合SCADA再验证。")
        if not recommendations:
            recommendations.append("预测在物理范围内，无需额外操作。")

        summary = (
            f"预测功率: {context.predicted_power:.2f}MW, 置信度: {context.confidence*100:.0f}%\n"
            f"LLM反思: {reflection['llm_feedback']}\n"
            f"主要警告: {', '.join(reflection['criticisms']) or '无'}"
        )

        return ExplanationResult(
            summary=summary,
            reasoning_chain=reasoning_chain,
            evidence=evidence,
            confidence_level=confidence_level,
            recommendations=recommendations,
        )

    def answer_question(
        self,
        question: str,
        context: Optional[PredictionContext] = None,
        prediction_result: Optional[PredictionResult] = None,
    ) -> str:
        self.conversation_history.append(ChatTurn(role="user", content=question))
        self._prune_history()
        evidence = self.knowledge_base.search(question)
        context_prompt = context.to_prompt() if context else "无上下文"
        result_prompt = (
            f"预测功率: {prediction_result.predicted_power:.2f}MW, "
            f"置信度: {prediction_result.confidence*100:.0f}%\n"
            f"主要警告: {prediction_result.warnings}"
            if prediction_result
            else "尚未执行预测"
        )
        evidence_prompt = "\n".join(
            f"- {doc.get('id', 'doc')} | {doc.get('title', doc.get('fault_type', 'N/A'))}: {doc.get('content', '')[:160]}"
            for doc in evidence
        ) or "- 无"
        prompt = (
            f"问题: {question}\n"
            f"上下文:\n{context_prompt}\n"
            f"预测结果:\n{result_prompt}\n"
            f"检索证据:\n{evidence_prompt}\n"
        )
        answer = self.llm.generate(
            prompt,
            system="风电场智能助理，参考对话历史回答，必要时指出不确定性。",
            messages=[*self.session_directives, *self.conversation_history],
            max_tokens=640,
        )
        self.conversation_history.append(ChatTurn(role="assistant", content=answer))
        self._prune_history()
        return answer

    def interactive_query(self, question: str, context: PredictionContext) -> str:
        return self.answer_question(question, context=context)

    def generate_report(self, context: PredictionContext, result: PredictionResult) -> str:
        evidence_lines = "\n".join(
            f"- [{doc.get('id','doc')}] {doc.get('title', doc.get('fault_type','N/A'))}"
            for doc in result.evidence
        ) or "- 无"
        reasoning = "\n".join(result.reasoning_chain)
        warnings = "\n".join(f"⚠️ {w}" for w in result.warnings) or "无"
        return (
            "========================================\n"
            "风电功率预测报告\n"
            "========================================\n"
            f"时间: {context.timestamp}\n"
            f"风机: T{context.turbine_id}\n"
            f"风速: {context.wind_speed} m/s\n"
            f"风向: {context.wind_direction}°\n\n"
            f"预测功率: {result.predicted_power:.2f} MW\n"
            f"置信度: {result.confidence*100:.0f}%\n"
            f"推理过程:\n{reasoning}\n\n"
            f"检索依据:\n{evidence_lines}\n\n"
            f"LLM反思:\n{result.narrative}\n\n"
            f"警告信息:\n{warnings}\n"
            "========================================\n"
        )

    def _prune_history(self):
        # First bound by number of turns
        if len(self.conversation_history) > self.conversation_max_turns:
            self.conversation_history = self.conversation_history[-self.conversation_max_turns :]

        # Then bound by character budget to avoid token explosions
        total_chars = sum(len(turn.content) for turn in self.conversation_history)
        while self.conversation_history and total_chars > self.conversation_max_chars:
            removed = self.conversation_history.pop(0)
            total_chars -= len(removed.content)


class ReportGenerator:
    """Lightweight reporter to summarize multiple PredictionContext entries."""

    def __init__(self, agent: WindAgent):
        self.agent = agent

    def generate_report(self, contexts: List[PredictionContext], report_type: str = "summary") -> str:
        lines = [
            "================ 风电预测报告 ================",
            f"报告类型: {report_type}",
            f"生成时间: {datetime.now()}",
            "----------------------------------------------",
        ]
        for ctx in contexts:
            lines.append(
                (
                    f"风机 T{ctx.turbine_id}: 预测 {ctx.predicted_power:.2f}MW, 置信度 {ctx.confidence*100:.0f}% | "
                    f"风速 {ctx.wind_speed:.1f}m/s, 风向 {ctx.wind_direction:.0f}°"
                )
            )
            lines.append(f"- 持久指令: {'; '.join(ctx.session_directives) if ctx.session_directives else '无'}")
        lines.append("==============================================")
        return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("Wind-Agent: LLM-Driven Intelligent Agent")
    print("=" * 60)

    agent = WindAgent(rated_power=2.0)

    context = PredictionContext(
        timestamp=datetime.now(),
        wind_speed=10.5,
        wind_direction=270,
        temperature=15.0,
        pressure=1013,
        turbine_id=5,
        historical_power=[1.2, 1.3, 1.4, 1.5, 1.6, 1.5],
    )

    print("\n1. Prediction with explanation:")
    result = agent.predict_with_explanation(context, wake_info={"upstream": ["T1", "T3"], "wake_loss": 0.15})
    print(f"Predicted: {result.predicted_power:.2f} MW, confidence {result.confidence*100:.0f}%")
    print(f"Reasoning steps: {len(result.reasoning_chain)} | Evidence: {len(result.evidence)} | Warnings: {len(result.warnings)}")

    print("\n2. Q&A: ")
    for q in [
        "为什么预测这个功率值？",
        "尾流会有什么影响？",
        "历史上有没有类似问题？",
        "这个预测可靠吗？",
    ]:
        print(f"Q: {q}")
        ans = agent.answer_question(q, context=context, prediction_result=result)
        print(f"A: {ans[:160]}...")

    print("\n3. Report:\n")
    print(agent.generate_report(context, result)[:400] + "...")

    print("\nDone.")
