"""RAGAS 评估集成 — 基于官方 ragas.evaluate() 实现

使用 RAGAS 官方 evaluate() 函数和官方 Metric 类，确保评估逻辑与 RAGAS 框架完全一致。
通过智谱/DeepSeek 的 OpenAI-compatible API 驱动 LLM-as-Judge。

核心指标：
- Faithfulness: 答案是否忠实于检索到的上下文（不编造）
- Answer Relevancy: 答案与问题的相关性
- Context Precision: 检索结果中相关上下文的排名质量
- Context Recall: 是否检索到了所有相关上下文
- Factual Correctness: 答案与 ground truth 的事实一致性
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import openai
from rich.console import Console

from config import config
from src.llm import LLM

console = Console()
_llm_logger = logging.getLogger("llm_calls")


# 需要禁用推理/思考的模型（RAGAS Judge 任务不需要推理）
ZHIPU_THINKING_MODELS = {"glm-4.7-flashx", "glm-4.7", "glm-4.7-plus"}
OPENAI_REASONING_MODELS = {"o3", "o3-mini", "o4-mini", "gpt-5-mini", "gpt-5"}


def _wrap_client_with_logging(client: openai.OpenAI, model_name: str, source: str) -> openai.OpenAI:
    """给 OpenAI client 的 chat.completions.create 添加日志拦截 + 推理模型自动禁用思考。

    同时处理以下兼容性问题：
    - 智谱推理模型（glm-4.7 等）：自动注入 thinking: disabled，避免消耗大量 tokens
    - OpenAI 推理模型（o3/o4-mini 等）：自动注入 reasoning: none
    - OpenAI 新版模型（gpt-5/gpt-4.1/o3/o4）：max_tokens → max_completion_tokens
    """
    original_create = client.chat.completions.create

    def logged_create(*args, **kwargs):
        model = kwargs.get("model", model_name)

        # 智谱推理模型：自动注入 thinking: disabled（RAGAS Judge 任务不需要推理）
        if model in ZHIPU_THINKING_MODELS:
            extra_body = kwargs.get("extra_body", {}) or {}
            if "thinking" not in extra_body:
                extra_body["thinking"] = {"type": "disabled"}
                kwargs["extra_body"] = extra_body

        # OpenAI 推理模型：自动注入 reasoning: none（避免消耗推理 token）
        if model in OPENAI_REASONING_MODELS:
            if "reasoning" not in kwargs:
                kwargs["reasoning"] = {"effort": "none"}

        # OpenAI 新版模型 API 改动：max_tokens 参数已重命名为 max_completion_tokens
        if "max_tokens" in kwargs and model.startswith(("gpt-5", "gpt-4.1", "o3", "o4")):
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

        t0 = time.time()
        response = original_create(*args, **kwargs)
        elapsed = time.time() - t0

        messages = kwargs.get("messages", args[0] if args else [])
        prompt, system = "", ""
        for msg in (messages or []):
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    prompt = str(msg.get("content", ""))
                elif msg.get("role") == "system":
                    system = str(msg.get("content", ""))

        content, usage = "", {}
        try:
            content = response.choices[0].message.content or ""
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                }
        except Exception:
            pass

        _llm_logger.debug(json.dumps({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": kwargs.get("model", model_name),
            "source": source,
            "prompt": prompt[:500],
            "system": system[:200],
            "output": content[:500],
            "output_len": len(content),
            "usage": usage,
            "elapsed_s": round(elapsed, 2),
        }, ensure_ascii=False))
        return response

    client.chat.completions.create = logged_create
    return client


@dataclass
class RAGASResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_correctness: float
    hallucination_rate: float


class RAGASEvaluator:
    """基于 RAGAS 官方 evaluate() 的评估器

    使用 ragas.metrics.collections 中的官方 Metric 实现，
    通过 llm_factory 桥接智谱/DeepSeek OpenAI-compatible API。
    """

    ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    def __init__(self, llm: LLM = None, embedding_model=None):
        """
        Args:
            llm: LLM 实例（用于确定 provider/model/api_key）
            embedding_model: 项目 EmbeddingModel（用于 Answer Relevancy）
        """
        self.llm = llm or LLM()
        self.embedding_model = embedding_model

        # 创建 RAGAS 兼容的 LLM 和 Embedding
        self._ragas_llm = self._create_ragas_llm()
        self._ragas_emb = self._create_ragas_embeddings()

        # 创建官方 metrics
        self._init_metrics()

    def _create_ragas_llm(self):
        """根据 LLM 实例的 provider 创建 RAGAS 兼容的 LLM（带日志拦截）"""
        from ragas.llms import llm_factory

        if self.llm.provider == "openai":
            client = openai.OpenAI(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
            )
        elif self.llm.provider == "deepseek":
            client = openai.OpenAI(
                api_key=config.deepseek_api_key,
                base_url=config.deepseek_base_url,
            )
        else:  # zhipu
            client = openai.OpenAI(
                api_key=config.zhipuai_api_key,
                base_url=self.ZHIPU_BASE_URL,
            )
        client = _wrap_client_with_logging(client, self.llm.model, "ragas_evaluate")
        return llm_factory(
            model=self.llm.model, provider="openai", client=client,
            max_tokens=4096,
        )

    def _create_ragas_embeddings(self):
        """创建 RAGAS 兼容的 Embedding（始终用智谱 embedding-3）

        使用 LangChain 的 OpenAIEmbeddings（提供 embed_query 方法），
        而非 ragas.embeddings.embedding_factory（缺少 embed_query，
        会导致 Answer Relevancy 指标报错）。
        """
        from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings
        return LCOpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=config.zhipuai_api_key,
            openai_api_base=self.ZHIPU_BASE_URL,
        )

    def _init_metrics(self):
        """初始化官方 RAGAS metrics（使用旧版导入，兼容 evaluate() 类型检查）"""
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

        from ragas.metrics import (
            faithfulness, answer_relevancy,
            context_precision, context_recall, answer_correctness,
        )
        self._full_metrics = [
            faithfulness, answer_relevancy,
            context_precision, context_recall, answer_correctness,
        ]
        self._retrieval_metrics = [context_precision, context_recall]

    def _run_ragas_evaluate(self, samples: list, metrics: list) -> dict:
        """调用官方 ragas.evaluate()，自动注入 LLM 和 Embedding"""
        from ragas import evaluate, EvaluationDataset
        from ragas.run_config import RunConfig

        dataset = EvaluationDataset(samples=samples)
        run_config = RunConfig(
            max_workers=config.max_concurrent_judge,
            timeout=180,
            max_retries=10,
            max_wait=30,
            log_tenacity=True,
        )

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self._ragas_llm,
            embeddings=self._ragas_emb,
            run_config=run_config,
            raise_exceptions=False,
            show_progress=True,
        )
        return result

    def _sample_to_result(self, row: dict, mode: str = "full") -> RAGASResult:
        """将 RAGAS evaluate() 的一行结果转为 RAGASResult"""
        faith = row.get("faithfulness", 0.0) or 0.0
        # 兼容新旧版 metric 列名
        ctx_prec = row.get("context_precision", row.get("context_precision_with_reference", 0.0)) or 0.0
        ctx_recall = row.get("context_recall", 0.0) or 0.0
        relevancy = row.get("answer_relevancy", 0.0) or 0.0
        correctness = row.get("answer_correctness", row.get("factual_correctness", 0.0)) or 0.0

        return RAGASResult(
            faithfulness=faith if mode == "full" else 0.0,
            answer_relevancy=relevancy if mode == "full" else 0.0,
            context_precision=ctx_prec,
            context_recall=ctx_recall,
            answer_correctness=correctness if mode == "full" else 0.0,
            hallucination_rate=1.0 - faith if mode == "full" else 0.0,
        )

    def evaluate(
        self, question: str, answer: str, contexts: list[str], ground_truth: str,
    ) -> RAGASResult:
        """单题完整评估（5 个官方指标）"""
        from ragas import SingleTurnSample
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        )
        result = self._run_ragas_evaluate([sample], self._full_metrics)
        df = result.to_pandas()
        if len(df) > 0:
            return self._sample_to_result(df.iloc[0].to_dict(), mode="full")
        return RAGASResult(0, 0, 0, 0, 0, 0)

    def evaluate_retrieval(
        self, question: str, contexts: list[str], ground_truth: str,
    ) -> RAGASResult:
        """单题纯检索评估（Context Precision + Context Recall）"""
        from ragas import SingleTurnSample
        sample = SingleTurnSample(
            user_input=question,
            response="",  # retrieval 模式无 answer
            retrieved_contexts=contexts,
            reference=ground_truth,
        )
        result = self._run_ragas_evaluate([sample], self._retrieval_metrics)
        df = result.to_pandas()
        if len(df) > 0:
            return self._sample_to_result(df.iloc[0].to_dict(), mode="retrieval")
        return RAGASResult(0, 0, 0, 0, 0, 0)

    def evaluate_batch(
        self,
        questions: list[str],
        answers: list[str],
        contexts_list: list[list[str]],
        ground_truths: list[str],
    ) -> tuple[dict[str, float], list[RAGASResult]]:
        """批量完整评估（官方 evaluate 一次性处理所有样本，内部自带并发）"""
        from ragas import SingleTurnSample

        t0 = time.time()
        samples = [
            SingleTurnSample(
                user_input=q, response=a, retrieved_contexts=c, reference=gt,
            )
            for q, a, c, gt in zip(questions, answers, contexts_list, ground_truths)
        ]

        console.print(f"    正在运行 RAGAS 官方评估（{len(samples)} 条样本，{len(self._full_metrics)} 个指标）...")
        result = self._run_ragas_evaluate(samples, self._full_metrics)
        df = result.to_pandas()
        elapsed = time.time() - t0

        results = []
        for i, row in df.iterrows():
            r = self._sample_to_result(row.to_dict(), mode="full")
            results.append(r)
            console.print(
                f"    RAGAS [{i+1}/{len(df)}] "
                f"faith={r.faithfulness:.2f} relev={r.answer_relevancy:.2f} "
                f"prec={r.context_precision:.2f} recall={r.context_recall:.2f} "
                f"correct={r.answer_correctness:.2f}"
            )

        console.print(f"    完成，耗时 {elapsed:.1f}s")

        n = len(results) or 1
        avg = {
            "faithfulness": sum(r.faithfulness for r in results) / n,
            "answer_relevancy": sum(r.answer_relevancy for r in results) / n,
            "context_precision": sum(r.context_precision for r in results) / n,
            "context_recall": sum(r.context_recall for r in results) / n,
            "answer_correctness": sum(r.answer_correctness for r in results) / n,
            "hallucination_rate": sum(r.hallucination_rate for r in results) / n,
        }
        return avg, results

    def evaluate_retrieval_batch(
        self,
        questions: list[str],
        contexts_list: list[list[str]],
        ground_truths: list[str],
    ) -> tuple[dict[str, float], list[RAGASResult]]:
        """批量纯检索评估（官方 evaluate 一次性处理，内部自带并发）"""
        from ragas import SingleTurnSample

        t0 = time.time()
        samples = [
            SingleTurnSample(
                user_input=q, response="", retrieved_contexts=c, reference=gt,
            )
            for q, c, gt in zip(questions, contexts_list, ground_truths)
        ]

        console.print(f"    正在运行 RAGAS 官方评估（{len(samples)} 条样本，检索指标模式）...")
        result = self._run_ragas_evaluate(samples, self._retrieval_metrics)
        df = result.to_pandas()
        elapsed = time.time() - t0

        results = []
        for i, row in df.iterrows():
            r = self._sample_to_result(row.to_dict(), mode="retrieval")
            results.append(r)
            console.print(
                f"    RAGAS-retrieval [{i+1}/{len(df)}] "
                f"ctx_prec={r.context_precision:.2f} ctx_recall={r.context_recall:.2f}"
            )

        console.print(f"    完成，耗时 {elapsed:.1f}s")

        n = len(results) or 1
        avg = {
            "context_precision": sum(r.context_precision for r in results) / n,
            "context_recall": sum(r.context_recall for r in results) / n,
        }
        return avg, results


# ════════════════════════════════════════════════════════════════════════════
# 以下为自实现的简化版 RAGAS Evaluator（已停用，保留供参考和对照学习）
#
# 停用原因：改用官方 ragas.evaluate()，指标与社区对齐、支持并发、结果更可靠。
#
# 与官方实现的差异：
# - 合并 prompt（每题 ~5 次调用 vs 官方 10+ 次），速度快但精度低
# - 正则解析输出（脆弱）vs 官方结构化 JSON 输出
# - 缺少 F1 计算、惩罚项等细节
#
# 如需恢复使用：将 RAGASEvaluator 重命名，然后取消下方注释并重命名为 RAGASEvaluator
# ════════════════════════════════════════════════════════════════════════════

# import re
# from src.embeddings import EmbeddingModel
#
# class RAGASEvaluatorSimplified:
#     """自实现的简化版 RAGAS 评估器（已停用）
#
#     优化要点：
#     1. 合并 prompt：将"提取声明 + 逐条验证"合并为单次 LLM 调用
#     2. 并发调用：5 个指标的 LLM 调用并发执行
#     3. 减少总调用次数：每道题 ~5 次 LLM + 1 次 Embedding（原来 ~20 次）
#     """
#
#     def __init__(self, llm: LLM = None, embedding_model: EmbeddingModel = None):
#         self.llm = llm or LLM()
#         self.embedding = embedding_model or EmbeddingModel()
#
#     def evaluate_faithfulness(self, answer: str, contexts: list[str]) -> float:
#         context_text = "\n---\n".join(contexts)
#         prompt = (
#             f"任务：判断回答中的每个事实声明是否能从参考资料中找到依据。\n\n"
#             f"参考资料：\n{context_text}\n\n"
#             f"回答：{answer}\n\n"
#             f"请先列出回答中的所有事实声明，然后逐条判断是否有依据。\n"
#             f"最后一行输出格式：支持数/总数 = X/Y"
#         )
#         result = self.llm.generate(prompt, temperature=0.1)
#         match = re.search(r"(\d+)\s*/\s*(\d+)", result)
#         if match:
#             supported, total = int(match.group(1)), int(match.group(2))
#             return supported / total if total > 0 else 1.0
#         return 0.5
#
#     def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
#         gen_prompt = (
#             f"请根据以下回答生成 3 个可能的原始问题，每行一个，只输出问题。\n\n"
#             f"回答：{answer}\n\n问题列表："
#         )
#         gen_text = self.llm.generate(gen_prompt, temperature=0.5)
#         generated_questions = [
#             q.strip().lstrip("0123456789.、)）-·").strip()
#             for q in gen_text.strip().split("\n")
#             if q.strip()
#         ][:3]
#         if not generated_questions:
#             return 0.0
#         original_emb = self.embedding.embed_query(question)
#         gen_embs = self.embedding.embed_texts(generated_questions)
#         similarities = [
#             self.embedding.cosine_similarity(original_emb, ge) for ge in gen_embs
#         ]
#         return sum(similarities) / len(similarities)
#
#     def evaluate_context_precision(self, question, contexts, ground_truth) -> float:
#         ctx_list = "\n".join(f"[上下文{i+1}]: {ctx[:200]}" for i, ctx in enumerate(contexts))
#         prompt = (
#             f"问题：{question}\n参考答案：{ground_truth}\n\n"
#             f"以下是检索到的上下文，请判断每个是否有助于回答问题。\n"
#             f"{ctx_list}\n\n"
#             f"请对每个上下文回答'相关'或'不相关'，格式：上下文1:相关"
#         )
#         result = self.llm.generate(prompt, temperature=0.1)
#         relevances = []
#         for i in range(len(contexts)):
#             pattern = f"上下文{i+1}"
#             relevances.append(1.0 if pattern in result and "相关" in result.split(pattern)[-1].split("\n")[0] and "不相关" not in result.split(pattern)[-1].split("\n")[0] else 0.0)
#         if not relevances or sum(relevances) == 0:
#             return 0.0
#         precision_sum = 0.0
#         relevant_count = 0
#         for k, rel in enumerate(relevances, 1):
#             relevant_count += rel
#             if rel == 1.0:
#                 precision_sum += relevant_count / k
#         return precision_sum / sum(relevances)
#
#     def evaluate_context_recall(self, contexts, ground_truth) -> float:
#         context_text = "\n---\n".join(contexts)
#         prompt = (
#             f"标准答案：{ground_truth}\n\n"
#             f"检索到的参考资料：\n{context_text}\n\n"
#             f"任务：判断标准答案中的关键信息有多少在参考资料中出现。\n"
#             f"先列出标准答案的关键信息点，再逐条判断是否被覆盖。\n"
#             f"最后一行输出格式：覆盖数/总数 = X/Y"
#         )
#         result = self.llm.generate(prompt, temperature=0.1)
#         match = re.search(r"(\d+)\s*/\s*(\d+)", result)
#         if match:
#             covered, total = int(match.group(1)), int(match.group(2))
#             return covered / total if total > 0 else 1.0
#         return 0.5
#
#     def evaluate_answer_correctness(self, answer, ground_truth) -> float:
#         prompt = (
#             f"请评估回答与标准答案的一致性，0-10 打分，只输出数字。\n"
#             f"10=完全一致，0=完全不同。\n\n"
#             f"标准答案：{ground_truth}\n回答：{answer}\n评分："
#         )
#         try:
#             score_text = self.llm.generate(prompt, temperature=0.1).strip()
#             numbers = re.findall(r"\d+\.?\d*", score_text)
#             if numbers:
#                 return min(float(numbers[0]) / 10.0, 1.0)
#         except (ValueError, TypeError):
#             pass
#         return 0.0
