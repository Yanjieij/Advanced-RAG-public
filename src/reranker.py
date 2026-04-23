"""重排序模块：基于 LLM 的重排序与上下文压缩

重排序（Rerank）：对初步检索的多个结果用 LLM 逐一评分，取 top-K。
上下文压缩（Compression）：对每个 chunk 提取与 query 最相关的片段，压缩输入长度。
"""

from src.retriever import RetrievalResult
from src.llm import LLM
from config import config


class LLMReranker:
    """基于 LLM 评分的重排序器。

    对初步检索的文档逐一调用 LLM 打分（0-10），按分数从高到低重排，
    只保留 top-K（config.rerank_top_k）个结果传入生成阶段。
    """

    def __init__(self, llm: LLM = None):
        self.llm = llm or LLM()

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = None
    ) -> list[RetrievalResult]:
        k = top_k or config.rerank_top_k
        if len(results) <= k:
            return results

        scored = []
        for r in results:
            relevance = self.llm.score_relevance(query, r.document)
            scored.append((r, relevance))

        scored.sort(key=lambda x: x[1], reverse=True)
        reranked = []
        for r, score in scored[:k]:
            reranked.append(RetrievalResult(
                id=r.id,
                document=r.document,
                metadata={**r.metadata, "rerank_score": score},
                score=score,
                strategy=r.strategy,
            ))
        return reranked


class ContextCompressor:
    """上下文压缩：提取 chunk 中与 query 最相关的片段"""

    def __init__(self, llm: LLM = None):
        self.llm = llm or LLM()

    def compress(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        compressed = []
        for r in results:
            prompt = (
                f"请从以下文本中提取与查询最相关的关键信息，只保留核心内容。\n\n"
                f"查询：{query}\n文本：{r.document}\n\n提取的关键信息："
            )
            extracted = self.llm.generate(prompt, temperature=0.1)
            compressed.append(RetrievalResult(
                id=r.id,
                document=extracted,
                metadata={**r.metadata, "compressed": True, "original_length": len(r.document)},
                score=r.score,
                strategy=r.strategy,
            ))
        return compressed
