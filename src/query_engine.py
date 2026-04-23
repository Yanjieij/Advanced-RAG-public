"""RAG 查询引擎：整合检索 + 重排 + 生成"""

import time
from dataclasses import dataclass, field
from src.retriever import UnifiedRetriever, RetrievalStrategy, RetrievalResult
from src.reranker import LLMReranker, ContextCompressor
from src.llm import LLM
from config import config


@dataclass
class RAGResponse:
    query: str
    answer: str
    contexts: list[str]
    retrieval_results: list[RetrievalResult]
    strategy: str
    timings: dict = field(default_factory=dict)
    rewritten_query: str = ""
    metadata: dict = field(default_factory=dict)


class QueryEngine:
    def __init__(
        self,
        retriever: UnifiedRetriever,
        llm: LLM,
        reranker: LLMReranker | None = None,
        compressor: ContextCompressor | None = None,
        use_query_rewrite: bool = False,
        use_rerank: bool = False,
        use_compression: bool = False,
    ):
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker
        self.compressor = compressor
        self.use_query_rewrite = use_query_rewrite
        self.use_rerank = use_rerank
        self.use_compression = use_compression

    def query(
        self, query: str, strategy: RetrievalStrategy, top_k: int = None
    ) -> RAGResponse:
        timings = {}
        k = top_k or config.top_k
        actual_query = query
        rewritten = ""

        # 1. Query Rewriting (可选)
        if self.use_query_rewrite and strategy != RetrievalStrategy.SEMANTIC_DUAL_PATH:
            t0 = time.time()
            rewritten = self.llm.rewrite_query(query)
            actual_query = rewritten
            timings["query_rewrite"] = time.time() - t0

        # 2. 检索
        t0 = time.time()
        results = self.retriever.retrieve(actual_query, strategy, k)
        timings["retrieval"] = time.time() - t0

        # 3. 重排序 (可选)
        if self.use_rerank and self.reranker and len(results) > config.rerank_top_k:
            t0 = time.time()
            results = self.reranker.rerank(query, results)
            timings["rerank"] = time.time() - t0

        # 4. 上下文压缩 (可选)
        if self.use_compression and self.compressor:
            t0 = time.time()
            results = self.compressor.compress(query, results)
            timings["compression"] = time.time() - t0

        # 5. 生成
        contexts = [r.document for r in results]
        t0 = time.time()
        answer = self.llm.generate_with_context(query, contexts)
        timings["generation"] = time.time() - t0

        timings["total"] = sum(timings.values())

        return RAGResponse(
            query=query,
            answer=answer,
            contexts=contexts,
            retrieval_results=results,
            strategy=strategy.value,
            timings=timings,
            rewritten_query=rewritten,
        )

    def retrieve_only(
        self, query: str, strategy: RetrievalStrategy, top_k: int = None
    ) -> RAGResponse:
        """纯检索模式：只执行 retrieve + rerank，跳过 LLM 生成。

        用于快速评估检索质量，无 LLM 调用开销。
        """
        timings = {}
        k = top_k or config.top_k
        actual_query = query
        rewritten = ""

        # 1. Query Rewriting (可选)
        if self.use_query_rewrite and strategy != RetrievalStrategy.SEMANTIC_DUAL_PATH:
            t0 = time.time()
            rewritten = self.llm.rewrite_query(query)
            actual_query = rewritten
            timings["query_rewrite"] = time.time() - t0

        # 2. 检索
        t0 = time.time()
        results = self.retriever.retrieve(actual_query, strategy, k)
        timings["retrieval"] = time.time() - t0

        # 3. 重排序 (可选)
        if self.use_rerank and self.reranker and len(results) > config.rerank_top_k:
            t0 = time.time()
            results = self.reranker.rerank(query, results)
            timings["rerank"] = time.time() - t0

        # 跳过 generation 和 compression
        timings["total"] = sum(timings.values())
        contexts = [r.document for r in results]

        return RAGResponse(
            query=query,
            answer="",  # 无生成
            contexts=contexts,
            retrieval_results=results,
            strategy=strategy.value,
            timings=timings,
            rewritten_query=rewritten,
        )
