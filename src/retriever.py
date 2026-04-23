"""多路召回统一接口 - 7种检索策略

策略列表:
1. naive_dense        - 智谱 Embedding-3 纯向量检索 (baseline)
2. sparse_bm25        - BM25 稀疏关键词检索
3. hybrid_rrf         - Dense + Sparse RRF 融合
4. bge_m3_dense       - BGE-M3 稠密向量检索
5. bge_m3_multivec    - BGE-M3 ColBERT-style 多向量检索
6. bge_m3_hybrid      - BGE-M3 三路融合 (Dense + Sparse + ColBERT)
7. multi_vector       - 原文 + 摘要 双路召回 RRF 融合
8. semantic_dual_path - HyDE + 原始 Query 双路检索 RRF 融合
"""

from dataclasses import dataclass
from enum import Enum
from config import config
from src.embeddings import EmbeddingModel
from src.vectorstore import VectorStore
from src.sparse_retriever import BM25Retriever
from src.bge_m3_retriever import BGEM3Retriever
from src.chunking import Chunk


class RetrievalStrategy(Enum):
    NAIVE_DENSE = "naive_dense"
    SPARSE_BM25 = "sparse_bm25"
    HYBRID_RRF = "hybrid_rrf"
    BGE_M3_DENSE = "bge_m3_dense"
    BGE_M3_MULTIVEC = "bge_m3_multivec"
    BGE_M3_HYBRID = "bge_m3_hybrid"
    MULTI_VECTOR = "multi_vector"
    SEMANTIC_DUAL_PATH = "semantic_dual_path"


@dataclass
class RetrievalResult:
    id: str
    document: str
    metadata: dict
    score: float
    strategy: str


class UnifiedRetriever:
    """统一检索接口，整合 8 种策略"""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vectorstore: VectorStore,
        bm25: BM25Retriever,
        bge_m3: BGEM3Retriever | None = None,
        llm=None,
        summary_vectorstore: VectorStore | None = None,
    ):
        self.embedding = embedding_model
        self.vectorstore = vectorstore
        self.bm25 = bm25
        self.bge_m3 = bge_m3
        self.llm = llm
        self.summary_vectorstore = summary_vectorstore

    def retrieve(
        self, query: str, strategy: RetrievalStrategy, top_k: int = None
    ) -> list[RetrievalResult]:
        k = top_k or config.top_k
        dispatch = {
            RetrievalStrategy.NAIVE_DENSE: self._naive_dense,
            RetrievalStrategy.SPARSE_BM25: self._sparse_bm25,
            RetrievalStrategy.HYBRID_RRF: self._hybrid_rrf,
            RetrievalStrategy.BGE_M3_DENSE: self._bge_m3_dense,
            RetrievalStrategy.BGE_M3_MULTIVEC: self._bge_m3_multivec,
            RetrievalStrategy.BGE_M3_HYBRID: self._bge_m3_hybrid,
            RetrievalStrategy.MULTI_VECTOR: self._multi_vector,
            RetrievalStrategy.SEMANTIC_DUAL_PATH: self._semantic_dual_path,
        }
        return dispatch[strategy](query, k)

    # ── 策略 1: Naive Dense (智谱 Embedding-3) ──
    def _naive_dense(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_emb = self.embedding.embed_query(query)
        results = self.vectorstore.query_by_text(query, query_emb, top_k)
        return [
            RetrievalResult(
                id=r["id"], document=r["document"],
                metadata=r["metadata"], score=r["score"],
                strategy="naive_dense",
            )
            for r in results
        ]

    # ── 策略 2: Sparse BM25 ──
    def _sparse_bm25(self, query: str, top_k: int) -> list[RetrievalResult]:
        results = self.bm25.search(query, top_k)
        return [
            RetrievalResult(
                id=r["id"], document=r["document"],
                metadata=r["metadata"], score=r["score"],
                strategy="sparse_bm25",
            )
            for r in results
        ]

    # ── 策略 3: Hybrid RRF (Dense + Sparse 融合) ──
    def _hybrid_rrf(self, query: str, top_k: int) -> list[RetrievalResult]:
        fetch_k = top_k * 3
        dense_results = self._naive_dense(query, fetch_k)
        sparse_results = self._sparse_bm25(query, fetch_k)
        return self._rrf_fuse(
            [dense_results, sparse_results], top_k, strategy_name="hybrid_rrf"
        )

    # ── 策略 4: BGE-M3 Dense (稠密向量) ──
    def _bge_m3_dense(self, query: str, top_k: int) -> list[RetrievalResult]:
        self._check_bge_m3()
        results = self.bge_m3.search(query, top_k, mode="dense")
        return self._bge_m3_to_results(results, "bge_m3_dense")

    # ── 策略 5: BGE-M3 MultiVec (ColBERT-style) ──
    def _bge_m3_multivec(self, query: str, top_k: int) -> list[RetrievalResult]:
        self._check_bge_m3()
        results = self.bge_m3.search(query, top_k, mode="colbert")
        return self._bge_m3_to_results(results, "bge_m3_multivec")

    # ── 策略 6: BGE-M3 Hybrid (Dense + Sparse + ColBERT 三路融合) ──
    def _bge_m3_hybrid(self, query: str, top_k: int) -> list[RetrievalResult]:
        self._check_bge_m3()
        results = self.bge_m3.search(
            query, top_k, mode="hybrid", weights=config.bge_m3_weights
        )
        return self._bge_m3_to_results(results, "bge_m3_hybrid")

    # ── 策略 7: Multi-Vector (原文 + 摘要双路) ──
    def _multi_vector(self, query: str, top_k: int) -> list[RetrievalResult]:
        """
        对每个 chunk 生成两种 embedding 表示：
        1. 原文 embedding（主 vectorstore）
        2. 摘要 embedding（summary_vectorstore）
        两路检索结果通过 RRF 融合
        """
        query_emb = self.embedding.embed_query(query)
        fetch_k = top_k * 2

        # 原文路
        original_results = self.vectorstore.query_by_text(query, query_emb, fetch_k)

        # 摘要路
        summary_results = []
        if self.summary_vectorstore:
            summary_results = self.summary_vectorstore.query_by_text(query, query_emb, fetch_k)

        # RRF 融合（无摘要的 chunk 自动降级为纯 dense 分数，不被惩罚）
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}
        for rank, r in enumerate(original_results):
            rrf_scores[r["id"]] = rrf_scores.get(r["id"], 0) + 1.0 / (config.rrf_k + rank + 1)
            doc_map[r["id"]] = r

        # 收集摘要路径命中的 original_chunk_id
        summary_hit_ids = set()
        for rank, r in enumerate(summary_results):
            orig_id = r["metadata"].get("original_chunk_id", r["id"])
            rrf_scores[orig_id] = rrf_scores.get(orig_id, 0) + 1.0 / (config.rrf_k + rank + 1)
            summary_hit_ids.add(orig_id)
            if orig_id not in doc_map:
                doc_map[orig_id] = r

        # 对原文路径中没有摘要命中的 chunk，补偿一个等价于摘要路径同排名的分数
        # 这样无摘要的 chunk 不会因为"少一路"而被系统性压低
        if summary_results:
            for rank, r in enumerate(original_results):
                if r["id"] not in summary_hit_ids:
                    rrf_scores[r["id"]] += 1.0 / (config.rrf_k + rank + 1)

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return [
            RetrievalResult(
                id=did, document=doc_map[did]["document"],
                metadata=doc_map[did]["metadata"], score=rrf_scores[did],
                strategy="multi_vector",
            )
            for did in sorted_ids if did in doc_map
        ]

    # ── 策略 8: Semantic Dual-Path (HyDE + 原始查询双路) ──
    def _semantic_dual_path(self, query: str, top_k: int) -> list[RetrievalResult]:
        """
        路径1: 用原始 query embedding 检索
        路径2: HyDE — 用 LLM 生成假设性回答，用其 embedding 检索
        两路 RRF 融合
        """
        import time
        if self.llm is None:
            raise RuntimeError("LLM required for Semantic Dual-Path (HyDE)")

        fetch_k = top_k * 2

        # 路径1: 原始 query
        t0 = time.time()
        original_emb = self.embedding.embed_query(query)
        path1 = self.vectorstore.query_by_text(query, original_emb, fetch_k)
        t1 = time.time()

        # 路径2: HyDE
        hypothetical = self.llm.generate_hypothetical_answer(query)
        t2 = time.time()
        hyde_emb = self.embedding.embed_query(hypothetical)
        path2 = self.vectorstore.query_by_text(query, hyde_emb, fetch_k)
        t3 = time.time()

        from rich.console import Console
        Console().print(
            f"[dim]  Dual-Path timings: "
            f"path1={t1-t0:.1f}s, HyDE_gen={t2-t1:.1f}s, path2={t3-t2:.1f}s[/dim]"
        )

        # RRF 融合
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}
        for rank, r in enumerate(path1):
            rrf_scores[r["id"]] = rrf_scores.get(r["id"], 0) + 1.0 / (config.rrf_k + rank + 1)
            doc_map[r["id"]] = r
        for rank, r in enumerate(path2):
            rrf_scores[r["id"]] = rrf_scores.get(r["id"], 0) + 1.0 / (config.rrf_k + rank + 1)
            if r["id"] not in doc_map:
                doc_map[r["id"]] = r

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return [
            RetrievalResult(
                id=did, document=doc_map[did]["document"],
                metadata=doc_map[did]["metadata"], score=rrf_scores[did],
                strategy="semantic_dual_path",
            )
            for did in sorted_ids if did in doc_map
        ]

    # ── 辅助方法 ──

    def _check_bge_m3(self):
        if self.bge_m3 is None or not self.bge_m3.indexed:
            raise RuntimeError(
                "BGE-M3 index not available. Run 'python main.py ingest' first."
            )

    @staticmethod
    def _bge_m3_to_results(raw: list[dict], strategy: str) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                id=r["id"], document=r["document"],
                metadata=r["metadata"], score=r["score"],
                strategy=strategy,
            )
            for r in raw
        ]

    @staticmethod
    def _rrf_fuse(
        rank_lists: list[list[RetrievalResult]], top_k: int, strategy_name: str
    ) -> list[RetrievalResult]:
        """通用 RRF 融合：将多个排名列表融合为一个"""
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, RetrievalResult] = {}

        for ranked in rank_lists:
            for rank, r in enumerate(ranked):
                rrf_scores[r.id] = rrf_scores.get(r.id, 0) + 1.0 / (config.rrf_k + rank + 1)
                if r.id not in doc_map:
                    doc_map[r.id] = r

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return [
            RetrievalResult(
                id=doc_map[did].id, document=doc_map[did].document,
                metadata=doc_map[did].metadata, score=rrf_scores[did],
                strategy=strategy_name,
            )
            for did in sorted_ids if did in doc_map
        ]
