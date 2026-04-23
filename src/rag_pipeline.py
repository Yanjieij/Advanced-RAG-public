"""RAG Pipeline 组装器：构建和管理不同配置的 RAG 流水线"""

import os
import json
import pickle
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from config import config
from src.document_loader import load_documents, preprocess, Document
from src.chunking import Chunk, FixedChunking, RecursiveChunking, SemanticChunking, is_low_quality_chunk
from src.embeddings import EmbeddingModel
from src.llm import LLM
from src.vectorstore import VectorStore
from src.sparse_retriever import BM25Retriever
from src.bge_m3_retriever import BGEM3Retriever
from src.retriever import UnifiedRetriever, RetrievalStrategy
from src.reranker import LLMReranker, ContextCompressor
from src.query_engine import QueryEngine, RAGResponse

console = Console()


class RAGPipeline:
    """可配置的 RAG Pipeline，支持不同检索策略的快速切换和对比

    通过 dataset 参数隔离不同数据集的向量库和索引缓存：
    - ChromaDB collections: {dataset}_original, {dataset}_summaries
    - BM25 缓存: .index_cache/{dataset}_bm25.pkl
    - BGE-M3 缓存: .index_cache/{dataset}_bge_m3.pkl
    """

    def __init__(self, dataset: str = "default"):
        self.dataset = dataset
        self.embedding_model = EmbeddingModel()
        self.llm = LLM()
        self.vectorstore = VectorStore(f"{dataset}_original")
        self.summary_vectorstore = VectorStore(f"{dataset}_summaries")
        self.bm25 = BM25Retriever(dataset=dataset)
        self.bge_m3 = BGEM3Retriever(
            model_name=config.bge_m3_model,
            use_fp16=config.bge_m3_use_fp16,
            dataset=dataset,
        )
        self.reranker = LLMReranker(self.llm)
        self.compressor = ContextCompressor(self.llm)
        self.chunks: list[Chunk] = []

        # 启动时自动从磁盘加载缓存的索引（BM25 和 BGE-M3）
        if self.bm25.load():
            console.print(f"[dim]BM25 索引已从缓存加载（dataset={dataset}）[/dim]")
        if self.bge_m3.load():
            console.print(f"[dim]BGE-M3 索引已从缓存加载（dataset={dataset}）[/dim]")

    def ingest(
        self,
        doc_dir: str = None,
        chunking_strategy: str = "recursive",
        reset: bool = False,
        skip_summary: bool = False,
    ):
        """文档入库流程：加载 -> 预处理 -> 分块 -> 编码 -> 存储"""
        doc_dir = doc_dir or config.documents_dir

        if reset:
            console.print("[yellow]正在重置向量库和索引缓存...[/yellow]")
            self.vectorstore.reset()
            self.summary_vectorstore.reset()
            self.bm25.clear()
            self.bge_m3.clear()
            # 清除 chunk 缓存
            if os.path.exists(self._chunks_cache_path()):
                os.remove(self._chunks_cache_path())

        # 1. 加载文档
        console.print(f"[cyan]正在从 {doc_dir} 加载文档...[/cyan]")
        docs = load_documents(doc_dir)
        console.print(f"  已加载 {len(docs)} 个文档")

        # 2. 预处理（去除多余空白等）
        for doc in docs:
            doc.content = preprocess(doc.content)

        # 3. 分块
        console.print(f"[cyan]正在分块（策略: {chunking_strategy}）...[/cyan]")
        chunker_map = {
            "fixed": FixedChunking(),
            "recursive": RecursiveChunking(),
            "semantic": SemanticChunking(self.embedding_model),
        }
        chunker = chunker_map[chunking_strategy]
        raw_chunks = []
        for doc in docs:
            raw_chunks.extend(chunker.split(doc))

        # 质量过滤：过短、参考文献密集、噪声 chunk
        self.chunks = [
            c for c in raw_chunks
            if not is_low_quality_chunk(c.content, c.metadata.get("chunk_type", "text"))
        ]
        filtered = len(raw_chunks) - len(self.chunks)
        console.print(
            f"  生成 {len(raw_chunks)} 个分块，"
            f"过滤 {filtered} 个低质量块 → 保留 {len(self.chunks)} 个"
        )

        # 4. Embedding（智谱 Embedding-3）并写入 ChromaDB
        console.print("[cyan]正在向量化并写入 ChromaDB...[/cyan]")
        texts = [c.content for c in self.chunks]
        ids = [c.chunk_id for c in self.chunks]
        metadatas = [c.metadata for c in self.chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        self.vectorstore.add_documents(ids, texts, embeddings, metadatas)
        console.print(f"  已写入 {self.vectorstore.count()} 个分块到 ChromaDB")

        # 5. 构建 BM25 索引
        console.print("[cyan]正在构建 BM25 索引...[/cyan]")
        self.bm25.index(self.chunks)

        # 6. 构建 BGE-M3 多粒度索引（Dense + Sparse + ColBERT 三合一）
        console.print("[cyan]正在构建 BGE-M3 多粒度索引...[/cyan]")
        try:
            self.bge_m3.index(texts, ids)
        except Exception as e:
            console.print(
                f"[yellow]  BGE-M3 索引构建失败: {e}[/yellow]\n"
                f"[dim]  (其他策略不受影响，可先用 naive_dense/hybrid_rrf 等)[/dim]"
            )

        if skip_summary:
            console.print(
                "[green]入库完成（--no-summary 跳过摘要）！"
                "7 种策略可用，multi_vector 降级为纯 dense 检索。[/green]"
            )
        else:
            console.print(
                "[green]核心入库完成！7 种策略已就绪（可立即运行 generate-eval / evaluate）[/green]\n"
            )

            # 7. 生成摘要 embedding（用于 Multi-Vector，最耗时，放最后）
            console.print("[cyan]正在为 Multi-Vector 生成摘要（可按 Ctrl+C 跳过）...[/cyan]")
            self._generate_summaries(self.chunks)

            console.print("[green]入库完成！8 种策略全部就绪。[/green]")

    def _generate_summaries(self, chunks: list[Chunk]):
        """为 chunks 生成摘要并存入 summary_vectorstore。

        支持断点续传：检查 summary_vectorstore 中已有的摘要数量，
        只对缺少摘要的 chunks 生成。中途 Ctrl+C 不影响其他策略。
        """
        # 检查已有摘要数量
        existing_summary_count = self.summary_vectorstore.count()
        total = len(chunks)

        if existing_summary_count >= total:
            console.print(
                f"  [dim]摘要已全部生成（{existing_summary_count}/{total}），跳过[/dim]"
            )
            return

        if existing_summary_count > 0:
            console.print(
                f"  [dim]续传摘要生成：已完成 {existing_summary_count}/{total}[/dim]"
            )
            # 跳过已有摘要的 chunks
            chunks_to_summarize = chunks[existing_summary_count:]
        else:
            chunks_to_summarize = chunks

        console.print(f"  正在为 {len(chunks_to_summarize)} 个分块生成摘要...")

        try:
            prompts = [
                f"请用一句话概括以下内容的核心要点：\n{chunk.content}"
                for chunk in chunks_to_summarize
            ]
            summaries = self.llm.batch_generate(
                prompts, temperature=0.1, max_tokens=100, model=config.ingest_llm_model
            )

            summary_ids = [f"summary_{c.chunk_id}" for c in chunks_to_summarize]
            summary_metadatas = [
                {**c.metadata, "original_chunk_id": c.chunk_id, "type": "summary"}
                for c in chunks_to_summarize
            ]
            summary_embeddings = self.embedding_model.embed_texts(summaries)
            self.summary_vectorstore.add_documents(
                summary_ids, summaries, summary_embeddings, summary_metadatas
            )
            console.print(
                f"  [green]摘要生成完成（{self.summary_vectorstore.count()}/{total}）[/green]"
            )
        except KeyboardInterrupt:
            console.print(
                f"\n  [yellow]摘要生成被中断 "
                f"（已完成 {self.summary_vectorstore.count()}/{total}）。"
                f"multi_vector 策略将仅使用原文路径。"
                f"重新运行 ingest（不加 --reset）可续传。[/yellow]"
            )

    def _chunks_cache_path(self) -> str:
        """chunk 缓存文件路径"""
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".index_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{self.dataset}_chunks.pkl")

    def _save_chunks(self, chunks: list[Chunk]):
        """将 chunks 持久化到本地"""
        with open(self._chunks_cache_path(), "wb") as f:
            pickle.dump(chunks, f)

    def _load_chunks(self) -> list[Chunk]:
        """从本地加载 chunks"""
        path = self._chunks_cache_path()
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return []

    def ingest_incremental(
        self,
        doc_dir: str = None,
        chunking_strategy: str = "recursive",
        reset: bool = False,
        skip_summary: bool = False,
    ):
        """逐文件增量入库：每个文件独立完成全部步骤，中断后从断点续传。

        流程：
        1. reset（可选）清空所有索引
        2. 扫描目录，逐文件处理：
           a. 加载 + 预处理 + 分块
           b. Embedding → ChromaDB
           c. 摘要 → ChromaDB summaries
           d. 保存 chunks 到本地缓存
        3. 所有文件入完后，从本地 chunk 缓存重建 BM25 + BGE-M3

        断点续传：不加 --reset 重新跑，已入库的 chunk 自动跳过。
        """
        doc_dir = doc_dir or config.documents_dir

        if reset:
            console.print("[yellow]正在重置向量库和索引缓存...[/yellow]")
            self.vectorstore.reset()
            self.summary_vectorstore.reset()
            self.bm25.clear()
            self.bge_m3.clear()
            # 清除 chunk 缓存
            if os.path.exists(self._chunks_cache_path()):
                os.remove(self._chunks_cache_path())

        # 加载全部文档列表
        docs = load_documents(doc_dir)
        console.print(f"[cyan]发现 {len(docs)} 个文档，目录：{doc_dir}[/cyan]")

        # 从本地缓存加载已有的 chunks（断点续传：跳过已入库的文档）
        cached_chunks = self._load_chunks()
        existing_ids = {c.chunk_id for c in cached_chunks}
        if existing_ids:
            console.print(f"  [dim]检测到 chunk 缓存：上次已处理 {len(existing_ids)} 个分块[/dim]")

        chunker_map = {
            "fixed": FixedChunking(),
            "recursive": RecursiveChunking(),
            "semantic": SemanticChunking(self.embedding_model),
        }
        chunker = chunker_map[chunking_strategy]

        for di, doc in enumerate(docs, 1):
            doc.content = preprocess(doc.content)
            raw_chunks = chunker.split(doc)
            # 质量过滤
            chunks = [c for c in raw_chunks if not is_low_quality_chunk(c.content)]

            # 过滤已入库的 chunk
            new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

            if not new_chunks:
                console.print(
                    f"[dim][{di}/{len(docs)}] {doc.doc_id} — "
                    f"{len(chunks)} 个分块（已全部缓存，跳过）[/dim]"
                )
                continue

            console.print(
                f"[cyan][{di}/{len(docs)}] {doc.doc_id} — "
                f"{len(new_chunks)} 个新分块（共 {len(chunks)} 个）[/cyan]"
            )

            # a. Embedding → ChromaDB（摘要统一在所有文件处理完后生成）
            texts = [c.content for c in new_chunks]
            ids = [c.chunk_id for c in new_chunks]
            metadatas = [c.metadata for c in new_chunks]
            embeddings = self.embedding_model.embed_texts(texts)
            self.vectorstore.add_documents(ids, texts, embeddings, metadatas)

            # b. 更新本地 chunk 缓存（每个文件处理完立即保存，确保中断后可续传）
            cached_chunks.extend(new_chunks)
            existing_ids.update(ids)
            self._save_chunks(cached_chunks)

            console.print(
                f"  [green]完成 — 累计 {len(cached_chunks)} 个分块 "
                f"（缓存已保存）[/green]"
            )

        # 从本地 chunk 缓存重建 BM25 + BGE-M3（全量重建以保证索引完整性）
        all_chunks = cached_chunks
        self.chunks = all_chunks

        console.print(f"[cyan]正在构建 BM25 索引（共 {len(all_chunks)} 个分块）...[/cyan]")
        self.bm25.index(all_chunks)

        console.print(f"[cyan]正在构建 BGE-M3 索引（共 {len(all_chunks)} 个分块）...[/cyan]")
        try:
            all_texts = [c.content for c in all_chunks]
            all_ids = [c.chunk_id for c in all_chunks]
            self.bge_m3.index(all_texts, all_ids)
        except Exception as e:
            console.print(
                f"[yellow]  BGE-M3 索引构建失败: {e}[/yellow]\n"
                f"[dim]  (其他策略不受影响)[/dim]"
            )

        if skip_summary:
            console.print(
                f"[green]增量入库完成（--no-summary 跳过摘要）！"
                f"共 {len(all_chunks)} 个分块，来自 {len(docs)} 个文档。"
                f"7 种策略可用，multi_vector 降级为纯 dense 检索。[/green]"
            )
        else:
            console.print(
                "[green]核心入库完成！7 种策略已就绪（可立即运行 generate-eval / evaluate）[/green]\n"
            )

            console.print("[cyan]正在为 Multi-Vector 生成摘要（可按 Ctrl+C 跳过）...[/cyan]")
            self._generate_summaries(all_chunks)

            console.print(
                f"[green]增量入库完成！"
                f"共 {len(all_chunks)} 个分块，来自 {len(docs)} 个文档[/green]"
            )

    def get_query_engine(
        self,
        use_query_rewrite: bool = False,
        use_rerank: bool = False,
        use_compression: bool = False,
    ) -> QueryEngine:
        retriever = UnifiedRetriever(
            embedding_model=self.embedding_model,
            vectorstore=self.vectorstore,
            bm25=self.bm25,
            bge_m3=self.bge_m3,
            llm=self.llm,
            summary_vectorstore=self.summary_vectorstore,
        )
        return QueryEngine(
            retriever=retriever,
            llm=self.llm,
            reranker=self.reranker if use_rerank else None,
            compressor=self.compressor if use_compression else None,
            use_query_rewrite=use_query_rewrite,
            use_rerank=use_rerank,
            use_compression=use_compression,
        )

    def query(
        self,
        query: str,
        strategy: str = "hybrid_rrf",
        use_rerank: bool = False,
        use_query_rewrite: bool = False,
        use_compression: bool = False,
    ) -> RAGResponse:
        engine = self.get_query_engine(use_query_rewrite, use_rerank, use_compression)
        strat = RetrievalStrategy(strategy)
        return engine.query(query, strat)

    def retrieve(
        self,
        query: str,
        strategy: str = "hybrid_rrf",
        use_rerank: bool = False,
        use_query_rewrite: bool = False,
    ) -> RAGResponse:
        """纯检索：只返回检索结果，不调用 LLM 生成。"""
        engine = self.get_query_engine(use_query_rewrite, use_rerank)
        strat = RetrievalStrategy(strategy)
        return engine.retrieve_only(query, strat)

    def compare_strategies(
        self, query: str, strategies: list[str] = None
    ) -> dict[str, RAGResponse]:
        """用同一个 query 对比所有策略"""
        if strategies is None:
            strategies = [s.value for s in RetrievalStrategy]
        results = {}
        engine = self.get_query_engine()
        for s in strategies:
            try:
                strat = RetrievalStrategy(s)
                resp = engine.query(query, strat)
                results[s] = resp
            except Exception as e:
                console.print(f"[yellow]Strategy {s} failed: {e}[/yellow]")
        return results

    def print_comparison(self, results: dict[str, RAGResponse]):
        table = Table(title="检索策略对比")
        table.add_column("策略", style="cyan")
        table.add_column("回答（截断预览）", style="white", max_width=60)
        table.add_column("上下文数", style="green", justify="center")
        table.add_column("总耗时", style="yellow", justify="right")
        for strategy, resp in results.items():
            answer_preview = resp.answer[:80] + "..." if len(resp.answer) > 80 else resp.answer
            table.add_row(
                strategy,
                answer_preview,
                str(len(resp.contexts)),
                f"{resp.timings.get('total', 0):.2f}s",
            )
        console.print(table)
