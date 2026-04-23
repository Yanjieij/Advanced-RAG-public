"""BM25 稀疏检索"""

import os
import pickle
import jieba
from rank_bm25 import BM25Okapi
from src.chunking import Chunk
from config import PROJECT_ROOT

# 持久化目录
INDEX_DIR = os.path.join(PROJECT_ROOT, ".index_cache")


class BM25Retriever:
    """基于 BM25 的稀疏检索器，使用 jieba 分词，支持磁盘持久化"""

    def __init__(self, dataset: str = "default"):
        self.dataset = dataset
        self.SAVE_PATH = os.path.join(INDEX_DIR, f"{dataset}_bm25.pkl")
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: list[list[str]] = []

    def index(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.tokenized_corpus = [
            list(jieba.cut(chunk.content)) for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._save()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.bm25 is None:
            return []
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                results.append({
                    "id": self.chunks[idx].chunk_id,
                    "document": self.chunks[idx].content,
                    "metadata": self.chunks[idx].metadata,
                    "score": float(scores[idx]),
                    "rank": rank,
                })
        return results

    def _save(self):
        """将索引持久化到磁盘"""
        os.makedirs(INDEX_DIR, exist_ok=True)
        data = {
            "chunks": self.chunks,
            "tokenized_corpus": self.tokenized_corpus,
        }
        with open(self.SAVE_PATH, "wb") as f:
            pickle.dump(data, f)

    def clear(self):
        """删除磁盘缓存并重置内存状态"""
        if os.path.exists(self.SAVE_PATH):
            os.remove(self.SAVE_PATH)
        self.chunks = []
        self.tokenized_corpus = []
        self.bm25 = None

    def load(self) -> bool:
        """从磁盘加载索引，成功返回 True"""
        if not os.path.exists(self.SAVE_PATH):
            return False
        try:
            with open(self.SAVE_PATH, "rb") as f:
                data = pickle.load(f)
            self.chunks = data["chunks"]
            self.tokenized_corpus = data["tokenized_corpus"]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            return True
        except Exception:
            return False
