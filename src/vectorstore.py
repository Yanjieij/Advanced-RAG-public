"""ChromaDB 向量库操作

每个数据集对应两个 Collection：
- {dataset}_original: 原文 chunk 的向量
- {dataset}_summaries: 摘要 chunk 的向量（Multi-Vector 策略专用）

相似度计算方式：Cosine（ChromaDB 默认返回 distance，转换为 similarity = 1 - distance）
"""

import chromadb
from config import config


class VectorStore:
    def __init__(self, collection_name: str = "rag_documents"):
        self.client = chromadb.HttpClient(
            host=config.chroma_host, port=config.chroma_port
        )
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        batch_size: int = 200,
    ):
        """分批写入 ChromaDB，避免单次请求过大触发 nginx 413 错误。"""
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            batch_meta = metadatas[i:end] if metadatas else None
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=batch_meta,
            )

    def query(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
    ) -> dict:
        """用向量检索，返回 ChromaDB 原始结果（含 ids/documents/metadatas/distances）"""
        k = top_k or config.top_k
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def query_by_text(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int | None = None,
    ) -> list[dict]:
        """用向量检索，返回格式化结果列表（含 id/document/metadata/score）

        score = 1 - cosine_distance，范围 [0, 1]，越高越相关。
        """
        raw = self.query(query_embedding, top_k)
        results = []
        for i in range(len(raw["ids"][0])):
            results.append({
                "id": raw["ids"][0][i],
                "document": raw["documents"][0][i],
                "metadata": raw["metadatas"][0][i] if raw["metadatas"] else {},
                "distance": raw["distances"][0][i],
                "score": 1.0 - raw["distances"][0][i],  # cosine distance → similarity
            })
        return results

    def reset(self):
        """删除并重建 Collection（清空所有数据）"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """返回 Collection 中的文档数量"""
        return self.collection.count()
