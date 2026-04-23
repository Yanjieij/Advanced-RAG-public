"""智谱 Embedding-3 封装

向量维度：2048（由 config.embedding_dim 配置）
单次请求上限：64 条（智谱 API 限制，内部自动分批）
"""

import numpy as np
from zhipuai import ZhipuAI
from config import config


class EmbeddingModel:
    def __init__(self):
        self.client = ZhipuAI(api_key=config.zhipuai_api_key)
        self.model = config.embedding_model
        self.dim = config.embedding_dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量生成文本向量，自动分批处理（每批最多 64 条）"""
        if not texts:
            return []
        # 清理：将 None 或空串替换为占位文本，避免 API 报错
        cleaned = [t if t and t.strip() else "(empty)" for t in texts]
        # 智谱 API 单次最多 64 条，超过则分批
        all_embeddings = []
        for i in range(0, len(cleaned), 64):
            batch = cleaned[i : i + 64]
            response = self.client.embeddings.create(model=self.model, input=batch)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """生成单条查询的向量（embed_texts 的快捷方式）"""
        return self.embed_texts([query])[0]

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """计算两个向量的余弦相似度，返回 [-1, 1]"""
        a_np, b_np = np.array(a), np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))
