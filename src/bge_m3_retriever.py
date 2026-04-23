"""BGE-M3 多粒度检索器

BGE-M3 (BAAI General Embedding - Multi-Functionality, Multi-Linguality, Multi-Granularity)
是智源研究院开发的多功能嵌入模型，核心特点是 **一个模型同时输出三种表示**：

1. Dense Embedding (稠密向量):
   - 与普通 embedding 模型一样，为整个文本生成一个固定维度的向量（1024 维）
   - 用于标准的语义相似度检索

2. Sparse Embedding (稀疏向量 / Learned Sparse):
   - 类似 BM25，但权重是模型学习得到的，而非统计计算
   - 输出一个 token → weight 的字典，权重越高表示该 token 越重要
   - 比传统 BM25 更智能：能学习到同义词的关联，且对新词有更好的泛化

3. ColBERT-style Multi-Vector (多向量):
   - 与 ColBERT 相同的 Late Interaction 机制
   - 为文本中的每个 token 生成一个向量
   - 检索时用 MaxSim 操作计算细粒度相似度
   - 精度最高，但计算和存储开销也最大

本模块封装了 BGE-M3 的三种检索模式及其融合方案。
"""

import os
import pickle
import numpy as np
from typing import Optional
from rich.console import Console
from config import PROJECT_ROOT

console = Console()

# 向量索引持久化目录
INDEX_DIR = os.path.join(PROJECT_ROOT, ".index_cache")


class BGEM3Retriever:
    """BGE-M3 多粒度检索器

    支持三种检索模式：
    - dense: 稠密向量检索（默认，最快）
    - sparse: 学习型稀疏检索（精确关键词 + 语义感知）
    - colbert: ColBERT-style token 级别细粒度检索（最精确）
    - hybrid: 三路融合（Dense + Sparse + ColBERT，综合最优）
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True, dataset: str = "default"):
        """
        Args:
            model_name: HuggingFace 模型名称
            use_fp16: 是否使用 FP16 推理（减少内存，速度更快）
            dataset: 数据集名称，用于隔离不同数据集的索引缓存
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.model = None
        self.dataset = dataset
        self.SAVE_PATH = os.path.join(INDEX_DIR, f"{dataset}_bge_m3.pkl")

        # 索引存储（三种表示分别保存）
        self._doc_dense: Optional[np.ndarray] = None   # 稠密向量矩阵，形状 (n_docs, 1024)
        self._doc_sparse: Optional[list[dict]] = None   # 学习型稀疏权重，每个元素为 {token_id: weight}
        self._doc_colbert: Optional[list] = None         # ColBERT 多向量，每个元素为 token 级别的向量数组
        self._documents: list[str] = []
        self._doc_ids: list[str] = []
        self.indexed = False

    def _load_model(self):
        """延迟加载模型（首次使用时才下载和加载，避免启动时长时间等待）"""
        if self.model is not None:
            return

        console.print(f"[cyan]正在加载 BGE-M3 模型（{self.model_name}）...[/cyan]")
        try:
            from FlagEmbedding import BGEM3FlagModel

            self.model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
            )
            console.print("[green]  BGE-M3 模型加载成功[/green]")
        except ImportError:
            raise RuntimeError(
                "FlagEmbedding 未安装，请执行: pip install FlagEmbedding"
            )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 12,
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = True,
    ) -> dict:
        """编码文本，同时生成三种表示

        Args:
            texts: 待编码的文本列表
            batch_size: 批处理大小
            return_dense: 是否返回稠密向量
            return_sparse: 是否返回稀疏向量
            return_colbert: 是否返回 ColBERT 多向量

        Returns:
            dict，包含键：'dense_vecs'（稠密向量）、'lexical_weights'（稀疏权重）、'colbert_vecs'（多向量）
        """
        self._load_model()
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=512,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert,
        )
        return output

    def index(
        self,
        documents: list[str],
        document_ids: list[str] | None = None,
        batch_size: int = 12,
    ):
        """为文档集合建立三种索引

        Args:
            documents: 文档文本列表
            document_ids: 文档 ID 列表
            batch_size: 编码批处理大小
        """
        self._load_model()
        self._documents = documents
        self._doc_ids = document_ids or [str(i) for i in range(len(documents))]

        console.print(f"  正在用 BGE-M3 编码 {len(documents)} 个文档块...")
        output = self.encode(
            documents,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert=True,
        )

        self._doc_dense = np.array(output["dense_vecs"])        # 稠密向量，形状 (n, 1024)
        self._doc_sparse = output["lexical_weights"]            # 稀疏权重，list[dict]
        self._doc_colbert = output["colbert_vecs"]              # ColBERT 多向量，list[np.ndarray]

        self.indexed = True
        self._save()
        console.print(
            f"  BGE-M3 索引构建完成：{len(documents)} 个文档块，"
            f"稠密向量={self._doc_dense.shape}，"
            f"稀疏向量={len(self._doc_sparse)} 条，"
            f"多向量={len(self._doc_colbert)} 条"
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        weights: dict | None = None,
    ) -> list[dict]:
        """检索

        Args:
            query: 查询文本
            top_k: 返回数量
            mode: 检索模式
                - "dense": 仅用稠密向量
                - "sparse": 仅用学习型稀疏向量
                - "colbert": 仅用 ColBERT 多向量
                - "hybrid": 三路加权融合（默认）
            weights: hybrid 模式下的权重, 默认 {"dense": 0.4, "sparse": 0.2, "colbert": 0.4}

        Returns:
            list[dict]，每条包含：id、document、metadata、score（相关性得分）、rank（排名）
        """
        if not self.indexed:
            return []

        self._load_model()

        # 默认三路融合权重（Dense:Sparse:ColBERT = 4:2:4）
        w = weights or {"dense": 0.4, "sparse": 0.2, "colbert": 0.4}

        # 编码 query（只编码当前模式所需的表示类型）
        q_output = self.encode(
            [query],
            return_dense=(mode in ("dense", "hybrid")),
            return_sparse=(mode in ("sparse", "hybrid")),
            return_colbert=(mode in ("colbert", "hybrid")),
        )

        scores = np.zeros(len(self._documents))

        # Dense 评分（点积 = cosine，因为向量已 L2 归一化）
        if mode in ("dense", "hybrid"):
            q_dense = np.array(q_output["dense_vecs"][0])
            dense_scores = self._doc_dense @ q_dense
            if mode == "dense":
                scores = dense_scores
            else:
                scores += w["dense"] * self._normalize_scores(dense_scores)

        # Sparse 评分（Learned Sparse，token 级别内积）
        if mode in ("sparse", "hybrid"):
            q_sparse = q_output["lexical_weights"][0]
            sparse_scores = self._compute_sparse_scores(q_sparse)
            if mode == "sparse":
                scores = sparse_scores
            else:
                scores += w["sparse"] * self._normalize_scores(sparse_scores)

        # ColBERT 评分（MaxSim，token 级别细粒度匹配）
        if mode in ("colbert", "hybrid"):
            q_colbert = q_output["colbert_vecs"][0]
            colbert_scores = self._compute_colbert_scores(q_colbert)
            if mode == "colbert":
                scores = colbert_scores
            else:
                scores += w["colbert"] * self._normalize_scores(colbert_scores)

        # 排序取 top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "id": self._doc_ids[idx],
                "document": self._documents[idx],
                "metadata": {"retrieval_mode": mode},
                "score": float(scores[idx]),
                "rank": rank,
            })
        return results

    def _compute_sparse_scores(self, q_sparse: dict) -> np.ndarray:
        """计算 Learned Sparse 检索分数

        原理：query 和 doc 都有一个 token→weight 的字典，
        score = sum(q_weight[token] * d_weight[token]) for 共有的 token
        """
        scores = np.zeros(len(self._documents))
        for i, d_sparse in enumerate(self._doc_sparse):
            score = 0.0
            for token_id, q_weight in q_sparse.items():
                if token_id in d_sparse:
                    score += q_weight * d_sparse[token_id]
            scores[i] = score
        return scores

    def _compute_colbert_scores(self, q_colbert: np.ndarray) -> np.ndarray:
        """计算 ColBERT MaxSim 分数

        原理：对 query 中的每个 token embedding，找到 document 中
        与其最相似的 token embedding (max similarity)，然后求和。

        score(Q, D) = Σ_i max_j sim(q_i, d_j)
        """
        q_vecs = np.array(q_colbert)  # (q_len, dim)
        scores = np.zeros(len(self._documents))

        for i, d_colbert in enumerate(self._doc_colbert):
            d_vecs = np.array(d_colbert)  # (d_len, dim)
            # sim_matrix[i][j] = cos_sim(q_i, d_j)
            sim_matrix = q_vecs @ d_vecs.T  # (q_len, d_len)
            # MaxSim: 每个 query token 取最大相似度，然后求和
            max_sims = sim_matrix.max(axis=1)  # (q_len,)
            scores[i] = max_sims.sum()

        return scores

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Min-Max 归一化到 [0, 1]，用于不同检索模式的分数融合"""
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s < 1e-8:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    def _save(self):
        """将索引数据（编码后的向量）持久化到磁盘，不含模型权重，文件体积远小于模型本身"""
        os.makedirs(INDEX_DIR, exist_ok=True)
        data = {
            "documents": self._documents,
            "doc_ids": self._doc_ids,
            "doc_dense": self._doc_dense,
            "doc_sparse": self._doc_sparse,
            "doc_colbert": self._doc_colbert,
        }
        with open(self.SAVE_PATH, "wb") as f:
            pickle.dump(data, f)

    def clear(self):
        """删除磁盘缓存并重置内存状态"""
        if os.path.exists(self.SAVE_PATH):
            os.remove(self.SAVE_PATH)
        self._doc_dense = None
        self._doc_sparse = None
        self._doc_colbert = None
        self._documents = []
        self._doc_ids = []
        self.indexed = False

    def load(self) -> bool:
        """从磁盘加载索引数据，成功返回 True。

        只加载向量数据（pkl），BGE-M3 模型本身在首次 search 时延迟加载。
        """
        if not os.path.exists(self.SAVE_PATH):
            return False
        try:
            with open(self.SAVE_PATH, "rb") as f:
                data = pickle.load(f)
            self._documents = data["documents"]
            self._doc_ids = data["doc_ids"]
            self._doc_dense = data["doc_dense"]
            self._doc_sparse = data["doc_sparse"]
            self._doc_colbert = data["doc_colbert"]
            self.indexed = True
            return True
        except Exception:
            return False
