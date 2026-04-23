"""分块策略：固定分块、递归分块、语义分块

支持表格感知分块：在递归分块前识别 HTML/Markdown 表格边界，
将完整表格作为独立 chunk，避免切割破坏表格结构。
"""

import re
from dataclasses import dataclass
from src.document_loader import Document
from src.embeddings import EmbeddingModel
from config import config
import numpy as np


@dataclass
class Chunk:
    content: str
    metadata: dict
    chunk_id: str


def is_low_quality_chunk(text: str, chunk_type: str = "text") -> bool:
    """判断 chunk 是否为低质量内容，应在 ingest 阶段过滤。

    Args:
        text: chunk 文本内容
        chunk_type: "text" 或 "table"，表格 chunk 跳过参考文献规则

    过滤规则：
    - 过短（<50 字符）：碎片、标题残留
    - 参考文献密集区（3+ 个 "et al." 或 2+ 个 "doi:"）— 仅 text 类型
    - 字母比例过低（<35%）：纯数字/符号/公式噪声 — 仅 text 类型
    """
    if len(text.strip()) < 50:
        return True
    # 表格 chunk 保留（内容有价值，只是格式特殊）
    if chunk_type == "table":
        return False
    if text.count("et al.") > 3 or text.count("doi:") > 2:
        return True
    alpha_ratio = sum(1 for ch in text if ch.isalpha()) / max(len(text), 1)
    if alpha_ratio < 0.35:
        return True
    return False


def _extract_tables(text: str) -> tuple[list[str], str]:
    """从文本中提取完整的 HTML 表格，返回 (表格列表, 去除表格后的剩余文本)。

    保留表格前面紧邻的标题行（如 "Table 10.1 | ..."），作为表格 chunk 的一部分。
    """
    # 匹配：可选的表格标题（Table X.X | ...）+ <table>...</table>
    pattern = r'((?:Table\s+\d+[\.\d]*\s*\|[^\n]*\n\s*)?<table>.*?</table>)'
    tables = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    # 从原文中移除表格，用占位符标记断点
    remaining = re.sub(pattern, '\n\n', text, flags=re.DOTALL | re.IGNORECASE)
    return tables, remaining


class FixedChunking:
    """固定大小分块 + 重叠"""

    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or config.chunk_size
        self.overlap = overlap or config.chunk_overlap

    def split(self, doc: Document) -> list[Chunk]:
        text = doc.content
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(Chunk(
                    content=chunk_text.strip(),
                    metadata={**doc.metadata, "chunk_strategy": "fixed", "chunk_index": idx},
                    chunk_id=f"{doc.doc_id}_fixed_{idx}",
                ))
                idx += 1
            start += self.chunk_size - self.overlap
        return chunks


class RecursiveChunking:
    """递归字符分割：按段落 -> 句子 -> 字符层级递归"""

    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or config.chunk_size
        self.overlap = overlap or config.chunk_overlap
        self.separators = ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", " "]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]
        sep = separators[0]
        parts = text.split(sep)
        result = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(part) > self.chunk_size:
                    result.extend(self._split_text(part, separators[1:]))
                else:
                    current = part
                    continue
                current = ""
        if current:
            result.append(current)
        return result

    def split(self, doc: Document) -> list[Chunk]:
        chunks = []
        idx = 0

        # 1. 表格感知：提取完整 HTML 表格作为独立 chunk
        tables, remaining_text = _extract_tables(doc.content)
        for table_text in tables:
            table_text = table_text.strip()
            if table_text:
                chunks.append(Chunk(
                    content=table_text,
                    metadata={
                        **doc.metadata,
                        "chunk_strategy": "recursive",
                        "chunk_type": "table",
                        "chunk_index": idx,
                    },
                    chunk_id=f"{doc.doc_id}_recursive_{idx}",
                ))
                idx += 1

        # 2. 对剩余文本做递归分块
        raw_chunks = self._split_text(remaining_text, self.separators)
        for text in raw_chunks:
            if text.strip():
                chunks.append(Chunk(
                    content=text.strip(),
                    metadata={
                        **doc.metadata,
                        "chunk_strategy": "recursive",
                        "chunk_type": "text",
                        "chunk_index": idx,
                    },
                    chunk_id=f"{doc.doc_id}_recursive_{idx}",
                ))
                idx += 1

        return chunks


class SemanticChunking:
    """基于 embedding 相似度的语义分块：
    将文本按句子切分后，计算相邻句子的 embedding 相似度，
    在相似度低于阈值的位置断开，形成语义连贯的分块。
    """

    def __init__(self, embedding_model: EmbeddingModel = None, threshold: float = None):
        self.embedding_model = embedding_model
        self.threshold = threshold or config.semantic_threshold

    def _split_sentences(self, text: str) -> list[str]:
        import re
        sentences = re.split(r"(?<=[。！？.!?\n])", text)
        return [s.strip() for s in sentences if s.strip()]

    def split(self, doc: Document) -> list[Chunk]:
        if self.embedding_model is None:
            self.embedding_model = EmbeddingModel()

        sentences = self._split_sentences(doc.content)
        if len(sentences) <= 1:
            return [Chunk(
                content=doc.content.strip(),
                metadata={**doc.metadata, "chunk_strategy": "semantic", "chunk_index": 0},
                chunk_id=f"{doc.doc_id}_semantic_0",
            )]

        embeddings = self.embedding_model.embed_texts(sentences)

        # 计算相邻句子的余弦相似度
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self.embedding_model.cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # 在相似度低于阈值的位置断开
        chunks = []
        current_sentences = [sentences[0]]
        idx = 0
        for i, sim in enumerate(similarities):
            if sim < self.threshold and len("".join(current_sentences)) > 50:
                chunks.append(Chunk(
                    content="".join(current_sentences).strip(),
                    metadata={
                        **doc.metadata,
                        "chunk_strategy": "semantic",
                        "chunk_index": idx,
                        "sentence_count": len(current_sentences),
                    },
                    chunk_id=f"{doc.doc_id}_semantic_{idx}",
                ))
                current_sentences = []
                idx += 1
            current_sentences.append(sentences[i + 1])

        if current_sentences:
            chunks.append(Chunk(
                content="".join(current_sentences).strip(),
                metadata={
                    **doc.metadata,
                    "chunk_strategy": "semantic",
                    "chunk_index": idx,
                    "sentence_count": len(current_sentences),
                },
                chunk_id=f"{doc.doc_id}_semantic_{idx}",
            ))
        return chunks


CHUNKING_STRATEGIES = {
    "fixed": FixedChunking,
    "recursive": RecursiveChunking,
    "semantic": SemanticChunking,
}
