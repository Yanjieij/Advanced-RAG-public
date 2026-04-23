"""文档加载与预处理

支持格式：.txt、.md（Markdown）、.pdf

主要函数：
- load_documents(directory): 扫描目录，加载所有支持格式的文档
- preprocess(text): 清理文本（压缩多余换行、空格）
"""

import os
import re
from dataclasses import dataclass, field


@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""


def load_txt(filepath: str) -> Document:
    """加载纯文本文件（UTF-8 编码）"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return Document(
        content=content,
        metadata={"source": filepath, "type": "txt"},
        doc_id=os.path.basename(filepath),
    )


def load_markdown(filepath: str) -> Document:
    """加载 Markdown 文件（保留原始格式，分块时会处理标题/段落边界）"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return Document(
        content=content,
        metadata={"source": filepath, "type": "markdown"},
        doc_id=os.path.basename(filepath),
    )


def load_pdf(filepath: str) -> Document:
    """加载 PDF 文件（逐页提取文本，页间用双换行分隔）"""
    from pypdf import PdfReader

    reader = PdfReader(filepath)
    pages = [page.extract_text() or "" for page in reader.pages]
    content = "\n\n".join(pages)
    return Document(
        content=content,
        metadata={"source": filepath, "type": "pdf", "pages": len(reader.pages)},
        doc_id=os.path.basename(filepath),
    )


# 格式 → 加载函数映射
LOADERS = {
    ".txt": load_txt,
    ".md": load_markdown,
    ".pdf": load_pdf,
}


def load_documents(directory: str) -> list[Document]:
    """扫描目录，按文件名排序加载所有支持的文档"""
    docs = []
    for filename in sorted(os.listdir(directory)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in LOADERS:
            filepath = os.path.join(directory, filename)
            docs.append(LOADERS[ext](filepath))
    return docs


def preprocess(text: str) -> str:
    """文本预处理：压缩多余换行（3+ → 2），合并行内多余空格"""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
