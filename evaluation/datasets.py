"""评估数据集加载与管理"""

import json
import re
from dataclasses import dataclass, field
from config import config


@dataclass
class EvalQuestion:
    id: int
    question: str
    ground_truth: str
    ground_truth_contexts: list[str]
    source_doc: str          # 统一为 string（多文档用 " | " 拼接）
    difficulty: str = "simple"


class EvalDataset:
    def __init__(self, path: str = None):
        self.path = path or config.eval_dataset_path
        self.questions: list[EvalQuestion] = []

    @staticmethod
    def _normalize_source_doc(raw) -> str:
        """source_doc 可能是 str 或 list[str]，统一为 str"""
        if isinstance(raw, list):
            return " | ".join(raw)
        return raw or ""

    @staticmethod
    def _clean_context(text: str) -> str:
        """清理 RAGAS 生成的 context 中的标签和多余空白

        去除：<1-hop>, <2-hop> 等标签；开头的 # 标题行；多余换行
        """
        # 去除 <N-hop> 标签
        text = re.sub(r'<\d+-hop>\s*', '', text)
        # 去除开头的 markdown 标题行（# Title）
        text = re.sub(r'^(?:#+\s+[^\n]*\n)+', '', text.strip())
        # 压缩多余换行
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def load(self) -> list[EvalQuestion]:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 兼容两种格式：{"questions": [...]} 或直接 [...]
        items = data["questions"] if isinstance(data, dict) else data
        self.questions = [
            EvalQuestion(
                id=q.get("id", i + 1),
                question=q["question"],
                ground_truth=q["ground_truth"],
                ground_truth_contexts=[
                    self._clean_context(c) for c in q.get("ground_truth_contexts", [])
                ],
                source_doc=self._normalize_source_doc(q.get("source_doc", "")),
                difficulty=q.get("difficulty", "simple"),
            )
            for i, q in enumerate(items)
        ]
        return self.questions

    def get_subset(self, n: int = None) -> list[EvalQuestion]:
        if not self.questions:
            self.load()
        if n is None:
            return self.questions
        return self.questions[:n]
