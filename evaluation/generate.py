"""基于 RAGAS TestsetGenerator 自动生成评估数据集

从已入库的 chunks 出发，利用 RAGAS 的知识图谱 + 多类型 synthesizer
自动生成 simple / inference / cross_document 三种难度的评估题目。

使用方法:
    python main.py generate-eval -d wg2 --n-questions 50
"""

import json
import logging
import os
import pickle
import random
import time
from datetime import datetime
from typing import Optional

from rich.console import Console

from config import config, PROJECT_ROOT

console = Console()

# chunk 缓存目录
INDEX_DIR = os.path.join(PROJECT_ROOT, ".index_cache")

# RAGAS synthesizer_name → difficulty 映射
SYNTHESIZER_DIFFICULTY_MAP = {
    "single_hop_specific": "simple",
    "SingleHopSpecificQuerySynthesizer": "simple",
    "multi_hop_abstract": "inference",
    "MultiHopAbstractQuerySynthesizer": "inference",
    "multi_hop_specific": "cross_document",
    "MultiHopSpecificQuerySynthesizer": "cross_document",
}


def _map_difficulty(synthesizer_name: str) -> str:
    """将 RAGAS synthesizer 名称映射为项目难度标签"""
    for key, diff in SYNTHESIZER_DIFFICULTY_MAP.items():
        if key.lower() in synthesizer_name.lower():
            return diff
    return "simple"


class EvalGenerator:
    """基于 RAGAS 的评估数据集生成器

    从本地 chunk 缓存加载已入库的文本，采样后通过 RAGAS TestsetGenerator
    生成多类型评估题目，输出为项目标准 EvalQuestion JSON 格式。
    """

    ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    def __init__(
        self,
        dataset: str,
        api_key: str,
        model: str = "glm-4-flash",
        embedding_model_name: str = "embedding-3",
        n_questions: int = 50,
        max_chunks: int = 300,
        max_workers: int = 5,
        seed: int = 42,
        output_path: Optional[str] = None,
        domain: Optional[str] = None,
    ):
        from evaluation.domains import load_domain_config
        self.dataset = dataset
        self.api_key = api_key
        self.model = model
        self.embedding_model_name = embedding_model_name
        self.n_questions = n_questions
        self.max_chunks = max_chunks
        self.max_workers = max_workers
        self.seed = seed
        self.output_path = output_path or os.path.join(
            PROJECT_ROOT, "evaluation", f"{dataset}_eval_dataset.json"
        )
        self.domain_config = load_domain_config(domain)

    def _load_chunks(self) -> list:
        """从本地缓存加载 chunks"""
        cache_path = os.path.join(INDEX_DIR, f"{self.dataset}_chunks.pkl")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Chunk cache not found: {cache_path}\n"
                f"Please run: python main.py ingest -d {self.dataset} --incremental"
            )
        with open(cache_path, "rb") as f:
            chunks = pickle.load(f)
        console.print(f"Loaded {len(chunks)} chunks from cache")
        return chunks

    @staticmethod
    def _is_low_quality(text: str) -> bool:
        """判断 chunk 是否为低质量内容（表格/参考文献/过短/噪声）"""
        # 1. 过短
        if len(text) < 100:
            return True
        # 2. HTML 表格（IPCC 报告常见）
        if "<td>" in text or "<tr>" in text or "<table" in text:
            return True
        # 3. 参考文献密集区（3+ 个 "et al." 或 2+ 个 "doi:"）
        if text.count("et al.") > 3 or text.count("doi:") > 2:
            return True
        # 4. 字母比例过低（数字/符号/公式噪声）
        alpha_ratio = sum(1 for ch in text if ch.isalpha()) / max(len(text), 1)
        if alpha_ratio < 0.35:
            return True
        # 5. Markdown 表格密集（超过 5 个 | 分隔符）
        if text.count("|") > 5:
            return True
        return False

    def _sample_chunks(self, chunks: list) -> list:
        """过滤低质量 chunk，连续段落采样 + 跨文档配对，保证局部关联性和全局多样性。

        采样策略：
        1. 按文档分组，保持 chunk 原始顺序（chunk_index）
        2. 每个文档内选取多个"窗口"（连续 chunk 段落），保证局部上下文关联
        3. 跨文档 round-robin 分配配额，保证全局覆盖
        """
        random.seed(self.seed)

        # 通用质量过滤
        valid = [c for c in chunks if not self._is_low_quality(c.content)]
        filtered = len(chunks) - len(valid)
        console.print(
            f"  Base quality filter: {len(valid)} passed, {filtered} removed "
            f"({filtered * 100 // max(len(chunks), 1)}%)"
        )

        # 领域专用过滤
        if self.domain_config.chunk_filter:
            before = len(valid)
            valid = [c for c in valid if not self.domain_config.chunk_filter(c.content)]
            domain_filtered = before - len(valid)
            console.print(
                f"  Domain filter ({self.domain_config.name}): "
                f"{domain_filtered} more removed → {len(valid)} remaining"
            )

        if len(valid) <= self.max_chunks:
            console.print(f"  Using all {len(valid)} valid chunks")
            return valid

        # 按文档分组，保持原始顺序
        doc_groups: dict[str, list] = {}
        for c in valid:
            doc_id = c.metadata.get("filename", c.metadata.get("doc_id", "unknown"))
            doc_groups.setdefault(doc_id, []).append(c)

        # 每组内按 chunk_index 排序（保持文档内顺序）
        for doc_id in doc_groups:
            doc_groups[doc_id].sort(
                key=lambda c: c.metadata.get("chunk_index", 0)
            )

        doc_ids = sorted(doc_groups.keys())
        n_docs = len(doc_ids)
        per_doc = max(1, self.max_chunks // n_docs)
        window_size = 5  # 每个窗口取 5 个连续 chunk

        sampled = []
        for doc_id in doc_ids:
            group = doc_groups[doc_id]

            if len(group) <= per_doc:
                # 文档 chunk 不足配额，全部取
                sampled.extend(group)
                continue

            # 计算需要多少个窗口
            n_windows = max(1, per_doc // window_size)
            # 窗口起始位置均匀分布在文档中
            step = max(1, (len(group) - window_size) // n_windows)
            starts = [i * step for i in range(n_windows)]

            doc_sampled = []
            seen = set()
            for start in starts:
                for j in range(start, min(start + window_size, len(group))):
                    if group[j].chunk_id not in seen:
                        doc_sampled.append(group[j])
                        seen.add(group[j].chunk_id)
                        if len(doc_sampled) >= per_doc:
                            break
                if len(doc_sampled) >= per_doc:
                    break

            sampled.extend(doc_sampled)

        # 如果还没到上限，从剩余中补充（优先取长 chunk）
        if len(sampled) < self.max_chunks:
            used_ids = {c.chunk_id for c in sampled}
            remaining = [c for c in valid if c.chunk_id not in used_ids]
            remaining.sort(key=lambda c: len(c.content), reverse=True)
            sampled.extend(remaining[: self.max_chunks - len(sampled)])

        sampled = sampled[: self.max_chunks]

        # 统计
        doc_counts = {}
        for c in sampled:
            did = c.metadata.get("filename", "?")
            doc_counts[did] = doc_counts.get(did, 0) + 1
        console.print(
            f"  Sampled {len(sampled)} chunks from {len(doc_counts)} documents "
            f"(window_size={window_size}, ~{per_doc}/doc)"
        )
        return sampled

    def _create_ragas_llm(self):
        """创建 RAGAS 兼容的 LLM（通过智谱 OpenAI-compatible endpoint，带日志）"""
        import openai as _openai
        from ragas.llms import llm_factory
        from evaluation.ragas_eval import _wrap_client_with_logging

        client = _openai.OpenAI(
            api_key=self.api_key,
            base_url=self.ZHIPU_BASE_URL,
        )
        client = _wrap_client_with_logging(client, self.model, "ragas_generate_eval")
        llm = llm_factory(
            model=self.model,
            provider="openai",
            client=client,
        )
        console.print(f"  LLM: {self.model} via ZhiPu OpenAI-compatible API (with logging)")
        return llm

    def _create_ragas_embeddings(self):
        """创建 RAGAS 兼容的 Embedding（通过智谱 OpenAI-compatible endpoint）"""
        import openai
        from ragas.embeddings import embedding_factory

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.ZHIPU_BASE_URL,
        )
        embeddings = embedding_factory(
            model=self.embedding_model_name,
            provider="openai",
            client=client,
        )
        console.print(f"  Embeddings: {self.embedding_model_name} via ZhiPu")
        return embeddings

    def _build_personas(self):
        """从 domain_config 构建 RAGAS Persona 对象列表"""
        if not self.domain_config.personas:
            return None  # None = RAGAS 自动生成
        from ragas.testset.persona import Persona
        personas = [
            Persona(name=name, role_description=desc)
            for name, desc in self.domain_config.personas
        ]
        console.print(f"  Personas: {', '.join(p.name for p in personas)}")
        return personas

    def _build_query_distribution(self, llm):
        """构建 RAGAS query distribution，比例从 domain_config 读取"""
        from ragas.testset.synthesizers import (
            SingleHopSpecificQuerySynthesizer,
            MultiHopAbstractQuerySynthesizer,
            MultiHopSpecificQuerySynthesizer,
        )

        s, i, c = self.domain_config.distribution
        distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=llm), s),
            (MultiHopAbstractQuerySynthesizer(llm=llm), i),
            (MultiHopSpecificQuerySynthesizer(llm=llm), c),
        ]
        console.print(
            f"  Distribution: simple({s:.0%}) + inference({i:.0%}) + cross_document({c:.0%})"
        )
        return distribution

    def _convert_to_eval_format(self, testset, chunk_metadata: dict) -> list[dict]:
        """将 RAGAS Testset 转换为项目 EvalQuestion 格式"""
        df = testset.to_pandas()
        questions = []

        for i, row in df.iterrows():
            # 提取 source_doc
            contexts = row.get("reference_contexts", [])
            if not isinstance(contexts, list):
                contexts = [str(contexts)] if contexts else []

            # synthesizer_name → difficulty
            synth_name = row.get("synthesizer_name", "")
            difficulty = _map_difficulty(synth_name)

            question = {
                "id": i + 1,
                "question": str(row.get("user_input", "")),
                "ground_truth": str(row.get("reference", "")),
                "ground_truth_contexts": contexts,
                "source_doc": synth_name,  # RAGAS 不直接提供 source_doc，用 synthesizer 类型标记
                "difficulty": difficulty,
            }

            # 跳过空问题
            if not question["question"] or not question["ground_truth"]:
                continue

            questions.append(question)

        return questions

    def _kg_cache_path(self) -> str:
        """知识图谱缓存路径"""
        return os.path.join(INDEX_DIR, f"{self.dataset}_knowledge_graph.json")

    def generate(self) -> str:
        """执行完整的评估数据集生成流程

        支持知识图谱缓存：首次运行构建并保存 KG（最耗时），
        后续运行直接加载 KG，只执行题目生成（几分钟）。

        Returns:
            输出文件路径
        """
        from ragas.testset import TestsetGenerator
        from ragas.testset.graph import KnowledgeGraph
        from ragas.run_config import RunConfig

        dc = self.domain_config
        console.print(f"\n[bold cyan]Generating evaluation dataset for '{self.dataset}'[/bold cyan]")
        console.print(f"  Target: {self.n_questions} questions")
        console.print(f"  Model: {self.model}")
        console.print(f"  Domain: {dc.name}")
        console.print(f"  Output: {self.output_path}\n")

        # 1. 加载并采样 chunks
        console.print("[cyan]Step 1: Loading and sampling chunks...[/cyan]")
        chunks = self._load_chunks()
        sampled = self._sample_chunks(chunks)

        # 保存 chunk metadata 用于后续映射
        chunk_metadata = {
            c.chunk_id: c.metadata.get("filename", c.metadata.get("doc_id", ""))
            for c in sampled
        }

        # 转为纯文本列表（RAGAS generate_with_chunks 接受 list[str]）
        chunk_texts = [c.content for c in sampled]

        # 2. 创建 RAGAS 组件
        console.print("\n[cyan]Step 2: Initializing RAGAS components...[/cyan]")
        llm = self._create_ragas_llm()
        embeddings = self._create_ragas_embeddings()
        distribution = self._build_query_distribution(llm)

        run_config = RunConfig(
            max_workers=self.max_workers,
            timeout=60,
            max_retries=10,
            max_wait=30,
            seed=self.seed,
        )

        # 3. 知识图谱：优先从缓存加载，否则构建并保存
        kg_path = self._kg_cache_path()
        if os.path.exists(kg_path):
            console.print(f"\n[cyan]Step 3a: Loading cached knowledge graph from {os.path.basename(kg_path)}...[/cyan]")
            kg = KnowledgeGraph.load(kg_path)
            console.print(f"  [green]KG loaded ({len(kg.nodes)} nodes, {len(kg.relationships)} relationships)[/green]")

            generator = TestsetGenerator(
                llm=llm, embedding_model=embeddings, knowledge_graph=kg,
                persona_list=self._build_personas(),
                llm_context=dc.llm_context or None,
            )
        else:
            console.print(
                f"\n[cyan]Step 3a: Building knowledge graph "
                f"(max_workers={self.max_workers}, this is the slow part)...[/cyan]"
            )
            from ragas.testset.transforms import default_transforms_for_prechunked, apply_transforms
            from ragas.testset.graph import Node, NodeType

            # 将 chunks 转为 RAGAS nodes
            nodes = []
            for text in chunk_texts:
                if text and text.strip():
                    nodes.append(Node(
                        type=NodeType.CHUNK,
                        properties={"page_content": text, "document_metadata": {}},
                    ))

            kg = KnowledgeGraph(nodes=nodes)
            transforms = default_transforms_for_prechunked(llm, embeddings)
            apply_transforms(kg, transforms, run_config=run_config)

            # 保存 KG 到缓存
            os.makedirs(INDEX_DIR, exist_ok=True)
            kg.save(kg_path)
            console.print(
                f"  [green]KG built and saved to {os.path.basename(kg_path)} "
                f"({len(kg.nodes)} nodes, {len(kg.relationships)} relationships)[/green]"
            )

            generator = TestsetGenerator(
                llm=llm, embedding_model=embeddings, knowledge_graph=kg,
                persona_list=self._build_personas(),
                llm_context=dc.llm_context or None,
            )

        # 4. 生成题目（多请求 30% 补偿 post_filter 损耗）
        request_size = self.n_questions
        if dc.post_filter:
            request_size = int(self.n_questions * 1.3)
            console.print(
                f"\n[cyan]Step 4: Generating {request_size} questions "
                f"(+30% for post-filter, target {self.n_questions})...[/cyan]"
            )
        else:
            console.print(
                f"\n[cyan]Step 4: Generating {request_size} questions from knowledge graph...[/cyan]"
            )
        testset = generator.generate(
            testset_size=request_size,
            query_distribution=distribution,
            run_config=run_config,
            raise_exceptions=False,
        )

        # 5. 转换格式
        console.print("\n[cyan]Step 5: Converting to evaluation format...[/cyan]")
        questions = self._convert_to_eval_format(testset, chunk_metadata)
        console.print(f"  Raw questions: {len(questions)}")

        # 6. 领域专用 post-filter
        if dc.post_filter:
            before = len(questions)
            questions = [
                q for q in questions
                if not dc.post_filter(q["question"], q["ground_truth"])
            ]
            post_filtered = before - len(questions)
            console.print(
                f"  Post-filter ({dc.name}): {post_filtered} removed → {len(questions)} remaining"
            )
            # 截断到目标数量
            if len(questions) > self.n_questions:
                questions = questions[:self.n_questions]

        # 7. 保存
        output = {
            "description": f"Auto-generated evaluation dataset for {self.dataset} ({len(questions)} questions)",
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "generator_model": self.model,
            "generator": "RAGAS TestsetGenerator",
            "domain": dc.name,
            "questions": questions,
        }

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        # 如果旧文件存在，加时间戳后缀备份，避免覆盖
        if os.path.exists(self.output_path):
            base, ext = os.path.splitext(self.output_path)
            backup_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{base}_{backup_ts}{ext}"
            os.rename(self.output_path, backup_path)
            console.print(f"  [dim]Existing file backed up to {os.path.basename(backup_path)}[/dim]")
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        # 统计难度分布
        from collections import Counter
        dist = Counter(q["difficulty"] for q in questions)
        console.print(f"\n[green]Generated {len(questions)} questions:[/green]")
        for diff, count in sorted(dist.items()):
            console.print(f"  {diff}: {count}")
        console.print(f"\n[green]Saved to {self.output_path}[/green]")

        return self.output_path
