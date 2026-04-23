"""多策略评估对比 & 全量数据保存 & 可视化

支持：
- RAGAS 结果缓存：基于 (judge_model, question, contexts_hash) 的细粒度缓存
- 策略级续传：中断后重跑自动跳过已完成 RAGAS 的策略
- --incremental 控制是否使用缓存
"""

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from config import config
from evaluation.datasets import EvalDataset, EvalQuestion
from evaluation.metrics import (
    compute_all_generation_metrics,
    compute_all_retrieval_metrics,
    match_contexts_by_text,
)
from evaluation.ragas_eval import RAGASEvaluator, RAGASResult
from src.query_engine import RAGResponse
from src.retriever import RetrievalStrategy

console = Console()

# ── RAGAS 结果缓存 ──

RAGAS_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".index_cache", "ragas_cache")


def _ragas_cache_path(dataset: str, strategy: str, mode: str, n_questions: int, judge_model: str) -> str:
    """生成 RAGAS 缓存文件路径，包含数据集/策略/模式/题数/模型信息"""
    os.makedirs(RAGAS_CACHE_DIR, exist_ok=True)
    filename = f"{dataset}_{strategy}_{mode}_n{n_questions}_{judge_model}.json"
    return os.path.join(RAGAS_CACHE_DIR, filename)


def _save_ragas_cache(path: str, results: list):
    """保存 RAGAS 结果到缓存"""
    data = [asdict(r) if r else None for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_ragas_cache(path: str) -> list[RAGASResult | None] | None:
    """从缓存加载 RAGAS 结果，不存在或格式错误返回 None"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            RAGASResult(**item) if item else None
            for item in data
        ]
    except Exception:
        return None


# ── Phase 1 检索结果缓存（保证 HyDE 等 LLM 依赖策略的一致性） ──

RETRIEVAL_CACHE_DIR = os.path.join(config.output_dir, "retrieval_cache")


def _retrieval_cache_path(dataset: str, strategy: str, mode: str, n_questions: int, llm_model: str = "") -> str:
    """检索结果缓存路径。full 模式包含 LLM 模型名（不同模型生成不同 answer）"""
    os.makedirs(RETRIEVAL_CACHE_DIR, exist_ok=True)
    if mode == "full" and llm_model:
        return os.path.join(RETRIEVAL_CACHE_DIR, f"{dataset}_{strategy}_{mode}_n{n_questions}_{llm_model}.json")
    return os.path.join(RETRIEVAL_CACHE_DIR, f"{dataset}_{strategy}_{mode}_n{n_questions}.json")


def _save_retrieval_cache(path: str, responses: list):
    """保存 Phase 1 检索结果"""
    data = []
    for r in responses:
        if r is None:
            data.append(None)
        else:
            data.append({
                "answer": r.answer,
                "contexts": r.contexts,
                "timings": r.timings,
                "strategy": r.strategy,
                "rewritten_query": r.rewritten_query,
            })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_retrieval_cache(path: str) -> list[RAGResponse | None] | None:
    """从缓存加载检索结果"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = []
        for item in data:
            if item is None:
                results.append(None)
            else:
                results.append(RAGResponse(
                    query="",
                    answer=item.get("answer", ""),
                    contexts=item.get("contexts", []),
                    retrieval_results=[],
                    strategy=item.get("strategy", ""),
                    timings=item.get("timings", {}),
                    rewritten_query=item.get("rewritten_query", ""),
                ))
        return results
    except Exception:
        return None


class StrategyComparator:
    """运行多策略评估对比实验，保存完整过程数据"""

    def __init__(self, pipeline, judge_llm=None):
        self.pipeline = pipeline
        self.ragas = RAGASEvaluator(judge_llm or pipeline.llm, pipeline.embedding_model)

    def run_evaluation(
        self,
        strategies: list[str] = None,
        n_questions: int = None,
        use_ragas: bool = True,
        mode: str = "full",  # "full" | "retrieval"
        incremental: bool = True,  # 是否使用 RAGAS 缓存
    ) -> dict:
        """运行评估，返回完整结果（含逐题明细）

        返回格式:
        {
            "meta": {...},
            "summary": {strategy: {metric: value}},
            "details": [
                {
                    "question_id": 1,
                    "question": "...",
                    "ground_truth": "...",
                    "difficulty": "simple",
                    "source_doc": "...",
                    "strategies": {
                        "naive_dense": {
                            "answer": "...",
                            "contexts": [...],
                            "timings": {...},
                            "ragas": {faithfulness: ..., ...},
                            "rouge_l": ...,
                            "bleu": ...,
                        },
                        ...
                    }
                },
                ...
            ]
        }
        """
        if strategies is None:
            strategies = [s.value for s in RetrievalStrategy]

        dataset = EvalDataset()
        questions = dataset.get_subset(n_questions)
        t_total_start = time.time()

        is_retrieval = mode == "retrieval"
        if is_retrieval:
            use_ragas = False  # retrieval 模式不需要 RAGAS
        mode_label = "retrieval-only" if is_retrieval else "full"
        ragas_label = " + RAGAS" if use_ragas else ""
        console.print(
            f"[bold cyan]Evaluation ({mode_label}): {len(questions)} questions × "
            f"{len(strategies)} strategies{ragas_label}[/bold cyan]\n"
        )

        # 中间保存目录
        checkpoint_dir = os.path.join(
            config.output_dir, "history",
            f"eval_{self.pipeline.dataset}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        def _save_checkpoint(label: str):
            """保存当前进度到中间文件，崩溃后不丢失"""
            ckpt = {
                "checkpoint": label,
                "strategy_responses": {
                    s: [
                        {"answer": r.answer, "contexts": r.contexts, "timings": r.timings}
                        if r else None
                        for r in resps
                    ]
                    for s, resps in strategy_responses.items()
                },
                "ragas_per_question": {
                    s: [asdict(r) if r else None for r in rlist]
                    for s, rlist in ragas_per_question.items()
                },
            }
            ckpt_path = os.path.join(checkpoint_dir, "checkpoint.json")
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(ckpt, f, ensure_ascii=False, indent=2)
            console.print(f"  [dim]Checkpoint saved: {label}[/dim]")

        # ── 阶段 1: 对每个策略跑全部问题（支持缓存，保证 HyDE 一致性） ──
        # strategy_responses[strategy][i] = RAGResponse
        strategy_responses: dict[str, list[RAGResponse | None]] = {}
        ragas_per_question: dict[str, list[RAGASResult | None]] = {}
        dataset_name = self.pipeline.dataset

        llm_model = self.pipeline.llm.model if hasattr(self.pipeline.llm, 'model') else ""

        for si, strategy in enumerate(strategies, 1):
            # 检查 Phase 1 检索结果缓存
            ret_cache_path = _retrieval_cache_path(
                dataset_name, strategy, mode, len(questions), llm_model
            )
            if incremental:
                cached_resps = _load_retrieval_cache(ret_cache_path)
                if cached_resps and len(cached_resps) == len(questions):
                    strategy_responses[strategy] = cached_resps
                    console.print(
                        f"[green]━━ Strategy {si}/{len(strategies)}: {strategy} "
                        f"━━ cached ({os.path.basename(ret_cache_path)})[/green]\n"
                    )
                    continue

            console.print(
                f"[yellow]━━ Strategy {si}/{len(strategies)}: {strategy} ━━[/yellow]"
            )
            t_strat = time.time()
            responses = []
            for qi, q in enumerate(questions, 1):
                t0 = time.time()
                try:
                    if is_retrieval:
                        resp = self.pipeline.retrieve(q.question, strategy=strategy)
                    else:
                        resp = self.pipeline.query(q.question, strategy=strategy)
                    responses.append(resp)
                    elapsed = time.time() - t0
                    console.print(
                        f"  [{qi}/{len(questions)}] {q.question[:40]}... "
                        f"({elapsed:.1f}s, {len(resp.contexts)} ctx)"
                    )
                except Exception as e:
                    responses.append(None)
                    console.print(f"  [red][{qi}/{len(questions)}] Error: {e}[/red]")
            strategy_responses[strategy] = responses

            # 保存检索结果缓存
            _save_retrieval_cache(ret_cache_path, responses)
            console.print(
                f"  [green]Done in {time.time() - t_strat:.1f}s[/green]"
            )
            console.print(f"  [dim]Retrieval cache saved: {os.path.basename(ret_cache_path)}[/dim]\n")
            _save_checkpoint(f"retrieval_done_{strategy}")

        # ── 阶段 2: RAGAS 评估（支持缓存和策略级续传） ──

        judge_model = self.ragas.llm.model if hasattr(self.ragas, 'llm') and self.ragas.llm else "unknown"
        ragas_mode = "retrieval" if is_retrieval else "full"

        if is_retrieval:
            for si, strategy in enumerate(strategies, 1):
                # 策略级缓存检查
                cache_path = _ragas_cache_path(
                    dataset_name, strategy, ragas_mode, len(questions), judge_model
                )
                if incremental:
                    cached = _load_ragas_cache(cache_path)
                    if cached and len(cached) == len(questions):
                        ragas_per_question[strategy] = cached
                        console.print(
                            f"[green]━━ RAGAS-retrieval {si}/{len(strategies)}: {strategy} "
                            f"━━ cached ({os.path.basename(cache_path)})[/green]\n"
                        )
                        continue

                console.print(
                    f"[cyan]━━ RAGAS-retrieval {si}/{len(strategies)}: {strategy} ━━[/cyan]"
                )
                resps = strategy_responses[strategy]
                qs, ctxs, gts = [], [], []
                valid_indices = []
                for i, (resp, q) in enumerate(zip(resps, questions)):
                    if resp is not None:
                        qs.append(q.question)
                        ctxs.append(resp.contexts)
                        gts.append(q.ground_truth)
                        valid_indices.append(i)

                if qs:
                    _, ragas_results = self.ragas.evaluate_retrieval_batch(qs, ctxs, gts)
                    full_ragas = [None] * len(questions)
                    for idx, rr in zip(valid_indices, ragas_results):
                        full_ragas[idx] = rr
                    ragas_per_question[strategy] = full_ragas
                else:
                    ragas_per_question[strategy] = [None] * len(questions)

                # 保存缓存
                _save_ragas_cache(cache_path, ragas_per_question[strategy])
                console.print(f"  [dim]RAGAS cache saved: {os.path.basename(cache_path)}[/dim]\n")
                _save_checkpoint(f"ragas_retrieval_done_{strategy}")

        elif use_ragas:
            for si, strategy in enumerate(strategies, 1):
                cache_path = _ragas_cache_path(
                    dataset_name, strategy, ragas_mode, len(questions), judge_model
                )
                if incremental:
                    cached = _load_ragas_cache(cache_path)
                    if cached and len(cached) == len(questions):
                        ragas_per_question[strategy] = cached
                        console.print(
                            f"[green]━━ RAGAS {si}/{len(strategies)}: {strategy} "
                            f"━━ cached ({os.path.basename(cache_path)})[/green]\n"
                        )
                        continue

                console.print(
                    f"[cyan]━━ RAGAS {si}/{len(strategies)}: {strategy} ━━[/cyan]"
                )
                resps = strategy_responses[strategy]
                qs, ans, ctxs, gts = [], [], [], []
                valid_indices = []
                for i, (resp, q) in enumerate(zip(resps, questions)):
                    if resp is not None:
                        qs.append(q.question)
                        ans.append(resp.answer)
                        ctxs.append(resp.contexts)
                        gts.append(q.ground_truth)
                        valid_indices.append(i)

                if qs:
                    _, ragas_results = self.ragas.evaluate_batch(qs, ans, ctxs, gts)
                    full_ragas = [None] * len(questions)
                    for idx, rr in zip(valid_indices, ragas_results):
                        full_ragas[idx] = rr
                    ragas_per_question[strategy] = full_ragas
                else:
                    ragas_per_question[strategy] = [None] * len(questions)

                _save_ragas_cache(cache_path, ragas_per_question[strategy])
                console.print(f"  [dim]RAGAS cache saved: {os.path.basename(cache_path)}[/dim]\n")
                _save_checkpoint(f"ragas_done_{strategy}")

        # ── 阶段 3: 组装完整结果 ──
        details = []
        for qi, q in enumerate(questions):
            question_detail = {
                "question_id": q.id,
                "question": q.question,
                "ground_truth": q.ground_truth,
                "ground_truth_contexts": q.ground_truth_contexts,
                "difficulty": q.difficulty,
                "source_doc": q.source_doc,
                "strategies": {},
            }
            for strategy in strategies:
                resp = strategy_responses[strategy][qi]
                if resp is None:
                    question_detail["strategies"][strategy] = {"error": "query failed"}
                    continue

                entry = {
                    "answer": resp.answer,
                    "contexts": resp.contexts,
                    "rewritten_query": resp.rewritten_query or None,
                    "timings": resp.timings,
                }

                if is_retrieval:
                    # 纯检索模式：计算检索指标
                    ret_ids, rel_ids = match_contexts_by_text(
                        resp.contexts, q.ground_truth_contexts
                    )
                    ret_metrics = compute_all_retrieval_metrics(
                        [ret_ids], [rel_ids], k=config.top_k
                    )
                    entry.update(ret_metrics)
                    # RAGAS 检索指标
                    if strategy in ragas_per_question:
                        rr = ragas_per_question[strategy][qi]
                        if rr:
                            entry["ragas_context_precision"] = rr.context_precision
                            entry["ragas_context_recall"] = rr.context_recall
                else:
                    # 完整模式：计算生成指标
                    gen = compute_all_generation_metrics(
                        [resp.answer], [q.ground_truth]
                    )
                    entry.update(gen)

                    # RAGAS 逐题
                    if use_ragas and strategy in ragas_per_question:
                        rr = ragas_per_question[strategy][qi]
                        if rr:
                            entry["ragas"] = asdict(rr)

                question_detail["strategies"][strategy] = entry
            details.append(question_detail)

        # ── 阶段 4: 汇总 summary ──
        summary = {}
        for strategy in strategies:
            resps = [
                strategy_responses[strategy][i]
                for i in range(len(questions))
                if strategy_responses[strategy][i] is not None
            ]
            if not resps:
                summary[strategy] = {"error": "all failed"}
                continue

            valid_qs = [
                questions[i]
                for i in range(len(questions))
                if strategy_responses[strategy][i] is not None
            ]

            latencies = {
                "avg_total_time": float(np.mean([r.timings.get("total", 0) for r in resps])),
                "avg_retrieval_time": float(np.mean([r.timings.get("retrieval", 0) for r in resps])),
            }

            if is_retrieval:
                # 纯检索模式：聚合检索指标
                all_ret_ids, all_rel_ids = [], []
                for resp, q in zip(resps, valid_qs):
                    ret_ids, rel_ids = match_contexts_by_text(
                        resp.contexts, q.ground_truth_contexts
                    )
                    all_ret_ids.append(ret_ids)
                    all_rel_ids.append(rel_ids)
                ret_metrics = compute_all_retrieval_metrics(
                    all_ret_ids, all_rel_ids, k=config.top_k
                )
                strat_summary = {**ret_metrics, **latencies, "n_success": len(resps)}
                # RAGAS 检索指标
                if strategy in ragas_per_question:
                    valid_ragas = [r for r in ragas_per_question[strategy] if r is not None]
                    if valid_ragas:
                        n = len(valid_ragas)
                        strat_summary["ragas_context_precision"] = sum(
                            r.context_precision for r in valid_ragas) / n
                        strat_summary["ragas_context_recall"] = sum(
                            r.context_recall for r in valid_ragas) / n
            else:
                # 完整模式
                predictions = [r.answer for r in resps]
                references = [q.ground_truth for q in valid_qs]
                gen_metrics = compute_all_generation_metrics(predictions, references)
                latencies["avg_generation_time"] = float(
                    np.mean([r.timings.get("generation", 0) for r in resps])
                )
                strat_summary = {**gen_metrics, **latencies, "n_success": len(resps)}

            if use_ragas and strategy in ragas_per_question:
                valid_ragas = [
                    r for r in ragas_per_question[strategy] if r is not None
                ]
                if valid_ragas:
                    n = len(valid_ragas)
                    strat_summary.update({
                        f"ragas_{k}": sum(getattr(r, k) for r in valid_ragas) / n
                        for k in [
                            "faithfulness", "answer_relevancy",
                            "context_precision", "context_recall",
                            "answer_correctness", "hallucination_rate",
                        ]
                    })

            summary[strategy] = strat_summary

        total_time = time.time() - t_total_start
        console.print(
            f"\n[bold green]Evaluation complete: {total_time:.0f}s total[/bold green]"
        )

        return {
            "meta": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "dataset": self.pipeline.dataset,
                "n_questions": len(questions),
                "strategies": strategies,
                "mode": mode,
                "use_ragas": use_ragas,
                "total_time_s": round(total_time, 1),
                "output_dir": checkpoint_dir,
            },
            "summary": summary,
            "details": details,
        }

    # ── 输出方法 ──

    def print_comparison_table(self, results: dict):
        """打印 summary 对比表"""
        summary = results.get("summary", results)  # 兼容旧格式

        all_metrics = set()
        for v in summary.values():
            all_metrics.update(v.keys())
        all_metrics -= {"error", "n_success"}
        all_metrics = sorted(all_metrics)

        table = Table(title="RAG Strategy Comparison", show_lines=True)
        table.add_column("Metric", style="cyan", min_width=25)
        for strategy in summary:
            table.add_column(strategy, justify="center", min_width=12)

        for metric in all_metrics:
            values = []
            for strategy in summary:
                val = summary[strategy].get(metric, None)
                if isinstance(val, (int, float)):
                    values.append((strategy, val))

            # 越小越好的指标
            LOWER_IS_BETTER = {"time", "hallucination", "latency", "error"}
            best_val = None
            if values:
                if any(kw in metric for kw in LOWER_IS_BETTER):
                    best_val = min(v for _, v in values)
                else:
                    best_val = max(v for _, v in values)

            row = [metric]
            for strategy in summary:
                val = summary[strategy].get(metric, "N/A")
                if isinstance(val, float):
                    fmt = f"{val:.4f}"
                    if best_val is not None and abs(val - best_val) < 1e-6:
                        fmt = f"[bold green]{fmt}[/bold green]"
                    row.append(fmt)
                elif isinstance(val, int):
                    row.append(str(val))
                else:
                    row.append(str(val))
            table.add_row(*row)

        console.print(table)

    def save_comparison_table(self, results: dict, path: str):
        """保存对齐的纯文本对比表格到文件"""
        summary = results.get("summary", results)

        all_metrics = set()
        for v in summary.values():
            all_metrics.update(v.keys())
        all_metrics -= {"error", "n_success"}
        all_metrics = sorted(all_metrics)

        strategies = list(summary.keys())

        # 计算列宽
        metric_width = max(len("Metric"), max(len(m) for m in all_metrics)) + 2
        col_width = max(14, max(len(s) for s in strategies) + 4)

        # 越小越好的指标
        LOWER_IS_BETTER = {"time", "hallucination", "latency", "error"}

        lines = []
        # 标题
        header = f"{'Metric':<{metric_width}}" + "".join(f"{s:>{col_width}}" for s in strategies)
        sep = "─" * len(header)
        lines.append(sep)
        lines.append(header)
        lines.append(sep)

        for metric in all_metrics:
            values = []
            for s in strategies:
                val = summary[s].get(metric, None)
                if isinstance(val, (int, float)):
                    values.append((s, val))

            best_val = None
            if values:
                if any(kw in metric for kw in LOWER_IS_BETTER):
                    best_val = min(v for _, v in values)
                else:
                    best_val = max(v for _, v in values)

            row = f"{metric:<{metric_width}}"
            for s in strategies:
                val = summary[s].get(metric, "N/A")
                if isinstance(val, float):
                    fmt = f"{val:.4f}"
                    if best_val is not None and abs(val - best_val) < 1e-6:
                        fmt = f"{fmt} *"  # 标记最优
                    row += f"{fmt:>{col_width}}"
                elif isinstance(val, int):
                    row += f"{val:>{col_width}}"
                else:
                    row += f"{str(val):>{col_width}}"
            lines.append(row)

        lines.append(sep)
        lines.append("")
        lines.append("* = 该指标最优值")

        # 附加元信息
        meta = results.get("meta", {})
        if meta:
            lines.append("")
            lines.append(f"数据集: {meta.get('dataset', 'N/A')}")
            lines.append(f"模式: {meta.get('mode', 'N/A')}")
            lines.append(f"题目数: {meta.get('n_questions', 'N/A')}")
            lines.append(f"总耗时: {meta.get('total_time_s', 0):.0f}s")
            lines.append(f"时间戳: {meta.get('timestamp', 'N/A')}")

        text = "\n".join(lines)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        console.print(f"[green]Comparison table saved to {path}[/green]")

    def save_results(self, results: dict, path: str = None):
        """保存完整评估结果（含逐题明细）到 JSON"""
        path = path or os.path.join(config.output_dir, "evaluation_results.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        console.print(f"[green]Full results saved to {path}[/green]")

    def save_details_csv(self, results: dict, path: str = None):
        """将逐题明细导出为 CSV，方便 pandas 分析"""
        path = path or os.path.join(config.output_dir, "evaluation_details.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        rows = []
        for item in results.get("details", []):
            for strategy, data in item.get("strategies", {}).items():
                if "error" in data:
                    continue
                row = {
                    "question_id": item["question_id"],
                    "question": item["question"],
                    "ground_truth": item["ground_truth"][:200],
                    "difficulty": item["difficulty"],
                    "source_doc": str(item["source_doc"]),
                    "strategy": strategy,
                    "answer": data.get("answer", "")[:200],
                    "n_contexts": len(data.get("contexts", [])),
                    "rouge_l": data.get("rouge_l"),
                    "bleu": data.get("bleu"),
                    "total_time": data.get("timings", {}).get("total"),
                    "retrieval_time": data.get("timings", {}).get("retrieval"),
                    "generation_time": data.get("timings", {}).get("generation"),
                }
                ragas = data.get("ragas", {})
                for k in ["faithfulness", "answer_relevancy", "context_precision",
                          "context_recall", "answer_correctness", "hallucination_rate"]:
                    row[f"ragas_{k}"] = ragas.get(k)
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        console.print(f"[green]Detail CSV saved to {path} ({len(rows)} rows)[/green]")
        return df

    def plot_comparison(self, results: dict, output_dir: str = None):
        """生成可视化图表，自动适配 full / retrieval 模式"""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False

        output_dir = output_dir or config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        summary = results.get("summary", results)
        details = results.get("details", [])
        meta = results.get("meta", {})
        mode = meta.get("mode", "full")
        strategies = [s for s in summary if "error" not in summary[s]]

        if not strategies:
            console.print("[yellow]No valid strategies to plot[/yellow]")
            return

        colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))
        strategy_colors = dict(zip(strategies, colors))
        chart_count = 0

        if mode == "retrieval":
            # ═══ Retrieval 模式图表 ═══

            # 图 1: 检索指标柱状图
            ret_keys = ["hit_rate", "mrr", f"precision@{config.top_k}",
                        f"recall@{config.top_k}", f"ndcg@{config.top_k}"]
            # 加上 RAGAS 检索指标（如果存在）
            if "ragas_context_precision" in summary[strategies[0]]:
                ret_keys += ["ragas_context_precision", "ragas_context_recall"]
            ret_labels = [k.replace("ragas_", "RAGAS\n") for k in ret_keys]

            fig, ax = plt.subplots(figsize=(max(12, len(ret_keys) * 2), 6))
            x = np.arange(len(ret_keys))
            width = 0.8 / len(strategies)
            for i, s in enumerate(strategies):
                vals = [summary[s].get(k, 0) for k in ret_keys]
                ax.bar(x + i * width, vals, width, label=s, color=strategy_colors[s])
            ax.set_xticks(x + width * (len(strategies) - 1) / 2)
            ax.set_xticklabels(ret_labels, fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
            ax.set_title("Retrieval Quality Metrics", fontsize=14, fontweight="bold")
            ax.legend(fontsize=8, ncol=min(4, len(strategies)))
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "01_retrieval_metrics.png"), dpi=150, bbox_inches="tight")
            plt.close()
            chart_count += 1

            # 图 2: 检索延迟柱状图
            fig, ax = plt.subplots(figsize=(max(8, len(strategies) * 1.5), 5))
            x = np.arange(len(strategies))
            times = [summary[s].get("avg_retrieval_time", 0) for s in strategies]
            bars = ax.bar(x, times, 0.5, color=[strategy_colors[s] for s in strategies])
            ax.set_xticks(x)
            ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("Time (s)")
            ax.set_title("Average Retrieval Latency", fontsize=14, fontweight="bold")
            for bar, t in zip(bars, times):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{t:.3f}s", ha="center", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "02_retrieval_latency.png"), dpi=150, bbox_inches="tight")
            plt.close()
            chart_count += 1

            # 图 3: Context Precision/Recall 热力图（逐题 × 策略）
            if details and "ragas_context_precision" in (details[0].get("strategies", {}).get(strategies[0], {})):
                heatmap_data = []
                q_labels = []
                for item in details:
                    q_labels.append(f"Q{item['question_id']}")
                    row = []
                    for s in strategies:
                        sdata = item["strategies"].get(s, {})
                        # 取 context_precision 和 context_recall 的均值作为综合检索分
                        cp = sdata.get("ragas_context_precision", 0)
                        cr = sdata.get("ragas_context_recall", 0)
                        row.append((cp + cr) / 2)
                    heatmap_data.append(row)

                fig, ax = plt.subplots(figsize=(max(8, len(strategies) * 1.5), max(6, len(q_labels) * 0.4)))
                im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
                ax.set_xticks(range(len(strategies)))
                ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=9)
                ax.set_yticks(range(len(q_labels)))
                ax.set_yticklabels(q_labels, fontsize=8)
                ax.set_title("Context Quality Heatmap (Precision+Recall avg)", fontsize=13, fontweight="bold")
                plt.colorbar(im, ax=ax, shrink=0.8)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "03_context_heatmap.png"), dpi=150, bbox_inches="tight")
                plt.close()
                chart_count += 1

            # 图 4: 难度分组柱状图
            if details:
                diff_data = {}
                for item in details:
                    d = item.get("difficulty", "unknown")
                    if d not in diff_data:
                        diff_data[d] = {s: [] for s in strategies}
                    for s in strategies:
                        sdata = item["strategies"].get(s, {})
                        # 检索模式用 hit_rate 作为分组得分
                        score = sdata.get("hit_rate", 0)
                        diff_data[d][s].append(score)

                difficulties = sorted(diff_data.keys())
                fig, ax = plt.subplots(figsize=(max(8, len(difficulties) * 3), 5))
                x = np.arange(len(difficulties))
                width = 0.8 / len(strategies)
                for i, s in enumerate(strategies):
                    means = [np.mean(diff_data[d][s]) if diff_data[d][s] else 0 for d in difficulties]
                    ax.bar(x + i * width, means, width, label=s, color=strategy_colors[s])
                ax.set_xticks(x + width * (len(strategies) - 1) / 2)
                ax.set_xticklabels(difficulties, fontsize=11)
                ax.set_ylabel("Hit Rate")
                ax.set_title("Retrieval Hit Rate by Difficulty", fontsize=14, fontweight="bold")
                ax.legend(fontsize=8, ncol=min(4, len(strategies)))
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "04_difficulty_breakdown.png"), dpi=150, bbox_inches="tight")
                plt.close()
                chart_count += 1

            # 图 5: 策略排名（综合检索得分）
            composite_keys = ["hit_rate", "mrr", f"ndcg@{config.top_k}"]
            if "ragas_context_precision" in summary[strategies[0]]:
                composite_keys += ["ragas_context_precision", "ragas_context_recall"]
            composite = {}
            for s in strategies:
                scores = [summary[s].get(k, 0) for k in composite_keys]
                composite[s] = np.mean(scores)
            sorted_strats = sorted(composite, key=composite.get, reverse=True)

            fig, ax = plt.subplots(figsize=(max(8, len(strategies) * 1.2), 5))
            bars = ax.barh(
                range(len(sorted_strats)),
                [composite[s] for s in sorted_strats],
                color=[strategy_colors[s] for s in sorted_strats],
            )
            ax.set_yticks(range(len(sorted_strats)))
            ax.set_yticklabels(sorted_strats, fontsize=10)
            ax.set_xlabel("Composite Retrieval Score")
            ax.set_title("Strategy Ranking (Retrieval)", fontsize=14, fontweight="bold")
            ax.set_xlim(0, 1)
            for bar, s in zip(bars, sorted_strats):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{composite[s]:.3f}", va="center", fontsize=10)
            ax.grid(axis="x", alpha=0.3)
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "05_strategy_ranking.png"), dpi=150, bbox_inches="tight")
            plt.close()
            chart_count += 1

        else:
            # ═══ Full 模式图表（原 6 张图不变） ═══

            # 图 1: RAGAS 雷达图
            ragas_keys = ["ragas_faithfulness", "ragas_answer_relevancy",
                          "ragas_context_precision", "ragas_context_recall",
                          "ragas_answer_correctness"]
            ragas_labels = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision",
                            "Context\nRecall", "Answer\nCorrectness"]
            has_ragas = any(k in summary[strategies[0]] for k in ragas_keys)

            if has_ragas:
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                angles = np.linspace(0, 2 * np.pi, len(ragas_keys), endpoint=False).tolist()
                angles += angles[:1]
                for s in strategies:
                    vals = [summary[s].get(k, 0) for k in ragas_keys]
                    vals += vals[:1]
                    ax.plot(angles, vals, "-o", label=s, linewidth=2, markersize=5,
                            color=strategy_colors[s])
                    ax.fill(angles, vals, alpha=0.08, color=strategy_colors[s])
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(ragas_labels, fontsize=10)
                ax.set_ylim(0, 1)
                ax.set_title("RAGAS Metrics Comparison", fontsize=14, fontweight="bold", pad=20)
                ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "01_ragas_radar.png"), dpi=150, bbox_inches="tight")
                plt.close()
                chart_count += 1

            # 图 2: 综合指标柱状图
            metric_keys = ["rouge_l", "bleu"]
            if has_ragas:
                metric_keys += ["ragas_faithfulness", "ragas_answer_correctness"]
            metric_labels = [k.replace("ragas_", "") for k in metric_keys]

            fig, ax = plt.subplots(figsize=(max(10, len(strategies) * 2), 6))
            x = np.arange(len(metric_keys))
            width = 0.8 / len(strategies)
            for i, s in enumerate(strategies):
                vals = [summary[s].get(k, 0) for k in metric_keys]
                ax.bar(x + i * width, vals, width, label=s, color=strategy_colors[s])
            ax.set_xticks(x + width * (len(strategies) - 1) / 2)
            ax.set_xticklabels(metric_labels, fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score")
            ax.set_title("Quality Metrics by Strategy", fontsize=14, fontweight="bold")
            ax.legend(fontsize=8, ncol=min(4, len(strategies)))
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "02_quality_metrics.png"), dpi=150, bbox_inches="tight")
            plt.close()
            chart_count += 1

            # 图 3: 延迟堆叠柱状图
            fig, ax = plt.subplots(figsize=(max(8, len(strategies) * 1.5), 5))
            retrieval_times = [summary[s].get("avg_retrieval_time", 0) for s in strategies]
            gen_times = [summary[s].get("avg_generation_time", 0) for s in strategies]
            x = np.arange(len(strategies))
            ax.bar(x, retrieval_times, 0.5, label="Retrieval", color="#4C78A8")
            ax.bar(x, gen_times, 0.5, bottom=retrieval_times, label="Generation", color="#F58518")
            ax.set_xticks(x)
            ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("Time (s)")
            ax.set_title("Average Latency Breakdown", fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "03_latency.png"), dpi=150, bbox_inches="tight")
            plt.close()
            chart_count += 1

            # 图 4: 逐题热力图（Faithfulness）
            if has_ragas and details:
                heatmap_data = []
                q_labels = []
                for item in details:
                    q_labels.append(f"Q{item['question_id']}")
                    row = []
                    for s in strategies:
                        sdata = item["strategies"].get(s, {})
                        ragas = sdata.get("ragas", {})
                        row.append(ragas.get("faithfulness", np.nan))
                    heatmap_data.append(row)

                fig, ax = plt.subplots(figsize=(max(8, len(strategies) * 1.5), max(6, len(q_labels) * 0.4)))
                im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
                ax.set_xticks(range(len(strategies)))
                ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=9)
                ax.set_yticks(range(len(q_labels)))
                ax.set_yticklabels(q_labels, fontsize=8)
                ax.set_title("Faithfulness Heatmap (per question × strategy)", fontsize=13, fontweight="bold")
                plt.colorbar(im, ax=ax, shrink=0.8)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "04_faithfulness_heatmap.png"), dpi=150, bbox_inches="tight")
                plt.close()
                chart_count += 1

            # 图 5: 难度分组柱状图
            if details:
                diff_data = {}
                for item in details:
                    d = item.get("difficulty", "unknown")
                    if d not in diff_data:
                        diff_data[d] = {s: [] for s in strategies}
                    for s in strategies:
                        sdata = item["strategies"].get(s, {})
                        ragas = sdata.get("ragas", {})
                        score = ragas.get("answer_correctness") or sdata.get("rouge_l", 0)
                        diff_data[d][s].append(score)

                difficulties = sorted(diff_data.keys())
                fig, ax = plt.subplots(figsize=(max(8, len(difficulties) * 3), 5))
                x = np.arange(len(difficulties))
                width = 0.8 / len(strategies)
                for i, s in enumerate(strategies):
                    means = [np.mean(diff_data[d][s]) if diff_data[d][s] else 0 for d in difficulties]
                    ax.bar(x + i * width, means, width, label=s, color=strategy_colors[s])
                ax.set_xticks(x + width * (len(strategies) - 1) / 2)
                ax.set_xticklabels(difficulties, fontsize=11)
                ax.set_ylabel("Score (correctness / rouge_l)")
                ax.set_title("Performance by Question Difficulty", fontsize=14, fontweight="bold")
                ax.legend(fontsize=8, ncol=min(4, len(strategies)))
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "05_difficulty_breakdown.png"), dpi=150, bbox_inches="tight")
                plt.close()
                chart_count += 1

            # 图 6: 策略总分排名
            if has_ragas:
                composite_keys = ["ragas_faithfulness", "ragas_answer_relevancy",
                                  "ragas_context_precision", "ragas_answer_correctness", "rouge_l"]
            else:
                composite_keys = ["rouge_l", "bleu"]

            composite = {}
            for s in strategies:
                scores = [summary[s].get(k, 0) for k in composite_keys]
                composite[s] = np.mean(scores)
            sorted_strats = sorted(composite, key=composite.get, reverse=True)

            fig, ax = plt.subplots(figsize=(max(8, len(strategies) * 1.2), 5))
            bars = ax.barh(
                range(len(sorted_strats)),
                [composite[s] for s in sorted_strats],
                color=[strategy_colors[s] for s in sorted_strats],
            )
            ax.set_yticks(range(len(sorted_strats)))
            ax.set_yticklabels(sorted_strats, fontsize=10)
            ax.set_xlabel("Composite Score")
            ax.set_title("Strategy Ranking (Composite Score)", fontsize=14, fontweight="bold")
            ax.set_xlim(0, 1)
            for bar, s in zip(bars, sorted_strats):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{composite[s]:.3f}", va="center", fontsize=10)
            ax.grid(axis="x", alpha=0.3)
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "06_strategy_ranking.png"), dpi=150, bbox_inches="tight")
            plt.close()
            chart_count += 1

        console.print(f"[green]{chart_count} charts saved to {output_dir}/[/green]")
