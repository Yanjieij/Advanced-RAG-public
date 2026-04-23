"""Advanced RAG - 主入口 CLI

支持的命令：
  python main.py ingest              # 文档入库
  python main.py query "问题"        # 单次查询
  python main.py evaluate            # 运行评估
  python main.py interactive         # 交互式查询
  python main.py dataset list        # 查看已有数据集
  python main.py dataset delete <名称>  # 删除数据集
"""

import argparse
import json
import sys
import os
import glob
from datetime import datetime

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 历史记录目录
HISTORY_DIR = os.path.join(PROJECT_ROOT, "output", "history")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import config, PROJECT_ROOT as PROJ_ROOT
from src.rag_pipeline import RAGPipeline
from src.retriever import RetrievalStrategy
from src.llm import set_log_context
from evaluation.compare import StrategyComparator

console = Console()

ALL_STRATEGIES = [s.value for s in RetrievalStrategy]

# 预定义数据集配置：名称 → (文档目录, 评估数据集)
DATASET_PRESETS = {
    "samples": {
        "doc_dir": os.path.join(PROJ_ROOT, "documents/samples"),
        "eval_dataset": os.path.join(PROJ_ROOT, "evaluation/samples_eval_dataset.json"),
        "description": "内置中英文技术文档 (9 篇, 50 题)",
    },
    "multihop": {
        "doc_dir": os.path.join(PROJ_ROOT, "documents/multihop_rag"),
        "eval_dataset": os.path.join(PROJ_ROOT, "evaluation/multihop_eval_dataset.json"),
        "description": "MultiHop-RAG 精选子集 (30 篇英文新闻, 30 题多跳推理)",
    },
}


def get_eval_dataset_path(dataset: str) -> str:
    """根据 dataset 名称获取评估数据集路径（自动匹配）

    查找顺序：
    1. evaluation/{dataset}_eval_dataset.json（约定命名，generate-eval 的输出位置）
    2. 预设数据集配置（samples → samples_eval_dataset.json, multihop → multihop_eval_dataset.json）
    3. 默认 evaluation/samples_eval_dataset.json
    """
    # 优先检查约定命名（generate-eval 生成的文件在此）
    convention_path = os.path.join(PROJ_ROOT, "evaluation", f"{dataset}_eval_dataset.json")
    if os.path.exists(convention_path):
        return convention_path
    if dataset in DATASET_PRESETS:
        return DATASET_PRESETS[dataset]["eval_dataset"]
    return config.eval_dataset_path


def get_doc_dir(dataset: str) -> str:
    """根据 dataset 名称获取默认文档目录"""
    if dataset in DATASET_PRESETS:
        return DATASET_PRESETS[dataset]["doc_dir"]
    return config.documents_dir


def save_query_history(response, strategy: str, dataset: str, extras: dict = None):
    """将单次查询结果追加到历史记录"""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "query": response.query,
        "strategy": strategy,
        "answer": response.answer,
        "contexts": response.contexts,
        "timings": response.timings,
        "rewritten_query": response.rewritten_query or None,
    }
    if extras:
        record.update(extras)

    history_file = os.path.join(HISTORY_DIR, "query_history.jsonl")
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def cmd_ingest(args):
    """文档入库"""
    dataset = args.dataset
    doc_dir = args.doc_dir or get_doc_dir(dataset)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_log_context("ingest", f"{dataset}_{ts}")

    console.print(Panel(
        f"数据集: [bold cyan]{dataset}[/bold cyan]\n文档目录: {doc_dir}",
        title="入库",
    ))

    pipeline = RAGPipeline(dataset=dataset)
    ingest_fn = pipeline.ingest_incremental if args.incremental else pipeline.ingest
    ingest_fn(
        doc_dir=doc_dir,
        chunking_strategy=args.chunking,
        reset=args.reset,
        skip_summary=args.no_summary,
    )


def cmd_query(args):
    """单次查询"""
    dataset = args.dataset
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_log_context("query", f"{dataset}_{ts}")
    pipeline = RAGPipeline(dataset=dataset)

    console.print(Panel(
        f"[bold]查询:[/bold] {args.question}\n"
        f"数据集: [cyan]{dataset}[/cyan]  策略: [cyan]{args.strategy}[/cyan]  "
        f"重排序: {'开' if args.rerank else '关'}  "
        f"查询改写: {'开' if args.rewrite else '关'}",
        title="Advanced RAG",
    ))

    response = pipeline.query(
        query=args.question,
        strategy=args.strategy,
        use_rerank=args.rerank,
        use_query_rewrite=args.rewrite,
        use_compression=args.compress,
    )

    if response.rewritten_query:
        console.print(f"[dim]改写后查询: {response.rewritten_query}[/dim]\n")

    console.print(Panel(response.answer, title="回答", border_style="green"))

    console.print(f"\n[dim]── 检索到 {len(response.contexts)} 个上下文 ──[/dim]")
    for i, ctx in enumerate(response.contexts, 1):
        preview = ctx[:120] + "..." if len(ctx) > 120 else ctx
        console.print(f"  {i}. {preview}")

    console.print(f"\n[dim]── 耗时统计 ──[/dim]")
    for stage, t in response.timings.items():
        console.print(f"  {stage}: {t:.3f}s")

    save_query_history(response, args.strategy, dataset, {
        "rerank": args.rerank, "rewrite": args.rewrite, "compress": args.compress,
    })
    console.print(f"[dim]结果已保存至 output/history/query_history.jsonl[/dim]")


def cmd_evaluate(args):
    """运行评估"""
    dataset = args.dataset
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_log_context("evaluate", f"{dataset}_{ts}")
    pipeline = RAGPipeline(dataset=dataset)

    # 切换评估数据集：--eval-file 优先，否则用数据集预设
    eval_path = args.eval_file if args.eval_file else get_eval_dataset_path(dataset)
    config.eval_dataset_path = eval_path

    strategies = args.strategies.split(",") if args.strategies != "all" else ALL_STRATEGIES

    # 创建 RAGAS Judge LLM（根据 .env 配置选择 zhipu / deepseek / openai）
    from src.llm import LLM
    judge_llm = LLM.create_judge()
    comparator = StrategyComparator(pipeline, judge_llm=judge_llm)

    use_cache = not args.no_cache
    judge_info = f"{config.ragas_judge_provider}:{judge_llm.model}"
    cache_info = "启用（加 --no-cache 强制重评）" if use_cache else "禁用"
    console.print(Panel(
        f"数据集: [bold cyan]{dataset}[/bold cyan]\n"
        f"模式: [bold]{'仅检索（无 LLM 生成）' if args.mode == 'retrieval' else '完整模式（检索 + 生成 + RAGAS）'}[/bold]\n"
        f"RAGAS Judge: [bold]{judge_info}[/bold]\n"
        f"RAGAS 缓存: {cache_info}\n"
        f"评估文件: {eval_path}\n"
        f"检索策略: {', '.join(strategies)}",
        title="评估",
    ))
    results = comparator.run_evaluation(
        strategies=strategies,
        n_questions=args.n_questions,
        use_ragas=not args.no_ragas,
        mode=args.mode,
        incremental=use_cache,
    )
    comparator.print_comparison_table(results)

    # 使用 run_evaluation 创建的目录（checkpoint 已在此目录中）
    eval_dir = results.get("meta", {}).get("output_dir")
    if not eval_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = os.path.join(config.output_dir, "history", f"eval_{dataset}_{ts}")
        os.makedirs(eval_dir, exist_ok=True)

    comparator.save_results(results, os.path.join(eval_dir, "full_results.json"))
    comparator.save_details_csv(results, os.path.join(eval_dir, "details.csv"))
    comparator.save_comparison_table(results, os.path.join(eval_dir, "comparison_table.txt"))

    if args.plot:
        comparator.plot_comparison(results, eval_dir)

    console.print(f"[green]评估结果已保存至 {eval_dir}/[/green]")


def cmd_interactive(args):
    """交互式查询模式"""
    dataset = args.dataset
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_log_context("interactive", f"{dataset}_{ts}")
    pipeline = RAGPipeline(dataset=dataset)
    strategy = args.strategy

    console.print(Panel(
        f"[bold]Advanced RAG 交互模式[/bold]\n"
        f"数据集: [cyan]{dataset}[/cyan] | 策略: [cyan]{strategy}[/cyan]\n"
        f"输入 'quit' 退出 | '/strategy <名称>' 切换检索策略",
        border_style="blue",
    ))

    while True:
        try:
            question = console.input("\n[bold green]Q:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.startswith("/strategy"):
            parts = question.split()
            if len(parts) > 1 and parts[1] in ALL_STRATEGIES:
                strategy = parts[1]
                console.print(f"已切换至 [cyan]{strategy}[/cyan]")
            else:
                console.print(f"可用策略: {', '.join(ALL_STRATEGIES)}")
            continue

        response = pipeline.query(question, strategy=strategy)
        console.print(f"\n[bold blue]A:[/bold blue] {response.answer}")
        console.print(f"[dim]({response.timings.get('total', 0):.2f}s, {len(response.contexts)} contexts)[/dim]")
        save_query_history(response, strategy, dataset)


def cmd_generate_eval(args):
    """自动生成评估数据集"""
    from evaluation.generate import EvalGenerator

    dataset = args.dataset
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_log_context("generate_eval", f"{dataset}_{ts}")

    model = args.model or config.generate_eval_llm_model
    # 并发数不超过 API 限制
    max_workers = min(args.max_workers, config.max_concurrent_generate)

    domain = args.domain
    console.print(Panel(
        f"数据集: [bold cyan]{dataset}[/bold cyan]\n"
        f"模型: {model}\n"
        f"领域: {domain or 'default'}\n"
        f"题目数: {args.n_questions} | 最大 chunk 数: {args.max_chunks} | "
        f"并发数: {max_workers}（上限: {config.max_concurrent_generate}）| 随机种子: {args.seed}",
        title="生成评估数据集",
    ))

    generator = EvalGenerator(
        dataset=dataset,
        api_key=config.zhipuai_api_key,
        model=model,
        n_questions=args.n_questions,
        max_chunks=args.max_chunks,
        max_workers=max_workers,
        seed=args.seed,
        output_path=args.output,
        domain=domain,
    )
    output = generator.generate()
    console.print(f"\n[green]完成！评估数据集已保存至 {output}[/green]")
    console.print(f"[dim]下一步: python main.py evaluate -d {dataset} --mode retrieval --plot[/dim]")


def cmd_dataset(args):
    """数据集管理"""
    import chromadb

    if args.action == "list":
        # 列出所有已入库的数据集
        try:
            client = chromadb.HttpClient(host=config.chroma_host, port=config.chroma_port)
            collections = client.list_collections()
        except Exception as e:
            console.print(f"[red]无法连接到 ChromaDB: {e}[/red]")
            return

        # 从 collection 名称提取 dataset 名称
        datasets = {}
        for col in collections:
            name = col.name if hasattr(col, 'name') else str(col)
            if name.endswith("_original"):
                ds_name = name.removesuffix("_original")
                if ds_name not in datasets:
                    datasets[ds_name] = {"collections": [], "chunks": 0}
                datasets[ds_name]["collections"].append(name)
                try:
                    datasets[ds_name]["chunks"] = client.get_collection(name).count()
                except Exception:
                    pass
            elif name.endswith("_summaries"):
                ds_name = name.removesuffix("_summaries")
                if ds_name not in datasets:
                    datasets[ds_name] = {"collections": [], "chunks": 0}
                datasets[ds_name]["collections"].append(name)

        # 检查本地缓存
        cache_dir = os.path.join(PROJ_ROOT, ".index_cache")
        cache_files = glob.glob(os.path.join(cache_dir, "*_bm25.pkl")) if os.path.exists(cache_dir) else []
        for f in cache_files:
            ds_name = os.path.basename(f).removesuffix("_bm25.pkl")
            if ds_name not in datasets:
                datasets[ds_name] = {"collections": [], "chunks": 0}

        table = Table(title="数据集列表", show_lines=True)
        table.add_column("数据集", style="cyan")
        table.add_column("Chunk 数", justify="center")
        table.add_column("ChromaDB Collection", style="dim")
        table.add_column("本地缓存", style="dim")
        table.add_column("预设说明", style="green")

        for ds_name in sorted(datasets):
            bm25_exists = os.path.exists(os.path.join(cache_dir, f"{ds_name}_bm25.pkl")) if os.path.exists(cache_dir) else False
            bge_exists = os.path.exists(os.path.join(cache_dir, f"{ds_name}_bge_m3.pkl")) if os.path.exists(cache_dir) else False
            cache_info = []
            if bm25_exists:
                cache_info.append("BM25")
            if bge_exists:
                cache_info.append("BGE-M3")

            preset_info = ""
            if ds_name in DATASET_PRESETS:
                preset_info = DATASET_PRESETS[ds_name]["description"]

            table.add_row(
                ds_name,
                str(datasets[ds_name]["chunks"]),
                ", ".join(datasets[ds_name]["collections"]),
                ", ".join(cache_info) or "-",
                preset_info or "-",
            )

        console.print(table)

    elif args.action == "delete":
        ds_name = args.name
        if not ds_name:
            console.print("[red]Usage: python main.py dataset delete <name>[/red]")
            return

        console.print(f"[yellow]正在删除数据集 '{ds_name}'...[/yellow]")

        # 删除 ChromaDB collections
        try:
            client = chromadb.HttpClient(host=config.chroma_host, port=config.chroma_port)
            for suffix in ["_original", "_summaries"]:
                col_name = f"{ds_name}{suffix}"
                try:
                    client.delete_collection(col_name)
                    console.print(f"  Deleted collection: {col_name}")
                except Exception:
                    pass
        except Exception as e:
            console.print(f"  [yellow]ChromaDB 连接失败: {e}[/yellow]")

        # 删除本地缓存
        cache_dir = os.path.join(PROJ_ROOT, ".index_cache")
        for suffix in ["_bm25.pkl", "_bge_m3.pkl", "_chunks.pkl", "_knowledge_graph.json"]:
            cache_path = os.path.join(cache_dir, f"{ds_name}{suffix}")
            if os.path.exists(cache_path):
                os.remove(cache_path)
                console.print(f"  Deleted cache: {cache_path}")

        # 删除 RAGAS 缓存
        ragas_cache_dir = os.path.join(cache_dir, "ragas_cache")
        if os.path.exists(ragas_cache_dir):
            import glob
            for f in glob.glob(os.path.join(ragas_cache_dir, f"{ds_name}_*.json")):
                os.remove(f)
                console.print(f"  Deleted RAGAS cache: {os.path.basename(f)}")

        console.print(f"[green]数据集 '{ds_name}' 已删除。[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced RAG - 混合精度检索实验平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 公共参数函数
    def add_dataset_arg(p, default="samples"):
        p.add_argument(
            "--dataset", "-d", default=default,
            help=f"数据集名称 (预设: {', '.join(DATASET_PRESETS.keys())}；也可用自定义名称)",
        )

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="文档入库")
    add_dataset_arg(p_ingest)
    p_ingest.add_argument("--doc-dir", default=None, help="文档目录（不指定则用数据集预设）")
    p_ingest.add_argument("--chunking", choices=["fixed", "recursive", "semantic"], default="recursive")
    p_ingest.add_argument("--reset", action="store_true", help="重置该数据集的向量库")
    p_ingest.add_argument("--incremental", action="store_true",
                          help="逐文件增量入库（支持断点续传，推荐用于大数据集）")
    p_ingest.add_argument("--no-summary", action="store_true",
                          help="跳过摘要生成（multi_vector 降级为纯 dense，其他 7 策略不受影响）")

    # query
    p_query = subparsers.add_parser("query", help="单次查询")
    add_dataset_arg(p_query)
    p_query.add_argument("question", help="查询问题")
    p_query.add_argument("--strategy", choices=ALL_STRATEGIES, default="hybrid_rrf")
    p_query.add_argument("--rerank", action="store_true")
    p_query.add_argument("--rewrite", action="store_true")
    p_query.add_argument("--compress", action="store_true")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="运行评估对比")
    add_dataset_arg(p_eval)
    p_eval.add_argument("--eval-file", default=None,
                        help="自定义评估数据集路径（JSON），不指定则用数据集预设")
    p_eval.add_argument("--mode", choices=["full", "retrieval"], default="full",
                        help="评估模式: full=完整评估(检索+生成+RAGAS), retrieval=纯检索评估(秒级)")
    p_eval.add_argument("--strategies", default="all")
    p_eval.add_argument("--n-questions", type=int, default=None)
    p_eval.add_argument("--no-ragas", action="store_true")
    p_eval.add_argument("--no-cache", action="store_true",
                        help="忽略 RAGAS 缓存，强制重新评估（默认使用缓存）")
    p_eval.add_argument("--plot", action="store_true")

    # interactive
    p_inter = subparsers.add_parser("interactive", help="交互式查询")
    add_dataset_arg(p_inter)
    p_inter.add_argument("--strategy", choices=ALL_STRATEGIES, default="hybrid_rrf")

    # generate-eval
    p_geneval = subparsers.add_parser("generate-eval",
                                       help="从已入库 chunks 自动生成评估数据集（基于 RAGAS）")
    add_dataset_arg(p_geneval)
    p_geneval.add_argument("--n-questions", "-n", type=int, default=50,
                           help="生成的问题数量")
    p_geneval.add_argument("--model", default=None,
                           help="LLM 模型覆盖（默认用 GENERATE_EVAL_LLM_MODEL）")
    p_geneval.add_argument("--max-chunks", type=int, default=300,
                           help="采样 chunk 上限（过多会导致知识图谱构建慢）")
    p_geneval.add_argument("--max-workers", type=int, default=5,
                           help="LLM 并发数（glm-4-flash 支持并发，默认 5）")
    p_geneval.add_argument("--seed", type=int, default=42,
                           help="随机种子（确保可复现）")
    p_geneval.add_argument("--domain", default=None,
                           help="领域配置（如 'ipcc'），加载 evaluation/domains/{name}.py")
    p_geneval.add_argument("--output", default=None,
                           help="输出路径（默认 evaluation/{dataset}_eval_dataset.json）")

    # dataset management
    p_ds = subparsers.add_parser("dataset", help="数据集管理")
    p_ds.add_argument("action", choices=["list", "delete"], help="操作")
    p_ds.add_argument("name", nargs="?", default=None, help="数据集名称 (delete 时必填)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "dataset":
        cmd_dataset(args)
        return

    if not config.zhipuai_api_key:
        console.print("[red]错误：ZHIPUAI_API_KEY 未设置，请创建 .env 文件并填写 API Key。[/red]")
        return

    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "evaluate": cmd_evaluate,
        "interactive": cmd_interactive,
        "generate-eval": cmd_generate_eval,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
