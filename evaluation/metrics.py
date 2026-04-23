"""自定义评估指标"""

import math
import numpy as np
from collections import Counter


def hit_rate(retrieved_ids: list[list[str]], relevant_ids: list[list[str]]) -> float:
    """Hit Rate: 至少命中一个相关文档的查询占比"""
    hits = 0
    for ret, rel in zip(retrieved_ids, relevant_ids):
        if any(r in rel for r in ret):
            hits += 1
    return hits / len(retrieved_ids) if retrieved_ids else 0.0


def mrr(retrieved_ids: list[list[str]], relevant_ids: list[list[str]]) -> float:
    """Mean Reciprocal Rank: 第一个相关文档排名的倒数的平均值"""
    rr_sum = 0.0
    for ret, rel in zip(retrieved_ids, relevant_ids):
        for rank, doc_id in enumerate(ret, 1):
            if doc_id in rel:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(retrieved_ids) if retrieved_ids else 0.0


def precision_at_k(retrieved_ids: list[list[str]], relevant_ids: list[list[str]], k: int = 5) -> float:
    """Precision@K: top-K 检索结果中相关文档的占比"""
    precisions = []
    for ret, rel in zip(retrieved_ids, relevant_ids):
        top_k = ret[:k]
        relevant_count = sum(1 for r in top_k if r in rel)
        precisions.append(relevant_count / k)
    return np.mean(precisions) if precisions else 0.0


def recall_at_k(retrieved_ids: list[list[str]], relevant_ids: list[list[str]], k: int = 5) -> float:
    """Recall@K: top-K 中命中的相关文档占所有相关文档的比例"""
    recalls = []
    for ret, rel in zip(retrieved_ids, relevant_ids):
        if not rel:
            continue
        top_k = ret[:k]
        relevant_count = sum(1 for r in top_k if r in rel)
        recalls.append(relevant_count / len(rel))
    return np.mean(recalls) if recalls else 0.0


def ndcg_at_k(retrieved_ids: list[list[str]], relevant_ids: list[list[str]], k: int = 5) -> float:
    """NDCG@K: 归一化折损累积增益"""
    ndcgs = []
    for ret, rel in zip(retrieved_ids, relevant_ids):
        # 计算 DCG
        dcg = 0.0
        for rank, doc_id in enumerate(ret[:k]):
            relevance = 1.0 if doc_id in rel else 0.0
            dcg += relevance / math.log2(rank + 2)
        # 计算 IDCG
        ideal_rels = sorted([1.0 if doc_id in rel else 0.0 for doc_id in ret[:k]], reverse=True)
        # 加上未检索到的相关文档
        remaining = max(0, len(rel) - sum(1 for d in ret[:k] if d in rel))
        ideal_rels = [1.0] * min(len(rel), k)
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))
        if idcg > 0:
            ndcgs.append(dcg / idcg)
        else:
            ndcgs.append(0.0)
    return np.mean(ndcgs) if ndcgs else 0.0


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L: 基于最长公共子序列的 F1 分数"""
    pred_tokens = list(prediction)
    ref_tokens = list(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS 长度
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_length = dp[m][n]

    precision = lcs_length / n if n > 0 else 0
    recall = lcs_length / m if m > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu_score(prediction: str, reference: str, max_n: int = 4) -> float:
    """简化版 BLEU 分数"""
    pred_tokens = list(prediction)
    ref_tokens = list(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(
            tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
        )
        clipped = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
        total = sum(pred_ngrams.values())
        if total == 0:
            scores.append(0.0)
        else:
            scores.append(clipped / total)

    if any(s == 0 for s in scores):
        return 0.0

    log_avg = sum(math.log(s) for s in scores) / len(scores)
    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))
    return bp * math.exp(log_avg)


def match_contexts_by_text(
    retrieved_contexts: list[str],
    ground_truth_contexts: list[str],
    threshold: float = 0.5,
) -> tuple[list[str], list[str]]:
    """用文本子串匹配将 retrieved contexts 映射为 ID，判断是否命中 ground truth。

    返回 (retrieved_ids, relevant_ids)，用于传入 retrieval metrics 函数。
    每个 ground_truth_context 视为一个 relevant "ID"（用索引标识）。
    如果某个 retrieved context 包含某个 ground_truth 片段（子串匹配），视为命中。
    """
    relevant_ids = [f"gt_{i}" for i in range(len(ground_truth_contexts))]
    retrieved_ids = []

    for ret_ctx in retrieved_contexts:
        matched = False
        for i, gt_ctx in enumerate(ground_truth_contexts):
            # 双向子串匹配：gt 是 ret 的子串，或 ret 是 gt 的子串
            if gt_ctx in ret_ctx or ret_ctx in gt_ctx:
                retrieved_ids.append(f"gt_{i}")
                matched = True
                break
            # 模糊匹配：计算重叠词比例
            gt_words = set(gt_ctx.lower().split())
            ret_words = set(ret_ctx.lower().split())
            if gt_words and ret_words:
                overlap = len(gt_words & ret_words) / len(gt_words)
                if overlap >= threshold:
                    retrieved_ids.append(f"gt_{i}")
                    matched = True
                    break
        if not matched:
            retrieved_ids.append(f"ret_{len(retrieved_ids)}")  # 非匹配项用唯一 ID

    return retrieved_ids, relevant_ids


def compute_all_retrieval_metrics(
    retrieved_ids: list[list[str]], relevant_ids: list[list[str]], k: int = 5
) -> dict[str, float]:
    return {
        "hit_rate": hit_rate(retrieved_ids, relevant_ids),
        "mrr": mrr(retrieved_ids, relevant_ids),
        f"precision@{k}": precision_at_k(retrieved_ids, relevant_ids, k),
        f"recall@{k}": recall_at_k(retrieved_ids, relevant_ids, k),
        f"ndcg@{k}": ndcg_at_k(retrieved_ids, relevant_ids, k),
    }


def compute_all_generation_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    rouge_scores = [rouge_l(p, r) for p, r in zip(predictions, references)]
    bleu_scores = [bleu_score(p, r) for p, r in zip(predictions, references)]
    return {
        "rouge_l": float(np.mean(rouge_scores)),
        "bleu": float(np.mean(bleu_scores)),
    }
