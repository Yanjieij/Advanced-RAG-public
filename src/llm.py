"""智谱 GLM LLM 封装"""

import json
import logging
import os
import time
from zhipuai import ZhipuAI
from zhipuai.core._errors import APIReachLimitError
from config import config, PROJECT_ROOT

# LLM 调用日志：按任务分文件，每次 set_log_context 切换输出文件
_LOG_DIR = os.path.join(PROJECT_ROOT, "output", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_llm_logger = logging.getLogger("llm_calls")
_llm_logger.setLevel(logging.DEBUG)
_llm_logger.propagate = False
# 默认 fallback 日志文件
if not _llm_logger.handlers:
    _fh = logging.FileHandler(os.path.join(_LOG_DIR, "llm_calls.jsonl"), encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(message)s"))
    _llm_logger.addHandler(_fh)


def set_log_context(task: str, run_id: str = ""):
    """切换 LLM 日志输出文件，按任务类型分文件。

    示例：
        set_log_context("ingest", "IPCC_20260331_160000")
        → output/logs/ingest_IPCC_20260331_160000.jsonl

        set_log_context("evaluate", "test_20260331_170000")
        → output/logs/evaluate_test_20260331_170000.jsonl
    """
    filename = f"{task}_{run_id}.jsonl" if run_id else f"{task}.jsonl"
    filepath = os.path.join(_LOG_DIR, filename)

    # 移除旧 handler，添加新的
    for h in _llm_logger.handlers[:]:
        h.close()
        _llm_logger.removeHandler(h)

    fh = logging.FileHandler(filepath, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    _llm_logger.addHandler(fh)


class LLM:
    def __init__(self, provider: str = "zhipu", model: str = None, api_key: str = None, base_url: str = None):
        """创建 LLM 实例。

        Args:
            provider: "zhipu" / "deepseek" / "openai" / "openai_compatible"
            model: 模型名（不指定则用 config.llm_model）
            api_key: API Key（不指定则从 config 读取）
            base_url: API Base URL
        """
        self.provider = provider

        if provider == "zhipu":
            self.client = ZhipuAI(api_key=api_key or config.zhipuai_api_key)
            self.model = model or config.llm_model
        elif provider == "openai":
            import openai
            self.client = openai.OpenAI(
                api_key=api_key or config.openai_api_key,
                base_url=base_url or config.openai_base_url,
            )
            self.model = model or "gpt-4o-mini"
        elif provider in ("deepseek", "openai_compatible"):
            import openai
            self.client = openai.OpenAI(
                api_key=api_key or config.deepseek_api_key,
                base_url=base_url or config.deepseek_base_url,
            )
            self.model = model or "deepseek-chat"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @staticmethod
    def create_judge() -> "LLM":
        """根据 config 创建 RAGAS Judge LLM 实例。

        读取 RAGAS_JUDGE_PROVIDER 和 RAGAS_JUDGE_MODEL 配置，
        支持 zhipu / deepseek / openai 三种 provider。
        """
        provider = config.ragas_judge_provider
        model = config.ragas_judge_model

        if provider == "openai":
            return LLM(
                provider="openai",
                model=model or "gpt-4o-mini",
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
            )
        elif provider == "deepseek":
            return LLM(
                provider="deepseek",
                model=model or "deepseek-chat",
                api_key=config.deepseek_api_key,
                base_url=config.deepseek_base_url,
            )
        else:  # zhipu (default)
            return LLM(
                provider="zhipu",
                model=model or config.llm_model,
            )

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_retries: int = 5,
        max_tokens: int = None,
        model: str = None,
    ) -> str:
        """调用 LLM，遇到 429 时指数退避重试。每次调用写入日志。"""
        use_model = model or self.model
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = dict(
            model=use_model,
            messages=messages,
            temperature=temperature,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        t0 = time.time()
        delay = 2.0
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                elapsed = time.time() - t0
                # 提取 token 用量
                usage = {}
                if hasattr(response, "usage") and response.usage:
                    usage = {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(response.usage, "completion_tokens", None),
                        "total_tokens": getattr(response.usage, "total_tokens", None),
                    }
                _llm_logger.debug(json.dumps({
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "provider": self.provider,
                    "model": use_model,
                    "prompt": prompt,
                    "system": system if system else "",
                    "output": content if content else "",
                    "output_len": len(content) if content else 0,
                    "usage": usage,
                    "elapsed_s": round(elapsed, 2),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }, ensure_ascii=False))
                return content
            except APIReachLimitError:
                # 智谱 429
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)
            except Exception as e:
                # DeepSeek / OpenAI 429 限流（openai.RateLimitError）
                if "rate" in str(e).lower() or "429" in str(e):
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    raise

    def batch_generate(
        self,
        prompts: list[str],
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = None,
        model: str = None,
    ) -> list[str]:
        """顺序批量调用 LLM，内置限流保护与指数退避重试。

        Args:
            prompts: 待处理的 prompt 列表
            system: 系统提示词
            temperature: 生成温度
            max_tokens: 限制每次调用的最大输出 token 数
            model: 覆盖默认模型（如摘要任务用 glm-4-flash）

        Returns:
            与 prompts 同序的结果列表
        """
        results = []
        total = len(prompts)
        t0 = time.time()

        for i, prompt in enumerate(prompts):
            try:
                result = self.generate(
                    prompt, system=system, temperature=temperature,
                    max_tokens=max_tokens, model=model,
                )
                results.append(result or "")
            except Exception as e:
                # 内容审核等不可恢复错误：跳过该条目，用空字符串占位
                print(f"    [!] Item {i+1}/{total} failed: {e}", flush=True)
                results.append("")

            done = i + 1
            if done == 1 or done % 10 == 0 or done == total:
                elapsed = time.time() - t0
                avg = elapsed / done
                eta = avg * (total - done)
                print(
                    f"    [{done}/{total}]  elapsed {elapsed:.0f}s  "
                    f"avg {avg:.1f}s/item  ETA {eta:.0f}s",
                    flush=True,
                )

        return results

    def generate_with_context(self, query: str, contexts: list[str]) -> str:
        context_text = "\n\n---\n\n".join(contexts)
        system = (
            "You are a knowledge-based QA assistant. Answer the user's question "
            "based ONLY on the provided reference materials. "
            "If the reference materials do not contain relevant information, state this clearly. "
            "Do not fabricate information. "
            "IMPORTANT: Answer in the SAME LANGUAGE as the user's question."
        )
        prompt = f"Reference materials:\n{context_text}\n\nQuestion: {query}\n\nAnswer based on the reference materials:"
        return self.generate(prompt, system=system, temperature=0.3)

    def rewrite_query(self, query: str) -> str:
        prompt = (
            f"请将以下用户查询改写为更适合信息检索的形式，"
            f"保持语义不变但更加明确具体。只输出改写后的查询，不要解释。\n\n"
            f"原始查询：{query}\n改写查询："
        )
        return self.generate(prompt, temperature=0.3).strip()

    def generate_hypothetical_answer(self, query: str) -> str:
        """HyDE: 生成假设性文档片段，用其 embedding 替代原始 query 进行检索。

        关键设计：
        - 不需要正确答案，只需要风格和用词接近真实文档
        - 严格限制长度（2-3句话），避免生成过长文本导致延迟爆炸
        - 低 temperature 减少随机性
        """
        prompt = (
            f"针对以下问题，写一段可能包含答案的文档片段（2-3句话，不超过100字）。"
            f"不需要完全正确，只需风格接近技术文档。\n\n"
            f"问题：{query}\n文档片段："
        )
        return self.generate(prompt, temperature=0.3)

    def score_relevance(self, query: str, document: str) -> float:
        prompt = (
            f"请评估以下文档与查询的相关性，打分范围 0-10，只输出数字。\n\n"
            f"查询：{query}\n文档：{document}\n相关性评分："
        )
        try:
            score_text = self.generate(prompt, temperature=0.1).strip()
            return float(score_text) / 10.0
        except (ValueError, TypeError):
            return 0.0
