"""默认领域配置 — 通用 RAG 评估数据集生成

不带任何领域专用的 persona、prompt 或过滤规则。
当 --domain 未指定时自动使用。
"""

from evaluation.domains import DomainConfig

DOMAIN_CONFIG = DomainConfig(
    name="default",
    distribution=(0.4, 0.3, 0.3),
    personas=[],
    llm_context="",
    chunk_filter=None,
    post_filter=None,
)
