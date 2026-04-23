"""领域配置加载器

每个领域（如 IPCC、医学、法律）有独立的 .py 文件定义 DomainConfig，
包含该领域专用的 persona、prompt 引导、chunk 过滤和生成后质量过滤规则。

使用方式:
    from evaluation.domains import load_domain_config
    config = load_domain_config("ipcc")   # 加载 evaluation/domains/ipcc.py
    config = load_domain_config(None)     # 加载 evaluation/domains/default.py

新增领域:
    1. 创建 evaluation/domains/my_domain.py
    2. 定义 DOMAIN_CONFIG = DomainConfig(...)
    3. 使用: python main.py generate-eval -d my_data --domain my_domain
"""

import importlib
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class DomainConfig:
    """领域专用评估数据集生成配置"""

    # 领域名称
    name: str = "default"

    # Query 难度分布: (single_hop, multi_hop_abstract, multi_hop_specific)
    # 三个值之和应为 1.0
    distribution: tuple[float, float, float] = (0.4, 0.3, 0.3)

    # Persona 列表: [(name, role_description), ...]
    # 空列表 = RAGAS 自动生成 persona
    personas: list[tuple[str, str]] = field(default_factory=list)

    # 生成引导 prompt，传给 TestsetGenerator 的 llm_context
    # 空字符串 = 无特殊引导
    llm_context: str = ""

    # 额外的 chunk 过滤器（在通用 _is_low_quality 之后追加）
    # callable(text: str) -> bool，返回 True 表示该 chunk 应被排除
    # None = 不追加额外过滤
    chunk_filter: Optional[Callable[[str], bool]] = None

    # 生成后质量过滤器
    # callable(question: str, ground_truth: str) -> bool，返回 True 表示该题应被排除
    # None = 不过滤
    post_filter: Optional[Callable[[str, str], bool]] = None


def load_domain_config(name: Optional[str] = None) -> DomainConfig:
    """加载领域配置

    Args:
        name: 领域名称（对应 evaluation/domains/{name}.py 中的 DOMAIN_CONFIG）
              None 或 "default" 加载默认配置

    Returns:
        DomainConfig 实例

    Raises:
        ImportError: 领域配置文件不存在
        AttributeError: 配置文件中缺少 DOMAIN_CONFIG
    """
    module_name = name or "default"

    try:
        module = importlib.import_module(f"evaluation.domains.{module_name}")
    except ImportError:
        available = _list_available_domains()
        raise ImportError(
            f"Domain config '{module_name}' not found.\n"
            f"Expected file: evaluation/domains/{module_name}.py\n"
            f"Available domains: {', '.join(available)}"
        )

    if not hasattr(module, "DOMAIN_CONFIG"):
        raise AttributeError(
            f"evaluation/domains/{module_name}.py must define a DOMAIN_CONFIG variable"
        )

    config = module.DOMAIN_CONFIG
    if not isinstance(config, DomainConfig):
        raise TypeError(
            f"DOMAIN_CONFIG in {module_name}.py must be a DomainConfig instance"
        )

    return config


def _list_available_domains() -> list[str]:
    """列出所有可用的领域配置"""
    import os
    domains_dir = os.path.dirname(__file__)
    return [
        f[:-3] for f in os.listdir(domains_dir)
        if f.endswith(".py") and f != "__init__.py"
    ]
