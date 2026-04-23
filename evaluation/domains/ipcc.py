"""IPCC 气候报告领域配置

针对 IPCC Working Group II Assessment Report (AR6) 的特点优化：
- 高密度学术文本，含大量数据表格、置信度评级和区域分析
- 文档结构噪声多（目录页、作者列表、参考文献区、图片引用）
- 真实用户需要定量、跨区域对比、因果机制、不确定性感知的问题

使用: python main.py generate-eval -d wg2 --domain ipcc --n-questions 50
"""

import re
from evaluation.domains import DomainConfig


# ── Persona（提问角色） ──

PERSONAS = [
    (
        "Climate Policy Analyst",
        "You work at a government think tank, extracting specific risk data, "
        "confidence levels, temperature thresholds, timelines, and monetary values "
        "from IPCC reports to support policy recommendations. "
        "You always ask about specific numbers, percentages, and how findings "
        "compare with previous assessment reports. You care about 'under what conditions', "
        "'what confidence level', and 'compared to AR5, what has changed'."
    ),
    (
        "Climate Adaptation Engineer",
        "You are responsible for climate resilience planning of urban infrastructure. "
        "You need to understand specific regional risks (coastal, urban, agricultural) "
        "under different warming scenarios (1.5°C, 2°C, 4°C), and which adaptation "
        "measures have been proven effective. "
        "You tend to ask cross-sectoral compound questions involving water resources, "
        "energy systems, food security, and public health simultaneously."
    ),
    (
        "Climate Vulnerability Researcher",
        "You specialize in studying how climate change disproportionately affects "
        "developing countries, low-income communities, Indigenous peoples, women, and youth. "
        "You look for evidence on inequality of exposure, differential adaptive capacity, "
        "and whether adaptation measures might exacerbate existing inequalities. "
        "You ask about distributional impacts, equity implications, and justice dimensions "
        "of climate policies."
    ),
]


# ── LLM Context（生成引导 Prompt，告诉 RAGAS 生成哪类问题） ──
# 注意：此字符串直接发送给 LLM，必须保持英文

LLM_CONTEXT = """You are generating evaluation questions for a RAG system based on the \
IPCC Working Group II Assessment Report on climate change impacts, adaptation, and vulnerability.

Generate questions that reflect REAL professional user needs. Prioritize these question types:

1. QUANTITATIVE: Ask about specific temperatures (1.5°C, 2°C, 4°C scenarios), percentages, \
   population numbers, monetary values (USD billions), timelines (by 2050, by 2100).
2. COMPARATIVE: Differences across regions (Africa vs Asia), sectors (water vs food), \
   warming levels (1.5°C vs 2°C), or assessment reports (AR5 vs AR6).
3. CAUSAL MECHANISM: HOW does X cause Y? What is the chain of impacts? \
   (not just "What is X?" but "How does sea level rise affect urban infrastructure?")
4. UNCERTAINTY-AWARE: Ask about confidence levels (high/medium/low confidence), \
   evidence quality, likelihood ranges, or agreement levels.
5. CROSS-SECTORAL: Questions linking multiple sectors, e.g., how does drought \
   affect both food security AND energy production AND migration patterns?

NEVER generate questions about:
- Author names, affiliations, or contributions
- Citation details or reference lists
- Document structure (table of contents, page numbers, section headings)
- Content requiring information NOT in the provided context
- Simple yes/no questions without depth
"""


# ── Chunk Filter（chunk 过滤器） ──

def ipcc_chunk_filter(text: str) -> bool:
    """IPCC 专用 chunk 过滤器，返回 True 表示应排除。

    在通用 _is_low_quality 之后追加，针对 IPCC 报告特有的噪声内容：
    目录页、作者列表、纯图片引用、附录/术语表、引用格式页。
    """
    # 1. 目录页检测：高密度三四位页码数字（如 "6.2.1 Risk Creation ... 921"）
    words = text.split()
    if words:
        page_number_density = sum(
            1 for w in words if re.match(r'^\d{3,4}$', w)
        ) / len(words)
        if page_number_density > 0.05:
            return True

    # 2. 作者列表页（IPCC 报告每章都有）
    author_patterns = [
        r'(?i)^Coordinating Lead Authors?:',
        r'(?i)^Lead Authors?:',
        r'(?i)^Contributing Authors?:',
        r'(?i)^Review Editors?:',
        r'(?i)^Chapter Scientist:',
    ]
    for pattern in author_patterns:
        if re.search(pattern, text[:200]):
            return True

    # 3. 纯图片引用行（Markdown 格式图片，无实质文本）
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines and all(re.match(r'^!\[.*?\]\(.*?\)$', l) for l in lines):
        return True

    # 4. 附录/术语表/缩略词页（短小且高度列表化，生成题目质量差）
    if re.match(r'(?i)^(Appendix|Glossary|Acronyms|Abbreviations)\s', text[:50]):
        if text.count('\n') > 5 and len(text) < 500:
            return True

    # 5. 引用格式密集区（"This chapter should be cited as:" 等引用提示）
    if 'should be cited as' in text[:200]:
        return True

    return False


# ── Post-Generation Quality Filter（生成后质量过滤器） ──

def ipcc_post_filter(question: str, ground_truth: str) -> bool:
    """IPCC 专用生成后质量过滤器，返回 True 表示该题应排除。

    过滤噪声题：作者/机构类、引用/参考文献类、目录结构类、
    ground truth 过短、以及模型承认无法回答的幻觉题。
    """
    q_lower = question.lower()
    gt_lower = ground_truth.lower() if ground_truth else ""

    # 1. 作者/机构类问题（询问谁写的/哪个机构）
    author_patterns = [
        r'(?i)who\s+(wrote|authored|contributed|led|coordinated)',
        r'(?i)which\s+(author|institution|university|organization)',
        r'(?i)name\s+the\s+(authors?|contributors?|editors?)',
        r'(?i)what\s+(is|are)\s+the\s+affiliation',
    ]
    for p in author_patterns:
        if re.search(p, question):
            return True

    # 2. 引用/参考文献类问题（询问哪篇文献/如何引用）
    ref_patterns = [
        r'(?i)(which|what)\s+(reference|citation|source|bibliography)',
        r'(?i)cite\s+(the|this|a)',
        r'(?i)according\s+to\s+which\s+(study|paper|report)',
        r'(?i)et\s+al\.\s+\(\d{4}\)',
    ]
    for p in ref_patterns:
        if re.search(p, question):
            return True

    # 3. 目录/文档结构类问题（询问章节号/页码/目录）
    structure_patterns = [
        r'(?i)table\s+of\s+contents',
        r'(?i)(chapter|section)\s+number',
        r'(?i)page\s+number',
        r'(?i)section\s+heading',
        r'(?i)how\s+many\s+(chapters|sections|pages)',
    ]
    for p in structure_patterns:
        if re.search(p, question):
            return True

    # 4. ground truth 过短（可能是幻觉或空洞回答）
    if len(ground_truth.strip()) < 20:
        return True

    # 5. 幻觉信号：模型在 ground truth 中承认不知道或无信息
    hallucination_signals = [
        r"(?i)i\s+don'?t\s+know",
        r'(?i)not\s+(mentioned|found|provided|stated)\s+in',
        r'(?i)no\s+(information|data|evidence)\s+(is\s+)?(available|provided|given)',
        r'(?i)the\s+(text|passage|context)\s+does\s+not',
        r'(?i)cannot\s+be\s+determined',
    ]
    for p in hallucination_signals:
        if re.search(p, gt_lower):
            return True

    # 6. 问题过短（通常是低质量或过于模糊的问题）
    if len(question.strip()) < 30:
        return True

    return False


# ── Domain Config（领域配置入口） ──

DOMAIN_CONFIG = DomainConfig(
    name="ipcc",
    distribution=(0.2, 0.4, 0.4),
    personas=PERSONAS,
    llm_context=LLM_CONTEXT,
    chunk_filter=ipcc_chunk_filter,
    post_filter=ipcc_post_filter,
)
