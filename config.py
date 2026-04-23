"""全局配置"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# 项目根目录（config.py 所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


@dataclass
class Config:
    # 智谱 API
    zhipuai_api_key: str = os.getenv("ZHIPUAI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "glm-4v-flash")
    ingest_llm_model: str = os.getenv("INGEST_LLM_MODEL", "glm-4-flash")  # 摘要生成专用
    generate_eval_llm_model: str = os.getenv("GENERATE_EVAL_LLM_MODEL", "glm-4-flash")  # 评估数据集生成
    # 分阶段并发控制（不同模型 QPS 不同）
    max_concurrent: int = int(os.getenv("MAX_CONCURRENT", "5"))  # 通用默认值（兜底）
    max_concurrent_generate: int = int(os.getenv("MAX_CONCURRENT_GENERATE", os.getenv("MAX_CONCURRENT", "5")))  # generate-eval KG 构建
    max_concurrent_judge: int = int(os.getenv("MAX_CONCURRENT_JUDGE", os.getenv("MAX_CONCURRENT", "10")))  # RAGAS Judge 评估
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "embedding-3")

    # RAGAS Judge 配置（支持 zhipu / deepseek / openai）
    ragas_judge_provider: str = os.getenv("RAGAS_JUDGE_PROVIDER", "zhipu")
    ragas_judge_model: str = os.getenv("RAGAS_JUDGE_MODEL", "")  # 留空则跟随 LLM_MODEL

    # DeepSeek API
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    # OpenAI API
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    embedding_dim: int = 2048  # embedding-3 输出维度

    # ChromaDB
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))

    # 分块参数（可通过环境变量覆盖）
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1024"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "128"))
    semantic_threshold: float = 0.75  # 语义分块的相似度阈值

    # 检索参数
    top_k: int = 5
    rerank_top_k: int = 3
    rrf_k: int = 60  # RRF 融合参数

    # BGE-M3 多粒度检索
    bge_m3_model: str = "BAAI/bge-m3"
    bge_m3_use_fp16: bool = True
    # hybrid 模式下 dense/sparse/colbert 三路融合权重
    bge_m3_weights: dict = None  # 默认在 BGEM3Retriever 中设为 {dense:0.4, sparse:0.2, colbert:0.4}

    def __post_init__(self):
        if self.bge_m3_weights is None:
            self.bge_m3_weights = {"dense": 0.4, "sparse": 0.2, "colbert": 0.4}

    # 路径（绝对路径，避免工作目录问题）
    documents_dir: str = os.path.join(PROJECT_ROOT, "documents/samples")
    eval_dataset_path: str = os.path.join(PROJECT_ROOT, "evaluation/samples_eval_dataset.json")
    output_dir: str = os.path.join(PROJECT_ROOT, "output")

    # 评估
    eval_llm_judge: bool = True  # 是否使用 LLM 作为评估 judge


config = Config()
