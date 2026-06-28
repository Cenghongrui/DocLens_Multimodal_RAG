"""应用配置，启动时从 .env 加载。"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ─── LLM 供应商 ───
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    qwen_api_key: str
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    rerank_url: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    # ─── 模型选择 ───
    llm_model: str = "deepseek-v4-flash"            # 生成模型
    judge_llm_model: str = "qwen3.5-flash"          # 评分/路由用
    vision_model: str = "qwen-vl-flash"             # 图片理解
    embedding_model: str = "text-embedding-v3"      # 向量化
    rerank_model: str = "qwen3-vl-rerank"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 10                  # Qwen API 上限

    # ─── 分片参数 ───
    chunk_size: int = 512       # 每块最大 token 数
    chunk_overlap: int = 128    # 相邻 chunk 尾部叠加字符数

    # ─── 检索参数 ───
    top_k: int = 3

    # ─── HyDE ───
    hyde_enabled: bool = True
    hyde_route_threshold: float = 0.5
    hyde_model: str = "qwen3.5-flash"
    hyde_max_tokens: int = 200

    # ─── 存储路径 ───
    data_dir: str = "data/"
    image_dir: str = "images/"
    vectordb_dir: str = "vectordb/"

    class Config:
        env_file = ".env"


settings = Settings()
