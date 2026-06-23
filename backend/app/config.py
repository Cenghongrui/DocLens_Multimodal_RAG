from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置，自动从 .env 文件加载"""

    # ─── LLM 配置 ───
    # DeepSeek
    deepseek_api_key: str  
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    # Qwen
    qwen_api_key: str
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    rerank_url: str = " https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    # ─── 模型参数 ───
    llm_model: str = "deepseek-v4-flash"  
    judge_llm_model: str = "qwen3.6-flash"
    vision_model: str = "qwen-vl-plus"  
    embedding_model: str = "text-embedding-v4" 
    rerank_model: str = "qwen3-vl-rerank"  # 用于 rerank 的 LLM 模型
    embedding_dimension: int = 1024  # text-embedding-v4 输出 1024 维
    embedding_batch_size: int = 10  # Qwen embedding API 单次最大 10 条

    # ─── 分块参数 ───
    chunk_size: int = 1000  # 每块最大字符数
    chunk_overlap: int = 200  # 相邻块重叠字符数

    # ─── 检索参数 ───
    top_k: int = 5  # 每次检索返回多少个结果

    # ─── 路径 ───
    data_dir: str = "data/"
    image_dir: str = "images/"
    vectordb_dir: str = "vectordb/"

    class Config:
        env_file = ".env"


settings = Settings()