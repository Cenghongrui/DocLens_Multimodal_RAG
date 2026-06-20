import os
from typing import List

from openai import OpenAI, AsyncOpenAI
from langchain_chroma import Chroma
from chromadb import PersistentClient

from app.config import settings

# 全局 client（embedding 只同步调，因为 LangChain 接口是同步的）
embedding_client = OpenAI(
    api_key=settings.qwen_api_key,
    base_url=settings.qwen_base_url,
)

# 异步 client（retriever.py 用）
async_embedding_client = AsyncOpenAI(
    api_key=settings.qwen_api_key,
    base_url=settings.qwen_base_url,
)



async def embed_text(text: str) -> List[float]:
    """单段文本"""
    response = await async_embedding_client.embeddings.create(
        model=settings.embedding_model,
        input=text,
        dimensions=settings.embedding_dimension,
    )
    return response.data[0].embedding


async def embed_batch(texts: List[str]) -> List[List[float]]:
    """分批批量文本"""
    all_embeddings = []
    for i in range(0, len(texts), settings.embedding_batch_size):
        batch = texts[i:i + settings.embedding_batch_size]
        response = await async_embedding_client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
            dimensions=settings.embedding_dimension,
        )
        all_embeddings.extend(item.embedding for item in response.data)
    return all_embeddings


# ─── ChromaDB 连接 ───

def get_or_create_collection() -> Chroma:
    """获取 ChromaDB collection（不存在则自动创建）"""
    os.makedirs(settings.vectordb_dir, exist_ok=True)

    chroma_client = PersistentClient(path=settings.vectordb_dir)

    vectorstore = Chroma(
        client=chroma_client,
        collection_name="doclens",
        embedding_function=_EmbeddingFunction(),
    )
    return vectorstore


# ─── LangChain Embedding 适配器 ───

class _EmbeddingFunction:
    """
    LangChain Chroma 要求 embedding function 实现两个方法。
    内部转调我们的 embedding 函数。
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量入库用（同步，分批处理）"""
        all_embeddings = []
        for i in range(0, len(texts), settings.embedding_batch_size):
            batch = texts[i:i + settings.embedding_batch_size]
            response = embedding_client.embeddings.create(
                model=settings.embedding_model,
                input=batch,
                dimensions=settings.embedding_dimension,
            )
            all_embeddings.extend(item.embedding for item in response.data)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """查询用（同步）"""
        response = embedding_client.embeddings.create(
            model=settings.embedding_model,
            input=text,
            dimensions=settings.embedding_dimension,
        )
        return response.data[0].embedding