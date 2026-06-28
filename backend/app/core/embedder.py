"""Qwen Embedding API 封装 + ChromaDB 连接。"""
import os
from typing import List
from openai import OpenAI, AsyncOpenAI
from langchain_chroma import Chroma
from chromadb import PersistentClient
from app.config import settings

# 同步 client：供 LangChain Chroma 内部调用（同步接口）
_sync = OpenAI(api_key=settings.qwen_api_key, base_url=settings.qwen_base_url)
# 异步 client：供检索管线调用
_async = AsyncOpenAI(api_key=settings.qwen_api_key, base_url=settings.qwen_base_url)


async def embed_text(text: str) -> List[float]:
    resp = await _async.embeddings.create(
        model=settings.embedding_model, input=text, dimensions=settings.embedding_dimension,
    )
    return resp.data[0].embedding


async def embed_batch(texts: List[str]) -> List[List[float]]:
    """分批批量 embedding，每批不超过 API 上限。"""
    all_embs = []
    for i in range(0, len(texts), settings.embedding_batch_size):
        batch = texts[i:i + settings.embedding_batch_size]
        resp = await _async.embeddings.create(
            model=settings.embedding_model, input=batch, dimensions=settings.embedding_dimension,
        )
        all_embs.extend(item.embedding for item in resp.data)
    return all_embs


# ─── ChromaDB ───


def get_or_create_collection() -> Chroma:
    os.makedirs(settings.vectordb_dir, exist_ok=True)
    return Chroma(
        client=PersistentClient(path=settings.vectordb_dir),
        collection_name="doclens",
        embedding_function=_EmbeddingFunction(),
    )


# ─── LangChain Embedding 适配器 ───


class _EmbeddingFunction:
    """LangChain Chroma 需要 embed_documents + embed_query 两个方法。"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embs = []
        for i in range(0, len(texts), settings.embedding_batch_size):
            batch = texts[i:i + settings.embedding_batch_size]
            resp = _sync.embeddings.create(
                model=settings.embedding_model, input=batch, dimensions=settings.embedding_dimension,
            )
            all_embs.extend(item.embedding for item in resp.data)
        return all_embs

    def embed_query(self, text: str) -> List[float]:
        resp = _sync.embeddings.create(
            model=settings.embedding_model, input=text, dimensions=settings.embedding_dimension,
        )
        return resp.data[0].embedding
