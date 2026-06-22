from typing import List

from langchain_core.documents import Document

from app.config import settings
from app.core.hybrid_retriever import hybrid_retrieve


async def retrieve(query: str, top_k: int = None, source: str = None) -> List[Document]:
    """检索入口--混合检索"""
    return await hybrid_retrieve(query, top_k=top_k, source=source)