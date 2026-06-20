from typing import List

from langchain_core.documents import Document

from app.config import settings
from app.core.embedder import get_or_create_collection, embed_text


async def retrieve(query: str, top_k: int = None, source: str = None) -> List[Document]:
    """检索与 query 最相关的文档片段，可指定 source 过滤"""
    if top_k is None:
        top_k = settings.top_k

    filter_dict: dict | None = None
    if source:
        filter_dict = {"source": source}

    query_embedding = await embed_text(query)
    vectorstore = get_or_create_collection()

    # Chroma 的 filter 放在 search_kwargs 里
    docs = vectorstore.similarity_search_by_vector(
        embedding=query_embedding,
        k=top_k,
        filter=filter_dict,
    )
    return docs