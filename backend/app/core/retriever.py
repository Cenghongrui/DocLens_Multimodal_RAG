from typing import List

from langchain_core.documents import Document

from app.config import settings
from app.core.embedder import get_or_create_collection, embed_text


async def retrieve(query: str, top_k: int = None) -> List[Document]:
    """检索与 query 最相关的文档片段"""
    if top_k is None:
        top_k = settings.top_k

    query_embedding = await embed_text(query)
    vectorstore = get_or_create_collection()

    docs = vectorstore.similarity_search_by_vector(
        embedding=query_embedding,
        k=top_k,
    )
    return docs