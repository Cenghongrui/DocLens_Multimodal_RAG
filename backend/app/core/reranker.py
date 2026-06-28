"""Cross-encoder 重排序：Qwen text-rerank API。"""
import httpx
from app.config import settings
from langchain_core.documents import Document


async def rerank(query: str, documents: list[Document], top_n: int = 5) -> list[Document]:
    """对候选文档按 query 相关性精排。

    Args:
        query: 原始查询。
        documents: 混合检索召回的候选项（建议 15-20 个）。
        top_n: 精排后保留数量。

    Returns:
        按相关性降序排列的 Document 列表。
    """
    if not documents:
        return []

    texts = [d.page_content for d in documents]
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            settings.rerank_url,
            headers={"Authorization": f"Bearer {settings.qwen_api_key}"},
            json={
                "model": settings.rerank_model,
                "input": {"query": query, "documents": texts},
                "parameters": {"return_documents": False, "top_n": min(top_n, len(texts))},
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results = data["output"]["results"]
    reranked = []
    for r in results:
        doc = documents[r["index"]]
        doc.metadata["rerank_score"] = r["relevance_score"]
        reranked.append(doc)
    return reranked
