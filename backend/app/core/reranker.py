"""Cross-encoder 重排序器。

"""
import httpx

from app.config import settings
from langchain_core.documents import Document

async def rerank(
    query: str,
    documents: list[Document],
    top_n: int = 5,
) -> list[Document]:
    """对 documents 按与 query 的相关性重排序，返回 top_n 个。

    documents: 混合检索召回的候选（建议 20 个左右）
    top_n: 精排后保留几个
    """
    if not documents:
        return []

    # 把 Document 转成纯文本列表发给 API
    texts = [d.page_content for d in documents]

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            settings.rerank_url,
            headers={"Authorization": f"Bearer {settings.qwen_api_key}"},
            json={
                "model": settings.rerank_model,
                "input": {
                    "query": query,
                    "documents": texts,
                },
                "parameters": {
                    "return_documents": False,
                    "top_n": min(top_n, len(texts)),
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()

    # 返回的 results 按 relevance_score 降序排好
    results = data["output"]["results"]
    reranked = []
    for r in results:
        idx = r["index"]          # 对应原 texts 的下标
        score = r["relevance_score"]
        doc = documents[idx]
        # 把 rerank 分数写进 metadata，调试/展示用
        doc.metadata["rerank_score"] = score
        reranked.append(doc)

    return reranked