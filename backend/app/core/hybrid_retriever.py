"""混合检索：向量检索（语义）+ BM25（关键词），加权融合。

"""
from typing import List

from langchain_core.documents import Document

from app.config import settings
from app.core.embedder import get_or_create_collection
from app.core.bm25_retriever import get_bm25_retriever
from app.core.reranker import rerank


def _min_max_normalize(scores: list[float]) -> list[float]:
    """把分数列表归一化到 0~1。
    公式：(x - min) / (max - min)。全相同则全部设为 1。"""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


async def hybrid_retrieve(
    query: str,
    top_k: int = None,
    source: str = None,
    # 权重：向量 vs BM25。0.5/0.5 是均衡起点，可调。
    vector_weight: float = 0.75,
    bm25_weight: float = 0.25,
    # 召回阶段多召回一些，留给 reranker 精排
    candidate_k: int = 20,
) -> List[Document]:
    """混合检索主函数。"""
    if top_k is None:
        top_k = settings.top_k

    # ─── 第一路：向量检索 ───
    vectorstore = get_or_create_collection()
    filter_dict = {"source": source} if source else None

    # 多召回，给融合更大候选池
    vec_pairs = vectorstore.similarity_search_with_score(
        query=query,
        k=candidate_k,
        filter=filter_dict,
    )
    # ChromaDB 返回的是"距离"（越小越相似），转成相似度并归一化
    vec_scores_raw = [1.0 / (1.0 + dist) for _, dist in vec_pairs]

    # ─── 第二路：BM25 检索 ───
    bm25 = get_bm25_retriever()
    bm25_docs, bm25_scores_raw = bm25.search(query, top_k=candidate_k, return_scores=True)

    # ─── 融合 ───
    # 用 chunk_id 作为唯一键，合并两路的分数
    fused: dict[str, dict] = {}

    vec_scores = _min_max_normalize(vec_scores_raw)
    for (doc, _), score in zip(vec_pairs, vec_scores):
        cid = doc.metadata.get("chunk_id", id(doc))
        fused.setdefault(cid, {"doc": doc, "score": 0.0})
        fused[cid]["score"] += vector_weight * score

    bm25_scores = _min_max_normalize(bm25_scores_raw)
    for doc, score in zip(bm25_docs, bm25_scores):
        cid = doc.metadata.get("chunk_id", id(doc))
        fused.setdefault(cid, {"doc": doc, "score": 0.0})
        fused[cid]["score"] += bm25_weight * score

    # 按融合分数排序，取 top_k
    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    candidates = [item["doc"] for item in ranked[:candidate_k]]

    # ─── cross-encoder 重排序 ───
    try:
        final_docs = await rerank(query, candidates, top_n=top_k)
    except Exception as e:
        # rerank 失败时降级：直接用混合检索的 top_k，保证可用性
        print(f"[WARN] rerank failed, fallback to hybrid: {e}")
        final_docs = candidates[:top_k]

    return final_docs