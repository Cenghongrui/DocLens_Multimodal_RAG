"""混合检索：向量检索（语义）+ BM25（关键词），加权融合。"""

from typing import List

from langchain_core.documents import Document

from app.config import settings
from app.core.embedder import get_or_create_collection
from app.core.bm25_retriever import get_bm25_retriever
from app.core.reranker import rerank
from app.core.logger import get_logger


def _min_max_normalize(scores: list[float]) -> list[float]:
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
    vector_weight: float = 0.65,
    bm25_weight: float = 0.35,
    candidate_k: int = 15,
    hyde_enabled: bool = None,
    trace_id: str = "",
) -> List[Document]:
    """混合检索主函数。

    Args:
        trace_id: 多智能体链路追踪 ID
    """
    log = get_logger(trace_id)
    if top_k is None:
        top_k = settings.top_k

    # ─── HyDE ───
    search_query = query
    use_hyde = settings.hyde_enabled if hyde_enabled is None else hyde_enabled
    if use_hyde:
        from app.core.query_router import should_use_hyde
        should_use, route_reason = await should_use_hyde(
            query,
            threshold=settings.hyde_route_threshold,
        )
        if should_use:
            try:
                from app.core.query_transform import hyde_transform
                search_query = await hyde_transform(query)
                log.info("[HyDE] ✓ %s → %s...", query[:40], search_query[:50])
            except Exception as e:
                log.warning("[HyDE] transform failed, use raw query: %s", e)
                search_query = query
        else:
            log.info("[HyDE] ✗ skipped: %s", route_reason)
    else:
        source_label = "globally disabled" if hyde_enabled is None else "per-query disabled"
        log.info("[HyDE] ⊘ %s", source_label)

    # ─── 向量检索 ───
    vectorstore = get_or_create_collection()
    filter_dict = {"source": source} if source else None

    vec_pairs = await vectorstore.asimilarity_search_with_score(
        query=search_query, k=candidate_k, filter=filter_dict,
    )
    vec_scores_raw = [1.0 / (1.0 + dist) for _, dist in vec_pairs]

    # ─── BM25 检索 ───
    bm25 = await get_bm25_retriever()
    bm25_docs, bm25_scores_raw = bm25.search(search_query, top_k=candidate_k, return_scores=True)

    # ─── 融合 ───
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

    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    candidates = [item["doc"] for item in ranked[:candidate_k]]

    # ─── cross-encoder 重排序 ───
    try:
        final_docs = await rerank(query, candidates, top_n=top_k)
    except Exception as e:
        log.warning("rerank failed, fallback to hybrid: %s", e)
        final_docs = candidates[:top_k]

    return final_docs
