"""
检索入口——混合检索。
带 TTL 缓存：多智能体重复相同 query 时复用检索结果，省 API 调用。
"""

import time
import hashlib
import json
from typing import List, Optional

from langchain_core.documents import Document

from app.config import settings
from app.core.hybrid_retriever import hybrid_retrieve

# ─── 简单 TTL 缓存（内存） ───

_cache: dict[str, tuple[float, list]] = {}  # key → (expire_at, docs)
CACHE_TTL = 60  # 缓存有效期（秒）


def _make_cache_key(query: str, top_k, source, **overrides) -> str:
    raw = json.dumps([query, top_k, source, overrides], sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw.encode()).hexdigest()


async def retrieve(
    query: str,
    top_k: int = None,
    source: str = None,
    hyde_enabled: bool = None,
    vector_weight: float = None,
    bm25_weight: float = None,
    candidate_k: int = None,
    trace_id: str = "",
    # ——— 缓存控制 ———
    skip_cache: bool = False,
) -> List[Document]:
    """检索入口——混合检索。

    多智能体场景下，同一 query 在短时间内重复检索会命中缓存，
    无需重复调 embedding API 和 reranker。

    Args:
        query: 用户问题
        top_k: 返回结果数（默认 config.top_k）
        source: 按文档名过滤
        hyde_enabled: 覆盖 config.hyde_enabled
        vector_weight: 向量检索融合权重
        bm25_weight: BM25 权重
        candidate_k: 召回候选数
        trace_id: 链路追踪 ID
        skip_cache: 跳过缓存
    """
    kwargs = {}
    if hyde_enabled is not None:
        kwargs["hyde_enabled"] = hyde_enabled
    if vector_weight is not None:
        kwargs["vector_weight"] = vector_weight
    if bm25_weight is not None:
        kwargs["bm25_weight"] = bm25_weight
    if candidate_k is not None:
        kwargs["candidate_k"] = candidate_k

    cache_key = _make_cache_key(query, top_k, source, **kwargs)

    # 命中缓存
    if not skip_cache and cache_key in _cache:
        expire_at, docs = _cache[cache_key]
        if time.time() < expire_at:
            return docs

    # 未命中 → 检索
    docs = await hybrid_retrieve(
        query, top_k=top_k, source=source, trace_id=trace_id, **kwargs
    )

    # 写入缓存
    _cache[cache_key] = (time.time() + CACHE_TTL, docs)
    return docs


def clear_cache():
    """清空检索缓存。ingest 新文档后可调用。"""
    _cache.clear()
