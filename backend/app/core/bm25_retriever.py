"""BM25 关键词检索器。
  线程安全：多智能体场景下并发调用不会竞争。
"""

import asyncio
import jieba
from rank_bm25 import BM25Okapi

from app.core.embedder import get_or_create_collection


def _tokenize(text: str) -> list[str]:
    """中文分词。BM25 是按词算分的，中文必须先切词。"""
    return [w for w in jieba.cut(text) if w.strip()]


class BM25Retriever:
    """基于 rank_bm25 的内存关键词检索器。

    每次实例化时从 ChromaDB 把所有 chunk 拉出来重建索引。
    """

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.docs: list = []
        self.tokenized_corpus: list[list[str]] = []

    def build_index(self):
        """从 ChromaDB 拉全部 chunk，构建 BM25 索引。"""
        from langchain_core.documents import Document

        vectorstore = get_or_create_collection()
        results = vectorstore.get(include=["documents", "metadatas"])

        self.docs = []
        self.tokenized_corpus = []
        for text, meta in zip(results["documents"], results["metadatas"]):
            if not text or not text.strip():
                continue
            self.docs.append(Document(page_content=text, metadata=meta))
            self.tokenized_corpus.append(_tokenize(text))

        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    def search(self, query: str, top_k: int = 5, return_scores: bool = False):
        """返回 top_k 个最相关的 Document。"""
        if not self.bm25:
            return ([], []) if return_scores else []
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_idx = ranked[:top_k]
        docs = [self.docs[i] for i in top_idx]
        top_scores = [float(scores[i]) for i in top_idx]
        return (docs, top_scores) if return_scores else docs


# ─── 带锁的全局单例（多智能体安全） ───

_bm25_retriever: BM25Retriever | None = None
_bm25_lock: asyncio.Lock = asyncio.Lock()
_bm25_ready: asyncio.Event = asyncio.Event()


async def get_bm25_retriever() -> BM25Retriever:
    """获取 BM25 检索器单例。第一次调用建索引，之后复用。
    多智能体并发安全：用 asyncio.Lock 保护。
    """
    global _bm25_retriever

    # 快速路径：已就绪，无需锁
    if _bm25_ready.is_set():
        return _bm25_retriever

    async with _bm25_lock:
        # 二次检查：可能在上一个等待者已建好
        if _bm25_retriever is None:
            _bm25_retriever = BM25Retriever()
            # 同步建索引（BM25 是 CPU 操作，不能 await）
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _bm25_retriever.build_index)
        _bm25_ready.set()
    return _bm25_retriever


def reset_bm25_index():
    """清空缓存，下次 get 时重建。ingest 新文档后调用。"""
    global _bm25_retriever
    _bm25_retriever = None
    _bm25_ready.clear()
