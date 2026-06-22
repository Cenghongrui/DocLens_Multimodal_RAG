"""BM25 关键词检索器。
"""

import jieba
from rank_bm25 import BM25Okapi

from app.core.embedder import get_or_create_collection


def _tokenize(text: str) -> list[str]:
    """中文分词。BM25 是按词算分的，中文必须先切词。"""
    return [w for w in jieba.cut(text) if w.strip()]


class BM25Retriever:
    """基于 rank_bm25 的内存关键词检索器。
s
    每次实例化时从 ChromaDB 把所有 chunk 拉出来重建索引。
    """

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.docs: list = []          # 存原始 Document，下标和 bm25 内部对齐
        self.tokenized_corpus: list[list[str]] = []

    def build_index(self):
        """从 ChromaDB 拉全部 chunk，构建 BM25 索引。"""
        vectorstore = get_or_create_collection()
        results = vectorstore.get(include=["documents", "metadatas"])

        self.docs = []
        self.tokenized_corpus = []
        for text, meta in zip(results["documents"], results["metadatas"]):
            if not text or not text.strip():
                continue
            # 用 LangChain Document 包一下，方便后续统一处理
            from langchain_core.documents import Document
            self.docs.append(Document(page_content=text, metadata=meta))
            self.tokenized_corpus.append(_tokenize(text))

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 5, return_scores: bool = False):
        """返回 top_k 个最相关的 Document。
        当 return_scores=True 时返回 (docs, scores) 元组。
        """
        if not self.bm25:
            return ([], []) if return_scores else []
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        # 按分数降序，取 top_k 的下标
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_idx = ranked[:top_k]
        docs = [self.docs[i] for i in top_idx]
        top_scores = [float(scores[i]) for i in top_idx]
        return (docs, top_scores) if return_scores else docs


# 模块级单例：整个进程只建一次索引，重复查询复用
_bm25_retriever: BM25Retriever | None = None


def get_bm25_retriever() -> BM25Retriever:
    """获取 BM25 检索器单例。第一次调用建索引，之后复用。"""
    global _bm25_retriever
    if _bm25_retriever is None:
        _bm25_retriever = BM25Retriever()
        _bm25_retriever.build_index()
    return _bm25_retriever


def reset_bm25_index():
    """清空缓存，下次 get 时重建。ingest 新文档后调用。"""
    global _bm25_retriever
    _bm25_retriever = None