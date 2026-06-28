"""语义分片：Chonkie SemanticChunker + Qwen embedding。"""
from typing import List
import numpy as np
from langchain_core.documents import Document
from chonkie import SemanticChunker
from chonkie.embeddings.base import BaseEmbeddings
from app.config import settings
from app.core.embedder import _sync as _embed_client


class _QwenEmbeddings(BaseEmbeddings):
    """将 Qwen text-embedding-v3 包装为 Chonkie 可用的 BaseEmbeddings。"""

    def __init__(self):
        self._client = _embed_client
        self._model = settings.embedding_model
        self._dim = settings.embedding_dimension
        self._tokenizer = None
        self._batch_size = min(settings.embedding_batch_size, 10)

    @property
    def dimension(self) -> int:
        return self._dim

    def get_tokenizer(self):
        if self._tokenizer is None:
            from chonkie.tokenizer import AutoTokenizer
            self._tokenizer = AutoTokenizer("gpt2")
        return self._tokenizer

    def embed(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(model=self._model, input=text, dimensions=self._dim)
        return resp.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        all_embs = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            resp = self._client.embeddings.create(model=self._model, input=batch, dimensions=self._dim)
            all_embs.extend(item.embedding for item in resp.data)
        return all_embs

    def similarity(self, u, v):
        u, v = np.array(u), np.array(v)
        return np.float32(np.dot(u, v.T) / (np.linalg.norm(u) * np.linalg.norm(v)))

    async def aembed(self, text: str) -> List[float]:
        return self.embed(text)

    async def aembed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.embed_batch(texts)


# ─── 全局分片器（单例） ───

_chunker: SemanticChunker | None = None


def _get_chunker() -> SemanticChunker:
    global _chunker
    if _chunker is None:
        _chunker = SemanticChunker(
            embedding_model=_QwenEmbeddings(),
            chunk_size=settings.chunk_size,
            threshold=0.5,
            similarity_window=3,
            min_sentences_per_chunk=1,
            min_characters_per_sentence=8,
            delim=["。", "！", "？", "\n\n", "\n", ". ", "! ", "? ", "；"],
            include_delim="prev",
        )
    return _chunker


def split_documents(documents: List[Document]) -> List[Document]:
    """语义分片入口。

    - 文本 Document → SemanticChunker 切分，相邻 chunk 尾部叠加 overlap
    - 图片 Document → 原样保留
    - 每个 chunk 写入 chunk_id 元数据（用于混合检索去重）
    """
    chunker = _get_chunker()
    result = []

    for doc in documents:
        if doc.metadata.get("type") == "image":
            result.append(doc)
            continue

        text = doc.page_content
        if not text.strip():
            continue

        raw = chunker(text)
        if not raw:
            continue

        texts = [c.text for c in raw]
        overlap = settings.chunk_overlap
        overlapped = []
        for i, t in enumerate(texts):
            if overlap > 0 and i < len(texts) - 1:
                t = t + "\n" + texts[i + 1][:overlap]
            overlapped.append(t)

        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)
        for i, t in enumerate(overlapped):
            d = Document(page_content=t, metadata={**doc.metadata})
            d.metadata["chunk_id"] = f"{source}_p{page}_{i}"
            result.append(d)

    return result
