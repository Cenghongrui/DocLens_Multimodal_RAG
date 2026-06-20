from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

def create_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # 优先级：段落 → 行 → 句子 → 空格 → 字符
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

def split_documents(documents: List[Document]) -> List[Document]:
    """
    文本 Document 切块，图片 Document 原样保留
    """
    splitter = create_splitter()
    result = []

    for doc in documents:
        if doc.metadata.get("type") == "image":
            result.append(doc)   # 图片不切
        else:
            chunks = splitter.split_documents([doc])
            result.extend(chunks)

    for i, chunk in enumerate(result):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        chunk.metadata["chunk_id"] = f"{source}_p{page}_{i}"

    return result