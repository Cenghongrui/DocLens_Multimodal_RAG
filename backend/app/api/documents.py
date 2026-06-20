#文档列表API
from fastapi import APIRouter

from app.models.schemas import DocumentItem, DocumentListResponse
from app.core.embedder import get_or_create_collection

router = APIRouter()


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """返回已索引的文档列表（按 source 聚合）"""
    vectorstore = get_or_create_collection()
    results = vectorstore.get(include=["metadatas"])

    if not results["metadatas"]:
        return DocumentListResponse(documents=[])

    # 按 source 聚合
    doc_map = {}
    for meta in results["metadatas"]:
        source = meta.get("source", "未知文档")
        if source not in doc_map:
            doc_map[source] = {"chunks": 0, "images": 0}

        if meta.get("type") == "image":
            doc_map[source]["images"] += 1
        else:
            doc_map[source]["chunks"] += 1

    documents = [
        DocumentItem(name=name, chunks=stats["chunks"], images=stats["images"])
        for name, stats in doc_map.items()
    ]

    return DocumentListResponse(documents=documents)