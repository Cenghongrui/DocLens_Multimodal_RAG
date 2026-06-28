"""文档列表 API。"""
from fastapi import APIRouter
from app.models.schemas import DocumentItem, DocumentListResponse
from app.core.embedder import get_or_create_collection

router = APIRouter()


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """返回已索引的文档列表（按 source 聚合）。"""
    meta = get_or_create_collection().get(include=["metadatas"])["metadatas"]
    if not meta:
        return DocumentListResponse(documents=[])

    index = {}
    for m in meta:
        s = m.get("source", "未知")
        index.setdefault(s, {"chunks": 0, "images": 0})
        index[s]["chunks" if m.get("type") != "image" else "images"] += 1

    return DocumentListResponse(documents=[
        DocumentItem(name=k, chunks=v["chunks"], images=v["images"]) for k, v in index.items()
    ])
