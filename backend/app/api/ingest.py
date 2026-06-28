"""文档上传与入库 API。"""
import os
from fastapi import APIRouter, UploadFile, File
from app.config import settings
from app.models.schemas import IngestResponse
from app.core.loader import load_file
from app.core.vision import describe_image
from app.core.splitter import split_documents
from app.core.embedder import get_or_create_collection
from app.core.bm25_retriever import reset_bm25_index
from app.core.retriever import clear_cache

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """上传 → 解析 → 图片理解 → 分片 → Embedding → 入库 + 刷新缓存。"""
    try:
        os.makedirs(settings.data_dir, exist_ok=True)
        file_path = os.path.join(settings.data_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        documents, _ = load_file(file_path)

        images = 0
        for doc in documents:
            if doc.metadata.get("type") == "image":
                img = doc.metadata.get("image_path", "")
                if img and os.path.exists(img):
                    doc.page_content = f"[图片] {await describe_image(img)}"
                    images += 1

        chunks = split_documents(documents)
        chunks = [c for c in chunks if c.page_content.strip()]
        if not chunks:
            return IngestResponse(status="error", file_name=file.filename, message="无可提取内容")

        get_or_create_collection().add_documents(chunks)
        reset_bm25_index()
        clear_cache()

        return IngestResponse(status="ok", file_name=file.filename, chunks_created=len(chunks), images_processed=images)

    except Exception as e:
        return IngestResponse(status="error", file_name=file.filename, message=str(e))
