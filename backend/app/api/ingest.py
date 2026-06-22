#文件上传API
import os

from fastapi import APIRouter, UploadFile, File

from app.config import settings
from app.models.schemas import IngestResponse
from app.core.loader import load_file
from app.core.vision import describe_image
from app.core.splitter import split_documents
from app.core.embedder import get_or_create_collection


router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """上传文档 → 解析 → 图片理解 → 分块 → embedding → 入库"""
    try:
        # 保存文件
        os.makedirs(settings.data_dir, exist_ok=True)
        file_path = os.path.join(settings.data_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # 加载文档
        documents, image_paths = load_file(file_path)

        # Vision：给图片 Document 填描述
        images_processed = 0
        for doc in documents:
            if doc.metadata.get("type") == "image":
                img_path = doc.metadata.get("image_path", "")
                if img_path and os.path.exists(img_path):
                    doc.page_content = f"[图片描述] {await describe_image(img_path)}"
                    images_processed += 1

        # 分块
        chunks = split_documents(documents)

        # 过滤空内容
        chunks = [c for c in chunks if c.page_content.strip()]
        if not chunks:
            return IngestResponse(
                status="error",
                file_name=file.filename,
                message="文档没有可提取的内容",
            )

        # 向量化 + 入库
        vectorstore = get_or_create_collection()
        vectorstore.add_documents(chunks)

        # 刷新 BM25 内存索引（新文档才能被关键词检索到）
        from app.core.bm25_retriever import reset_bm25_index
        reset_bm25_index()

        return IngestResponse(
            status="ok",
            file_name=file.filename,
            chunks_created=len(chunks),
            images_processed=images_processed,
        )

    except Exception as e:
        return IngestResponse(
            status="error",
            file_name=file.filename,
            message=str(e),
        )