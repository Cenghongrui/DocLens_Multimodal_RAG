from pydantic import BaseModel
from typing import List, Optional


# ─── 请求 ───
class ChatRequest(BaseModel):
    """POST /api/chat 的请求体"""
    message: str


# ─── 响应 ───
class SourceItem(BaseModel):
    """单个引用来源"""
    content: str
    source: str
    page: int = 0
    type: str = "text"          # "text" 或 "image"
    image_url: Optional[str] = None


class ChatResponse(BaseModel):
    """POST /api/chat 的响应体"""
    answer: str
    sources: List[SourceItem] = []


class IngestResponse(BaseModel):
    """POST /api/ingest 的响应体"""
    status: str                 # "ok" 或 "error"
    file_name: str
    chunks_created: int = 0
    images_processed: int = 0
    message: str = ""


class DocumentItem(BaseModel):
    """GET /api/documents 返回的单条文档"""
    name: str
    chunks: int
    images: int
    ingested_at: str = ""


class DocumentListResponse(BaseModel):
    """GET /api/documents 的响应体"""
    documents: List[DocumentItem] = []