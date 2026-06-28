"""请求/响应模型定义。"""
from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):
    message: str
    source: Optional[str] = None


class SourceItem(BaseModel):
    content: str
    source: str
    page: int = 0
    type: str = "text"
    image_url: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = []


class IngestResponse(BaseModel):
    status: str
    file_name: str
    chunks_created: int = 0
    images_processed: int = 0
    message: str = ""


class DocumentItem(BaseModel):
    name: str
    chunks: int
    images: int
    ingested_at: str = ""


class DocumentListResponse(BaseModel):
    documents: List[DocumentItem] = []
