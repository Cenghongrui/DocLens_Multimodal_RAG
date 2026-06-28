"""聊天 API：检索 + 生成。"""
import uuid
from pathlib import Path
from fastapi import APIRouter
from app.models.schemas import ChatRequest, ChatResponse, SourceItem
from app.core.retriever import retrieve
from app.core.generator import generate

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """检索 → 生成 → 返回答案与引用来源。"""
    tid = str(uuid.uuid4())[:8]
    docs = await retrieve(request.message, source=request.source, trace_id=tid)
    answer = await generate(request.message, docs)

    sources = []
    for d in docs:
        item = SourceItem(
            content=d.page_content[:200],
            source=d.metadata.get("source", "未知"),
            page=d.metadata.get("page", 0),
            type=d.metadata.get("type", "text"),
        )
        if d.metadata.get("type") == "image":
            img = d.metadata.get("image_path", "")
            if img:
                item.image_url = f"/images/{Path(img).name}"
        sources.append(item)

    return ChatResponse(answer=answer, sources=sources)
