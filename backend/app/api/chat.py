#聊天API
from pathlib import Path

from fastapi import APIRouter

from app.models.schemas import ChatRequest, ChatResponse, SourceItem
from app.core.retriever import retrieve
from app.core.generator import generate

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """对话：检索 + 生成"""

    # 检索
    retrieved_docs = await retrieve(request.message)

    # 生成
    answer = await generate(request.message, retrieved_docs)

    # 组装来源引用
    sources = []
    for doc in retrieved_docs:
        item = SourceItem(
            content=doc.page_content[:200],
            source=doc.metadata.get("source", "未知"),
            page=doc.metadata.get("page", 0),
            type=doc.metadata.get("type", "text"),
        )

        if doc.metadata.get("type") == "image":
            image_path = doc.metadata.get("image_path", "")
            if image_path:
                item.image_url = f"/images/{Path(image_path).name}"

        sources.append(item)

    return ChatResponse(answer=answer, sources=sources)