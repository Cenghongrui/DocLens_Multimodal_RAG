from typing import List
from openai import AsyncOpenAI

from langchain_core.documents import Document

from app.config import settings

llm_client = AsyncOpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
)


def build_system_prompt(retrieved_docs: List[Document]) -> str:
    """组装 system prompt：角色定义 + 规则 + 文档内容"""

    # ─── 组装检索到的文档内容 ───
    parts = []
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "N/A")
        doc_type = doc.metadata.get("type", "text")

        icon = "📷" if doc_type == "image" else "📄"
        parts.append(
            f"[来源 {i+1}] {icon}（{source}，第 {page} 页）：\n{doc.page_content}"
        )

    context = "\n\n---\n\n".join(parts)

    return f"""你是一个知识库助手。你的回答必须基于以下检索到的文档内容。

【规则】
1. 优先使用【文档内容】回答问题
2. 引用时注明来源编号，如"根据 [来源 1]..."
3. 文档内容不足以回答时，直接说"根据已有文档，我无法回答这个问题"，不要编造
4. 用中文回答，简洁准确

【文档内容】
{context}
"""


async def generate(query: str, retrieved_docs: List[Document]) -> str:
    """调 DeepSeek 生成回答"""
    system_prompt = build_system_prompt(retrieved_docs)

    response = await llm_client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.1,
        max_tokens=2000,
    )
    return response.choices[0].message.content