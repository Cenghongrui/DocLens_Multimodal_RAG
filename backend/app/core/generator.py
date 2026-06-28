"""DeepSeek 回答生成：基于检索结果的 System Prompt + LLM 调用。"""
from typing import List
from openai import AsyncOpenAI
from langchain_core.documents import Document
from app.config import settings

_llm = AsyncOpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)


def _build_prompt(docs: List[Document]) -> str:
    """组装 system prompt：角色定义 + 文档上下文 + 回答约束。"""
    parts = []
    for i, d in enumerate(docs):
        src = d.metadata.get("source", "未知")
        pg = d.metadata.get("page", "N/A")
        parts.append(f"[来源 {i+1}]（{src}，第 {pg} 页）：\n{d.page_content}")
    context = "\n\n---\n\n".join(parts)

    return f"""你是一个知识库助手。你的回答必须基于以下检索到的文档内容。

规则：
1. 优先使用【文档内容】回答问题
2. 引用时注明来源编号，如"根据 [来源 1]..."
3. 文档内容不足以回答时，直接说"根据已有文档，我无法回答这个问题"，不要编造
4. 用用户提问的语言回答
5. 如果用户语言与文档语言不一致，先翻译用户提问，再检索，最后将答案翻译回来

【文档内容】
{context}"""


async def generate(
    query: str,
    retrieved_docs: List[Document],
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
) -> str:
    """调 DeepSeek 生成回答。

    Args:
        query: 用户问题。
        retrieved_docs: 检索结果。
        model: 覆盖模型名。
        temperature: 采样温度（默认 0.05）。
        max_tokens: 最大输出长度。
    """
    resp = await _llm.chat.completions.create(
        model=model or settings.llm_model,
        messages=[
            {"role": "system", "content": _build_prompt(retrieved_docs)},
            {"role": "user", "content": query},
        ],
        temperature=0.05 if temperature is None else temperature,
        max_tokens=2000 if max_tokens is None else max_tokens,
    )
    return resp.choices[0].message.content
