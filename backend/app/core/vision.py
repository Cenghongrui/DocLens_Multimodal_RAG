"""Qwen-VL 图片描述。"""
import base64
from openai import AsyncOpenAI
from app.config import settings

_client = AsyncOpenAI(api_key=settings.qwen_api_key, base_url=settings.qwen_base_url)


async def describe_image(image_path: str) -> str:
    """调用 Qwen-VL 生成图片的中文描述。"""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")

    resp = await _client.chat.completions.create(
        model=settings.vision_model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": "请详细描述这张图片的内容，包括数据结构、架构组件等。用中文回答。"},
            ],
        }],
        max_tokens=1000,
    )
    return resp.choices[0].message.content
