"""独立测试 rerank 接口，确认格式正确后再集成。"""
import asyncio
import sys
import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings


async def main():
    resp = await httpx.AsyncClient(timeout=30).post(
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/"
        "text-rerank/text-rerank",
        headers={"Authorization": f"Bearer {settings.qwen_api_key}"},
        json={
            "model": "gte-rerank-v2",
            "input": {
                "query": "什么是深度学习",
                "documents": [
                    "深度学习是机器学习的一个分支，使用多层神经网络。",
                    "今天天气不错，适合出去玩。",
                    "神经网络通过反向传播算法训练权重。",
                ],
            },
            "parameters": {"return_documents": False, "top_n": 3},
        },
    )
    print("状态码:", resp.status_code)
    print("返回:", resp.text)


if __name__ == "__main__":
    asyncio.run(main())