"""ImageAgent — 配图/封面生成 (DALL-E 3 / Stable Diffusion)"""

import os
import logging
import httpx
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


async def handle_image(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    topic = task_input.get("topic", task_input.get("query", ""))
    style = task_input.get("style", "vivid")
    size = task_input.get("size", "1024x1024")
    provider = os.getenv("IMAGE_PROVIDER", "")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        "https://api.openai.com/v1/images/generations",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": "dall-e-3", "prompt": f"Illustration for: {topic}", "n": 1, "size": size, "style": style},
                    )
                    data = resp.json()
                    image_url = data.get("data", [{}])[0].get("url", "")
                    if image_url:
                        return {
                            "status": "completed", "agent_type": "ImageAgent",
                            "output": {"image_url": image_url, "alt": f"Illustration for {topic}", "provider": "openai"},
                            "tokens_used": 0,
                            "thinking_steps": [{"agent": "ImageAgent", "action": "配图生成完成", "reasoning": f"DALL-E 3 为「{topic}」生成配图", "output_summary": "DALL-E 3 生成成功", "status": "done"}],
                        }
            except Exception as e:
                logger.warning(f"DALL-E failed: {e}")

    if provider == "stability":
        api_key = os.getenv("STABILITY_API_KEY")
        if api_key:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"text_prompts": [{"text": f"Illustration: {topic}", "weight": 1}], "cfg_scale": 7, "height": 1024, "width": 1024, "samples": 1},
                    )
                    data = resp.json()
                    image_b64 = data.get("artifacts", [{}])[0].get("base64")
                    if image_b64:
                        return {
                            "status": "completed", "agent_type": "ImageAgent",
                            "output": {"image_url": f"data:image/png;base64,{image_b64}", "alt": f"Illustration for {topic}", "provider": "stability"},
                            "tokens_used": 0,
                            "thinking_steps": [{"agent": "ImageAgent", "action": "配图生成完成", "reasoning": f"SDXL 为「{topic}」生成配图", "output_summary": "Stable Diffusion 生成成功", "status": "done"}],
                        }
            except Exception as e:
                logger.warning(f"Stability failed: {e}")

    return {
        "status": "completed", "agent_type": "ImageAgent",
        "output": {"image_url": "", "alt": f"Illustration for {topic}", "mock": True, "note": "未配置图片生成 API Key"},
        "tokens_used": 0,
        "thinking_steps": [{"agent": "ImageAgent", "action": "配图生成完成", "reasoning": f"为「{topic}」生成配图", "output_summary": "未配置图片 API，降级处理", "status": "done"}],
    }
