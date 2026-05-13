"""ImageGen — LangChain BaseTool 版本"""
from __future__ import annotations
import json
import httpx
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class ImageGenInput(BaseModel):
    prompt: str = Field(description="图片描述")
    style: str = Field(default="minimalist", description="风格: realistic, cartoon, minimalist, tech")
    size: str = Field(default="1024x1024", description="尺寸: 1024x1024, 1792x1024")


class ImageGenTool(BaseTool):
    name: str = "ImageGen"
    description: str = "根据描述生成配图或封面"
    args_schema: Type[BaseModel] = ImageGenInput
    api_key: str = ""
    provider: str = "mock"

    async def _arun(self, prompt: str, style: str = "minimalist", size: str = "1024x1024", **kwargs) -> str:
        if self.provider == "openai" and self.api_key:
            return await self._openai_generate(prompt, size)
        return json.dumps(self._mock_generate(prompt, style), ensure_ascii=False)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async _arun")

    async def _openai_generate(self, prompt: str, size: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": "dall-e-3", "prompt": prompt, "size": size, "n": 1},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    url = data["data"][0]["url"]
                    return json.dumps({
                        "success": True,
                        "data": {"url": url, "prompt": prompt, "alt": f"Illustration: {prompt[:50]}"},
                        "summary": "图片生成成功",
                    }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
        return json.dumps(self._mock_generate(prompt), ensure_ascii=False)

    def _mock_generate(self, prompt: str, style: str = "minimalist") -> dict:
        return {
            "success": True,
            "data": {
                "url": f"https://placehold.co/1024x1024/3b82f6/ffffff?text={prompt[:20].replace(' ', '+')}",
                "prompt": prompt,
                "style": style,
                "alt": f"Illustration: {prompt[:50]}",
                "mock": True,
            },
            "summary": f"[Mock] 配图生成：{prompt[:30]}...",
        }
