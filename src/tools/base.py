"""工具基类 — Agent 可调用的工具抽象"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    success: bool
    output: Any = None
    error: str = ""
    metadata: dict | None = None


class BaseTool(ABC):
    """所有工具的基类"""

    name: str = "base_tool"
    description: str = ""

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """执行工具，返回 ToolResult"""
        ...

    def to_openai_function(self) -> dict:
        """转换为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
            },
        }
