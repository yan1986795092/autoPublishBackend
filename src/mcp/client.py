"""MCP Web Search 客户端 — 管理 MCP Server 进程生命周期，暴露 LangChain Tools

用法：
1. 单例模式（推荐）：agent 启动时启动 MCP server，关闭时停止
2. 每次调用模式：按需启动/停止（开销较大）

MCP Server 支持两种 transport：
- stdio: 进程内通信，适合单例模式
- streamable-http: HTTP 通信，适合独立部署
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)

# 延迟导入，避免启动时必须有 mcp 包
_langchain_mcp_adapters_available = True
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    _langchain_mcp_adapters_available = False
    MultiServerMCPClient = None


class MCPWebSearchClient:
    """MCP Web Search 客户端 — 管理 MCP Server 进程 + LangChain Tools

    使用 stdio transport，MCP Server 作为子进程运行。
    通过 langchain-mcp-adapters 加载 MCP Tools 并转换为 LangChain Tools。
    """

    def __init__(
        self,
        mcp_server_path: str | Path | None = None,
        timeout: float = 60.0,
    ):
        self._mcp_server_path = mcp_server_path
        self._timeout = timeout
        self._client: MultiServerMCPClient | None = None
        self._tools: list[BaseTool] = []
        self._started = False

        if not _langchain_mcp_adapters_available:
            logger.warning(
                "langchain-mcp-adapters not installed. "
                "MCP Web Search will be disabled. Install with: pip install langchain-mcp-adapters"
            )

    # ── 生命周期管理 ──

    async def start(self) -> bool:
        """启动 MCP Server 并加载工具，返回是否成功

        MultiServerMCPClient 会自动通过 stdio transport 启动子进程。
        """
        if self._started:
            return True

        if not _langchain_mcp_adapters_available:
            return False

        try:
            server_path = self._resolve_server_path()
            if not server_path:
                logger.warning(f"MCP server not found at {self._mcp_server_path}")
                return False

            backend_root = str(Path(__file__).parent.parent.parent.absolute())
            logger.info(f"Connecting to MCP WebSearch server: {server_path}")

            # MultiServerMCPClient 自动通过 stdio 启动子进程
            self._client = MultiServerMCPClient(
                {
                    "websearch": {
                        "command": sys.executable,
                        "args": [str(server_path), "stdio"],
                        "transport": "stdio",
                        "env": self._build_env(),
                        "cwd": backend_root,
                    }
                },
            )

            # 加载 MCP Tools → LangChain Tools
            raw_tools = await self._client.get_tools()
            self._tools = [_wrap_mcp_tool(t) for t in raw_tools]

            self._started = True
            logger.info(f"MCP WebSearch connected with {len(self._tools)} tools: {[t.name for t in self._tools]}")
            return True

        except Exception as e:
            logger.error(f"Failed to start MCP WebSearch server: {e}")
            await self._cleanup()
            return False

    async def stop(self) -> None:
        """停止 MCP Server 进程"""
        await self._cleanup()
        self._started = False

    async def _cleanup(self) -> None:
        """清理资源"""
        if self._client:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
        self._tools = []

    # ── 工具访问 ──

    def get_tools(self) -> list[BaseTool]:
        """返回已加载的 LangChain Tools"""
        return self._tools

    def get_tool(self, name: str) -> BaseTool | None:
        for t in self._tools:
            if t.name == name:
                return t
        return None

    @property
    def is_started(self) -> bool:
        return self._started and bool(self._tools)

    # ── 内部 ──

    def _resolve_server_path(self) -> Path | None:
        if self._mcp_server_path:
            return Path(self._mcp_server_path)
        # 默认路径：src/mcp/web_search_server.py
        default = Path(__file__).parent / "web_search_server.py"
        if default.exists():
            return default
        return None

    def _build_env(self) -> dict[str, str] | None:
        """传递必要的环境变量给 MCP Server 子进程"""
        env_keys = [
            "BING_API_KEY",
            "SERPAPI_API_KEY",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "OPENAI_API_KEY",       # 透传给可能的 LLM 调用
            "ZHIPU_API_KEY",
        ]
        env = {}
        for key in env_keys:
            val = os.getenv(key)
            if val:
                env[key] = val
        return env if env else None


# ──────────────────────────────────────────
# LangChain Tool 适配层
# ──────────────────────────────────────────

def _wrap_mcp_tool(mcp_tool: Any) -> BaseTool:
    """将 MCP Tool 包装为标准 LangChain BaseTool

    MCP Tool 的结构：
    - name: str
    - description: str
    - args_schema: pydantic model
    - fn: callable

    LangChain BaseTool 需要：
    - name
    - description
    - _run(args, **kwargs)
    - _arun(args, **kwargs)
    """
    name = getattr(mcp_tool, "name", None) or getattr(mcp_tool, "tool_name", "unknown")
    description = getattr(mcp_tool, "description", "")

    async def _arun(**kwargs) -> str:
        try:
            result = await mcp_tool.ainvoke(kwargs)
            if hasattr(result, "content"):
                return result.content
            if isinstance(result, dict):
                import json
                return json.dumps(result, ensure_ascii=False, indent=2)
            return str(result)
        except Exception as e:
            logger.warning(f"MCP tool {name} failed: {e}")
            return f"工具执行失败：{e}"

    def _run(**kwargs) -> str:
        logger.warning(f"MCP tool {name} called in sync mode, use async")
        return f"工具 {name} 需要异步调用"

    # 使用 langchain_core.tools.tool 装饰器创建
    @tool(name, description=description)
    def wrapped_tool(args: str = "") -> str:  # type: ignore[valid-type]
        return f"工具 {name} 需要通过异步方式调用"

    # 直接设置属性（绕过装饰器）
    wrapped_tool.name = name
    wrapped_tool.description = description
    wrapped_tool._arun = _arun
    wrapped_tool._run = _run

    if hasattr(mcp_tool, "args_schema") and mcp_tool.args_schema:
        wrapped_tool.args_schema = mcp_tool.args_schema

    return wrapped_tool  # type: ignore[return-value]


# ──────────────────────────────────────────
# 全局单例（延迟初始化）
# ──────────────────────────────────────────

_mcp_client: MCPWebSearchClient | None = None
_mcp_lock = asyncio.Lock()


async def get_mcp_client() -> MCPWebSearchClient:
    """获取全局 MCP WebSearch 客户端（单例）"""
    global _mcp_client
    async with _mcp_lock:
        if _mcp_client is None:
            _mcp_client = MCPWebSearchClient()
            await _mcp_client.start()
        return _mcp_client


async def shutdown_mcp_client() -> None:
    """关闭全局 MCP 客户端"""
    global _mcp_client
    if _mcp_client:
        await _mcp_client.stop()
        _mcp_client = None
