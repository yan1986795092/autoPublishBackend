"""MCP Client — 多 MCP Server 统一管理

重构自 MCPWebSearchClient，支持同时管理多个 MCP Server：
- Search Server（网页/新闻搜索 + 页面抓取）
- Brave Search Server（隐私搜索，AI 优化结果）
- Jina Reader Server（网页 Markdown 化抓取）
- Tavily Search Server（AI 新闻/博客优化搜索）
- Pexels Server（免费高清图片搜索）
- GitHub Server（代码/Issues/PR 搜索）

用法：
    client = MCPClient()
    await client.start()
    tools = client.get_agent_tools()  # → list[AgentTool]
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

_langchain_mcp_adapters_available = True
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    _langchain_mcp_adapters_available = False
    MultiServerMCPClient = None


class MCPServerConfig:
    """单个 MCP Server 的配置"""

    def __init__(
        self,
        name: str,
        module: str | Path | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        transport: str = "stdio",
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        category: str = "search",
        enabled: bool = True,
    ):
        self.name = name
        self.module = module
        self.command = command or sys.executable
        self.args = args or []
        self.transport = transport
        self.env = env or {}
        self.cwd = cwd
        self.category = category
        self.enabled = enabled


class MCPClient:
    """MCP 多 Server 客户端

    通过配置驱动，自动加载所有已启用的 MCP Server。
    每个 Server 的工具统一通过 AgentTool.from_langchain() 适配。

    支持 stdio 和 streamable-http 两种 transport：
    - stdio: Server 作为子进程，通过 stdin/stdout 通信（推荐，本地开发）
    - streamable-http: Server 作为独立 HTTP 服务（生产环境推荐）
    """

    def __init__(self, configs: list[MCPServerConfig] | None = None):
        self._configs = configs or []
        self._client: MultiServerMCPClient | None = None
        self._tools: list[BaseTool] = []
        self._agent_tools: list = []
        self._started = False
        self._last_health_check: float = 0.0
        self._health_check_interval: float = 30.0
        self._reconnect_attempts: int = 0
        self._max_reconnect_attempts: int = 3

        if not _langchain_mcp_adapters_available:
            logger.warning(
                "langchain-mcp-adapters not installed. "
                "MCP will be disabled. Install with: pip install langchain-mcp-adapters"
            )

    def add_server(self, config: MCPServerConfig) -> None:
        """添加一个 MCP Server 配置"""
        self._configs.append(config)

    @property
    def is_started(self) -> bool:
        return self._started and bool(self._tools)

    async def start(self) -> bool:
        """启动所有已配置的 MCP Server"""
        if self._started:
            return True

        if not _langchain_mcp_adapters_available:
            return False

        enabled_configs = [c for c in self._configs if c.enabled]
        if not enabled_configs:
            logger.info("No MCP servers configured, skipping start")
            return True

        try:
            # 构建 MultiServerMCPClient 配置
            server_configs = {}
            backend_root = str(Path(__file__).parent.parent.parent.absolute())

            for config in enabled_configs:
                # 解析 module path
                if config.module:
                    module_path = Path(config.module)
                    if not module_path.is_absolute():
                        module_path = Path(__file__).parent / config.module
                else:
                    module_path = Path(__file__).parent / f"{config.name}_server.py"

                # 构建 server config dict
                server_config: dict[str, Any] = {
                    "command": config.command,
                    "args": config.args or [str(module_path), "stdio"],
                    "transport": config.transport,
                }

                # 环境变量
                env = dict(config.env) if config.env else {}
                for key, val in env.items():
                    if val == "from_env":
                        env[key] = os.getenv(key, "")
                server_config["env"] = env if env else None

                # 工作目录
                server_config["cwd"] = config.cwd or backend_root

                server_configs[config.name] = server_config

            logger.info(
                "Starting MCP servers: %s",
                ", ".join(f"{c.name}({c.category})" for c in enabled_configs),
            )

            self._client = MultiServerMCPClient(server_configs)
            raw_tools = await self._client.get_tools()
            self._tools = list(raw_tools)
            self._agent_tools = self._to_agent_tools(self._tools)

            self._started = True
            self._reconnect_attempts = 0
            self._last_health_check = time.monotonic()

            logger.info(
                "MCP started: %d tools total — %s",
                len(self._tools),
                ", ".join(f"{t.name}" for t in self._agent_tools),
            )
            return True

        except Exception as e:
            logger.error("Failed to start MCP: %s", e, exc_info=True)
            await self._cleanup()
            return False

    async def stop(self) -> None:
        """停止所有 MCP Server"""
        await self._cleanup()
        self._started = False

    async def _cleanup(self) -> None:
        if self._client:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
        self._tools = []
        self._agent_tools = []

    async def is_healthy(self) -> bool:
        now = time.monotonic()
        if now - self._last_health_check < self._health_check_interval:
            return self._started
        self._last_health_check = now
        return self._started and bool(self._tools)

    async def ensure_alive(self) -> bool:
        if await self.is_healthy():
            return True
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error("MCP reconnect exhausted (%d attempts)", self._reconnect_attempts)
            return False
        self._reconnect_attempts += 1
        logger.info("MCP reconnect attempt %d/%d...", self._reconnect_attempts, self._max_reconnect_attempts)
        await self._cleanup()
        success = await self.start()
        if success:
            logger.info("MCP reconnected successfully")
        return success

    def get_tools(self) -> list[BaseTool]:
        """返回原始 LangChain BaseTool 列表（向后兼容）"""
        return self._tools

    def get_agent_tools(self) -> list:
        """返回 AgentTool 列表（直接用于 ToolRegistry）"""
        return self._agent_tools

    def get_tools_by_category(self, category: str) -> list:
        """按分类获取工具"""
        return [t for t in self._agent_tools if getattr(t, "category", "") == category]

    def _to_agent_tools(self, tools: list[BaseTool]) -> list:
        from ..agent.tools import AgentTool, AgentToolMeta

        result = []
        for t in tools:
            # 根据工具名推断 category
            name = t.name.lower()
            if any(kw in name for kw in ["search", "web", "news", "fetch", "tavily", "brave"]):
                category = "search"
            elif any(kw in name for kw in ["image", "photo", "pexels", "unsplash", "picture"]):
                category = "media"
            elif any(kw in name for kw in ["github", "git", "repo", "pr", "issue"]):
                category = "code"
            else:
                category = "search"

            meta = AgentToolMeta(
                category=category,
                builtin=False,
                visible=True,
            )
            agent_tool = AgentTool.from_langchain(t, meta=meta)
            result.append(agent_tool)
        return result


# ──────────────────────────────────────────
# 全局单例（延迟初始化）
# ──────────────────────────────────────────

_mcp_client: MCPClient | None = None
_mcp_lock = asyncio.Lock()


def _build_default_configs() -> list[MCPServerConfig]:
    """根据环境变量构建默认 MCP Server 配置

    通过设置环境变量来启用/禁用各个 MCP Server。
    """
    configs: list[MCPServerConfig] = []
    backend_root = str(Path(__file__).parent.parent.parent.absolute())

    # ── 1. Open WebSearch Server（网页/新闻搜索 + 页面抓取，DuckDuckGo 免费）
    if os.getenv("MCP_ENABLE_SEARCH", "true").lower() != "false":
        configs.append(MCPServerConfig(
            name="websearch",
            module=Path(__file__).parent / "web_search_server.py",
            transport="stdio",
            env={
                "BING_API_KEY": "from_env",
                "SERPAPI_API_KEY": "from_env",
                "HTTP_PROXY": "from_env",
                "HTTPS_PROXY": "from_env",
            },
            cwd=backend_root,
            category="search",
            enabled=True,
        ))

    # ── 2. Brave Search（隐私优先搜索，AI 友好）
    if os.getenv("MCP_ENABLE_BRAVE", "false").lower() != "false":
        configs.append(MCPServerConfig(
            name="brave_search",
            module=Path(__file__).parent / "brave_search_server.py",
            transport="stdio",
            env={
                "BRAVE_API_KEY": "from_env",
            },
            cwd=backend_root,
            category="search",
            enabled=bool(os.getenv("BRAVE_API_KEY")),
        ))

    # ── 3. Jina Reader（网页 Markdown 化抓取，完全免费）
    if os.getenv("MCP_ENABLE_JINA", "true").lower() != "false":
        configs.append(MCPServerConfig(
            name="jina_reader",
            module=Path(__file__).parent / "jina_reader_server.py",
            transport="stdio",
            env={},
            cwd=backend_root,
            category="search",
            enabled=True,
        ))

    # ── 4. Tavily Search（AI 优化新闻/博客搜索）
    if os.getenv("MCP_ENABLE_TAVILY", "false").lower() != "false":
        configs.append(MCPServerConfig(
            name="tavily_search",
            module=Path(__file__).parent / "tavily_search_server.py",
            transport="stdio",
            env={
                "TAVILY_API_KEY": "from_env",
            },
            cwd=backend_root,
            category="search",
            enabled=bool(os.getenv("TAVILY_API_KEY")),
        ))

    # ── 5. Pexels（免费高清图片搜索）
    if os.getenv("MCP_ENABLE_PEXELS", "false").lower() != "false":
        configs.append(MCPServerConfig(
            name="pexels",
            module=Path(__file__).parent / "pexels_server.py",
            transport="stdio",
            env={
                "PEXELS_API_KEY": "from_env",
            },
            cwd=backend_root,
            category="media",
            enabled=bool(os.getenv("PEXELS_API_KEY")),
        ))

    # ── 6. GitHub（代码/Issues/PR 搜索）
    if os.getenv("MCP_ENABLE_GITHUB", "false").lower() != "false":
        configs.append(MCPServerConfig(
            name="github",
            module=Path(__file__).parent / "github_server.py",
            transport="stdio",
            env={
                "GITHUB_TOKEN": "from_env",
            },
            cwd=backend_root,
            category="code",
            enabled=bool(os.getenv("GITHUB_TOKEN")),
        ))

    return configs


async def get_mcp_client() -> MCPClient:
    """获取全局 MCP Client（单例，自动配置所有 Server）"""
    global _mcp_client
    async with _mcp_lock:
        if _mcp_client is None:
            _mcp_client = MCPClient(configs=_build_default_configs())
            await _mcp_client.start()
        return _mcp_client


async def shutdown_mcp_client() -> None:
    """关闭全局 MCP Client"""
    global _mcp_client
    if _mcp_client:
        await _mcp_client.stop()
        _mcp_client = None


# 向后兼容别名
MCPWebSearchClient = MCPClient
