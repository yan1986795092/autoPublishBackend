"""MCP Server 模块 — 多 MCP Server 统一管理

提供多个免费开源的 MCP 工具服务，统一通过 AgentTool 体系暴露给 Agent 系统。

核心组件：
    MCPClient           — 多 Server 客户端，统一生命周期管理
    get_mcp_client()    — 全局单例获取（延迟初始化，自动加载所有 Server）
    shutdown_mcp_client() — 全局单例关闭

可用 MCP Server：
    1. web_search_server   — 网页/新闻搜索 + 页面抓取（DuckDuckGo 免费，无需 Key）
    2. brave_search_server  — Brave Search 隐私搜索（需 BRAVE_API_KEY）
    3. jina_reader_server   — 网页 Markdown 化抓取（完全免费，无需 Key）
    4. tavily_search_server — AI 优化新闻/博客搜索（需 TAVILY_API_KEY）
    5. pexels_server       — 免费高清图片搜索（需 PEXELS_API_KEY）
    6. github_server       — GitHub 代码/Issues/PR 搜索（可选 GITHUB_TOKEN）

启用方式：设置对应环境变量即可（详见 .env.example）
"""
from .client import (
    MCPClient,
    MCPServerConfig,
    get_mcp_client,
    shutdown_mcp_client,
    _build_default_configs,
)

# 向后兼容别名
MCPWebSearchClient = MCPClient

__all__ = [
    "MCPClient",
    "MCPServerConfig",
    "get_mcp_client",
    "shutdown_mcp_client",
    "_build_default_configs",
]
