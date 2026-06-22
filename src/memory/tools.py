"""记忆检索工具 — Agent 通过 tool call 主动获取记忆

使用方式（在 handler 中）：
    from src.memory.tools import run_memory_tool
    result = await run_memory_tool("memory_search", query="用户偏好")
"""

from __future__ import annotations
from typing import Any

from langchain_core.tools import StructuredTool

from .manager import memory_manager


async def memory_search(query: str, user_id: str = "",
                        conversation_id: str = "", top_k: int = 3) -> str:
    """语义搜索：查询相关历史记忆

    Agent 在需要参考历史内容、知识或经验时调用此工具。

    Args:
        query: 搜索关键词
        user_id: 用户 ID（自动传入）
        conversation_id: 会话 ID（自动传入）
        top_k: 返回条数

    Returns:
        格式化后的相关记忆文本
    """
    if not memory_manager.semantic.available:
        return "（记忆检索不可用）"

    meta_filter = {}
    if user_id:
        meta_filter["user_id"] = user_id
    if conversation_id:
        meta_filter["conversation_id"] = conversation_id

    result = memory_manager.semantic.retrieve_formatted(
        query=query, namespace=None, top_k=top_k,
        max_chars_per_item=400, metadata_filter=meta_filter or None,
    )
    return result or "（未找到相关记忆）"


async def user_preferences(user_id: str = "", conversation_id: str = "") -> str:
    """获取当前用户偏好

    Agent 在需要了解用户风格、平台偏好、常用语调时调用。

    Returns:
        用户偏好描述文本
    """
    if not user_id:
        return "（无用户信息）"

    text = ""
    if memory_manager.semantic.available:
        meta_filter = {"user_id": user_id}
        if conversation_id:
            meta_filter["conversation_id"] = conversation_id
        result = memory_manager.semantic.retrieve_formatted(
            query="用户偏好 写作平台 风格 语调",
            namespace="user_conversation",
            top_k=5, max_chars_per_item=200,
            metadata_filter=meta_filter,
        )
        if result:
            text = result

    if not text:
        cache_key = f"{user_id}:{conversation_id}" if conversation_id else user_id
        cached = memory_manager._local_preferences.get(cache_key)
        if cached:
            text = "## 用户偏好\n" + "\n".join(f"- {p}" for p in cached[-5:])

    return text or "（暂无偏好记录）"


async def recent_generations(user_id: str = "", conversation_id: str = "",
                              limit: int = 5) -> str:
    """查询最近的生成记录

    Agent 在需要参考之前生成的内容时调用。

    Returns:
        最近生成的记录列表
    """
    if not user_id:
        return "（无用户信息）"

    try:
        from src.database import SessionLocal
        from sqlalchemy import text

        where = "user_id = :uid AND status = 'completed'"
        params: dict = {"uid": user_id, "limit": limit}
        if conversation_id:
            where += " AND conversation_id = :cid"
            params["cid"] = conversation_id

        db = SessionLocal()
        try:
            rows = db.execute(
                text(f"""
                    SELECT prompt, total_tokens, created_at
                    FROM agent_tasks
                    WHERE {where}
                    ORDER BY created_at DESC LIMIT :limit
                """),
                params,
            ).fetchall()
        finally:
            db.close()
    except Exception as e:
        return f"（查询失败: {e}）"

    if not rows:
        return "（暂无历史生成记录）"

    items = []
    for r in rows:
        prompt = (r.prompt or "")[:80].replace("\n", " ")
        if prompt:
            items.append(f"  - {prompt}（{r.total_tokens or 0} tokens）")

    return "## 历史生成\n" + "\n".join(items) if items else "（暂无记录）"


# ═══════════════════════════════════════════
# LangChain Tool 定义
# ═══════════════════════════════════════════

# 定义 memory_search 工具
memory_search_tool = StructuredTool(
    name="memory_search",
    description="搜索相关历史记忆、知识和经验。需要参考之前生成的内容或知识时使用。",
    coroutine=memory_search,
    args_schema={
        "query": {
            "title": "query",
            "type": "string",
            "description": "搜索关键词",
        },
        "top_k": {
            "title": "top_k",
            "type": "integer",
            "description": "返回条数（默认 3）",
            "default": 3,
        },
    },
)

# 定义 user_preferences 工具
user_preferences_tool = StructuredTool(
    name="user_preferences",
    description="获取当前用户的写作偏好，包括常用平台、风格、语调等。需要了解用户偏好时使用。",
    coroutine=user_preferences,
    args_schema={
        "limit": {
            "title": "limit",
            "type": "integer",
            "description": "返回偏好条数",
            "default": 5,
        },
    },
)

# 定义 recent_generations 工具
recent_generations_tool = StructuredTool(
    name="recent_generations",
    description="查询最近生成的内容记录。需要参考最近创作过的内容时使用。",
    coroutine=recent_generations,
    args_schema={
        "limit": {
            "title": "limit",
            "type": "integer",
            "description": "返回条数（默认 5）",
            "default": 5,
        },
    },
)

# 所有记忆工具的列表，供 handlers 注册使用
MEMORY_TOOLS: list[StructuredTool] = [
    memory_search_tool,
    user_preferences_tool,
    recent_generations_tool,
]

# 工具名称 → 函数 映射（供 run_memory_tool 使用）
_MEMORY_TOOL_MAP = {
    "memory_search": memory_search,
    "user_preferences": user_preferences,
    "recent_generations": recent_generations,
}


async def run_memory_tool(tool_name: str, user_id: str = "",
                           conversation_id: str = "", **kwargs: Any) -> str:
    """在 handler 中调用记忆工具的统一入口

    用法:
        result = await run_memory_tool("memory_search", query="甜品教程",
                                        user_id=uid, conversation_id=cid)
    """
    fn = _MEMORY_TOOL_MAP.get(tool_name)
    if not fn:
        return f"（未知工具: {tool_name}）"
    return await fn(user_id=user_id, conversation_id=conversation_id, **kwargs)
