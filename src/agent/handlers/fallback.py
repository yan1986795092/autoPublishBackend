"""Fallback Agent — 未注册 Agent 类型的通用 LLM 回退处理

支持工具调用。
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from .tool_utils import call_llm


async def handle_fallback(task: dict, llm: ChatOpenAI, _state: dict, mcp_tools: list | None = None) -> dict:
    agent_type = task.get("agent_type", "Unknown")
    task_input = task.get("input", {})

    if not llm:
        return {
            "status": "completed", "agent_type": agent_type,
            "output": {"content": f"[Mock] {agent_type} completed", "type": "text"},
            "tokens_used": 0, "thinking_steps": [],
        }

    tool_registry = _state.get("_tool_registry") if isinstance(_state, dict) else None

    try:
        lc_messages = [
            SystemMessage(content=f"You are acting as a {agent_type}. Complete the assigned task."),
            HumanMessage(content=json.dumps(task_input, ensure_ascii=False)),
        ]
        response_text = await call_llm(
            llm, lc_messages,
            tool_registry=tool_registry,
            agent_type=agent_type,
            stream=False,
        )
        return {
            "status": "completed", "agent_type": agent_type,
            "output": {"content": response_text, "type": "text"},
            "tokens_used": len(response_text) // 4, "thinking_steps": [],
        }
    except Exception as e:
        return {
            "status": "failed", "agent_type": agent_type,
            "error": str(e), "tokens_used": 0, "thinking_steps": [],
        }
