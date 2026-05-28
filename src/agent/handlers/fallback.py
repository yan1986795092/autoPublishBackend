"""Fallback Agent — 未注册 Agent 类型的通用 LLM 回退处理

支持工具调用。
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from .tool_utils import call_llm
from src.utils.tokenizer import estimate_tokens


async def handle_fallback(task: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    agent_type = task.get("agent_type", "Unknown")
    task_input = task.get("input", {})

    if not llm:
        return {
            "status": "partial", "agent_type": agent_type,
            "output": {"content": f"[Mock] {agent_type} completed (no LLM available)", "type": "text"},
            "tokens_used": 0, "thinking_steps": [],
            "warning": f"No LLM configured for {agent_type}, returning mock result",
        }

    tool_registry = task.get("_tool_registry")

    try:
        skill_system_prompt = task_input.get("_skill_system_prompt", "")
        skill_note = f"\n\n=== 创作指导 (Skill: {task_input.get('_skill_name', 'Custom')}) ===\n{skill_system_prompt}\n===" if skill_system_prompt else ""
        lc_messages = [
            SystemMessage(content=f"You are acting as a {agent_type}. Complete the assigned task.{skill_note}"),
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
            "tokens_used": estimate_tokens(response_text), "thinking_steps": [],
        }
    except Exception as e:
        return {
            "status": "failed", "agent_type": agent_type,
            "error": str(e), "tokens_used": 0, "thinking_steps": [],
        }
