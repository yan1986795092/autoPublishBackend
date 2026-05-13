"""Fallback Agent — 未注册 Agent 类型的通用 LLM 回退处理"""

import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI


async def handle_fallback(task: dict, llm: ChatOpenAI, _state: dict) -> dict:
    agent_type = task.get("agent_type", "Unknown")
    task_input = task.get("input", {})

    if not llm:
        return {
            "status": "completed", "agent_type": agent_type,
            "output": {"content": f"[Mock] {agent_type} completed", "type": "text"},
            "tokens_used": 0, "thinking_steps": [],
        }

    try:
        response = await llm.ainvoke([
            SystemMessage(content=f"You are acting as a {agent_type}. Complete the assigned task."),
            HumanMessage(content=json.dumps(task_input, ensure_ascii=False)),
        ])
        return {
            "status": "completed", "agent_type": agent_type,
            "output": {"content": response.content, "type": "text"},
            "tokens_used": len(response.content) // 4, "thinking_steps": [],
        }
    except Exception as e:
        return {
            "status": "failed", "agent_type": agent_type,
            "error": str(e), "tokens_used": 0, "thinking_steps": [],
        }
