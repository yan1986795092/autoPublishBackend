"""SubAgent — 带独立 ReAct 循环的子 Agent

核心职责：
1. run_with_react(task, context) — 完整 ReAct 循环（Think → Decide → Act → Observe）
2. _think() — 调用 LLM 决定下一步（工具调用 / peer_call / 完成）
3. _act() — 执行工具调用或 peer_call
4. _observe() — 收集结果并决定是否继续循环
5. _peer_call() — 通过 MessageBus 向对等 Agent 发消息并等待回复
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Optional

import structlog

from ..agents.base_agent import BaseAgent
from ..agents.profile import AgentProfile
from ..schemas import ReActStep, PeerCallResult, SubAgentConfig
from ..communication.bus import MessageBus
from ..tools.registry import ToolRegistry
from langchain_openai import ChatOpenAI

logger = structlog.get_logger(__name__)


class SubAgent(BaseAgent):
    """带独立 ReAct 循环的子 Agent

    继承 BaseAgent（保留 profile + handler + execute 能力），
    新增 run_with_react() 方法实现多步自主推理与对等协作。

    对等协作流程：
      Agent A 执行 run_with_react()
        → A 的 LLM 决定需要 B 的能力
        → A 调用 _peer_call("AgentB", task)
        → MessageBus 投递消息给 B
        → A 等待 B 的回复（超时 peer_call_timeout）
        → 收到回复 → A 继续 ReAct 循环
    """

    def __init__(
        self,
        profile: AgentProfile,
        llm: Optional[ChatOpenAI] = None,
        tool_registry: Optional[ToolRegistry] = None,
        message_bus: Optional[MessageBus] = None,
        config: Optional[SubAgentConfig] = None,
    ):
        super().__init__(profile=profile, llm=llm)
        self._tool_registry = tool_registry
        self._message_bus = message_bus
        self._config = config or SubAgentConfig()
        self._peer_call_count = 0

    # ── 核心入口 ──────────────────────────────────────────────────────────────

    async def run_with_react(
        self,
        task: str | dict,
        context: Optional[dict] = None,
    ) -> dict:
        """执行任务（ReAct 循环）

        Args:
            task: 任务描述（字符串或 dict）
            context: 额外上下文（来自 Manager 的依赖结果等）

        Returns:
            dict，包含：
              - output: 最终输出内容
              - react_steps: ReAct 步骤列表（用于审计）
              - peer_calls: 对等调用记录列表
              - status: completed / partial / failed
        """
        self._peer_call_count = 0
        task_str = task if isinstance(task, str) else str(task)
        session_id = (context or {}).get("session_id", "default")
        run_id = str(uuid.uuid4())

        system_prompt = self._build_react_system_prompt()
        user_message = self._build_task_message(task_str, context)

        react_steps: list[ReActStep] = []
        max_steps = self._config.max_react_steps
        final_output = ""
        peer_calls: list[PeerCallResult] = []
        step_index = 0

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        logger.info(
            "SubAgent[%s] ReAct 循环开始",
            self.name,
            task_preview=task_str[:100],
            max_steps=max_steps,
        )

        try:
            while step_index < max_steps:
                step_index += 1
                step_start = time.time()

                thought_result = await self._think(messages, step_index)
                if thought_result is None:
                    final_output = self._extract_final_output(messages)
                    break

                action, action_input, thought_text, raw_response = thought_result

                if action == "peer_call":
                    target_agent = action_input.get("target_agent", "")
                    content = action_input.get("content", "")
                    peer_result = await self._peer_call(target_agent, content, run_id)
                    peer_calls.append(peer_result)
                    observation = self._format_peer_result(peer_result)
                else:
                    observation = await self._act_tool(action, action_input)
                    observation = f"[{action}] {observation}"

                step_elapsed = round((time.time() - step_start) * 1000, 2)
                react_steps.append(ReActStep(
                    step_index=step_index,
                    thought=thought_text,
                    action=action,
                    action_input=action_input,
                    observation=observation,
                    raw_response=raw_response,
                    tool_name=action if action != "peer_call" else None,
                    peer_target=action_input.get("target_agent") if action == "peer_call" else None,
                ))

                messages.append({"role": "assistant", "content": thought_text})
                messages.append({"role": "user", "content": f"Observation: {observation}"})

                logger.debug(
                    "SubAgent[%s] 步骤 %d/%d: %s",
                    self.name, step_index, max_steps, action,
                )

            if not final_output:
                final_output = self._extract_final_output(messages)

        except asyncio.TimeoutError:
            logger.warning("SubAgent[%s] ReAct 超时（%ds）", self.name, self._config.react_timeout)
            final_output = self._extract_final_output(messages)
        except Exception as e:
            logger.error("SubAgent[%s] ReAct 异常", self.name, exc_info=True)
            final_output = f"执行异常：{e}"

        total_steps = len(react_steps)
        status = "completed" if final_output and "异常" not in final_output else "partial"

        logger.info(
            "SubAgent[%s] ReAct 完成: %d 步, peer_calls=%d, output_len=%d",
            self.name, total_steps, len(peer_calls), len(final_output),
        )

        return {
            "output": {"content": final_output},
            "react_steps": [
                {
                    "step_index": s.step_index,
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation,
                }
                for s in react_steps
            ],
            "peer_calls": [
                {
                    "target": p.target_agent,
                    "call_id": p.call_id,
                    "success": p.success,
                    "response_preview": str(p.response)[:200] if p.response else "",
                    "error": p.error,
                }
                for p in peer_calls
            ],
            "status": status,
            "agent_type": self.name,
            "total_steps": total_steps,
        }

    # ── 内部：Think ────────────────────────────────────────────────────────────

    async def _think(
        self,
        messages: list[dict],
        step_index: int,
    ) -> Optional[tuple[str, dict, str, str]]:
        if self._llm is None:
            return self._rule_based_decide(messages, step_index)

        try:
            response = await self._llm.ainvoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)
            return self._parse_llm_decision(raw)
        except Exception as e:
            logger.warning("SubAgent[%s] LLM 决策失败: %s", self.name, e)
            return self._rule_based_decide(messages, step_index)

    def _parse_llm_decision(self, raw: str) -> Optional[tuple[str, dict, str, str]]:
        raw_stripped = raw.strip()

        json_start = raw_stripped.find("```json")
        if json_start == -1:
            json_start = raw_stripped.find("{")
        else:
            json_start += 7

        if json_start != -1:
            json_end = raw_stripped.rfind("```")
            if json_end == -1:
                json_end = len(raw_stripped)
            json_str = raw_stripped[json_start:json_end].strip()
            try:
                obj = json.loads(json_str)
                action = obj.get("action", "")
                action_input = obj.get("action_input", {})
                thought = obj.get("thought", obj.get("reasoning", raw_stripped[:200]))
                if action in ("finish", "done", "complete"):
                    return None
                return (action, action_input, thought, raw_stripped)
            except json.JSONDecodeError:
                pass

        if "finish" in raw_stripped.lower() or "完成" in raw_stripped:
            return None

        return self._rule_based_decide_from_raw(raw_stripped)

    def _rule_based_decide(
        self,
        messages: list[dict],
        step_index: int,
    ) -> Optional[tuple[str, dict, str, str]]:
        last_content = messages[-1].get("content", "") if messages else ""
        return self._rule_based_decide_from_raw(last_content)

    def _rule_based_decide_from_raw(self, text: str) -> Optional[tuple[str, dict, str, str]]:
        if self._peer_call_count >= self._config.max_peer_calls:
            return None

        peer_signals = ["需要", "请", "呼叫", "协作", "编辑", "审核", "核查", "润色", "事实"]
        for signal in peer_signals:
            if signal in text:
                target = self._guess_peer_target(text)
                return (
                    "peer_call",
                    {"target_agent": target, "content": f"请协助处理：{text[:300]}"},
                    f"检测到需要 {target} 协作",
                    f"rule_engine:peer_signal={signal}",
                )

        return None

    def _guess_peer_target(self, text: str) -> str:
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["审核", "质量", "评审", "检查"]):
            return "ReviewAgent"
        if any(kw in text_lower for kw in ["编辑", "润色", "改写", "优化"]):
            return "EditorAgent"
        if any(kw in text_lower for kw in ["事实", "数据", "搜索", "核查"]):
            return "ResearcherAgent"
        if any(kw in text_lower for kw in ["关键词", "seo", "排名"]):
            return "SEOAgent"
        return "ReviewAgent"

    # ── 内部：Act ─────────────────────────────────────────────────────────────

    async def _act_tool(self, tool_name: str, tool_input: dict) -> str:
        if self._tool_registry is None:
            return f"[{tool_name}] 工具注册中心未就绪"

        tool = self._tool_registry.get(tool_name)
        if tool is None:
            return f"[{tool_name}] 工具不存在"

        try:
            result = await tool.execute(**tool_input)
            return str(result) if result else "执行成功（无输出）"
        except Exception as e:
            return f"[{tool_name}] 执行失败: {e}"

    # ── 内部：Peer Call ──────────────────────────────────────────────────────

    async def _peer_call(
        self,
        target_agent: str,
        content: str,
        run_id: str,
    ) -> PeerCallResult:
        if not self._config.enable_peer_collaboration:
            return PeerCallResult(
                target_agent=target_agent,
                call_id=run_id,
                response={},
                success=False,
                error="对等协作未启用",
            )

        if self._message_bus is None:
            return PeerCallResult(
                target_agent=target_agent,
                call_id=run_id,
                response={},
                success=False,
                error="MessageBus 未就绪",
            )

        if self._peer_call_count >= self._config.max_peer_calls:
            return PeerCallResult(
                target_agent=target_agent,
                call_id=run_id,
                response={},
                success=False,
                error=f"已达到最大 peer_call 次数（{self._config.max_peer_calls}）",
            )

        self._peer_call_count += 1
        call_id = f"{self.name}→{target_agent}:{run_id[:8]}"

        from ..communication.models import Message, MessageType
        msg = Message(
            msg_id=call_id,
            from_agent=self.name,
            to_agent=target_agent,
            msg_type=MessageType.REQUEST,
            content={"text": content, "task_context": {}},
            conversation_id=f"peer-{run_id}",
        )

        logger.info("SubAgent[%s] → peer_call → %s", self.name, target_agent)

        try:
            reply = await self._message_bus.send_and_wait(
                msg,
                timeout=self._config.peer_call_timeout,
            )
            if reply is None:
                return PeerCallResult(
                    target_agent=target_agent,
                    call_id=call_id,
                    response={},
                    success=False,
                    error=f"peer_call 超时（{self._config.peer_call_timeout}s）",
                )
            return PeerCallResult(
                target_agent=target_agent,
                call_id=call_id,
                response=reply.content or {},
                success=True,
            )
        except Exception as e:
            return PeerCallResult(
                target_agent=target_agent,
                call_id=call_id,
                response={},
                success=False,
                error=str(e),
            )

    def _format_peer_result(self, result: PeerCallResult) -> str:
        if result.success:
            content = result.response.get("text", "") if isinstance(result.response, dict) else str(result.response)
            preview = content[:500] if content else "（无内容）"
            return f"[{result.target_agent} 回复]: {preview}"
        return f"[{result.target_agent} 调用失败]: {result.error}"

    # ── 内部：输出提取 ──────────────────────────────────────────────────────

    def _extract_final_output(self, messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                content = msg["content"]
                if len(content) > 50:
                    return content
        return messages[-1]["content"] if messages else ""

    # ── 内部：Prompt 构建 ──────────────────────────────────────────────────

    def _build_react_system_prompt(self) -> str:
        base = self._build_base_role_prompt()
        tools_section = self._build_tools_section()
        peer_section = self._build_peer_section()

        return f"""{base}

## ReAct 执行模式

你将以以下循环完成任务：
1. Think: 分析当前状态，决定下一步
2. Act: 调用工具或向对等 Agent 发请求
3. Observe: 收集结果，决定是否继续

## 输出格式

每次决策请输出 JSON：
```json
{{
  "thought": "你的思考过程（为什么这样做）",
  "action": "工具名 | peer_call | finish",
  "action_input": {{ "工具参数字典 或 peer_call 的 {{'target_agent': 'Agent名', 'content': '内容'}} }}
}}
```

{tools_section}

{peer_section}

重要：
- 优先使用工具完成具体任务
- 当需要其他 Agent 能力时使用 peer_call
- 确认任务完成后使用 action: finish
- 不要在 observation 中重复思考，只记录客观结果
""".strip()

    def _build_base_role_prompt(self) -> str:
        profile = self.profile
        parts = [f"# 角色: {profile.name}"]
        if profile.role:
            parts.append(f"**角色**: {profile.role}")
        if profile.description:
            parts.append(f"**描述**: {profile.description}")
        if profile.persona:
            parts.append(f"**人格**: {profile.persona}")
        if profile.expertise:
            parts.append(f"**专长**: {', '.join(profile.expertise)}")
        if profile.communication_style:
            parts.append(f"**沟通风格**: {profile.communication_style}")
        return "\n\n".join(parts)

    def _build_tools_section(self) -> str:
        if self._tool_registry is None:
            return "## 可用工具\n（未注册工具）"

        tools = self._tool_registry.get_for_agent(self.name)
        if not tools:
            return "## 可用工具\n（无工具）"

        lines = ["## 可用工具"]
        for t in tools:
            lines.append(f"- **{t.name}**: {t.description}")
        return "\n".join(lines)

    def _build_peer_section(self) -> str:
        if not self._config.enable_peer_collaboration:
            return ""

        peers = []
        if self._message_bus and hasattr(self._message_bus, "_registry") and self._message_bus._registry:
            peers = self._message_bus._registry.list_types()
        else:
            peers = ["WritingAgent", "EditorAgent", "ReviewAgent", "SEOAgent", "ResearcherAgent"]

        peers = [p for p in peers if p != self.name]
        if not peers:
            return ""

        return f"""## 对等协作

你可以向以下 Agent 发起 peer_call 请求协作：
{', '.join(peers)}

使用 peer_call 时：
- target_agent: 目标 Agent 名称
- content: 简要描述你需要的帮助内容
- 等待回复后，继续根据回复内容决定下一步

示例：
```json
{{
  "thought": "这段文字需要事实核查，我的知识可能过时了",
  "action": "peer_call",
  "action_input": {{
    "target_agent": "ResearcherAgent",
    "content": "请核实以下事实：赛博朋克2077的发行日期和开发商"
  }}
}}
```"""

    # ── 内部：任务消息构建 ──────────────────────────────────────────────────

    def _build_task_message(self, task: str, context: Optional[dict]) -> str:
        parts = [f"## 任务\n{task}"]

        if context:
            deps = context.get("dependencies", {})
            if deps:
                parts.append("\n## 依赖任务结果")
                for dep_id, dep_data in deps.items():
                    content = dep_data.get("content", "")
                    agent = dep_data.get("agent", "未知")
                    if content:
                        parts.append(f"\n### [{agent}] {dep_id[:8]}\n{content[:500]}")

            extra = context.get("extra_context", "")
            if extra:
                parts.append(f"\n## 额外上下文\n{extra}")

        return "\n".join(parts)
