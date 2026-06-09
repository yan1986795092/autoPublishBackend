"""ReplayEngine — 基于快照的执行回放引擎"""
from __future__ import annotations
import logging
import time as _time
from datetime import datetime
from typing import Optional

from .models import ExecutionSnapshot, PromptMessageSnapshot
from .storage import SnapshotStorage
from .collector import SnapshotCollector, set_collector

logger = logging.getLogger(__name__)


class ReplayEngine:
    """执行回放引擎

    基于 ExecutionSnapshot 重建当时的 prompt 上下文，
    支持换模型/换 prompt 版本重跑。
    """

    def __init__(self, storage: SnapshotStorage | None = None):
        self._storage = storage or SnapshotStorage()

    def load(self, snapshot_id: str) -> Optional[ExecutionSnapshot]:
        """加载快照。"""
        return self._storage.load(snapshot_id)

    def rebuild_prompt(self, snapshot_id: str,
                       registry=None) -> list[dict]:
        """根据快照重建当时的 prompt messages。"""
        snapshot = self.load(snapshot_id)
        if not snapshot:
            raise ValueError(f"快照未找到: {snapshot_id}")

        messages = []
        for msg in snapshot.prompt_messages:
            content = msg.resolved_content

            # 如果有 registry 且可以重建（从模板重新插值）
            if registry and msg.template_name and msg.variables and msg.template_hash:
                try:
                    rendered = registry.render(
                        msg.template_name,
                        msg.variables,
                        pin_hash=msg.template_hash,
                        strict=False,
                    )
                    if rendered.success and rendered.text:
                        content = rendered.text
                except Exception:
                    logger.warning(
                        "重建失败: %s@%s，使用缓存内容",
                        msg.template_name, msg.template_hash,
                    )

            messages.append({
                "role": msg.role,
                "content": content,
            })

        return messages

    async def rerun(self, snapshot_id: str, *,
                    llm=None,
                    prompt_overrides: dict[str, dict] | None = None,
                    variable_overrides: dict[str, dict] | None = None,
                    ) -> Optional[ExecutionSnapshot]:
        """基于快照重新执行。

        Args:
            snapshot_id: 原始快照 ID
            llm: 传 None 仅重建 prompt（dry-run），传 LLM 实例则重跑
            prompt_overrides: {"writing.task": {"pin": "v2.2"}}
            variable_overrides: {"writing.task": {"topic": "new topic"}}

        Returns:
            新生成的 ExecutionSnapshot（parent_snapshot_id 关联到原始快照）
        """
        snapshot = self.load(snapshot_id)
        if not snapshot:
            raise ValueError(f"Snapshot not found: {snapshot_id}")

        prompt_overrides = prompt_overrides or {}
        variable_overrides = variable_overrides or {}

        # 重建 prompt messages
        messages = self._apply_overrides(snapshot, prompt_overrides, variable_overrides)

        # 如果没有 LLM，只返回重建的 prompt（dry-run）
        if llm is None:
            return ExecutionSnapshot(
                snapshot_id=f"snap_dry_{_time.time_ns()}",
                agent_name=snapshot.agent_name,
                timestamp=datetime.utcnow(),
                task_id=snapshot.task_id,
                parent_snapshot_id=snapshot_id,
                prompt_messages=[
                    PromptMessageSnapshot(
                        role=m["role"],
                        source="rebuilt",
                        resolved_content=m["content"],
                    ) for m in messages
                ],
                result_summary="dry-run (无 LLM 调用)",
            )

        # 执行 LLM 调用
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            elif m["role"] == "assistant":
                lc_messages.append(AIMessage(content=m["content"]))
            else:
                lc_messages.append(HumanMessage(content=m["content"]))

        # 创建 collector 记录 rerun
        collector = SnapshotCollector(
            agent_name=snapshot.agent_name,
            profile=snapshot.context.profile_snapshot if snapshot.context else {},
            task_id=snapshot.task_id,
            parent_snapshot_id=snapshot_id,
        )

        for m in messages:
            collector.record_prompt_message(PromptMessageSnapshot(
                role=m["role"],
                source="rebuilt",
                resolved_content=m["content"],
            ))

        set_collector(collector)
        try:
            start = _time.monotonic()
            response = await llm.ainvoke(lc_messages)
            latency_ms = int((_time.monotonic() - start) * 1000)
            content = response.content if hasattr(response, 'content') else str(response)

            finish_reason = getattr(response, 'finish_reason', None)
            if finish_reason is None and hasattr(response, 'response_metadata'):
                finish_reason = response.response_metadata.get('finish_reason', 'stop')

            usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata if response.usage_metadata else {}
            elif hasattr(response, 'usage'):
                usage = response.usage if response.usage else {}

            collector.record_llm_call(
                model=getattr(llm, 'model_name', '未知'),
                params={"temperature": getattr(llm, 'temperature', None)},
                messages=messages,
                response_content=content,
                finish_reason=finish_reason or 'stop',
                prompt_tokens=usage.get('input_tokens', usage.get('prompt_tokens', 0)),
                completion_tokens=usage.get('output_tokens', usage.get('completion_tokens', 0)),
                total_tokens=usage.get('total_tokens', 0),
                latency_ms=latency_ms,
            )

            result = {"status": "completed", "content": content}
            new_snapshot = collector.finalize(result)
            self._storage.save(new_snapshot)
            return new_snapshot
        finally:
            set_collector(None)

    def _apply_overrides(self, snapshot, prompt_overrides, variable_overrides):
        """根据 overrides 重建 prompt 消息列表。"""
        from ...prompts import prompt_engine

        messages = []
        for msg in snapshot.prompt_messages:
            content = msg.resolved_content

            if msg.template_name:
                override = prompt_overrides.get(msg.template_name, {})
                var_override = variable_overrides.get(msg.template_name, {})

                if override or var_override:
                    variables = dict(msg.variables or {})
                    variables.update(var_override)
                    try:
                        rendered = prompt_engine.render(
                            msg.template_name,
                            variables,
                            pin=override.get("pin"),
                            pin_hash=override.get("pin_hash"),
                            strict=False,
                        )
                        if rendered.success and rendered.text:
                            content = rendered.text
                    except Exception:
                        pass

            messages.append({"role": msg.role, "content": content})

        return messages
