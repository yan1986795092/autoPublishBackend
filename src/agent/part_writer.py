"""PartWriter — 统一的 Part 写入器：内存缓冲 + 异步节流落盘

spec: docs/superpowers/specs/2026-07-02-chat-parts-model.md §5.1

设计原则：
  - append / update：立即写入内存，异步节流刷 DB
  - server_id 幂等：重复 append 不重复写
  - append_text：O(1) 追加（用 list 而非字符串拼接）
  - final_flush：message_end 时强制同步落盘
  - 每个 PartWriter 实例绑定一个 message_id，不跨消息共享
"""

from __future__ import annotations
import asyncio
import json
import time
import uuid
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# 200ms batch flush — 平衡延迟和 DB 写入压力
_FLUSH_INTERVAL_SEC = 0.2


class PartWriter:
    """Part 写入器（spec §5.1）"""

    def __init__(self, message_id: str, db_session: "Session") -> None:
        self.message_id = message_id
        self._db: Session = db_session
        # server_id → ORM ChatMessagePart instance
        self._parts: dict[str, "ChatMessagePart"] = {}
        self._seq_counter: int = 0
        # server_id → partial input string（tool_input_delta 累加）
        self._partial_inputs: dict[str, str] = {}
        # server_id → list of text chunks（text_delta 累加）
        self._text_chunks: dict[str, list[str]] = {}
        # 待 flush 的 part id → True
        self._flush_queue: set[str] = set()
        self._flush_task: asyncio.Task | None = None
        self._finalized: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    async def append(
        self,
        part_type: str,
        server_id: str,
        payload: dict,
    ) -> None:
        """追加一个新 Part（幂等：server_id 已存在则忽略）"""
        if self._finalized:
            logger.warning(
                "PartWriter.finalize() 后仍尝试 append，忽略",
                message_id=self.message_id,
                server_id=server_id,
            )
            return
        if server_id in self._parts:
            return  # 幂等：已存在

        from src.models import ChatMessagePart

        part = ChatMessagePart(
            id=str(uuid.uuid4()),
            message_id=self.message_id,
            seq=self._seq_counter,
            type=part_type,
            state="streaming",
            payload_json=json.dumps(payload, ensure_ascii=False),
            server_id=server_id,
        )
        self._parts[server_id] = part
        self._seq_counter += 1

        # 初始化累加器
        self._partial_inputs.setdefault(server_id, "")
        self._text_chunks.setdefault(server_id, [])

        self._enqueue_flush(server_id)
        await self._schedule_flush()

    async def update(self, server_id: str, patch: dict) -> None:
        """更新已有 Part 的 payload（部分更新 + state/completed_at）"""
        if self._finalized:
            return
        part = self._parts.get(server_id)
        if not part:
            logger.debug(
                "PartWriter.update: server_id 不存在，忽略",
                message_id=self.message_id,
                server_id=server_id,
            )
            return

        current = json.loads(part.payload_json)
        current.update(patch)
        part.payload_json = json.dumps(current, ensure_ascii=False)

        # state / completed_at 联动
        if patch.get("state") in ("done", "error", "cancelled"):
            part.state = patch["state"]
            part.completed_at = part.completed_at or _utcnow()

        self._enqueue_flush(server_id)
        await self._schedule_flush()

    async def append_text(self, server_id: str, delta: str) -> None:
        """流式追加文本 delta（O(1)，用 list 避免字符串拼接 O(n²)）

        策略：payload['content'] = ''.join(_chunks)，给前端用的 join 结果
        """
        if self._finalized:
            return

        part = self._parts.get(server_id)
        if not part:
            # 惰性创建空的 text part
            await self.append("text", server_id, {"content": ""})
            part = self._parts[server_id]

        chunks = self._text_chunks.setdefault(server_id, [])
        chunks.append(delta)
        current = json.loads(part.payload_json)
        current["content"] = "".join(chunks)
        part.payload_json = json.dumps(current, ensure_ascii=False)

        self._enqueue_flush(server_id)
        await self._schedule_flush()

    async def append_reasoning(
        self,
        server_id: str,
        content_delta: str,
        extra_payload: dict | None = None,
    ) -> None:
        """追加 reasoning delta（惰性创建 + 累加 content）"""
        if self._finalized:
            return

        part = self._parts.get(server_id)
        if not part:
            payload = {"content": content_delta, "collapsed_by_default": True}
            if extra_payload:
                payload.update(extra_payload)
            await self.append("reasoning", server_id, payload)
        else:
            chunks = self._text_chunks.setdefault(server_id, [])
            chunks.append(content_delta)
            current = json.loads(part.payload_json)
            current["content"] = "".join(chunks)
            part.payload_json = json.dumps(current, ensure_ascii=False)
            self._enqueue_flush(server_id)
            await self._schedule_flush()

    async def append_tool_call(
        self,
        server_id: str,
        tool_name: str,
        display_name: str,
        icon: str,
        round_: int = 0,
    ) -> None:
        """追加 tool_call part（来自 stage:started 事件）"""
        if self._finalized:
            return
        await self.append("tool_call", server_id, {
            "tool_name": tool_name,
            "display_name": display_name,
            "icon": icon,
            "input": {},
            "round": round_,
        })

    async def update_tool_input_delta(
        self,
        server_id: str,
        delta_json: str,
    ) -> None:
        """tool_input_delta：部分 JSON 合并（容错：解析失败时用 raw string）"""
        if self._finalized:
            return

        part = self._parts.get(server_id)
        if not part:
            # 惰性创建
            await self.append("tool_call", server_id, {
                "tool_name": "",
                "display_name": "",
                "icon": "",
                "input": {},
            })
            part = self._parts[server_id]

        partial = self._partial_inputs.setdefault(server_id, "")
        partial += delta_json

        # 尝试解析
        try:
            delta_obj = json.loads(delta_json)
            current = json.loads(part.payload_json)
            input_dict = current.get("input", {})
            input_dict.update(delta_obj)
            current["input"] = input_dict
            part.payload_json = json.dumps(current, ensure_ascii=False)
            # 解析成功：保存解析后的字符串（去掉已解析的部分）
            self._partial_inputs[server_id] = partial[len(delta_json):]
        except json.JSONDecodeError:
            # 解析失败：直接累加到 payload 的 _raw_partial
            current = json.loads(part.payload_json)
            input_dict = current.get("input", {})
            input_dict["_raw_partial"] = partial
            part.payload_json = json.dumps(current, ensure_ascii=False)

        self._enqueue_flush(server_id)
        await self._schedule_flush()

    async def finalize(self) -> None:
        """message_end 时强制同步 flush 所有 pending parts

        必须在 SSE message_end 事件 yield 之前调用。
        """
        self._finalized = True

        if self._flush_task:
            self._flush_task.cancel()
            self._flush_task = None

        # 同步刷所有 pending
        await self._do_flush(set(self._parts.keys()))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _enqueue_flush(self, server_id: str) -> None:
        self._flush_queue.add(server_id)

    async def _schedule_flush(self) -> None:
        """启动/重置 200ms 延迟 flush timer"""
        if self._flush_task is not None and not self._flush_task.done():
            return  # timer 已在跑

        async def _delayed_flush() -> None:
            await asyncio.sleep(_FLUSH_INTERVAL_SEC)
            await self._do_flush(set(self._flush_queue))

        loop = asyncio.get_running_loop()
        self._flush_task = loop.create_task(_delayed_flush())

    async def _do_flush(self, server_ids: set[str]) -> None:
        """批量写 DB"""
        if not server_ids:
            return

        to_flush = [
            self._parts[sid]
            for sid in server_ids
            if sid in self._parts
        ]
        if not to_flush:
            return

        for sid in server_ids:
            self._flush_queue.discard(sid)

        try:
            for part in to_flush:
                self._db.add(part)
            self._db.commit()
            logger.debug(
                "PartWriter flush",
                message_id=self.message_id,
                count=len(to_flush),
            )
        except Exception:
            self._db.rollback()
            logger.error(
                "PartWriter flush 失败",
                message_id=self.message_id,
                count=len(to_flush),
                exc_info=True,
            )

    def get_part(self, server_id: str) -> dict | None:
        """读取单个 part 的 payload（供 SSE v3 事件使用）"""
        part = self._parts.get(server_id)
        if not part:
            return None
        return json.loads(part.payload_json)

    def get_all_parts(self) -> list[dict]:
        """获取当前 message 的所有 part payload（有序）"""
        parts = sorted(self._parts.values(), key=lambda p: p.seq)
        return [
            {
                "id": p.id,
                "server_id": p.server_id,
                "type": p.type,
                "state": p.state,
                "payload": json.loads(p.payload_json),
                "seq": p.seq,
            }
            for p in parts
        ]


def _utcnow() -> "datetime":
    from datetime import datetime
    return datetime.utcnow()
