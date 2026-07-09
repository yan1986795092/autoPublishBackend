"""统一的上下文构建器 — 对话引擎和 Agent 执行共用

按优先级逐段构建 LLM 上下文，每段有独立 token 预算和容错。
"""

from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 2)


def _truncate(text: str, budget_tokens: int, label: str = "") -> tuple[str, int]:
    """截断到 budget，带压缩标记；低于 60 字符返回空"""
    if not text:
        return "", 0
    estimated = _estimate_tokens(text)
    if estimated <= budget_tokens:
        return text, estimated
    budget_chars = budget_tokens * 2
    if budget_chars <= 60:
        return "", 0
    truncated = text[:budget_chars - 30]
    note = f"\n（{label}共 {len(text)} 字，显示 {len(truncated)} 字）"
    result = truncated + note
    return result, _estimate_tokens(result)


def _compress_history(history: list[dict], budget_tokens: int) -> str:
    """压缩对话历史：早期轮次摘要 + 最近 5 轮原文"""
    if not history:
        return ""

    keep_raw = 5
    raw_part = history[-keep_raw:]
    summary_part = history[:-keep_raw]
    sections = []

    if summary_part:
        lines = []
        for m in summary_part:
            role = "用户" if m.get("role") == "user" else "助手"
            content = (m.get("content", "") or "")[:60].replace("\n", " ")
            if content:
                lines.append(f"{role}: {content}")
        if lines:
            summary = "; ".join(lines)
            if len(summary) > 400:
                summary = summary[:397] + "..."
            sections.append(f"[历史对话摘要]\n{summary}")

    if raw_part:
        lines = []
        for m in raw_part:
            role = "用户" if m.get("role") == "user" else "助手"
            content = (m.get("content", "") or "")
            lines.append(f"{role}: {content}")
        sections.append(f"[最近对话]\n" + "\n".join(lines))

    text = "\n\n".join(sections)
    text, _ = _truncate(text, budget_tokens, "对话历史")
    return text


class ContextBuilder:
    """统一上下文构建器

    用法:
        builder = ContextBuilder(memory_manager)
        context = await builder.build(
            agent_type="conversation",
            user_id=user_id,
            query=message,
            history=history,
        )
    """

    # 各段配置: (名称, 构建方法, 最小 token, 最大 token)
    TIERS = []

    def __init__(self, memory_manager):
        self.mm = memory_manager

    async def build(
        self,
        *,
        agent_type: str = "conversation",
        user_id: str = "",
        query: str = "",
        history: list[dict] | None = None,
        conversation_id: str = "",
        max_tokens: int = 4000,
        skip_tiers: set[str] | None = None,  # 跳过哪些 tier，避免重复
    ) -> str:
        remaining = max_tokens
        sections: list[str] = []
        history = history or []
        skip_tiers = skip_tiers or set()

        tiers = [
            ("用户偏好",   self._tier_user_profile,   50,  250),
            ("当前请求",   self._tier_current_query,  30,  100),
            ("最近对话",   self._tier_recent_history, 100, 800),
            ("相关内容",   self._tier_semantic,       200, 1500),
        ]

        for name, fn, min_tok, max_tok in tiers:
            if name in skip_tiers:
                continue
            if remaining < min_tok:
                continue
            budget = min(max_tok, remaining)
            try:
                text, used = await fn(
                    user_id=user_id, query=query, history=history,
                    budget=budget, conversation_id=conversation_id,
                )
                if text:
                    sections.append(text)
                    remaining -= used
            except Exception as e:
                logger.warning("ContextBuilder: %s 构建失败: %s", name, e)
                continue

        if not sections:
            return ""

        result = "\n\n".join(sections)
        total_used = max_tokens - remaining
        logger.info(
            "ContextBuilder: 构建完成 (%dt / %dt, %d 段)",
            total_used, max_tokens, len(sections),
        )
        return result

    # ── Tier 构建器 ──

    async def _tier_user_profile(
        self, user_id: str, query: str, history: list[dict],
        budget: int, conversation_id: str = "",
    ) -> tuple[str, int]:
        if not user_id:
            return "", 0
        text = ""
        if self.mm.semantic.available:
            meta_filter = {"user_id": user_id}
            if conversation_id:
                meta_filter["conversation_id"] = conversation_id
            result = self.mm.semantic.retrieve_formatted(
                query="用户偏好 写作平台 风格 语调 目标读者",
                namespace="user_conversation",
                top_k=5, max_chars_per_item=200,
                metadata_filter=meta_filter,
            )
            if result:
                text = result
        if not text:
            cache_key = f"{user_id}:{conversation_id}" if conversation_id else user_id
            cached = self.mm._local_preferences.get(cache_key)
            if cached:
                text = "## 用户偏好\n" + "\n".join(f"- {p}" for p in cached[-5:])
        return _truncate(text, budget, "用户偏好")

    async def _tier_current_query(
        self, user_id: str, query: str, history: list[dict],
        budget: int, conversation_id: str = "",
    ) -> tuple[str, int]:
        if not query:
            return "", 0
        return _truncate(f"## 当前请求\n{query}", budget, "当前请求")

    async def _tier_recent_history(
        self, user_id: str, query: str, history: list[dict],
        budget: int, conversation_id: str = "",
    ) -> tuple[str, int]:
        if not history:
            return "", 0
        text = _compress_history(history, budget)
        if not text:
            return "", 0
        return text, _estimate_tokens(text)

    async def _tier_semantic(
        self, user_id: str, query: str, history: list[dict],
        budget: int, conversation_id: str = "",
    ) -> tuple[str, int]:
        if not query or not self.mm.semantic.available:
            return "", 0
        meta_filter = {"user_id": user_id} if user_id else None
        if meta_filter and conversation_id:
            meta_filter["conversation_id"] = conversation_id
        result = self.mm.semantic.retrieve_formatted(
            query=query[:200], namespace=None, top_k=3, max_chars_per_item=300,
            metadata_filter=meta_filter,
        )
        if not result:
            return "", 0
        return _truncate(result, budget, "相关内容")

    async def _tier_history_gens(
        self, user_id: str, query: str, history: list[dict],
        budget: int, conversation_id: str = "",
    ) -> tuple[str, int]:
        if not user_id:
            return "", 0
        try:
            from src.database import SessionLocal
            from sqlalchemy import text as sql_text
            db = SessionLocal()
            try:
                where = "user_id = :uid AND status = 'completed'"
                params: dict = {"uid": user_id}
                if conversation_id:
                    where += " AND conversation_id = :cid"
                    params["cid"] = conversation_id
                rows = db.execute(
                    sql_text(f"""
                        SELECT prompt, total_tokens, created_at
                        FROM agent_tasks
                        WHERE {where}
                        ORDER BY created_at DESC LIMIT 8
                    """),
                    params,
                ).fetchall()
            finally:
                db.close()
        except Exception as e:
            logger.warning("ContextBuilder: 历史查询失败: %s", e)
            return "", 0

        if not rows:
            return "", 0
        items = []
        for r in rows:
            prompt = (r.prompt or "")[:80].replace("\n", " ")
            if prompt:
                items.append(f"  - {prompt}（{r.total_tokens or 0} tokens）")
        if not items:
            return "", 0
        return _truncate("## 历史生成\n" + "\n".join(items), budget, "历史生成")
