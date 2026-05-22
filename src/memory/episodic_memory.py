"""情景记忆 — 从历史任务中学习经验

从已有的 agent_tasks / agent_subtasks 数据库表中读取历史记录，
为 Agent 提供"过去类似场景是怎么处理的"参考。
"""

from __future__ import annotations
import logging
from typing import Optional

from .schemas import Episode

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """情景记忆：查询历史任务记录，提取可复用的经验

    数据来源：已有的 agent_tasks + agent_subtasks 表。
    所有数据库查询都有 try/except 兜底，不会影响 Agent 主流程。
    """

    def __init__(self):
        self._cache: dict[str, list[Episode]] = {}

    # ── 写入 ──

    def store(self, agent_type: str, input_data: dict,
              output_summary: str, success: bool = True,
              tokens_used: int = 0, total_steps: int = 0) -> None:
        """记录一条经验到内存缓存

        数据库持久化由 persistence.py 自动完成，
        这里只做内存预热，供后续快速检索。
        """
        input_summary = str(input_data).replace("\n", " ")[:200]
        episode = Episode(
            agent_type=agent_type,
            input_summary=input_summary,
            output_summary=output_summary[:200],
            success=success,
            tokens_used=tokens_used,
            total_steps=total_steps,
        )
        cache_key = agent_type
        if cache_key not in self._cache:
            self._cache[cache_key] = []
        self._cache[cache_key].append(episode)
        # 只保留最近 20 条
        if len(self._cache[cache_key]) > 20:
            self._cache[cache_key] = self._cache[cache_key][-20:]

    # ── 检索 ──

    def retrieve_similar(self, agent_type: str, query: str = "",
                         top_k: int = 3) -> list[Episode]:
        """查找同类 Agent 的历史执行记录

        检索来源（按优先级）：
        1. 内存缓存（当前会话）
        2. 数据库（历史任务）
        """
        # 1. 内存缓存
        cached = self._cache.get(agent_type, [])[-top_k:]
        if len(cached) >= top_k:
            return cached

        # 2. 数据库查询
        db_episodes = self._query_db(agent_type=agent_type, top_k=top_k)
        return db_episodes or cached

    def retrieve_user_preferences(self, user_id: str) -> dict:
        """从用户历史任务中学习偏好

        Returns:
            {
                "writing_style": str,        # 常用写作风格
                "preferred_platforms": list,  # 常用平台
                "avg_task_steps": int,        # 平均任务步数
                "total_tasks": int,           # 历史任务数
            }
        """
        if not user_id:
            return {}

        try:
            from src.database import SessionLocal
            from src.models import AgentTask

            with SessionLocal() as db:
                records = (
                    db.query(AgentTask)
                    .filter(
                        AgentTask.user_id == user_id,
                        AgentTask.status == "completed",
                    )
                    .order_by(AgentTask.created_at.desc())
                    .limit(50)
                    .all()
                )
        except Exception as e:
            logger.warning(f"EpisodicMemory: 查询用户偏好失败: {e}")
            return {}

        if not records:
            return {}

        total_steps = sum(r.total_steps or 0 for r in records)
        total_tokens = sum(r.total_tokens or 0 for r in records)

        return {
            "total_tasks": len(records),
            "avg_task_steps": round(total_steps / len(records), 1),
            "total_tokens": total_tokens,
            "avg_tokens_per_task": round(total_tokens / len(records), 1),
        }

    def retrieve_formatted(self, agent_type: str, query: str = "",
                           top_k: int = 2) -> str:
        """检索并格式化为 prompt 可用的字符串"""
        episodes = self.retrieve_similar(
            agent_type=agent_type, query=query, top_k=top_k
        )
        if not episodes:
            return ""

        parts = ["## 历史经验"]
        for ep in episodes:
            parts.append(f"  {ep.to_prompt()}")

        return "\n".join(parts)

    def retrieve_user_formatted(self, user_id: str) -> str:
        """检索用户偏好并格式化为 prompt 可用的字符串"""
        prefs = self.retrieve_user_preferences(user_id)
        if not prefs:
            return ""

        return (
            "## 用户画像\n"
            f"  历史任务: {prefs.get('total_tasks', 0)} 次 | "
            f"平均步数: {prefs.get('avg_task_steps', 0)} | "
            f"平均 Token: {prefs.get('avg_tokens_per_task', 0)}"
        )

    # ── 内部 ──

    def _query_db(self, agent_type: str, top_k: int = 5) -> list[Episode]:
        """从数据库查询历史记录"""
        try:
            from src.database import SessionLocal
            from src.models import SubTask

            with SessionLocal() as db:
                records = (
                    db.query(SubTask)
                    .filter(
                        SubTask.agent_type == agent_type,
                        SubTask.status == "completed",
                    )
                    .order_by(SubTask.created_at.desc())
                    .limit(top_k)
                    .all()
                )
        except Exception as e:
            logger.warning(f"EpisodicMemory: 数据库查询失败: {e}")
            return []

        results = []
        for r in records:
            input_summary = str(r.input_json or "")[:200]
            output_summary = str(r.output_json or "")[:200]
            results.append(Episode(
                agent_type=r.agent_type,
                input_summary=input_summary,
                output_summary=output_summary,
                success=r.status == "completed",
                tokens_used=r.tokens_used or 0,
                created_at=r.created_at.isoformat() if r.created_at else "",
            ))

        return results

    @property
    def cache_size(self) -> int:
        return sum(len(v) for v in self._cache.values())
