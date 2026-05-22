"""MemoryManager — 统一记忆入口

所有 Agent 不直接调用下层存储，通过 MemoryManager 统一路由。
职责：
  - build_agent_context(): Agent 执行前，聚合相关记忆注入 prompt
  - remember(): Agent 执行后，选择性沉淀高质量产出
  - consolidate(): 后台记忆巩固任务
"""

from __future__ import annotations
import logging
from typing import Optional

from .working_memory import WorkingMemory
from .semantic_memory import SemanticMemory
from .episodic_memory import EpisodicMemory

logger = logging.getLogger(__name__)

# ── Agent → 语义命名空间 映射 ──
AGENT_NAMESPACE_MAP: dict[str, str] = {
    "WritingAgent": "content",
    "WebSearch": "knowledge",
    "ReviewAgent": "feedback",
    "SEOAgent": "domain",
    "DiagramAgent": "content",
    "PlatformAdapter": "template",
    "ImageAgent": "content",
    "HotTopicAgent": "knowledge",
    "CodeAgent": "content",
    "TranslationAgent": "content",
}

# ── 哪些 Agent 需要注入语义记忆上下文 ──
AGENT_MEMORY_ENABLED = {
    "WritingAgent", "SEOAgent", "ReviewAgent",
    "DiagramAgent", "CodeAgent", "TranslationAgent",
}

# ── 哪些 Agent 的输出值得存入语义记忆 ──
AGENT_REMEMBER_ENABLED = {
    "WritingAgent", "WebSearch",
    "DiagramAgent", "SEOAgent",
}


class MemoryManager:
    """记忆管理器 — Agent 的唯一记忆入口"""

    def __init__(self):
        self.working = WorkingMemory()
        self.semantic = SemanticMemory()
        self.episodic = EpisodicMemory()
        self._session_task_id: Optional[str] = None

    # ═══════════════════════════════════════════
    # Agent 执行前：构建上下文
    # ═══════════════════════════════════════════

    async def build_agent_context(
        self,
        agent_type: str,
        task_input: dict,
        user_id: str = "",
        max_tokens: int = 2000,
    ) -> str:
        """为 Agent 构建增强 Prompt 上下文

        聚合三类记忆，按重要性排序后裁剪到 token budget。
        返回空字符串表示无可用记忆。

        三类记忆的权重：
          语义记忆（最相关）> 情景记忆（历史经验）> 工作记忆（当前对话）
        """
        if agent_type not in AGENT_MEMORY_ENABLED:
            return ""

        parts: list[tuple[str, str, int]] = []  # (tag, content, priority)

        # 1. 语义记忆 — 向量检索
        query = self._build_search_query(agent_type, task_input)
        if query and self.semantic.available:
            ns = AGENT_NAMESPACE_MAP.get(agent_type)
            semantic_content = self.semantic.retrieve_formatted(
                query=query, namespace=ns, top_k=3,
            )
            if semantic_content:
                parts.append(("semantic", semantic_content, 1))

        # 2. 情景记忆 — 历史经验（Manager/Review 这类决策型 Agent 更需要）
        if agent_type in ("ReviewAgent", "ManagerAgent", "SEOAgent"):
            episodic_content = self.episodic.retrieve_formatted(
                agent_type=agent_type, query=query or str(task_input),
            )
            if episodic_content:
                parts.append(("episodic", episodic_content, 2))

        # 3. 用户偏好
        if user_id:
            user_content = self.episodic.retrieve_user_formatted(user_id)
            if user_content:
                parts.append(("user", user_content, 3))

        # 4. 工作记忆 — 当前对话上下文（权重最低）
        working_content = self.working.get_recent_context(max_chars=800)
        if working_content:
            parts.append(("working", working_content, 4))

        if not parts:
            return ""

        # 按优先级排序，优先级数字越小越重要
        parts.sort(key=lambda x: x[2])

        # Token Budget 裁剪（粗略按字符估算）
        result_parts = []
        total_chars = 0
        for _, content, _ in parts:
            if total_chars + len(content) > max_tokens * 4:  # 1 token ≈ 4 chars
                break
            result_parts.append(content)
            total_chars += len(content)

        context = "\n\n".join(result_parts)
        if context:
            logger.info(
                "MemoryManager: 为 %s 注入记忆上下文 (%d chars, 来源: %s)",
                agent_type, total_chars,
                ", ".join(p[0] for p in parts),
            )

        return context

    def _build_search_query(self, agent_type: str, task_input: dict) -> str:
        """从 task_input 提取语义检索关键词"""
        if agent_type == "WritingAgent":
            return (
                task_input.get("topic", "")
                or task_input.get("query", "")
                or ""
            )
        if agent_type in ("ReviewAgent", "SEOAgent"):
            return (
                task_input.get("topic", "")
                or task_input.get("content", "")[:100]
                or ""
            )
        if agent_type == "WebSearch":
            return task_input.get("query", "")
        return task_input.get("topic", "") or str(task_input)[:100]

    # ═══════════════════════════════════════════
    # Agent 执行后：选择性沉淀记忆
    # ═══════════════════════════════════════════

    async def remember(
        self,
        agent_type: str,
        task_input: dict,
        result: dict,
        user_id: str = "",
    ) -> None:
        """Agent 执行完成后，选择性写入记忆

        写入规则：
        - 只写入 status=completed 的产出
        - 只写入 AGENT_REMEMBER_ENABLED 中的 Agent 类型
        - 只写入长度 >= 100 字符的内容
        """
        if agent_type not in AGENT_REMEMBER_ENABLED:
            return
        if result.get("status") != "completed":
            return

        # 提取输出内容
        output = result.get("output", {}) or {}
        content = output.get("content", "")
        if not content or len(content.strip()) < 100:
            return

        content = content.strip()

        # 1. 写入语义记忆
        namespace = AGENT_NAMESPACE_MAP.get(agent_type, "general")
        self.semantic.store(
            text=content,
            namespace=namespace,
            metadata={
                "agent_type": agent_type,
                "user_id": user_id,
                "source": "agent_output",
                "task_type": task_input.get("content_type", ""),
            },
        )

        # 2. 写入情景记忆
        self.episodic.store(
            agent_type=agent_type,
            input_data=task_input,
            output_summary=content[:200],
            success=True,
            tokens_used=result.get("tokens_used", 0),
        )

        # 3. 追踪工作记忆 token
        tokens = result.get("tokens_used", 0)
        if tokens:
            self.working.track_token(agent_type, tokens)

    # ═══════════════════════════════════════════
    # 会话管理
    # ═══════════════════════════════════════════

    def bind_session(self, task_id: str) -> None:
        """绑定到当前会话"""
        self._session_task_id = task_id
        self.working.bind_task(task_id)

    def add_user_message(self, content: str) -> None:
        """记录用户消息到工作记忆"""
        self.working.add_user_message(content)

    def add_assistant_message(self, content: str) -> None:
        """记录助手回复到工作记忆"""
        self.working.add_assistant_message(content)

    def clear_session(self) -> None:
        """清空当前会话记忆"""
        self.working.clear_session()
        self._session_task_id = None

    # ═══════════════════════════════════════════
    # 记忆巩固（后台维护）
    # ═══════════════════════════════════════════

    async def consolidate(self) -> dict:
        """后台记忆巩固任务

        执行：
        1. 从最近完成的 AgentTask 中提取高质量内容，向量化存入
        2. 清理低分记忆（预留）
        3. 返回统计数据

        应由外部 cron/定时任务触发，每小时一次。
        """
        stats: dict[str, int] = {"semantic_stored": 0}

        try:
            from src.database import SessionLocal
            from src.models import AgentTask as AgentTaskModel

            with SessionLocal() as db:
                recent_tasks = (
                    db.query(AgentTaskModel)
                    .filter(
                        AgentTaskModel.status == "completed",
                        AgentTaskModel.total_tokens > 0,
                    )
                    .order_by(AgentTaskModel.completed_at.desc())
                    .limit(20)
                    .all()
                )
        except Exception as e:
            logger.warning(f"MemoryManager.consolidate: 查询失败: {e}")
            return stats

        for task in recent_tasks:
            result_json = getattr(task, "result_json", None)
            if not result_json:
                continue
            try:
                import json
                result_data = json.loads(result_json)
                sections = result_data.get("sections", [])
                for sec in sections:
                    output = sec.get("output", {})
                    content = (output or {}).get("content", "")
                    if content and len(content) > 100:
                        self.semantic.store(
                            text=content.strip(),
                            namespace="content",
                            metadata={
                                "source": "consolidation",
                                "agent_type": sec.get("agent", ""),
                            },
                        )
                        stats["semantic_stored"] += 1
            except Exception:
                continue

        logger.info(f"MemoryManager: 巩固完成 — {stats}")
        return stats


# ── 模块级单例 ──
memory_manager = MemoryManager()
