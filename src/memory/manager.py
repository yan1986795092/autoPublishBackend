"""MemoryManager — 统一记忆入口

所有 Agent 不直接调用下层存储，通过 MemoryManager 统一路由。
职责：
  - build_agent_context(): Agent 执行前，聚合相关记忆注入 prompt
  - remember(): Agent 执行后，选择性沉淀高质量产出
  - consolidate(): 后台记忆巩固任务
"""

from __future__ import annotations
import logging
import re
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
    "SearchAnalyzer": "knowledge",
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
        # 本地偏好缓存: key = "user_id:conversation_id"
        self._local_preferences: dict[str, list[str]] = {}

    # ═══════════════════════════════════════════
    # Agent 执行前：构建上下文
    # ═══════════════════════════════════════════

    async def build_agent_context(
        self,
        agent_type: str,
        task_input: dict,
        user_id: str = "",
        conversation_id: str = "",
        max_tokens: int = 2000,
    ) -> str:
        """为 Agent 构建增强 Prompt 上下文

        委托给 ContextBuilder，统一上下文构建逻辑。
        """
        if agent_type not in AGENT_MEMORY_ENABLED:
            return ""

        query = self._build_search_query(agent_type, task_input)
        history = task_input.get("_history", [])
        from src.memory.context_builder import ContextBuilder
        builder = ContextBuilder(self)
        context = await builder.build(
            agent_type=agent_type,
            user_id=user_id,
            query=query,
            history=history,
            conversation_id=conversation_id,
            max_tokens=max_tokens,
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
        conversation_id: str = "",
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
        if not content or not isinstance(content, str):
            return
        stripped = content.strip()
        if len(stripped) < 100:
            return

        content = stripped

        # 1. 写入语义记忆（按 conversation_id 隔离）
        namespace = AGENT_NAMESPACE_MAP.get(agent_type, "general")
        meta: dict[str, str] = {
            "agent_type": agent_type,
            "user_id": user_id,
            "source": "agent_output",
            "task_type": task_input.get("content_type", ""),
        }
        if conversation_id:
            meta["conversation_id"] = conversation_id
        self.semantic.store(
            text=content,
            namespace=namespace,
            metadata=meta,
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

    # ═══════════════════════════════════════════
    # 对话层记忆（新增）
    # ═══════════════════════════════════════════

    async def get_user_profile(self, user_id: str) -> str:
        """获取用户画像描述，注入对话 LLM 的 System Prompt

        检索优先级：语义记忆（向量）> 本地缓存（进程内存）> 工作记忆（当前会话）
        """
        if not user_id:
            return ""

        # 1. 语义记忆（向量搜索，按 user_id 隔离）
        if self.semantic.available:
            result = self.semantic.retrieve_formatted(
                query="用户偏好 写作平台 风格 语调 目标读者",
                namespace="user_conversation",
                top_k=5,
                max_chars_per_item=300,
                metadata_filter={"user_id": user_id},
            )
            if result:
                return result

        # 2. 本地缓存（向量搜索不可用时的兜底）
        cached = self._local_preferences.get(user_id)
        if cached:
            lines = ["## 用户历史偏好（本地缓存）"]
            lines.extend(f"- {p}" for p in cached[-5:])  # 最近 5 条
            return "\n".join(lines)

        # 3. 工作记忆（当前会话上下文）
        working = self.working.get_recent_context(max_chars=500)
        if working:
            return f"近期对话上下文：\n{working}"
        return ""

    async def remember_conversation_preference(
        self,
        user_id: str,
        preferences: dict,
        conversation_id: str = "",
    ) -> None:
        """将对话中发现的用户偏好存入语义记忆

        偏好按 conversation_id 隔离，每个窗口独立保存。
        **兜底 trigger**（_fallback=True）不写入记忆，避免污染。
        """
        if not user_id or not preferences:
            return

        # ── 兜底 trigger：仍然写入 topic（用于"重做"复用主题），
        # 但跳过 platform/style 等粗略字段（避免污染用户偏好画像）
        _is_fallback = preferences.get("_fallback", False)

        # ── topic 清洗：去掉 ## 思考/## 正文 等元描述 ──
        raw_topic = (preferences.get("topic") or "").strip()
        _meta_re = re.compile(r'(?m)^#{1,3}\s*(思考|正文|标题|引言|结论)\s*[:：]?\s*')
        cleaned_topic = _meta_re.sub('', raw_topic).strip()
        cleaned_topic = re.sub(r'\s+', ' ', cleaned_topic)
        if not cleaned_topic or len(cleaned_topic) < 3 or len(cleaned_topic) > 200:
            logger.info(
                "MemoryManager: topic 不合法，跳过整个 trigger 记忆 raw_topic=%r cleaned=%r",
                raw_topic[:50], cleaned_topic[:50],
            )
            return

        # ── 新增：过滤"纯动作"topic ──
        # 兜底分支可能会把"重新生成""直接生成""开始创作""再来"这种动作词当 topic，
        # 一旦存进记忆，下次"重新生成"就会读到这种垃圾，形成死循环。
        # 仅对兜底触发启用过滤（正常 LLM trigger 已通过规则 C 校验）。
        _action_only_re = re.compile(
            r'^[直接的马快立刻帮我给我]*(?:'
            r'生成|写|创作|开始|再来|重写|重新)'
            r'(?:一份|一篇|一下|个|吧|创作|写|生成)?'
            r'[的吧！!。.，,"\'\s]*$'
        )
        _placeholder_re = re.compile(r'^\[[^\]]+\]$')
        if _action_only_re.match(cleaned_topic) or _placeholder_re.match(cleaned_topic):
            logger.info(
                "MemoryManager: topic 为纯动作词或占位符，不写入记忆（避免污染）raw_topic=%r",
                cleaned_topic[:50],
            )
            return

        # 写回清洗后的 topic
        preferences = {**preferences, "topic": cleaned_topic}

        field_labels = {
            "topic": "主题",
            "platform": "平台",
            "style": "风格",
            "length": "篇幅",
            "tone": "语调",
            "audience": "目标读者",
        }

        # 兜底时只写 topic；正常 trigger 写全部字段
        if _is_fallback:
            _keys_to_write = ("topic",)
        else:
            _keys_to_write = tuple(field_labels.keys())

        parts = []
        for key in _keys_to_write:
            label = field_labels[key]
            val = preferences.get(key)
            if val and isinstance(val, str) and val.strip():
                parts.append(f"{label}={val}")

        keywords = preferences.get("keywords", [])
        if keywords and isinstance(keywords, list) and len(keywords) > 0:
            kw = [k for k in keywords if k]
            if kw:
                parts.append(f"关键词={', '.join(kw)}")

        if not parts:
            return

        short = "用户偏好: " + ", ".join(parts)

        # 1. 写入语义记忆（向量搜索可用时）
        if self.semantic.available:
            detail_lines = ["用户的创作偏好记录："]
            detail_lines.extend(f"- {p}" for p in parts)
            detail = "\n".join(detail_lines)
            meta = {
                "user_id": user_id,
                "source": "conversation",
                "summary": short,
            }
            if conversation_id:
                meta["conversation_id"] = conversation_id
            self.semantic.store(
                text=detail,
                namespace="user_conversation",
                metadata=meta,
            )

        # 2. 始终写入本地缓存（按 conversation_id 隔离）
        cache_key = f"{user_id}:{conversation_id}" if conversation_id else user_id
        if cache_key not in self._local_preferences:
            self._local_preferences[cache_key] = []
        self._local_preferences[cache_key].append(short)
        if len(self._local_preferences[cache_key]) > 20:
            self._local_preferences[cache_key] = self._local_preferences[cache_key][-20:]

        logger.info(
            "MemoryManager: 记录对话偏好 — %s",
            short,
        )


    def get_last_conversation_topic(
        self,
        user_id: str,
        conversation_id: str = "",
    ) -> str:
        """从本地缓存读取最近一次对话偏好的 topic。

        用于"重新生成"等重做场景：复用上次主题，不让兜底 topic 提取
        从 history 随便取一条 user 消息（可能是 [内容生成] 占位）。
        """
        if not user_id:
            return ""
        cache_key = f"{user_id}:{conversation_id}" if conversation_id else user_id
        cached = self._local_preferences.get(cache_key, [])
        if not cached:
            return ""

        # 过滤掉"纯动作词" / 占位符（如 历史中污染的 "重新生成"、"直接生成" 等）
        # 避免重做循环死锁。保留真正的 topic。
        _action_only_re = re.compile(
            r'^[直接的马快立刻帮我给我]*(?:'
            r'生成|写|创作|开始|再来|重写|重新)'
            r'(?:一份|一篇|一下|个|吧|创作|写|生成)?'
            r'[的吧！!。.，,"\'\s]*$'
        )
        _placeholder_re = re.compile(r'^\[[^\]]+\]$')

        # 倒序找最新一条含 topic= 的偏好
        for entry in reversed(cached):
            if not isinstance(entry, str):
                continue
            for part in entry.split(","):
                part = part.strip()
                if part.startswith("主题="):
                    topic = part[len("主题="):].strip()
                    if topic and topic != "null":
                        if _action_only_re.match(topic) or _placeholder_re.match(topic):
                            continue
                        return topic
        return ""


# ── 模块级单例 ──
memory_manager = MemoryManager()
