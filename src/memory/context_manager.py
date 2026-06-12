"""ContextManager —— 混合路由、会话管理、Token 预算监控。"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .context_store import (
    SharedContextStore,
    CreativeAnchors,
    ProjectManifest,
    ProjectStage,
    VALID_TRANSITIONS,
    ThinkingStep,
)

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.75
DEFAULT_COMPRESSION_THRESHOLD_PCT = 0.80
DEFAULT_TARGET_BUDGET_PCT = 0.70


def _iso_now() -> str:
    return datetime.now().isoformat()


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _keyword_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    overlap = len(words_a & words_b)
    return overlap / min(len(words_a), len(words_b))


def _serialize_thinking_trace(steps: List[ThinkingStep]) -> str:
    return "\n".join(f"{s.agent}:{s.action}:{s.detail}" for s in steps)


def _serialize_creative_log(log: List) -> str:
    return "\n".join(f"{d.agent}:{d.decision_text}" for d in log)


class ContextManager:
    """
    管理跨对话的多轮上下文。

    职责：
    - 混合路由：相似度 + 显式覆盖，决定新建还是继续会话
    - 会话生命周期：创建、跟踪、归档
    - Token 预算监控与上下文压缩
    - 带验证的阶段转换
    """

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        compression_threshold_pct: float = DEFAULT_COMPRESSION_THRESHOLD_PCT,
        target_budget_pct: float = DEFAULT_TARGET_BUDGET_PCT,
    ):
        self.similarity_threshold = similarity_threshold
        self.compression_threshold_pct = compression_threshold_pct
        self.target_budget_pct = target_budget_pct
        self._active_sessions: Dict[str, SharedContextStore] = {}
        self._session_history: List[SharedContextStore] = []

    # ═══════════════════════════════════════════
    # 消息处理 —— 路由
    # ═══════════════════════════════════════════

    def handle_message(
        self,
        user_message: str,
        conversation_id: str,
        project_id: Optional[str] = None,
        project_directive: Optional[str] = None,
    ) -> SharedContextStore:
        """
        主入口。判断是继续现有会话还是新建会话，
        然后返回相应的 SharedContextStore。
        """
        if project_directive == "new_session":
            return self._start_new_session(user_message, conversation_id, project_id)

        existing = self._get_active_session(conversation_id)
        if existing is None:
            return self._start_new_session(user_message, conversation_id, project_id)

        similarity = self._compute_similarity(user_message, existing.user_request)
        existing.similarity_score = similarity

        if similarity >= self.similarity_threshold:
            return self._continue_session(existing, user_message)
        else:
            return self._start_new_session(user_message, conversation_id, project_id)

    def _get_active_session(self, conversation_id: str) -> Optional[SharedContextStore]:
        for store in reversed(list(self._active_sessions.values())):
            if store.conversation_id == conversation_id:
                return store
        return None

    def _start_new_session(
        self,
        user_message: str,
        conversation_id: str,
        project_id: Optional[str],
    ) -> SharedContextStore:
        old = self._get_active_session(conversation_id)
        if old:
            self._archive_session(old)

        session_id = str(uuid.uuid4())
        manifest = ProjectManifest(
            project_id=project_id or session_id,
            title="",
            stage=ProjectStage.CONCEPTUALIZATION,
            created_at=_iso_now(),
            turn_count=0,
        )

        store = SharedContextStore(
            session_id=session_id,
            conversation_id=conversation_id,
            user_request=user_message,
            project_manifest=manifest,
            is_new_session=True,
        )
        self._active_sessions[session_id] = store
        logger.info(f"ContextManager: 新建会话 {session_id}，所属对话 {conversation_id}")
        return store

    def _continue_session(
        self,
        store: SharedContextStore,
        user_message: str,
    ) -> SharedContextStore:
        store.user_request = user_message
        store.is_new_session = False
        if store.project_manifest:
            store.project_manifest.turn_count += 1
        store.add_thinking_step(agent="ContextManager", action="route", detail="continue_session")
        logger.debug(
            f"ContextManager: 继续会话 {store.session_id}，"
            f"轮次 {store.project_manifest.turn_count if store.project_manifest else 0}"
        )
        return store

    def _archive_session(self, store: SharedContextStore) -> None:
        self._session_history.append(store)
        if store.session_id in self._active_sessions:
            del self._active_sessions[store.session_id]
        logger.info(f"ContextManager: 已归档会话 {store.session_id}")

    # ═══════════════════════════════════════════
    # 相似度计算
    # ═══════════════════════════════════════════

    def _compute_similarity(self, message_a: str, message_b: str) -> float:
        """
        计算两条消息之间的相似度。
        如果嵌入向量不可用，回退到关键词重叠。
        """
        try:
            from .semantic_memory import SemanticMemory
            sm = SemanticMemory()
            if sm.available:
                vec_a = sm._embed_text(message_a)
                vec_b = sm._embed_text(message_b)
                if vec_a is not None and vec_b is not None:
                    return _cosine_similarity(vec_a, vec_b)
        except Exception:
            pass

        return _keyword_similarity(message_a, message_b)

    # ═══════════════════════════════════════════
    # 阶段转换
    # ═══════════════════════════════════════════

    def transition_stage(
        self,
        store: SharedContextStore,
        new_stage: ProjectStage,
    ) -> bool:
        """尝试进行项目阶段转换。如果有效返回 True，否则返回 False。"""
        manifest = store.project_manifest
        if not manifest:
            logger.warning("transition_stage: 没有项目清单")
            return False

        current = manifest.stage
        if new_stage not in VALID_TRANSITIONS.get(current, set()):
            logger.warning(
                f"无效的阶段转换: {current.value} -> {new_stage.value}"
            )
            store.add_thinking_step(
                agent="ContextManager",
                action="stage_transition_rejected",
                detail=f"{current.value} -> {new_stage.value}",
            )
            return False

        manifest.stage = new_stage
        store.add_thinking_step(
            agent="ContextManager",
            action="stage_transition",
            detail=f"{current.value} -> {new_stage.value}",
        )
        logger.info(f"ContextManager: 阶段转换 {current.value} -> {new_stage.value}")
        return True

    # ═══════════════════════════════════════════
    # Token 预算和压缩
    # ═══════════════════════════════════════════

    def estimate_token_count(self, store: SharedContextStore) -> int:
        """使用字符数 / 4 进行粗略的 Token 估算。"""
        parts = [
            store.user_request or "",
            _str_of(store.project_manifest),
            _str_of(store.creative_anchors),
            _serialize_thinking_trace(store.thinking_trace),
            _serialize_creative_log(store.creative_log),
            _str_of_list(store.writing_content),
            _str_of_list(store.review_feedback),
            _str_of_list(store.seo_suggestions),
        ]
        total_chars = sum(len(p) for p in parts)
        return total_chars // 4

    def should_compress(self, store: SharedContextStore, threshold_pct: float = 0.80) -> bool:
        """检查上下文是否超过 Token 预算阈值。"""
        tokens = self.estimate_token_count(store)
        budget = 32000
        return (tokens / budget) >= threshold_pct

    # ── 按阶段保留限制（用于压缩） ─────────────────────────
    _STAGE_RETENTION: dict = {
        # (priority_field, max_items_per_stage)
        "thinking_trace": {
            ProjectStage.CONCEPTUALIZATION: 30,
            ProjectStage.OUTLINING: 25,
            ProjectStage.DRAFTING: 20,
            ProjectStage.REVISION: 15,
            ProjectStage.POLISHING: 10,
            ProjectStage.COMPLETED: 5,
            ProjectStage.ABANDONED: 0,
        },
        "writing_content": {
            ProjectStage.CONCEPTUALIZATION: -1,  # -1 = 保留全部
            ProjectStage.OUTLINING: 10,
            ProjectStage.DRAFTING: 5,
            ProjectStage.REVISION: 3,
            ProjectStage.POLISHING: 2,
            ProjectStage.COMPLETED: 2,
            ProjectStage.ABANDONED: 0,
        },
        # review_feedback 和 seo_suggestions 始终与阶段无关
        "review_feedback": {
            stage: 2 for stage in ProjectStage
        },
        "seo_suggestions": {
            stage: 1 for stage in ProjectStage
        },
    }

    def compress_context(self, store: SharedContextStore) -> None:
        """
        按阶段感知的 4 个优先级级别应用压缩。

        优先级 1: thinking_trace —— 按阶段决定保留数量（概念阶段保留更长）
        优先级 2: review_feedback —— 保留最近 2 个块
        优先级 3: seo_suggestions —— 保留最近 1 个块
        优先级 4: writing_content —— 按阶段决定保留数量（早期阶段保留更多）

        永不压缩的内容：
        - creative_anchors —— 不可变的创作状态
        - creative_log —— 决策审计追踪
        - user_request —— 当前活跃任务
        """
        manifest = store.project_manifest
        stage = manifest.stage if manifest else ProjectStage.DRAFTING

        store.add_thinking_step(
            agent="ContextManager",
            action="compress_start",
            detail=f"stage={stage.value}",
        )

        def _cap(items: list, limit: int) -> list:
            """保留最近 N 项；-1 表示保留全部。"""
            if limit == -1 or len(items) <= limit:
                return items
            return items[-limit:]

        # ── 优先级 1: thinking_trace ──────────────────────────────────────
        tt_limit = self._STAGE_RETENTION["thinking_trace"].get(stage, 20)
        if len(store.thinking_trace) > tt_limit:
            old_len = len(store.thinking_trace)
            store.thinking_trace = _cap(list(store.thinking_trace), tt_limit)
            logger.debug(
                "compress: thinking_trace %d → %d (limit=%d, stage=%s)",
                old_len, len(store.thinking_trace), tt_limit, stage.value,
            )

        # ── 优先级 2: review_feedback —— 保留最近 2 个 ─────────────────────
        fb_limit = self._STAGE_RETENTION["review_feedback"].get(stage, 2)
        if len(store.review_feedback) > fb_limit:
            store.review_feedback = _cap(list(store.review_feedback), fb_limit)

        # ── 优先级 3: seo_suggestions —— 保留最近 1 个 ──────────────────────
        seo_limit = self._STAGE_RETENTION["seo_suggestions"].get(stage, 1)
        if len(store.seo_suggestions) > seo_limit:
            store.seo_suggestions = _cap(list(store.seo_suggestions), seo_limit)

        # ── 优先级 4: writing_content —— 按阶段决定 ──────────────────
        wc_limit = self._STAGE_RETENTION["writing_content"].get(stage, 5)
        if wc_limit != -1 and len(store.writing_content) > wc_limit:
            old_len = len(store.writing_content)
            store.writing_content = _cap(list(store.writing_content), wc_limit)
            logger.debug(
                "compress: writing_content %d → %d (limit=%d, stage=%s)",
                old_len, len(store.writing_content), wc_limit, stage.value,
            )

        store.add_thinking_step(
            agent="ContextManager",
            action="compress_end",
            detail=(
                f"trace={len(store.thinking_trace)} "
                f"content={len(store.writing_content)} "
                f"feedback={len(store.review_feedback)} "
                f"seo={len(store.seo_suggestions)}"
            ),
        )
        logger.info(
            "compress_context: 完成, stage=%s, trace=%d, content=%d",
            stage.value, len(store.thinking_trace), len(store.writing_content),
        )

    # ═══════════════════════════════════════════
    # 会话管理
    # ═══════════════════════════════════════════

    def get_active_session(self, conversation_id: str) -> Optional[SharedContextStore]:
        return self._get_active_session(conversation_id)

    def clear_session(self, session_id: str) -> None:
        if session_id in self._active_sessions:
            store = self._active_sessions[session_id]
            self._archive_session(store)

    def get_session_history(self) -> List[SharedContextStore]:
        return self._session_history


def _str_of(obj: Any) -> str:
    if obj is None:
        return ""
    if hasattr(obj, "__dict__"):
        return str(obj.__dict__)
    return str(obj)


def _str_of_list(items: Any) -> str:
    """将数据类项目列表序列化，用于 Token 计数。"""
    if not items:
        return ""
    if isinstance(items, list):
        return "\n".join(_str_of(item) for item in items)
    return _str_of(items)
