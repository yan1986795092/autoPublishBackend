"""语义记忆 — 基于向量检索的长期知识存储

包装已有的 VectorSearchTool，提供带命名空间的语义存储和检索。
"""

from __future__ import annotations
import logging
from typing import Optional

from .schemas import MemoryItem

logger = logging.getLogger(__name__)


# 命名空间定义：按内容类型隔离，防止互相干扰
NAMESPACES = {
    "content":  "已生成的高质量文章内容",
    "knowledge": "WebSearch 沉淀的领域知识",
    "style":    "用户写作风格样本",
    "feedback": "用户修正反馈和审核经验",
    "domain":   "领域专业知识（技术/营销/...）",
    "template": "高复用 Prompt 模板",
}


class SemanticMemory:
    """语义记忆：向量化写入和检索，多命名空间隔离

    使用方式：
        mem = SemanticMemory()
        mem.store("好内容...", namespace="content")
        results = mem.retrieve("查询", namespace="content")
    """

    def __init__(self):
        self._vs = None
        self._ready = False
        self._init_vector_search()

    def _init_vector_search(self) -> None:
        """延迟初始化向量搜索引擎"""
        try:
            from src.tools.vector_search import vector_search
            # 触发一次空查询验证连接
            vector_search.search(query="__init__", top_k=1)
            self._vs = vector_search
            self._ready = True
            logger.info("SemanticMemory: VectorSearchTool 就绪")
        except Exception as e:
            logger.warning(f"SemanticMemory: VectorSearchTool 不可用 ({e})，降级为纯文本模式")

    @property
    def available(self) -> bool:
        return self._ready

    # ── 写入 ──

    def store(self, text: str, namespace: str = "general",
              metadata: Optional[dict] = None, skip_small: bool = True) -> bool:
        """存入一条语义记忆

        Args:
            text: 要存储的文本
            namespace: 命名空间（见 NAMESPACES）
            metadata: 附加元数据
            skip_small: 跳过过短的文本（<50 字符）

        Returns:
            是否成功写入
        """
        if not text or not text.strip():
            return False
        if skip_small and len(text.strip()) < 50:
            return False
        if namespace not in NAMESPACES:
            namespace = "general"

        if not self._ready:
            return False

        meta = {
            "namespace": namespace,
            "source": metadata.get("source", "") if metadata else "",
            "agent_type": metadata.get("agent_type", "") if metadata else "",
            "user_id": metadata.get("user_id", "") if metadata else "",
        }

        try:
            self._vs.add_documents(
                texts=[text.strip()],
                metadatas=[meta],
            )
            return True
        except Exception as e:
            logger.warning(f"SemanticMemory.store 失败: {e}")
            return False

    # ── 检索 ──

    def retrieve(self, query: str, namespace: Optional[str] = None,
                 top_k: int = 5, min_score: float = 0.0) -> list[MemoryItem]:
        """检索语义记忆

        Args:
            query: 查询文本
            namespace: 筛选命名空间（None 为全部）
            top_k: 返回条数
            min_score: 最低相似度阈值

        Returns:
            按相关性降序排列的 MemoryItem 列表
        """
        if not self._ready or not query:
            return []

        try:
            results = self._vs.search(query=query, top_k=top_k * 2)
            matches = results.get("matches", [])
        except Exception as e:
            logger.warning(f"SemanticMemory.retrieve 失败: {e}")
            return []

        items = []
        for m in matches:
            doc = m.get("content", "")
            dist = m.get("distance", 0)
            meta = m.get("metadata", {}) or {}
            score = 1.0 - dist  # 距离越小越相似

            if min_score and score < min_score:
                continue
            if namespace and meta.get("namespace") != namespace:
                continue

            items.append(MemoryItem(
                text=doc,
                namespace=meta.get("namespace", "general"),
                score=score,
                metadata=meta,
            ))

        # 按分数降序，取 top_k
        items.sort(key=lambda x: x.score, reverse=True)
        return items[:top_k]

    def retrieve_formatted(self, query: str, namespace: Optional[str] = None,
                           top_k: int = 3, max_chars_per_item: int = 400) -> str:
        """检索并格式化为 prompt 可用的字符串

        Returns:
            格式化后的文本段落，空字符串表示无结果
        """
        items = self.retrieve(query=query, namespace=namespace, top_k=top_k)
        if not items:
            return ""

        parts = ["## 参考知识"]
        for item in items:
            ns_label = NAMESPACES.get(item.namespace, item.namespace)
            content = item.to_prompt(max_chars=max_chars_per_item)
            parts.append(f"  [{ns_label}] {content}")

        return "\n".join(parts)

    # ── 管理 ──

    def count(self, namespace: Optional[str] = None) -> int:
        """统计记忆数量（目前仅支持全局计数）"""
        if not self._ready:
            return 0
        return 0  # ChromaDB 无轻量计数接口，预留

    def clear(self) -> None:
        """清空记忆"""
        self._vs = None
        self._ready = False
        self._init_vector_search()
