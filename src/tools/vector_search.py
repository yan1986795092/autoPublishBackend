"""VectorSearch 工具 — 语义相似内容检索（基于 pgvector）"""

from __future__ import annotations
import json
import logging
from typing import Any, Optional

from sqlalchemy import text

from src.database import SessionLocal
from src.config import get_settings

logger = logging.getLogger(__name__)

# 嵌入模型配置（2026/07/03 起从 Zhipu embedding-2 切换到 Qwen text-embedding-v3）
# 两者均为 1024 维、OpenAI 协议兼容，无需迁移数据即可切换。
# 通过 settings.embedding_provider 字段可回退到 zhipu（保留旧的 Zhipu API Key 路径）。


def _resolve_embedding_config():
    """根据 settings.embedding_provider 返回 embedding 配置。

    Returns:
        (api_key, base_url, model, dim, provider_label)
    """
    settings = get_settings()
    provider = (settings.embedding_provider or "qwen").lower()

    if provider == "qwen":
        return (
            settings.qwen_api_key,
            settings.qwen_base_url,
            settings.qwen_embedding_model,
            settings.embedding_dim,
            "Qwen",
        )

    # 回退：Zhipu（不推荐 — embedding-2 已过期 / 余额耗尽）
    return (
        settings.zhipu_api_key,
        settings.zhipu_base_url,
        "embedding-2",
        1024,
        "Zhipu",
    )


class VectorSearchTool:
    """语义向量搜索工具，使用 PostgreSQL + pgvector + Qwen Embedding"""

    name = "vector_search"
    description = "使用语义相似度在知识库中搜索相似内容"

    def __init__(self):
        self._embeddings = None
        self._pgvector_available = False
        self._embedding_dim: int | None = None
        self._provider_label: str | None = None

    def _ensure_pgvector(self) -> bool:
        """检查 pgvector 扩展是否可用"""
        if self._pgvector_available:
            return True
        try:
            __import__("pgvector")
            self._pgvector_available = True
            return True
        except ImportError:
            logger.warning("pgvector 未安装（pip install pgvector），向量搜索不可用")
            return False

    def _ensure_embeddings(self):
        """按 settings.embedding_provider 初始化 embedder。"""
        if self._embeddings is not None:
            return

        api_key, base_url, model_name, dim, label = _resolve_embedding_config()
        self._embedding_dim = dim
        self._provider_label = label

        if not api_key:
            logger.warning(
                "未配置 %s_api_key，向量搜索不可用。"
                "请在 .env 中设置 QWEN_API_KEY（或 ZHIPU_API_KEY 回退）",
                label,
            )
            return

        try:
            from langchain_openai import OpenAIEmbeddings

            self._embeddings = OpenAIEmbeddings(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
            )
            logger.info(
                "VectorSearch: 使用 %s Embedding（模型=%s, 维度=%d）",
                label, model_name, dim,
            )
        except Exception as e:
            logger.warning("%s Embedding 初始化失败: %s", label, e)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim or 1024

    def _to_pgvector_str(self, embedding: list[float]) -> str:
        return "[" + ",".join(str(x) for x in embedding) + "]"

    # ── 写入 ──────────────────────────────────────────────

    def add_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> bool:
        self._ensure_embeddings()
        if not self._embeddings or not texts:
            return False
        if not self._ensure_pgvector():
            return False

        # 修复 BUG-2：清洗输入 — Qwen Embedding 拒绝非 str / 空字符串 / 控制字符，
        #     直接传会让 API 抛 'contents is neither str nor list of str' 然后整批失败。
        #     过滤非字符串 + 空串 + 过长文本（>8000 字符 Qwen 也会拒）。
        cleaned_texts: list[str] = []
        cleaned_metas: list[dict] = []
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                logger.warning("VectorSearch 跳过非字符串文本 (index=%s, type=%s)", i, type(t).__name__)
                continue
            t2 = "".join(ch for ch in t if ch == "\n" or ch == "\t" or ch >= " ")
            t2 = t2.strip()
            if not t2:
                continue
            if len(t2) > 8000:
                t2 = t2[:8000]
            cleaned_texts.append(t2)
            if metadatas and i < len(metadatas):
                cleaned_metas.append(metadatas[i])
        if not cleaned_texts:
            logger.warning("VectorSearch 全部文本清洗后为空，跳过写入")
            return False

        try:
            embeddings = self._embeddings.embed_documents(cleaned_texts)
            metas_to_use = cleaned_metas if cleaned_metas else ([{}] * len(cleaned_texts))

            db = SessionLocal()
            try:
                for i, doc_text in enumerate(cleaned_texts):
                    embedding_str = self._to_pgvector_str(embeddings[i])
                    meta_json = json.dumps(metas_to_use[i], ensure_ascii=False) if metas_to_use[i] else "{}"
                    namespace = metas_to_use[i].get("namespace", "general") if metas_to_use[i] else "general"

                    db.execute(
                        text(
                            "INSERT INTO vector_search (text, metadata, embedding, namespace) "
                            f"VALUES (:text, CAST(:metadata AS jsonb), '{embedding_str}'::vector, :namespace)"
                        ),
                        {
                            "text": doc_text,
                            "metadata": meta_json,
                            "namespace": namespace,
                        },
                    )
                db.commit()
                return True
            except Exception as e:
                db.rollback()
                logger.warning("VectorSearch 写入失败: %s", e)
                return False
            finally:
                db.close()

        except Exception as e:
            logger.warning("VectorSearch 嵌入生成失败: %s", e)
            return False

    # ── 检索 ──────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        namespace: Optional[str] = None,
        metadata_filter: Optional[dict[str, str]] = None,
    ) -> dict:
        self._ensure_embeddings()
        if not self._embeddings:
            return {"query": query, "matches": [], "total": 0, "fallback": "embedding_unavailable"}
        if not self._ensure_pgvector():
            return {"query": query, "matches": [], "total": 0, "fallback": "pgvector_unavailable"}

        # 修复 BUG-Qwen-400：query 必须是字符串且非空，否则 Qwen 抛 'contents is neither
        # str nor list of str' 400 错误，整批失败。统一清洗再交给 embed_query。
        if not isinstance(query, str) or not query.strip():
            logger.warning(
                "VectorSearch search() 跳过：query 非法 (type=%s, empty=%s)",
                type(query).__name__, not query,
            )
            return {"query": str(query) if query is not None else "",
                    "matches": [], "total": 0, "fallback": "invalid_query"}
        cleaned_query = "".join(
            ch for ch in query if ch == "\n" or ch == "\t" or ch >= " "
        ).strip()
        if not cleaned_query:
            return {"query": "", "matches": [], "total": 0, "fallback": "empty_query"}
        if len(cleaned_query) > 8000:
            cleaned_query = cleaned_query[:8000]

        try:
            query_embedding = self._embeddings.embed_query(cleaned_query)
            embedding_str = self._to_pgvector_str(query_embedding)

            db = SessionLocal()
            try:
                where_clauses: list[str] = []
                params: dict[str, Any] = {"limit": top_k}
                if namespace:
                    where_clauses.append("namespace = :namespace")
                    params["namespace"] = namespace
                if metadata_filter:
                    for idx, (key, value) in enumerate(metadata_filter.items()):
                        param_name = f"meta_{idx}"
                        where_clauses.append(f"metadata ->> '{key}' = :{param_name}")
                        params[param_name] = str(value)
                where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
                rows = db.execute(
                    text(
                        "SELECT id, text, metadata, namespace, "
                        f"  (embedding <=> '{embedding_str}'::vector) AS distance "
                        f"FROM vector_search "
                        f"{where_sql} "
                        f"ORDER BY embedding <=> '{embedding_str}'::vector "
                        f"LIMIT :limit"
                    ),
                    params,
                ).fetchall()

                matches = []
                for row in rows:
                    meta = {}
                    try:
                        meta = json.loads(row.metadata) if row.metadata else {}
                    except (json.JSONDecodeError, TypeError):
                        pass
                    matches.append({
                        "content": row.text,
                        "distance": float(row.distance),
                        "metadata": meta,
                        "namespace": row.namespace,
                    })

                return {"query": query, "matches": matches, "total": len(matches)}

            except Exception as e:
                logger.warning("VectorSearch SQL 查询失败: %s", e)
                return {"query": query, "matches": [], "total": 0, "error": str(e)}
            finally:
                db.close()

        except Exception as e:
            logger.warning("VectorSearch 查询嵌入生成失败: %s", e)
            return {"query": query, "matches": [], "total": 0, "error": str(e)}

    # ── 管理 ──────────────────────────────────────────────

    def count(self, namespace: Optional[str] = None) -> int:
        db = SessionLocal()
        try:
            if namespace:
                row = db.execute(
                    text("SELECT COUNT(*) FROM vector_search WHERE namespace = :ns"),
                    {"ns": namespace},
                ).scalar()
            else:
                row = db.execute(text("SELECT COUNT(*) FROM vector_search")).scalar()
            return row or 0
        except Exception as e:
            logger.warning("VectorSearch count 失败: %s", e)
            return 0
        finally:
            db.close()

    def clear(self, namespace: Optional[str] = None) -> None:
        db = SessionLocal()
        try:
            if namespace:
                db.execute(text("DELETE FROM vector_search WHERE namespace = :ns"), {"ns": namespace})
            else:
                db.execute(text("DELETE FROM vector_search"))
            db.commit()
        except Exception as e:
            logger.warning("VectorSearch clear 失败: %s", e)
            db.rollback()
        finally:
            db.close()


# 单例
vector_search = VectorSearchTool()
