"""VectorSearch 工具 — 语义相似内容检索（基于 pgvector）

支持多个 embedding 提供商，自动检测可用的 API key。
维度根据提供商自动适配。
"""

from __future__ import annotations
import json
import logging
from typing import Any, Optional

from sqlalchemy import text

from src.database import SessionLocal
from src.config import get_settings

logger = logging.getLogger(__name__)

# 固定使用 Zhipu embedding-2 模型（1024 维）
# 用户已配置 zhipu_api_key，且保持一致维度便于存储和检索
_EMBEDDING_DIM = 1024
_EMBEDDING_MODEL = "embedding-2"
_EMBEDDING_PROVIDER = "zhipu"


class VectorSearchTool:
    """语义向量搜索工具，使用 PostgreSQL + pgvector + Zhipu Embedding"""

    name = "vector_search"
    description = "使用语义相似度在知识库中搜索相似内容"

    def __init__(self):
        self._embeddings = None
        self._pgvector_available = False

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
        """使用 Zhipu embedding-2 模型初始化"""
        if self._embeddings is not None:
            return

        settings = get_settings()
        api_key = settings.zhipu_api_key
        if not api_key:
            logger.warning(
                "未配置 zhipu_api_key，向量搜索不可用。"
                "请在 .env 中设置 ZHIPU_API_KEY"
            )
            return

        try:
            from langchain_openai import OpenAIEmbeddings

            self._embeddings = OpenAIEmbeddings(
                model=_EMBEDDING_MODEL,
                openai_api_key=api_key,
                openai_api_base=settings.zhipu_base_url,
            )
            logger.info(
                "VectorSearch: 使用 Zhipu Embedding（模型=%s, 维度=%d）",
                _EMBEDDING_MODEL, _EMBEDDING_DIM,
            )
        except Exception as e:
            logger.warning("Zhipu Embedding 初始化失败: %s", e)

    @property
    def embedding_dim(self) -> int:
        return _EMBEDDING_DIM

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

        try:
            embeddings = self._embeddings.embed_documents(texts)
            metadatas = metadatas or [{}] * len(texts)

            db = SessionLocal()
            try:
                for i, doc_text in enumerate(texts):
                    embedding_str = self._to_pgvector_str(embeddings[i])
                    meta_json = json.dumps(metadatas[i], ensure_ascii=False) if metadatas[i] else "{}"
                    namespace = metadatas[i].get("namespace", "general") if metadatas[i] else "general"

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

    def search(self, query: str, top_k: int = 5) -> dict:
        self._ensure_embeddings()
        if not self._embeddings:
            return {"query": query, "matches": [], "total": 0, "fallback": "embedding_unavailable"}
        if not self._ensure_pgvector():
            return {"query": query, "matches": [], "total": 0, "fallback": "pgvector_unavailable"}

        try:
            query_embedding = self._embeddings.embed_query(query)
            embedding_str = self._to_pgvector_str(query_embedding)

            db = SessionLocal()
            try:
                rows = db.execute(
                    text(
                        "SELECT id, text, metadata, namespace, "
                        f"  (embedding <=> '{embedding_str}'::vector) AS distance "
                        f"FROM vector_search "
                        f"ORDER BY embedding <=> '{embedding_str}'::vector "
                        f"LIMIT :limit"
                    ),
                    {"limit": top_k},
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
