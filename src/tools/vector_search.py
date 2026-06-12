"""VectorSearch 工具 — 语义相似内容检索（基于 pgvector）

直接在 PostgreSQL 中存储和检索向量，无需额外向量数据库。
依赖 PostgreSQL 的 pgvector 扩展 + langchain 的 OpenAIEmbeddings。
"""

from __future__ import annotations
import json
import logging
import os
from typing import Any, Optional

from sqlalchemy import text

from src.database import SessionLocal

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 1536  # OpenAI text-embedding-ada-002


class VectorSearchTool:
    """语义向量搜索工具，使用 PostgreSQL + pgvector + OpenAI Embeddings"""

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
        if self._embeddings is not None:
            return
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from langchain_openai import OpenAIEmbeddings
                self._embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            except Exception as e:
                logger.warning("嵌入模型初始化失败：%s", e)
        else:
            logger.warning("未配置 OPENAI_API_KEY，向量搜索不可用")

    def _to_pgvector_str(self, embedding: list[float]) -> str:
        """将 embedding 列表格式化为 pgvector 可用的字符串"""
        return "[" + ",".join(str(x) for x in embedding) + "]"

    # ── 写入 ──────────────────────────────────────────────

    def add_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> bool:
        """向向量库添加文档

        Args:
            texts: 文档文本列表
            metadatas: 可选元数据列表
            ids: 可选 ID 列表（仅用于兼容旧接口，pgvector 使用 SERIAL 自增）

        Returns:
            是否成功写入
        """
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
                            "VALUES (:text, :metadata::jsonb, :embedding::vector, :namespace)"
                        ),
                        {
                            "text": doc_text,
                            "metadata": meta_json,
                            "embedding": embedding_str,
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
        """搜索相似内容

        Args:
            query: 查询文本
            top_k: 返回条数

        Returns:
            {"query": str, "matches": [...], "total": int}
        """
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
                        "  (embedding <=> :query_vec::vector) AS distance "
                        "FROM vector_search "
                        "ORDER BY embedding <=> :query_vec2::vector "
                        "LIMIT :limit"
                    ),
                    {"query_vec": embedding_str, "query_vec2": embedding_str, "limit": top_k},
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
        """统计条目数"""
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
        """清空数据"""
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
