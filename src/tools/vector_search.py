"""VectorSearch 工具 — 语义相似内容检索（基于 pgvector）"""

from __future__ import annotations
import json
import logging
from typing import Any, Optional

import httpx
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


# DashScope 原生 embedding 端点
_DASHSCOPE_EMBEDDING_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"


class _DashScopeEmbedder:
    """DashScope 原生 embedding 封装，替代 LangChain OpenAIEmbeddings。

    DashScope 的 text-embedding-v3 不在 /v1/embeddings 兼容端点开放，
    必须用原生端点 POST .../text-embedding/text-embedding。
    """

    def __init__(self, api_key: str, model: str, dim: int):
        self._api_key = api_key
        self._model = model
        self._dim = dim

    def embed_query(self, text: str) -> list[float]:
        return self._embed_single(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch(texts)

    def _embed_single(self, text: str) -> list[float]:
        # 单条: input.texts 是数组，批量: input 直接是数组
        resp = httpx.post(
            _DASHSCOPE_EMBEDDING_URL,
            json={"input": {"texts": [text]}, "model": self._model, "parameters": {"dimension": self._dim}},
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            timeout=30.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope embedding failed {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        emb = data.get("output", {}).get("embeddings", [{}])[0].get("embedding")
        if not emb or len(emb) != self._dim:
            raise RuntimeError(f"DashScope embedding dim mismatch or empty: {data}")
        return emb

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        # DashScope: 单条和批量都用 {"input": {"texts": [...]}}
        resp = httpx.post(
            _DASHSCOPE_EMBEDDING_URL,
            json={"input": {"texts": texts}, "model": self._model, "parameters": {"dimension": self._dim}},
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            timeout=60.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope batch embedding failed {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        # 响应格式: {"output": {"embeddings": [{"embedding": [...], "text_index": 0}, ...]}
        embeddings_out = data.get("output", {}).get("embeddings", [])
        if not embeddings_out:
            raise RuntimeError(f"DashScope batch embedding format error: {data}")
        results = []
        for item in embeddings_out:
            emb = item.get("embedding", [])
            results.append(emb if emb else [0.0] * self._dim)
        return results


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
        """按 settings.embedding_provider 初始化 embedder。

        注意：DashScope 的 text-embedding-v3 不在 /v1/embeddings 兼容端点开放，
        LangChain OpenAIEmbeddings 会失败。统一使用自定义 DashScope 原生 API embedder。
        """
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

        # 使用自定义 DashScope 原生 embedder（不接受 LangChain OpenAIEmbeddings）
        self._embeddings = _DashScopeEmbedder(api_key, model_name, dim)
        logger.info(
            "VectorSearch: 使用 DashScope Embedding（模型=%s, 维度=%d）",
            model_name, dim,
        )

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim or 1024

    def _embed_query_fallback(self, text: str) -> list[float] | None:
        """直连 HTTP 调 DashScope 原生 Embedding API（langchain 失败时兜底）"""
        api_key, base_url, model_name, dim, label = _resolve_embedding_config()
        if not api_key:
            return None
        # DashScope 原生 embedding 端点（不是 /v1/embeddings 兼容端点）
        url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        try:
            resp = httpx.post(
                url,
                json={"input": {"texts": [text]}, "model": model_name, "parameters": {"dimension": dim}},
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            if resp.status_code != 200:
                logger.warning(
                    "DashScope embedding fallback HTTP %d: %s",
                    resp.status_code, resp.text[:300],
                )
                return None
            data = resp.json()
            # DashScope 原生格式: {"output": {"embeddings": [{"text": "...", "embedding": [...]}]}}
            embeddings = data.get("output", {}).get("embeddings", [])
            if embeddings and len(embeddings) > 0:
                emb = embeddings[0].get("embedding")
                if emb and len(emb) == (dim or 1024):
                    return emb
                # 兼容另一种格式：embedding 直接在 output 下
                emb = data.get("output", {}).get("embedding")
                if emb and len(emb) == (dim or 1024):
                    return emb
            logger.warning("DashScope embedding fallback 返回格式异常: %s", str(data)[:200])
            return None
        except Exception as e:
            logger.warning("DashScope embedding fallback 异常: %s", e)
            return None

    def _embed_documents_fallback(self, texts: list[str]) -> list[list[float]] | None:
        """直连 HTTP 批量嵌入（langchain 失败时兜底）"""
        api_key, base_url, model_name, dim, label = _resolve_embedding_config()
        if not api_key:
            return None
        try:
            resp = httpx.post(
                _DASHSCOPE_EMBEDDING_URL,
                json={"input": {"texts": texts}, "model": model_name, "parameters": {"dimension": dim}},
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
            if resp.status_code != 200:
                logger.warning(
                    "DashScope documents embedding fallback HTTP %d: %s",
                    resp.status_code, resp.text[:300],
                )
                return None
            data = resp.json()
            embeddings_out = data.get("output", {}).get("embeddings", [])
            if not embeddings_out:
                logger.warning("DashScope batch embedding fallback 返回格式异常: %s",
                             str(data)[:200])
                return None
            results = []
            for item in embeddings_out:
                emb = item.get("embedding", [])
                if emb and len(emb) == (dim or 1024):
                    results.append(emb)
                else:
                    results.append([0.0] * (dim or 1024))
            return results if results else None
        except Exception as e:
            logger.warning("DashScope documents embedding fallback 异常: %s", e)
            return None

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

        # 清洗输入 — 过滤非字符串 + 空串 + 过长文本（>8000 DashScope 截断）
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

        # 获取嵌入：优先用 embedder（DashScope 原生 API），失败时用 fallback
        embeddings: list[list[float]] | None = None
        try:
            embeddings = self._embeddings.embed_documents(cleaned_texts)
        except Exception as embed_err:
            logger.warning("VectorSearch embed_documents 失败，尝试 fallback: %s", embed_err)
            embeddings = self._embed_documents_fallback(cleaned_texts)
            if embeddings is None:
                logger.warning("VectorSearch 嵌入全部失败，跳过写入")
                return False

        metas_to_use = cleaned_metas if cleaned_metas else ([{}] * len(cleaned_texts))

        db = SessionLocal()
        try:
            for i, doc_text in enumerate(cleaned_texts):
                emb = embeddings[i] if i < len(embeddings) else [0.0] * self.embedding_dim
                embedding_str = self._to_pgvector_str(emb)
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
        except Exception as embed_err:
            logger.warning("VectorSearch embed_query 失败，尝试 fallback: %s", embed_err)
            query_embedding = self._embed_query_fallback(cleaned_query)
            if query_embedding is None:
                return {"query": query, "matches": [], "total": 0, "error": f"embedding failed: {embed_err}"}

        try:
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
