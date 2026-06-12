"""VectorSearch 工具 — 语义相似内容检索"""

from __future__ import annotations
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VectorSearchTool:
    """语义向量搜索工具，用于在知识库中检索相似内容"""

    name = "vector_search"
    description = "使用语义相似度在知识库中搜索相似内容"

    def __init__(self, persist_path: str = "./data/vector_db"):
        self._client = None
        self._collection = None
        self._embeddings = None
        self._persist_path = persist_path

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=self._persist_path)
            self._collection = self._client.get_or_create_collection("content_vectors")
        except ImportError:
            logger.warning("chromadb 未安装，VectorSearch 不可用")
        except Exception as e:
            logger.warning(f"ChromaDB 初始化失败：{e}")

    def _ensure_embeddings(self):
        if self._embeddings is not None:
            return
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from langchain_openai import OpenAIEmbeddings
                self._embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            except Exception as e:
                logger.warning(f"嵌入模型初始化失败：{e}")

    def add_documents(self, texts: list[str], metadatas: Optional[list[dict]] = None, ids: Optional[list[str]] = None):
        """向向量库添加文档"""
        self._ensure_client()
        self._ensure_embeddings()
        if not self._collection or not self._embeddings:
            return False
        try:
            embeddings = self._embeddings.embed_documents(texts)
            self._collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas or [{}] * len(texts),
                ids=ids or [str(i) for i in range(len(texts))],
            )
            return True
        except Exception as e:
            logger.warning(f"VectorSearch 添加失败：{e}")
            return False

    def search(self, query: str, top_k: int = 5) -> dict:
        """搜索相似内容"""
        self._ensure_client()
        self._ensure_embeddings()

        if not self._collection or not self._embeddings:
            return {"query": query, "matches": [], "total": 0, "fallback": "vector_search_unavailable"}

        try:
            query_embedding = self._embeddings.embed_query(query)
            results = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)

            matches = []
            if results.get("documents") and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    dist = results.get("distances", [[0]])[0][i] if results.get("distances") else 0
                    meta = results.get("metadatas", [[{}]])[0][i] if results.get("metadatas") else {}
                    matches.append({"content": doc, "distance": dist, "metadata": meta})

            return {"query": query, "matches": matches, "total": len(matches)}
        except Exception as e:
            logger.warning(f"VectorSearch 查询失败：{e}")
            return {"query": query, "matches": [], "total": 0, "error": str(e)}


# 单例
vector_search = VectorSearchTool()
