"""Shared helpers for the atomic MCP retrieval tools.

The helpers intentionally construct collection-specific retrievers for each
request. This matches ``query_knowledge_hub``'s freshness policy so data
written by the Dashboard becomes visible without restarting the MCP server.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.core.query_engine.dense_retriever import DenseRetriever, create_dense_retriever
from src.core.query_engine.sparse_retriever import SparseRetriever, create_sparse_retriever
from src.core.response.response_builder import MCPToolResponse, ResponseBuilder
from src.core.settings import Settings, resolve_path
from src.core.trace import TraceCollector
from src.core.types import RetrievalResult
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
MAX_TOP_K = 20
DEFAULT_COLLECTION = "default"


def validate_search_inputs(query: str, top_k: Optional[int]) -> int:
    """Validate common atomic-search inputs and return the effective top-k."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("查询不能为空")
    if top_k is None:
        return DEFAULT_TOP_K
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k 必须是整数")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k 必须在 1 到 {MAX_TOP_K} 之间")
    return top_k


def create_collection_dense_retriever(
    settings: Settings,
    collection: str,
    embedding_client: Optional[Any] = None,
) -> DenseRetriever:
    """Create a dense retriever bound to a collection-specific vector store."""
    if embedding_client is None:
        embedding_client = EmbeddingFactory.create(settings)
    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    return create_dense_retriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )


def create_collection_sparse_retriever(
    settings: Settings,
    collection: str,
) -> SparseRetriever:
    """Create a sparse retriever bound to collection BM25 and vector data."""
    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    bm25_indexer = BM25Indexer(
        index_dir=str(resolve_path(f"data/db/bm25/{collection}")),
    )
    retriever = create_sparse_retriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
    retriever.default_collection = collection
    return retriever


def resolve_collection(
    explicit_collection: Optional[str],
    filters: Dict[str, Any],
) -> str:
    """Resolve the target collection and remove it from metadata filters."""
    query_collection = filters.pop("collection", None)
    collection = explicit_collection or query_collection or DEFAULT_COLLECTION
    if not isinstance(collection, str) or not collection.strip():
        raise ValueError("collection 必须是非空字符串")
    return collection


def apply_metadata_filters(
    results: List[RetrievalResult],
    filters: Dict[str, Any],
) -> List[RetrievalResult]:
    """Keep results whose metadata satisfies all query-derived filters."""
    if not filters:
        return results

    def matches(result: RetrievalResult) -> bool:
        for key, expected in filters.items():
            actual = result.metadata.get(key)
            if key == "tags":
                actual_tags = actual if isinstance(actual, list) else [actual]
                if not all(tag in actual_tags for tag in expected):
                    return False
            elif actual != expected:
                return False
        return True

    return [result for result in results if matches(result)]


def collect_trace_safely(trace: Any) -> None:
    """Persist a trace without allowing observability failures to break MCP calls."""
    try:
        TraceCollector().collect(trace)
    except Exception:
        logger.exception("Failed to collect retrieval trace")


def build_search_response(
    response_builder: ResponseBuilder,
    results: List[RetrievalResult],
    query: str,
    collection: str,
    retrieval_mode: str,
) -> MCPToolResponse:
    """Build a standard MCP response and identify its retrieval mode."""
    response = response_builder.build(
        results=results,
        query=query,
        collection=collection,
    )
    response.metadata["retrieval_mode"] = retrieval_mode
    return response


def build_search_error_response(
    query: str,
    collection: str,
    retrieval_mode: str,
) -> MCPToolResponse:
    """Build a safe client-facing response for a retrieval infrastructure error."""
    return MCPToolResponse(
        content=(
            "## 查询失败\n\n"
            f"查询: **{query}**\n"
            f"集合: `{collection}`\n\n"
            "检索服务暂时不可用，请检查集合、索引和服务配置后重试。"
        ),
        metadata={
            "query": query,
            "collection": collection,
            "retrieval_mode": retrieval_mode,
            "error": "retrieval_failed",
        },
        is_empty=True,
    )
