"""MCP Tool: semantic_search.

Exposes vector-embedding retrieval so an MCP client can explicitly select a
semantic strategy without invoking BM25, hybrid fusion, or reranking.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Dict, Optional

from mcp import types

from src.core.query_engine.query_processor import QueryProcessor
from src.core.response.response_builder import MCPToolResponse, ResponseBuilder
from src.core.settings import Settings, load_settings
from src.core.trace import TraceContext
from src.mcp_server.tools.retrieval_support import (
    apply_metadata_filters,
    build_search_error_response,
    build_search_response,
    collect_trace_safely,
    create_collection_dense_retriever,
    resolve_collection,
    validate_search_inputs,
)
from src.libs.embedding.embedding_factory import EmbeddingFactory

logger = logging.getLogger(__name__)

TOOL_NAME = "semantic_search"
TOOL_DESCRIPTION = """Search the knowledge base using semantic vector retrieval only.

Use this tool for conceptual questions and meaning-based similarity. It does
not use BM25 keyword retrieval, hybrid fusion, or reranking.
"""
TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "A query or question to search for semantically.",
        },
        "top_k": {
            "type": "integer",
            "description": "Maximum number of results to return.",
            "default": 5,
            "minimum": 1,
            "maximum": 20,
        },
        "collection": {
            "type": "string",
            "description": "Optional collection name to limit the search scope.",
        },
    },
    "required": ["query"],
}


class SemanticSearchTool:
    """Execute collection-scoped vector searches through the MCP protocol."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        query_processor: Optional[QueryProcessor] = None,
        response_builder: Optional[ResponseBuilder] = None,
    ) -> None:
        self._settings = settings
        self._query_processor = query_processor or QueryProcessor()
        self._response_builder = response_builder or ResponseBuilder()
        self._embedding_client = None

    @property
    def settings(self) -> Settings:
        """Load application settings only when the tool first executes."""
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    async def execute(
        self,
        query: str,
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
    ) -> MCPToolResponse:
        """Execute semantic-only retrieval and return a formatted MCP response."""
        effective_top_k = validate_search_inputs(query, top_k)
        processed_query = self._query_processor.process(query)
        effective_collection = resolve_collection(collection, processed_query.filters)

        trace = TraceContext(trace_type="query")
        trace.metadata.update({
            "query": query[:200],
            "top_k": effective_top_k,
            "collection": effective_collection,
            "source": "mcp",
            "retrieval_mode": "semantic",
        })

        try:
            search_query = re.sub(r"\b\w+:[^\s]+", "", query)
            search_query = " ".join(search_query.split())
            if not search_query:
                raise ValueError("查询不能只包含过滤条件")

            trace.metadata["filters"] = processed_query.filters
            initialization_started = time.monotonic()
            if self._embedding_client is None:
                self._embedding_client = await asyncio.to_thread(
                    EmbeddingFactory.create,
                    self.settings,
                )
            retriever = await asyncio.to_thread(
                create_collection_dense_retriever,
                self.settings,
                effective_collection,
                self._embedding_client,
            )
            trace.record_stage(
                "initialization",
                {"collection": effective_collection, "retrieval_mode": "semantic"},
                elapsed_ms=(time.monotonic() - initialization_started) * 1000.0,
            )

            retrieval_started = time.monotonic()
            results = await asyncio.to_thread(
                retriever.retrieve,
                search_query,
                effective_top_k,
                processed_query.filters or None,
                trace,
            )
            results = apply_metadata_filters(results, processed_query.filters)
            trace.record_stage(
                "semantic_retrieval",
                {
                    "query": search_query[:200],
                    "result_count": len(results),
                },
                elapsed_ms=(time.monotonic() - retrieval_started) * 1000.0,
            )
            trace.metadata["final_results"] = [
                {
                    "chunk_id": result.chunk_id,
                    "score": round(result.score, 4),
                    "text": result.text or "",
                    "source": result.metadata.get(
                        "source_path", result.metadata.get("source", ""),
                    ),
                }
                for result in results
            ]
            return build_search_response(
                self._response_builder,
                results,
                query,
                effective_collection,
                "semantic",
            )
        except ValueError:
            raise
        except Exception:
            logger.exception("semantic_search failed")
            return build_search_error_response(
                query,
                effective_collection,
                "semantic",
            )
        finally:
            collect_trace_safely(trace)


_tool_instance: Optional[SemanticSearchTool] = None


def get_tool_instance(settings: Optional[Settings] = None) -> SemanticSearchTool:
    """Get or lazily create the module-level semantic-search tool instance."""
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = SemanticSearchTool(settings=settings)
    return _tool_instance


async def semantic_search_handler(
    query: str,
    top_k: int = 5,
    collection: Optional[str] = None,
) -> types.CallToolResult:
    """Handle an MCP ``semantic_search`` tool call."""
    tool = get_tool_instance()
    try:
        response = await tool.execute(query=query, top_k=top_k, collection=collection)
        return types.CallToolResult(
            content=response.to_mcp_content(),
            isError="error" in response.metadata,
        )
    except ValueError as exc:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"参数错误: {exc}")],
            isError=True,
        )
    except Exception:
        logger.exception("semantic_search handler error")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="内部错误: 查询处理失败")],
            isError=True,
        )


def register_tool(protocol_handler) -> None:
    """Register ``semantic_search`` with the protocol handler."""
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=semantic_search_handler,
    )
    logger.info("Registered MCP tool: %s", TOOL_NAME)
