"""MCP Tool: keyword_search.

Exposes BM25-only retrieval so an MCP client can explicitly select a keyword
strategy without invoking hybrid fusion or reranking.
"""

from __future__ import annotations

import asyncio
import logging
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
    create_collection_sparse_retriever,
    resolve_collection,
    validate_search_inputs,
)

logger = logging.getLogger(__name__)

TOOL_NAME = "keyword_search"
TOOL_DESCRIPTION = """Search the knowledge base using BM25 keyword retrieval only.

Use this tool for exact terms, names, identifiers, filenames, and other
keyword-oriented queries. It does not use semantic embeddings, hybrid fusion,
or reranking.
"""
TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Keywords or a question to search for.",
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


class KeywordSearchTool:
    """Execute collection-scoped BM25 searches through the MCP protocol."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        query_processor: Optional[QueryProcessor] = None,
        response_builder: Optional[ResponseBuilder] = None,
    ) -> None:
        self._settings = settings
        self._query_processor = query_processor or QueryProcessor()
        self._response_builder = response_builder or ResponseBuilder()

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
        """Execute keyword-only retrieval and return a formatted MCP response."""
        effective_top_k = validate_search_inputs(query, top_k)
        processed_query = self._query_processor.process(query)
        effective_collection = resolve_collection(collection, processed_query.filters)

        trace = TraceContext(trace_type="query")
        trace.metadata.update({
            "query": query[:200],
            "top_k": effective_top_k,
            "collection": effective_collection,
            "source": "mcp",
            "retrieval_mode": "keyword",
        })

        try:
            trace.metadata["keywords"] = processed_query.keywords
            trace.metadata["filters"] = processed_query.filters
            if not processed_query.keywords:
                return build_search_response(
                    self._response_builder,
                    [],
                    query,
                    effective_collection,
                    "keyword",
                )

            initialization_started = time.monotonic()
            retriever = await asyncio.to_thread(
                create_collection_sparse_retriever,
                self.settings,
                effective_collection,
            )
            trace.record_stage(
                "initialization",
                {"collection": effective_collection, "retrieval_mode": "keyword"},
                elapsed_ms=(time.monotonic() - initialization_started) * 1000.0,
            )

            retrieval_started = time.monotonic()
            results = await asyncio.to_thread(
                retriever.retrieve,
                processed_query.keywords,
                effective_top_k,
                effective_collection,
                trace,
            )
            results = apply_metadata_filters(results, processed_query.filters)
            trace.record_stage(
                "keyword_retrieval",
                {
                    "keywords": processed_query.keywords,
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
                "keyword",
            )
        except ValueError:
            raise
        except Exception:
            logger.exception("keyword_search failed")
            return build_search_error_response(
                query,
                effective_collection,
                "keyword",
            )
        finally:
            collect_trace_safely(trace)


_tool_instance: Optional[KeywordSearchTool] = None


def get_tool_instance(settings: Optional[Settings] = None) -> KeywordSearchTool:
    """Get or lazily create the module-level keyword-search tool instance."""
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = KeywordSearchTool(settings=settings)
    return _tool_instance


async def keyword_search_handler(
    query: str,
    top_k: int = 5,
    collection: Optional[str] = None,
) -> types.CallToolResult:
    """Handle an MCP ``keyword_search`` tool call."""
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
        logger.exception("keyword_search handler error")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="内部错误: 查询处理失败")],
            isError=True,
        )


def register_tool(protocol_handler) -> None:
    """Register ``keyword_search`` with the protocol handler."""
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=keyword_search_handler,
    )
    logger.info("Registered MCP tool: %s", TOOL_NAME)
