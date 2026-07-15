"""Unit tests for the semantic_search MCP tool."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from src.core.types import RetrievalResult
from src.mcp_server.protocol_handler import ProtocolHandler
from src.mcp_server.tools.semantic_search import (
    TOOL_INPUT_SCHEMA,
    TOOL_NAME,
    SemanticSearchTool,
    semantic_search_handler,
    register_tool,
)


@pytest.fixture
def response_builder() -> Mock:
    builder = Mock()
    response = Mock()
    response.metadata = {"query": "concept", "result_count": 1}
    response.to_mcp_content.return_value = []
    builder.build.return_value = response
    return builder


@pytest.mark.asyncio
async def test_execute_uses_dense_retrieval_only(response_builder: Mock) -> None:
    """Semantic search delegates only to DenseRetriever with parsed filters."""
    retriever = Mock()
    retriever.retrieve.return_value = [
        RetrievalResult(
            chunk_id="chunk-1",
            score=0.9,
            text="Semantic retrieval result",
            metadata={"source_path": "guide.md", "doc_type": "pdf"},
        ),
        RetrievalResult(
            chunk_id="chunk-2",
            score=0.8,
            text="Unfiltered result",
            metadata={"source_path": "other.md", "doc_type": "markdown"},
        ),
    ]
    tool = SemanticSearchTool(settings=Mock(), response_builder=response_builder)

    with patch(
        "src.mcp_server.tools.semantic_search.create_collection_dense_retriever",
        return_value=retriever,
    ) as create_retriever, patch(
        "src.mcp_server.tools.semantic_search.EmbeddingFactory.create",
        return_value=Mock(),
    ), patch(
        "src.mcp_server.tools.semantic_search.collect_trace_safely",
    ):
        response = await tool.execute("collection:docs doc_type:pdf explain retrieval", top_k=3)

    create_retriever.assert_called_once_with(tool.settings, "docs", tool._embedding_client)
    assert retriever.retrieve.call_args.args[0] == "explain retrieval"
    assert retriever.retrieve.call_args.args[1:3] == (3, {"doc_type": "pdf"})
    assert response_builder.build.call_args.kwargs["results"] == retriever.retrieve.return_value[:1]
    assert response.metadata["retrieval_mode"] == "semantic"
    response_builder.build.assert_called_once()


@pytest.mark.asyncio
async def test_execute_records_retrieval_error_without_leaking_details() -> None:
    """Infrastructure failures create a safe error response and collect a trace."""
    retriever = Mock()
    retriever.retrieve.side_effect = RuntimeError("embedding provider secret failure")
    tool = SemanticSearchTool(settings=Mock())

    with patch(
        "src.mcp_server.tools.semantic_search.create_collection_dense_retriever",
        return_value=retriever,
    ), patch(
        "src.mcp_server.tools.semantic_search.EmbeddingFactory.create",
        return_value=Mock(),
    ), patch("src.mcp_server.tools.semantic_search.collect_trace_safely") as collect:
        response = await tool.execute("explain retrieval", collection="docs")

    assert response.metadata["error"] == "retrieval_failed"
    assert "secret" not in response.content
    collect.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("query, top_k", [("", 5), ("query", 0), ("query", 21)])
async def test_handler_reports_invalid_parameters(query: str, top_k: int) -> None:
    """Invalid semantic-search input returns a typed MCP parameter error."""
    tool = SemanticSearchTool(settings=Mock())
    with patch(
        "src.mcp_server.tools.semantic_search.get_tool_instance",
        return_value=tool,
    ), patch("src.mcp_server.tools.semantic_search.collect_trace_safely"):
        result = await semantic_search_handler(query=query, top_k=top_k)

    assert result.isError is True
    assert "参数错误" in result.content[0].text


def test_schema_and_registration() -> None:
    """Tool metadata is discoverable through the standard protocol registry."""
    assert TOOL_INPUT_SCHEMA["required"] == ["query"]
    assert TOOL_INPUT_SCHEMA["properties"]["top_k"]["minimum"] == 1

    protocol_handler = ProtocolHandler(server_name="test", server_version="0.0.0")
    register_tool(protocol_handler)

    assert TOOL_NAME in protocol_handler.tools
