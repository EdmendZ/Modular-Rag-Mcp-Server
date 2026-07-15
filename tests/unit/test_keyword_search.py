"""Unit tests for the keyword_search MCP tool."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from src.core.types import RetrievalResult
from src.mcp_server.protocol_handler import ProtocolHandler
from src.mcp_server.tools.keyword_search import (
    TOOL_INPUT_SCHEMA,
    TOOL_NAME,
    KeywordSearchTool,
    keyword_search_handler,
    register_tool,
)


@pytest.fixture
def response_builder() -> Mock:
    builder = Mock()
    response = Mock()
    response.metadata = {"query": "Azure", "result_count": 1}
    response.to_mcp_content.return_value = []
    builder.build.return_value = response
    return builder


@pytest.mark.asyncio
async def test_execute_uses_sparse_retrieval_only(response_builder: Mock) -> None:
    """Keyword search extracts terms and delegates only to SparseRetriever."""
    retriever = Mock()
    retriever.retrieve.return_value = [
        RetrievalResult(
            chunk_id="chunk-1",
            score=0.8,
            text="Azure OpenAI configuration",
            metadata={"source_path": "guide.md", "doc_type": "pdf"},
        ),
        RetrievalResult(
            chunk_id="chunk-2",
            score=0.7,
            text="Azure OpenAI changelog",
            metadata={"source_path": "changes.md", "doc_type": "markdown"},
        ),
    ]
    tool = KeywordSearchTool(settings=Mock(), response_builder=response_builder)

    with patch(
        "src.mcp_server.tools.keyword_search.create_collection_sparse_retriever",
        return_value=retriever,
    ) as create_retriever, patch(
        "src.mcp_server.tools.keyword_search.collect_trace_safely",
    ):
        response = await tool.execute(
            "doc_type:pdf 如何配置 Azure OpenAI",
            top_k=3,
            collection="docs",
        )

    create_retriever.assert_called_once_with(tool.settings, "docs")
    keywords = retriever.retrieve.call_args.args[0]
    assert "Azure" in keywords
    assert "OpenAI" in keywords
    assert retriever.retrieve.call_args.args[1:3] == (3, "docs")
    assert response_builder.build.call_args.kwargs["results"] == retriever.retrieve.return_value[:1]
    assert response.metadata["retrieval_mode"] == "keyword"
    response_builder.build.assert_called_once()


@pytest.mark.asyncio
async def test_execute_without_keywords_returns_successful_empty_response(
    response_builder: Mock,
) -> None:
    """A query containing only stopwords is a successful no-result search."""
    tool = KeywordSearchTool(settings=Mock(), response_builder=response_builder)

    with patch(
        "src.mcp_server.tools.keyword_search.create_collection_sparse_retriever",
    ) as create_retriever, patch(
        "src.mcp_server.tools.keyword_search.collect_trace_safely",
    ):
        response = await tool.execute("如何", collection="docs")

    create_retriever.assert_not_called()
    response_builder.build.assert_called_once_with(
        results=[],
        query="如何",
        collection="docs",
    )
    assert response.metadata["retrieval_mode"] == "keyword"


@pytest.mark.asyncio
@pytest.mark.parametrize("query, top_k", [("", 5), ("   ", 5), ("query", 0), ("query", 21)])
async def test_handler_reports_invalid_parameters(query: str, top_k: int) -> None:
    """Invalid user input produces an MCP parameter error rather than a crash."""
    tool = KeywordSearchTool(settings=Mock())
    with patch(
        "src.mcp_server.tools.keyword_search.get_tool_instance",
        return_value=tool,
    ), patch("src.mcp_server.tools.keyword_search.collect_trace_safely"):
        result = await keyword_search_handler(query=query, top_k=top_k)

    assert result.isError is True
    assert "参数错误" in result.content[0].text


def test_schema_and_registration() -> None:
    """Tool metadata is discoverable through the standard protocol registry."""
    assert TOOL_INPUT_SCHEMA["required"] == ["query"]
    assert TOOL_INPUT_SCHEMA["properties"]["top_k"]["maximum"] == 20

    protocol_handler = ProtocolHandler(server_name="test", server_version="0.0.0")
    register_tool(protocol_handler)

    assert TOOL_NAME in protocol_handler.tools
