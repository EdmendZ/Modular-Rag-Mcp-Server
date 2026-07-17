"""Unit tests for the list_directory MCP tool."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.mcp_server.protocol_handler import ProtocolHandler
from src.mcp_server.tools.list_directory import (
    TOOL_INPUT_SCHEMA,
    TOOL_NAME,
    ListDirectoryTool,
    list_directory_handler,
    register_tool,
)


@pytest.fixture
def documents() -> list[dict[str, object]]:
    return [
        {
            "source_path": "docs/guide.pdf",
            "source_hash": "0123456789abcdef0123456789abcdef",
            "collection": "research",
            "chunk_count": 12,
            "image_count": 3,
            "processed_at": "2026-07-15T10:30:00+00:00",
        }
    ]


@pytest.mark.asyncio
async def test_execute_lists_documents(documents: list[dict[str, object]]) -> None:
    """Directory listings include stable IDs and useful document metadata."""
    data_service = Mock()
    data_service.list_documents.return_value = documents
    tool = ListDirectoryTool(data_service=data_service)

    result = await tool.execute(collection="research")

    assert result.isError is False
    assert data_service.list_documents.call_args.args == ("research",)
    content = result.content[0].text
    assert "doc_0123456789abcdef" in content
    assert "docs/guide.pdf" in content
    assert "Chunks:** 12" in content
    assert "Images:** 3" in content


@pytest.mark.asyncio
async def test_execute_lists_all_documents_without_collection() -> None:
    """Omitting collection delegates an all-document listing to DataService."""
    data_service = Mock()
    data_service.list_documents.return_value = []
    tool = ListDirectoryTool(data_service=data_service)

    result = await tool.execute()

    assert result.isError is False
    assert data_service.list_documents.call_args.args == (None,)
    assert result.content[0].text == "No documents found in the knowledge base."


@pytest.mark.asyncio
@pytest.mark.parametrize("collection", ["", "   "])
async def test_handler_rejects_blank_collection(collection: str) -> None:
    """Blank collection names are MCP validation errors, not storage calls."""
    tool = ListDirectoryTool(data_service=Mock())

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "src.mcp_server.tools.list_directory.get_tool_instance",
            lambda: tool,
        )
        result = await list_directory_handler(collection=collection)

    assert result.isError is True
    assert "参数错误" in result.content[0].text
    tool._data_service.list_documents.assert_not_called()


@pytest.mark.asyncio
async def test_execute_hides_storage_exception() -> None:
    """Backend failures return a safe error without exposing implementation details."""
    data_service = Mock()
    data_service.list_documents.side_effect = RuntimeError("storage password leaked")
    tool = ListDirectoryTool(data_service=data_service)

    result = await tool.execute()

    assert result.isError is True
    assert "password" not in result.content[0].text


def test_schema_and_registration() -> None:
    """The document-directory tool is exposed through the standard registry."""
    assert TOOL_INPUT_SCHEMA["required"] == []
    assert "collection" in TOOL_INPUT_SCHEMA["properties"]

    protocol_handler = ProtocolHandler(server_name="test", server_version="0.0.0")
    register_tool(protocol_handler)

    assert TOOL_NAME in protocol_handler.tools
