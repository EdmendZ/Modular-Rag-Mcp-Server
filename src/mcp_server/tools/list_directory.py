"""MCP Tool: list_directory.

Exposes a read-only directory of successfully ingested documents so an Agent
can discover stable document identifiers before previewing or querying them.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from mcp import types

from src.observability.dashboard.services.data_service import DataService

logger = logging.getLogger(__name__)

TOOL_NAME = "list_directory"
TOOL_DESCRIPTION = """List documents available in the knowledge base.

Returns stable document IDs, source paths, collection names, and document
statistics. Use this tool to discover documents before calling
get_document_summary or a retrieval tool.
"""
TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "collection": {
            "type": "string",
            "description": "Optional collection name used to limit the document listing.",
        },
    },
    "required": [],
}


class ListDirectoryTool:
    """List ingested document records through the MCP protocol."""

    def __init__(self, data_service: Optional[DataService] = None) -> None:
        self._data_service = data_service or DataService()

    async def execute(self, collection: Optional[str] = None) -> types.CallToolResult:
        """List documents, optionally limited to one collection."""
        if collection is not None and (not isinstance(collection, str) or not collection.strip()):
            raise ValueError("collection 必须是非空字符串")

        try:
            documents = await asyncio.to_thread(
                self._data_service.list_documents,
                collection,
            )
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=self.format_response(documents),
                    )
                ],
                isError=False,
            )
        except Exception:
            logger.exception("list_directory failed")
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=(
                            "## Error\n\n"
                            "Unable to list documents. Please check the collection "
                            "and storage configuration."
                        ),
                    )
                ],
                isError=True,
            )

    @staticmethod
    def format_response(documents: List[Dict[str, Any]]) -> str:
        """Format document records as concise, agent-readable Markdown."""
        if not documents:
            return "No documents found in the knowledge base."

        lines = [f"## Documents ({len(documents)} total)", ""]
        for document in documents:
            source_hash = str(document.get("source_hash", ""))
            document_id = f"doc_{source_hash[:16]}" if source_hash else "unknown"
            source_path = document.get("source_path", "Unknown source")
            collection = document.get("collection", "default")
            chunk_count = document.get("chunk_count", 0)
            image_count = document.get("image_count", 0)
            processed_at = document.get("processed_at", "Unknown")

            lines.extend(
                [
                    f"- **Document ID:** `{document_id}`",
                    f"  **Source:** `{source_path}`",
                    f"  **Collection:** `{collection}`",
                    f"  **Source hash:** `{source_hash}`",
                    f"  **Chunks:** {chunk_count}",
                    f"  **Images:** {image_count}",
                    f"  **Processed:** `{processed_at}`",
                    "",
                ]
            )
        return "\n".join(lines).rstrip()


_tool_instance: Optional[ListDirectoryTool] = None


def get_tool_instance() -> ListDirectoryTool:
    """Get or lazily create the module-level document-directory tool."""
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = ListDirectoryTool()
    return _tool_instance


async def list_directory_handler(
    collection: Optional[str] = None,
) -> types.CallToolResult:
    """Handle an MCP ``list_directory`` tool call."""
    try:
        return await get_tool_instance().execute(collection=collection)
    except ValueError as exc:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"参数错误: {exc}")],
            isError=True,
        )
    except Exception:
        logger.exception("list_directory handler error")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="内部错误: 目录读取失败")],
            isError=True,
        )


def register_tool(protocol_handler) -> None:
    """Register ``list_directory`` with the protocol handler."""
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=list_directory_handler,
    )
    logger.info("Registered MCP tool: %s", TOOL_NAME)
