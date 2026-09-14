"""Filesystem tools: read, write, list, search, mkdir, info, delete — all sandboxed."""
from __future__ import annotations

from core.tools.registry import ToolRegistry
from tools.filesystem.delete import DeleteFileTool
from tools.filesystem.info import FileInfoTool
from tools.filesystem.list_dir import ListDirectoryTool
from tools.filesystem.mkdir import CreateDirectoryTool
from tools.filesystem.read import ReadFileTool
from tools.filesystem.search import SearchFilesTool
from tools.filesystem.write import WriteFileTool

ALL_TOOLS = [
    ReadFileTool(),
    WriteFileTool(),
    ListDirectoryTool(),
    SearchFilesTool(),
    CreateDirectoryTool(),
    FileInfoTool(),
    DeleteFileTool(),
]


def register_all(registry: ToolRegistry) -> None:
    for tool in ALL_TOOLS:
        registry.register(tool)
