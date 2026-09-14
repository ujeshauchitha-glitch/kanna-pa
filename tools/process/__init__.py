from __future__ import annotations

from core.tools.registry import ToolRegistry
from tools.process.run_process import ProcessTool

ALL_TOOLS = [ProcessTool()]


def register_all(registry: ToolRegistry) -> None:
    for tool in ALL_TOOLS:
        registry.register(tool)
