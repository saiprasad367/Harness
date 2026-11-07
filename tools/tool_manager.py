"""Asynchronous Tool and Action Channel for the Voice Harness."""
import asyncio
from typing import Any, Callable, Dict, Optional


class ToolManager:
    """Dispatches asynchronous external tools without interrupting audio streaming."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register_tool(self, name: str, func: Callable):
        self._tools[name] = func

    async def execute_tool_async(self, name: str, **kwargs) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found.")
        func = self._tools[name]
        if asyncio.iscoroutinefunction(func):
            return await func(**kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(**kwargs))

