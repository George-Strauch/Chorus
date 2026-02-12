"""Tool registry — central catalog of all tools available to agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@dataclass
class ToolDefinition:
    """Metadata and handler for a single tool."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: Callable[..., Awaitable[Any]]


class ToolRegistry:
    """Stores and retrieves tool definitions by name."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_all(self) -> list[ToolDefinition]:
        return list(self._tools.values())


def create_default_registry() -> ToolRegistry:
    """Build a registry with the built-in tools pre-registered."""
    from chorus.tools.bash import bash_execute
    from chorus.tools.file_ops import create_file, str_replace, view

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="create_file",
            description=(
                "Create or overwrite a file in the agent workspace. "
                "Intermediate directories are created automatically."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path within the workspace",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content (UTF-8)",
                    },
                },
                "required": ["path", "content"],
            },
            handler=create_file,
        )
    )

    registry.register(
        ToolDefinition(
            name="str_replace",
            description=(
                "Replace exactly one occurrence of a string in a file. "
                "Fails if the string is not found or appears more than once."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path within the workspace",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Exact string to find (must be unique)",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement string",
                    },
                },
                "required": ["path", "old_str", "new_str"],
            },
            handler=str_replace,
        )
    )

    registry.register(
        ToolDefinition(
            name="view",
            description=(
                "View a file's contents with line numbers. "
                "Supports optional offset and limit for large files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path within the workspace",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based line number to start from",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of lines to return",
                    },
                },
                "required": ["path"],
            },
            handler=view,
        )
    )

    registry.register(
        ToolDefinition(
            name="bash",
            description=(
                "Execute a shell command in the agent's workspace directory. "
                "The command runs with a sanitized environment and configurable timeout."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default 120)",
                    },
                },
                "required": ["command"],
            },
            handler=bash_execute,
        )
    )

    return registry
