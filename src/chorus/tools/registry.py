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
    from chorus.tools.git import (
        git_branch,
        git_checkout,
        git_commit,
        git_diff,
        git_init,
        git_log,
        git_merge_request,
        git_push,
    )

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
                        "description": "File path — relative paths resolve within workspace, absolute paths (starting with /) used as-is",
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
                        "description": "File path — relative paths resolve within workspace, absolute paths (starting with /) used as-is",
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
                        "description": "File path — relative paths resolve within workspace, absolute paths (starting with /) used as-is",
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

    # -- Git tools -----------------------------------------------------------

    registry.register(
        ToolDefinition(
            name="git_init",
            description="Initialize a git repository in the agent workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Agent name for git user config",
                    },
                },
                "required": ["agent_name"],
            },
            handler=git_init,
        )
    )

    registry.register(
        ToolDefinition(
            name="git_commit",
            description=(
                "Stage files and create a git commit. "
                "Stages all changes unless specific files are given."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Commit message",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific files to stage (default: all)",
                    },
                },
                "required": ["message"],
            },
            handler=git_commit,
        )
    )

    registry.register(
        ToolDefinition(
            name="git_push",
            description="Push commits to a remote repository.",
            parameters={
                "type": "object",
                "properties": {
                    "remote": {
                        "type": "string",
                        "description": "Remote name (e.g. origin)",
                    },
                    "branch": {
                        "type": "string",
                        "description": "Branch name to push",
                    },
                },
                "required": ["remote", "branch"],
            },
            handler=git_push,
        )
    )

    registry.register(
        ToolDefinition(
            name="git_branch",
            description="Create, list, or delete git branches.",
            parameters={
                "type": "object",
                "properties": {
                    "branch_name": {
                        "type": "string",
                        "description": "Branch name (omit to list all branches)",
                    },
                    "delete": {
                        "type": "boolean",
                        "description": "Delete the branch instead of creating it",
                    },
                },
                "required": [],
            },
            handler=git_branch,
        )
    )

    registry.register(
        ToolDefinition(
            name="git_checkout",
            description="Checkout a branch, tag, or commit.",
            parameters={
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Branch, tag, or commit to checkout",
                    },
                    "create": {
                        "type": "boolean",
                        "description": "Create a new branch (git checkout -b)",
                    },
                },
                "required": ["ref"],
            },
            handler=git_checkout,
        )
    )

    registry.register(
        ToolDefinition(
            name="git_diff",
            description="Show git diff — working tree vs HEAD, or between two refs.",
            parameters={
                "type": "object",
                "properties": {
                    "ref1": {
                        "type": "string",
                        "description": "First ref (optional)",
                    },
                    "ref2": {
                        "type": "string",
                        "description": "Second ref (optional)",
                    },
                },
                "required": [],
            },
            handler=git_diff,
        )
    )

    registry.register(
        ToolDefinition(
            name="git_log",
            description="Show git commit log.",
            parameters={
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of commits to show (default: 20)",
                    },
                    "oneline": {
                        "type": "boolean",
                        "description": "Use one-line format",
                    },
                },
                "required": [],
            },
            handler=git_log,
        )
    )

    registry.register(
        ToolDefinition(
            name="git_merge_request",
            description=(
                "Create a merge/pull request on GitHub or GitLab. "
                "Detects the forge from the origin remote URL."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "PR/MR title",
                    },
                    "description": {
                        "type": "string",
                        "description": "PR/MR description",
                    },
                    "source_branch": {
                        "type": "string",
                        "description": "Source (head) branch",
                    },
                    "target_branch": {
                        "type": "string",
                        "description": "Target (base) branch",
                    },
                },
                "required": ["title", "description", "source_branch", "target_branch"],
            },
            handler=git_merge_request,
        )
    )

    # -- Info tools -------------------------------------------------------------

    from chorus.tools.info import list_models

    registry.register(
        ToolDefinition(
            name="list_models",
            description=(
                "List all available LLM models discovered from API keys. "
                "Returns model IDs that can be used with self_edit_model."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=list_models,
        )
    )

    # -- Self-edit tools -------------------------------------------------------

    from chorus.agent.self_edit import (
        edit_docs,
        edit_model,
        edit_permissions,
        edit_system_prompt,
        edit_web_search,
    )

    registry.register(
        ToolDefinition(
            name="self_edit_system_prompt",
            description="Update this agent's system prompt. Takes effect on the next LLM call.",
            parameters={
                "type": "object",
                "properties": {
                    "new_prompt": {
                        "type": "string",
                        "description": "The new system prompt text",
                    },
                },
                "required": ["new_prompt"],
            },
            handler=edit_system_prompt,
        )
    )

    registry.register(
        ToolDefinition(
            name="self_edit_docs",
            description=(
                "Create or update a file in this agent's docs/ directory. "
                "These docs are included in the system prompt context."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Relative path within docs/ "
                            "(e.g. 'README.md', 'guides/setup.md')"
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "File content (UTF-8)",
                    },
                },
                "required": ["path", "content"],
            },
            handler=edit_docs,
        )
    )

    registry.register(
        ToolDefinition(
            name="self_edit_permissions",
            description=(
                "Update this agent's permission profile. "
                "Available presets: 'locked', 'standard', 'open'. "
                "'open' requires admin privileges."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "string",
                        "description": "Permission preset name ('locked', 'standard', or 'open')",
                    },
                },
                "required": ["profile"],
            },
            handler=edit_permissions,
        )
    )

    registry.register(
        ToolDefinition(
            name="self_edit_model",
            description=(
                "Switch this agent's LLM model. "
                "Validates against available models from /settings validate-keys."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": (
                            "Model identifier "
                            "(e.g. 'claude-sonnet-4-20250514', 'gpt-4o')"
                        ),
                    },
                },
                "required": ["model"],
            },
            handler=edit_model,
        )
    )

    registry.register(
        ToolDefinition(
            name="self_edit_web_search",
            description=(
                "Enable or disable web search for this agent. "
                "When enabled, the agent can search the web during conversations "
                "(Anthropic models only)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "Whether to enable web search",
                    },
                },
                "required": ["enabled"],
            },
            handler=edit_web_search,
        )
    )

    return registry
