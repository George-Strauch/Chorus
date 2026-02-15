"""Tests for chorus.tools.registry — tool registration and lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

if TYPE_CHECKING:
    from pathlib import Path

from chorus.tools.registry import ToolDefinition, ToolRegistry


class TestToolRegistry:
    def test_registers_tool(self) -> None:
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            handler=AsyncMock(),
        )
        registry.register(tool)
        assert registry.get("test_tool") is tool

    def test_lists_all_tools(self) -> None:
        registry = ToolRegistry()
        t1 = ToolDefinition(
            name="tool_a",
            description="Tool A",
            parameters={},
            handler=AsyncMock(),
        )
        t2 = ToolDefinition(
            name="tool_b",
            description="Tool B",
            parameters={},
            handler=AsyncMock(),
        )
        registry.register(t1)
        registry.register(t2)
        all_tools = registry.list_all()
        assert len(all_tools) == 2
        names = {t.name for t in all_tools}
        assert names == {"tool_a", "tool_b"}

    def test_get_tool_by_name(self) -> None:
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="finder",
            description="Find things",
            parameters={},
            handler=AsyncMock(),
        )
        registry.register(tool)
        assert registry.get("finder") is tool
        assert registry.get("nonexistent") is None

    def test_tool_has_name_and_description(self) -> None:
        tool = ToolDefinition(
            name="my_tool",
            description="Does stuff",
            parameters={"type": "object"},
            handler=AsyncMock(),
        )
        assert tool.name == "my_tool"
        assert tool.description == "Does stuff"

    def test_tool_has_parameter_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
            },
            "required": ["path"],
        }
        tool = ToolDefinition(
            name="schema_tool",
            description="Tool with schema",
            parameters=schema,
            handler=AsyncMock(),
        )
        assert tool.parameters == schema
        assert "path" in tool.parameters["properties"]


class TestRegistryContainsFileTools:
    def test_file_tools_registered(self) -> None:
        from chorus.tools.registry import create_default_registry

        registry = create_default_registry()
        assert registry.get("create_file") is not None
        assert registry.get("str_replace") is not None
        assert registry.get("view") is not None

    def test_bash_tool_registered(self) -> None:
        from chorus.tools.registry import create_default_registry

        registry = create_default_registry()
        tool = registry.get("bash")
        assert tool is not None
        assert tool.name == "bash"
        assert "command" in tool.parameters["properties"]


class TestRegistryContainsGitTools:
    def test_all_git_tools_registered(self) -> None:
        from chorus.tools.registry import create_default_registry

        registry = create_default_registry()
        expected = [
            "git_init",
            "git_commit",
            "git_push",
            "git_branch",
            "git_checkout",
            "git_diff",
            "git_log",
            "git_merge_request",
        ]
        for name in expected:
            tool = registry.get(name)
            assert tool is not None, f"Tool {name!r} not registered"
            assert tool.name == name

    def test_git_tool_count(self) -> None:
        from chorus.tools.registry import create_default_registry

        registry = create_default_registry()
        git_tools = [t for t in registry.list_all() if t.name.startswith("git_")]
        assert len(git_tools) == 9


class TestRegistryContainsInfoTools:
    def test_list_models_registered(self) -> None:
        from chorus.tools.registry import create_default_registry

        registry = create_default_registry()
        tool = registry.get("list_models")
        assert tool is not None
        assert tool.name == "list_models"

    def test_list_models_has_no_required_params(self) -> None:
        from chorus.tools.registry import create_default_registry

        registry = create_default_registry()
        tool = registry.get("list_models")
        assert tool is not None
        assert tool.parameters["required"] == []


class TestListModelsHandler:
    async def test_list_models_returns_cached_models(self, tmp_path: "Path") -> None:
        import json
        from pathlib import Path

        from chorus.tools.info import list_models

        chorus_home = tmp_path / "chorus-home"
        chorus_home.mkdir()
        cache = {
            "providers": {
                "anthropic": {
                    "valid": True,
                    "models": ["claude-opus-4-20250514", "claude-sonnet-4-20250514"],
                },
                "openai": {
                    "valid": True,
                    "models": ["gpt-4o", "gpt-4o-mini"],
                },
            },
        }
        (chorus_home / "available_models.json").write_text(json.dumps(cache))

        result_str = await list_models(chorus_home=chorus_home)
        result = json.loads(result_str)
        assert "models" in result
        assert "claude-opus-4-20250514" in result["models"]
        assert "gpt-4o" in result["models"]
        assert result["count"] == 4

    async def test_list_models_no_chorus_home(self) -> None:
        import json

        from chorus.tools.info import list_models

        result_str = await list_models(chorus_home=None)
        result = json.loads(result_str)
        assert "error" in result

    async def test_list_models_empty_cache(self, tmp_path: "Path") -> None:
        import json
        from pathlib import Path

        from chorus.tools.info import list_models

        chorus_home = tmp_path / "chorus-home"
        chorus_home.mkdir()
        # No cache file — should return empty models
        result_str = await list_models(chorus_home=chorus_home)
        result = json.loads(result_str)
        assert result["models"] == []
        assert "message" in result
