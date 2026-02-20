"""Tests for chorus.agent.communication — inter-agent communication."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from chorus.agent.communication import (
    _extract_first_paragraph,
    list_agents,
    read_agent_docs,
    send_to_agent,
)
from chorus.permissions.engine import PermissionResult, check, format_action, get_preset

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from chorus.storage.db import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chorus_home(tmp_path: Path) -> Path:
    """Create a chorus home with some agents."""
    home = tmp_path / "chorus-home"
    (home / "agents").mkdir(parents=True)
    return home


def _make_agent(
    chorus_home: Path,
    name: str,
    *,
    model: str | None = "claude-sonnet-4-20250514",
    readme_content: str | None = "# Agent\n\nA helpful agent.",
    extra_docs: dict[str, str] | None = None,
) -> Path:
    """Helper to create an agent directory under chorus_home."""
    agent_dir = chorus_home / "agents" / name
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "workspace").mkdir(exist_ok=True)
    (agent_dir / "docs").mkdir(exist_ok=True)

    agent_json: dict[str, Any] = {
        "name": name,
        "channel_id": 100,
        "model": model,
        "system_prompt": "You are an agent.",
        "permissions": "standard",
    }
    (agent_dir / "agent.json").write_text(json.dumps(agent_json, indent=4))

    if readme_content is not None:
        (agent_dir / "docs" / "README.md").write_text(readme_content)

    if extra_docs:
        for doc_path, doc_content in extra_docs.items():
            full = agent_dir / "docs" / doc_path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(doc_content)

    return agent_dir


@pytest.fixture
async def db(tmp_path: Path) -> AsyncGenerator[Database, None]:
    """Temporary database."""
    from chorus.storage.db import Database

    d = Database(tmp_path / "db" / "chorus.db")
    await d.init()
    yield d
    await d.close()


# ---------------------------------------------------------------------------
# TestSendToAgent
# ---------------------------------------------------------------------------


class TestSendToAgent:
    @pytest.mark.asyncio
    async def test_delivery(self, chorus_home: Path) -> None:
        _make_agent(chorus_home, "sender")
        _make_agent(chorus_home, "target")

        bot = AsyncMock()
        bot.spawn_agent_branch = AsyncMock()

        result = await send_to_agent(
            "target",
            "Hello, target!",
            agent_name="sender",
            bot=bot,
            chorus_home=chorus_home,
        )
        data = json.loads(result)
        assert data["delivered"] is True
        assert data["target"] == "target"
        bot.spawn_agent_branch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_self_send_rejected(self, chorus_home: Path) -> None:
        _make_agent(chorus_home, "sender")

        bot = AsyncMock()
        result = await send_to_agent(
            "sender",
            "Hello, self!",
            agent_name="sender",
            bot=bot,
            chorus_home=chorus_home,
        )
        data = json.loads(result)
        assert "error" in data
        assert "self" in data["error"].lower() or "own" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_target(self, chorus_home: Path) -> None:
        _make_agent(chorus_home, "sender")

        bot = AsyncMock()
        result = await send_to_agent(
            "nonexistent",
            "Hello!",
            agent_name="sender",
            bot=bot,
            chorus_home=chorus_home,
        )
        data = json.loads(result)
        assert "error" in data
        assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_no_bot_returns_error(self, chorus_home: Path) -> None:
        _make_agent(chorus_home, "sender")
        _make_agent(chorus_home, "target")

        result = await send_to_agent(
            "target",
            "Hello!",
            agent_name="sender",
            bot=None,
            chorus_home=chorus_home,
        )
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_spawn_called_with_attributed_message(self, chorus_home: Path) -> None:
        _make_agent(chorus_home, "sender")
        _make_agent(chorus_home, "target")

        bot = AsyncMock()
        bot.spawn_agent_branch = AsyncMock()

        await send_to_agent(
            "target",
            "Do the thing",
            agent_name="sender",
            bot=bot,
            chorus_home=chorus_home,
        )
        call_args = bot.spawn_agent_branch.call_args
        # The message should be attributed
        message_arg = call_args[1].get("message") or call_args[0][1]
        assert "sender" in message_arg
        assert "Do the thing" in message_arg

    @pytest.mark.asyncio
    async def test_spawn_failure(self, chorus_home: Path) -> None:
        _make_agent(chorus_home, "sender")
        _make_agent(chorus_home, "target")

        bot = AsyncMock()
        bot.spawn_agent_branch = AsyncMock(side_effect=Exception("spawn failed"))

        result = await send_to_agent(
            "target",
            "Hello!",
            agent_name="sender",
            bot=bot,
            chorus_home=chorus_home,
        )
        data = json.loads(result)
        assert "error" in data
        assert "spawn" in data["error"].lower() or "deliver" in data["error"].lower()


# ---------------------------------------------------------------------------
# TestReadAgentDocs
# ---------------------------------------------------------------------------


class TestReadAgentDocs:
    @pytest.mark.asyncio
    async def test_content_returned(self, chorus_home: Path) -> None:
        _make_agent(chorus_home, "reader")
        _make_agent(chorus_home, "target", readme_content="# Target Agent\n\nI do things.")

        result = await read_agent_docs(
            "target",
            agent_name="reader",
            chorus_home=chorus_home,
        )
        data = json.loads(result)
        assert "README.md" in data["docs"]
        assert "Target Agent" in data["docs"]["README.md"]

    @pytest.mark.asyncio
    async def test_self_read_rejected(self, chorus_home: Path) -> None:
        _make_agent(chorus_home, "reader")

        result = await read_agent_docs(
            "reader",
            agent_name="reader",
            chorus_home=chorus_home,
        )
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_missing_agent(self, chorus_home: Path) -> None:
        _make_agent(chorus_home, "reader")

        result = await read_agent_docs(
            "nonexistent",
            agent_name="reader",
            chorus_home=chorus_home,
        )
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_multiple_files(self, chorus_home: Path) -> None:
        _make_agent(chorus_home, "reader")
        _make_agent(
            chorus_home,
            "target",
            readme_content="# README",
            extra_docs={"guide.md": "# Guide\n\nSetup instructions."},
        )

        result = await read_agent_docs(
            "target",
            agent_name="reader",
            chorus_home=chorus_home,
        )
        data = json.loads(result)
        assert "README.md" in data["docs"]
        assert "guide.md" in data["docs"]

    @pytest.mark.asyncio
    async def test_empty_docs(self, chorus_home: Path) -> None:
        _make_agent(chorus_home, "reader")
        _make_agent(chorus_home, "target", readme_content=None)

        result = await read_agent_docs(
            "target",
            agent_name="reader",
            chorus_home=chorus_home,
        )
        data = json.loads(result)
        assert data["docs"] == {}


# ---------------------------------------------------------------------------
# TestListAgents
# ---------------------------------------------------------------------------


class TestListAgents:
    @pytest.mark.asyncio
    async def test_lists_agents(self, chorus_home: Path) -> None:
        _make_agent(chorus_home, "alpha")
        _make_agent(chorus_home, "beta")
        _make_agent(chorus_home, "lister")

        result = await list_agents(
            agent_name="lister",
            chorus_home=chorus_home,
        )
        data = json.loads(result)
        names = [a["name"] for a in data["agents"]]
        assert "alpha" in names
        assert "beta" in names

    @pytest.mark.asyncio
    async def test_excludes_self(self, chorus_home: Path) -> None:
        _make_agent(chorus_home, "alpha")
        _make_agent(chorus_home, "lister")

        result = await list_agents(
            agent_name="lister",
            chorus_home=chorus_home,
        )
        data = json.loads(result)
        names = [a["name"] for a in data["agents"]]
        assert "lister" not in names

    @pytest.mark.asyncio
    async def test_includes_description_and_model(self, chorus_home: Path) -> None:
        _make_agent(
            chorus_home,
            "alpha",
            model="gpt-4o",
            readme_content="# Alpha\n\nA coding agent that writes Python.",
        )
        _make_agent(chorus_home, "lister")

        result = await list_agents(
            agent_name="lister",
            chorus_home=chorus_home,
        )
        data = json.loads(result)
        alpha = next(a for a in data["agents"] if a["name"] == "alpha")
        assert alpha["model"] == "gpt-4o"
        assert "coding agent" in alpha["description"]

    @pytest.mark.asyncio
    async def test_empty_agents_dir(self, chorus_home: Path) -> None:
        _make_agent(chorus_home, "lister")

        result = await list_agents(
            agent_name="lister",
            chorus_home=chorus_home,
        )
        data = json.loads(result)
        assert data["agents"] == []


# ---------------------------------------------------------------------------
# TestExtractFirstParagraph
# ---------------------------------------------------------------------------


class TestExtractFirstParagraph:
    def test_heading_skip(self) -> None:
        text = "# Title\n\nThis is the first paragraph."
        assert _extract_first_paragraph(text) == "This is the first paragraph."

    def test_italic_skip(self) -> None:
        text = "# Title\n\n*Status: active*\n\nActual description here."
        result = _extract_first_paragraph(text)
        assert result == "Actual description here."

    def test_empty_content(self) -> None:
        assert _extract_first_paragraph("") == ""
        assert _extract_first_paragraph("# Just a heading") == ""

    def test_truncation(self) -> None:
        long_text = "# Title\n\n" + "x" * 300
        result = _extract_first_paragraph(long_text)
        assert len(result) <= 200
        assert result.endswith("...")

    def test_plain_paragraph(self) -> None:
        text = "A simple paragraph without headings."
        assert _extract_first_paragraph(text) == "A simple paragraph without headings."

    def test_bold_skip(self) -> None:
        text = "# Title\n\n**Bold line**\n\nReal description."
        result = _extract_first_paragraph(text)
        assert result == "Real description."

    def test_blockquote_skip(self) -> None:
        text = "# Title\n\n> Status: pending\n\nThe real content."
        result = _extract_first_paragraph(text)
        assert result == "The real content."


# ---------------------------------------------------------------------------
# TestPermissionIntegration
# ---------------------------------------------------------------------------


class TestPermissionIntegration:
    def test_tools_registered(self) -> None:
        from chorus.tools.registry import create_default_registry

        registry = create_default_registry()
        for name in ("send_to_agent", "read_agent_docs", "list_agents"):
            assert registry.get(name) is not None, f"Tool {name!r} not registered"

    def test_allowed_in_standard(self) -> None:
        profile = get_preset("standard")
        send = check(format_action("agent_comm", "send target"), profile)
        assert send is PermissionResult.ALLOW
        read = check(format_action("agent_comm", "read_docs target"), profile)
        assert read is PermissionResult.ALLOW
        lst = check(format_action("agent_comm", "list"), profile)
        assert lst is PermissionResult.ALLOW

    def test_allowed_in_open(self) -> None:
        profile = get_preset("open")
        assert check(format_action("agent_comm", "send target"), profile) is PermissionResult.ALLOW

    def test_category_mapping(self) -> None:
        from chorus.llm.tool_loop import _build_action_string

        action = _build_action_string("send_to_agent", {"target_agent": "alpha", "message": "hi"})
        assert action == "tool:agent_comm:send alpha"

        action = _build_action_string("read_agent_docs", {"target_agent": "alpha"})
        assert action == "tool:agent_comm:read_docs alpha"

        action = _build_action_string("list_agents", {})
        assert action == "tool:agent_comm:list"

    def test_action_strings(self) -> None:
        # Verify the action string format matches the preset patterns
        profile = get_preset("standard")
        send_action = format_action("agent_comm", "send my-agent")
        assert check(send_action, profile) is PermissionResult.ALLOW


# ---------------------------------------------------------------------------
# TestToolExecution
# ---------------------------------------------------------------------------


class TestToolExecution:
    @pytest.mark.asyncio
    async def test_send_via_execute_tool(self, chorus_home: Path) -> None:
        """End-to-end: dispatch send_to_agent through _execute_tool."""
        from chorus.llm.tool_loop import ToolExecutionContext, _execute_tool
        from chorus.tools.registry import create_default_registry

        _make_agent(chorus_home, "sender")
        _make_agent(chorus_home, "target")

        bot = AsyncMock()
        bot.spawn_agent_branch = AsyncMock()

        registry = create_default_registry()
        tool = registry.get("send_to_agent")
        assert tool is not None

        ctx = ToolExecutionContext(
            workspace=chorus_home / "agents" / "sender" / "workspace",
            profile=get_preset("open"),
            agent_name="sender",
            chorus_home=chorus_home,
            bot=bot,
        )
        result_str = await _execute_tool(
            tool,
            {"target_agent": "target", "message": "Hello from test!"},
            ctx,
        )
        data = json.loads(result_str)
        assert data["delivered"] is True
        bot.spawn_agent_branch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_via_execute_tool(self, chorus_home: Path) -> None:
        """End-to-end: dispatch list_agents through _execute_tool."""
        from chorus.llm.tool_loop import ToolExecutionContext, _execute_tool
        from chorus.tools.registry import create_default_registry

        _make_agent(chorus_home, "sender")
        _make_agent(chorus_home, "other")

        registry = create_default_registry()
        tool = registry.get("list_agents")
        assert tool is not None

        ctx = ToolExecutionContext(
            workspace=chorus_home / "agents" / "sender" / "workspace",
            profile=get_preset("open"),
            agent_name="sender",
            chorus_home=chorus_home,
        )
        result_str = await _execute_tool(tool, {}, ctx)
        data = json.loads(result_str)
        names = [a["name"] for a in data["agents"]]
        assert "other" in names
        assert "sender" not in names

    @pytest.mark.asyncio
    async def test_read_docs_via_execute_tool(self, chorus_home: Path) -> None:
        """End-to-end: dispatch read_agent_docs through _execute_tool."""
        from chorus.llm.tool_loop import ToolExecutionContext, _execute_tool
        from chorus.tools.registry import create_default_registry

        _make_agent(chorus_home, "sender")
        _make_agent(chorus_home, "target", readme_content="# Target\n\nDocs here.")

        registry = create_default_registry()
        tool = registry.get("read_agent_docs")
        assert tool is not None

        ctx = ToolExecutionContext(
            workspace=chorus_home / "agents" / "sender" / "workspace",
            profile=get_preset("open"),
            agent_name="sender",
            chorus_home=chorus_home,
        )
        result_str = await _execute_tool(
            tool, {"target_agent": "target"}, ctx
        )
        data = json.loads(result_str)
        assert "Docs here." in data["docs"]["README.md"]
