"""Tests for chorus.agent.context — context window management."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

if TYPE_CHECKING:
    from chorus.storage.db import Database

from chorus.agent.context import (
    MAX_INPUT_TOKENS,
    ContextManager,
    _get_context_limit,
    _truncate_to_budget,
    build_llm_context,
    estimate_tokens,
)
from chorus.agent.threads import ThreadManager
from chorus.models import Agent, SessionNotFoundError


@pytest.fixture
def sessions_dir(tmp_path: Path) -> Path:
    d = tmp_path / "sessions"
    d.mkdir()
    return d


@pytest.fixture
async def ctx_manager(context_db: Database, sessions_dir: Path) -> ContextManager:
    """ContextManager with a registered agent in the DB."""
    now = datetime.now(UTC).isoformat()
    await context_db.register_agent("test-agent", 12345, 99999, None, "standard", now)
    return ContextManager("test-agent", context_db, sessions_dir)


class TestMessagePersistence:
    async def test_message_persisted_to_sqlite(self, ctx_manager: ContextManager) -> None:
        await ctx_manager.persist_message(role="user", content="Hello")
        msgs = await ctx_manager._db.get_all_messages("test-agent")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Hello"

    async def test_message_includes_agent_name_and_thread_id(
        self, ctx_manager: ContextManager
    ) -> None:
        await ctx_manager.persist_message(role="user", content="Hi", thread_id=42)
        msgs = await ctx_manager._db.get_all_messages("test-agent")
        assert msgs[0]["agent_name"] == "test-agent"
        assert msgs[0]["thread_id"] == 42

    async def test_message_includes_timestamp(self, ctx_manager: ContextManager) -> None:
        await ctx_manager.persist_message(role="user", content="Hi")
        msgs = await ctx_manager._db.get_all_messages("test-agent")
        assert msgs[0]["timestamp"] is not None
        # Should be parseable as ISO timestamp
        datetime.fromisoformat(msgs[0]["timestamp"])

    async def test_messages_queryable_by_agent(
        self, context_db: Database, sessions_dir: Path
    ) -> None:
        now = datetime.now(UTC).isoformat()
        await context_db.register_agent("agent-a", 100, 99999, None, "standard", now)
        await context_db.register_agent("agent-b", 200, 99999, None, "standard", now)

        cm_a = ContextManager("agent-a", context_db, sessions_dir)
        cm_b = ContextManager("agent-b", context_db, sessions_dir)

        await cm_a.persist_message(role="user", content="From A")
        await cm_b.persist_message(role="user", content="From B")

        msgs_a = await context_db.get_all_messages("agent-a")
        assert len(msgs_a) == 1
        assert msgs_a[0]["content"] == "From A"


class TestRollingWindow:
    async def test_rolling_window_returns_messages_within_window(
        self, context_db: Database, sessions_dir: Path
    ) -> None:
        now = datetime.now(UTC)
        created = (now - timedelta(hours=2)).isoformat()
        await context_db.register_agent("test-agent", 12345, 99999, None, "standard", created)

        cm = ContextManager("test-agent", context_db, sessions_dir, rolling_window=7200)

        # Message within the 2-hour window
        recent_ts = (now - timedelta(minutes=30)).isoformat()
        await context_db.persist_message("test-agent", "user", recent_ts, content="Recent")

        msgs = await cm.get_context()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Recent"

    async def test_rolling_window_excludes_old_messages(
        self, context_db: Database, sessions_dir: Path
    ) -> None:
        now = datetime.now(UTC)
        created = (now - timedelta(days=5)).isoformat()
        await context_db.register_agent("test-agent", 12345, 99999, None, "standard", created)

        cm = ContextManager("test-agent", context_db, sessions_dir, rolling_window=3600)

        # Old message (2 hours ago, window is 1 hour)
        old_ts = (now - timedelta(hours=2)).isoformat()
        await context_db.persist_message("test-agent", "user", old_ts, content="Old")

        # Recent message (10 min ago)
        recent_ts = (now - timedelta(minutes=10)).isoformat()
        await context_db.persist_message("test-agent", "user", recent_ts, content="New")

        msgs = await cm.get_context()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "New"

    async def test_rolling_window_default_is_one_day(
        self, context_db: Database, sessions_dir: Path
    ) -> None:
        now = datetime.now(UTC)
        created = (now - timedelta(days=3)).isoformat()
        await context_db.register_agent("test-agent", 12345, 99999, None, "standard", created)

        cm = ContextManager("test-agent", context_db, sessions_dir)
        assert cm._rolling_window == 86400

    async def test_rolling_window_configurable(
        self, context_db: Database, sessions_dir: Path
    ) -> None:
        now = datetime.now(UTC).isoformat()
        await context_db.register_agent("test-agent", 12345, 99999, None, "standard", now)

        cm = ContextManager("test-agent", context_db, sessions_dir, rolling_window=3600)
        assert cm._rolling_window == 3600


class TestClear:
    async def test_clear_advances_last_clear_time(self, ctx_manager: ContextManager) -> None:
        before = datetime.now(UTC)
        await ctx_manager.clear()
        clear_time = await ctx_manager._db.get_last_clear_time("test-agent")
        assert clear_time is not None
        parsed = datetime.fromisoformat(clear_time)
        assert parsed >= before

    async def test_clear_excludes_messages_before_clear_time(
        self, ctx_manager: ContextManager
    ) -> None:
        # Persist a message
        await ctx_manager.persist_message(role="user", content="Before clear")

        # Clear
        await ctx_manager.clear()

        # Messages before clear should not be in context
        msgs = await ctx_manager.get_context()
        assert len(msgs) == 0

    async def test_clear_does_not_delete_messages_from_db(
        self, ctx_manager: ContextManager
    ) -> None:
        await ctx_manager.persist_message(role="user", content="Preserved")
        await ctx_manager.clear()

        # Still in the DB
        all_msgs = await ctx_manager._db.get_all_messages("test-agent")
        assert len(all_msgs) == 1
        assert all_msgs[0]["content"] == "Preserved"

    async def test_messages_after_clear_are_included(
        self, ctx_manager: ContextManager
    ) -> None:
        await ctx_manager.persist_message(role="user", content="Before")
        await ctx_manager.clear()
        await ctx_manager.persist_message(role="user", content="After")

        msgs = await ctx_manager.get_context()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "After"

    async def test_multiple_clears_use_most_recent(
        self, ctx_manager: ContextManager
    ) -> None:
        await ctx_manager.persist_message(role="user", content="Msg 1")
        await ctx_manager.clear()
        await ctx_manager.persist_message(role="user", content="Msg 2")
        await ctx_manager.clear()
        await ctx_manager.persist_message(role="user", content="Msg 3")

        msgs = await ctx_manager.get_context()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Msg 3"


class TestContextAssembly:
    async def test_context_includes_system_prompt(
        self, ctx_manager: ContextManager
    ) -> None:
        agent = Agent(name="test-agent", channel_id=12345, system_prompt="You are helpful.")
        tm = ThreadManager("test-agent")
        docs_dir = None

        result = await build_llm_context(agent, 1, ctx_manager, tm, docs_dir)
        # System prompt should be the first message
        assert result[0]["role"] == "system"
        assert "You are helpful" in result[0]["content"]

    async def test_context_includes_agent_docs(
        self, ctx_manager: ContextManager, tmp_path: Path
    ) -> None:
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "README.md").write_text("# My Agent\nI help with tasks.")

        agent = Agent(name="test-agent", channel_id=12345)
        tm = ThreadManager("test-agent")

        result = await build_llm_context(agent, 1, ctx_manager, tm, docs_dir)
        # Docs should appear in context
        docs_content = [
            m for m in result
            if m["role"] == "system" and "My Agent" in m.get("content", "")
        ]
        assert len(docs_content) > 0

    async def test_context_includes_thread_status(
        self, ctx_manager: ContextManager
    ) -> None:
        agent = Agent(name="test-agent", channel_id=12345)
        tm = ThreadManager("test-agent")
        t = tm.create_thread({"role": "user", "content": "hello"})
        t.summary = "Auth refactor"
        t.metrics.begin_step("Working")

        result = await build_llm_context(agent, t.id, ctx_manager, tm, None)
        # Thread status should be in context
        status_msgs = [m for m in result if "thread" in m.get("content", "").lower()]
        assert len(status_msgs) > 0

    async def test_context_includes_rolling_window_messages(
        self, ctx_manager: ContextManager
    ) -> None:
        await ctx_manager.persist_message(role="user", content="What's up?")
        await ctx_manager.persist_message(role="assistant", content="Not much!")

        agent = Agent(name="test-agent", channel_id=12345)
        tm = ThreadManager("test-agent")

        result = await build_llm_context(agent, None, ctx_manager, tm, None)
        contents = [m.get("content", "") for m in result]
        assert any("What's up?" in c for c in contents)
        assert any("Not much!" in c for c in contents)

    async def test_context_window_bounded_by_clear_time_or_rolling_window(
        self, context_db: Database, sessions_dir: Path
    ) -> None:
        now = datetime.now(UTC)
        created = (now - timedelta(days=3)).isoformat()
        await context_db.register_agent("test-agent", 12345, 99999, None, "standard", created)

        cm = ContextManager("test-agent", context_db, sessions_dir, rolling_window=86400)

        # Message from 2 hours ago — within 24h window
        ts1 = (now - timedelta(hours=2)).isoformat()
        await context_db.persist_message("test-agent", "user", ts1, content="In window")

        # Clear 1 hour ago — more recent than rolling window start
        clear_ts = (now - timedelta(hours=1)).isoformat()
        await context_db.set_last_clear_time("test-agent", clear_ts)

        # Message from 30 min ago — after clear
        ts2 = (now - timedelta(minutes=30)).isoformat()
        await context_db.persist_message("test-agent", "user", ts2, content="After clear")

        msgs = await cm.get_context()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "After clear"


class TestSaveSnapshot:
    async def test_save_creates_session_json_file(
        self, ctx_manager: ContextManager, sessions_dir: Path
    ) -> None:
        await ctx_manager.persist_message(role="user", content="Save this")
        meta = await ctx_manager.save_snapshot(description="Test save")
        file_path = sessions_dir / f"{meta.session_id}.json"
        assert file_path.exists()

    async def test_save_stores_metadata_in_sqlite(
        self, ctx_manager: ContextManager
    ) -> None:
        await ctx_manager.persist_message(role="user", content="Persist me")
        meta = await ctx_manager.save_snapshot(description="DB test")
        session = await ctx_manager._db.get_session(meta.session_id)
        assert session is not None
        assert session["description"] == "DB test"
        assert session["agent_name"] == "test-agent"
        assert session["message_count"] >= 1

    async def test_save_generates_summary(
        self, ctx_manager: ContextManager
    ) -> None:
        await ctx_manager.persist_message(role="user", content="Do something")

        async def mock_summarizer(messages: list[dict]) -> str:
            return "Summary of conversation"

        meta = await ctx_manager.save_snapshot(description="Summarized", summarizer=mock_summarizer)
        assert meta.summary == "Summary of conversation"

    async def test_save_does_not_clear_context(
        self, ctx_manager: ContextManager
    ) -> None:
        await ctx_manager.persist_message(role="user", content="Still here")
        await ctx_manager.save_snapshot(description="Snapshot")

        msgs = await ctx_manager.get_context()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Still here"

    async def test_save_filename_format(
        self, ctx_manager: ContextManager, sessions_dir: Path
    ) -> None:
        await ctx_manager.persist_message(role="user", content="Hello")
        meta = await ctx_manager.save_snapshot(description="Check filename")
        # File should exist in sessions_dir with .json extension
        json_files = list(sessions_dir.glob("*.json"))
        assert len(json_files) == 1
        assert json_files[0].name == f"{meta.session_id}.json"

    async def test_save_with_custom_description(
        self, ctx_manager: ContextManager
    ) -> None:
        await ctx_manager.persist_message(role="user", content="Work")
        meta = await ctx_manager.save_snapshot(description="Auth refactor session")
        assert meta.description == "Auth refactor session"

    async def test_save_session_file_contents(
        self, ctx_manager: ContextManager, sessions_dir: Path
    ) -> None:
        await ctx_manager.persist_message(role="user", content="Hello!")
        await ctx_manager.persist_message(role="assistant", content="Hi there!")
        meta = await ctx_manager.save_snapshot(description="Full test")

        file_path = sessions_dir / f"{meta.session_id}.json"
        data = json.loads(file_path.read_text())
        assert data["session_id"] == meta.session_id
        assert data["description"] == "Full test"
        assert len(data["messages"]) == 2
        assert data["message_count"] == 2

    async def test_save_summary_fallback_on_no_summarizer(
        self, ctx_manager: ContextManager
    ) -> None:
        await ctx_manager.persist_message(role="user", content="Test")
        meta = await ctx_manager.save_snapshot(description="No summarizer")
        assert meta.summary == ""


class TestRestore:
    async def test_restore_loads_saved_messages(
        self, ctx_manager: ContextManager
    ) -> None:
        await ctx_manager.persist_message(role="user", content="Original msg")
        meta = await ctx_manager.save_snapshot(description="To restore")
        await ctx_manager.clear()

        # After clear, context is empty
        msgs = await ctx_manager.get_context()
        assert len(msgs) == 0

        # Restore brings messages back
        await ctx_manager.restore_snapshot(meta.session_id)
        msgs = await ctx_manager.get_context()
        assert len(msgs) >= 1
        contents = [m["content"] for m in msgs]
        assert "Original msg" in contents

    async def test_restore_nonexistent_id_raises(
        self, ctx_manager: ContextManager
    ) -> None:
        with pytest.raises(SessionNotFoundError):
            await ctx_manager.restore_snapshot("nonexistent-id")


class TestHistory:
    async def test_list_sessions_returns_all_snapshots(
        self, ctx_manager: ContextManager
    ) -> None:
        await ctx_manager.persist_message(role="user", content="A")
        await ctx_manager.save_snapshot(description="First")
        await ctx_manager.persist_message(role="user", content="B")
        await ctx_manager.save_snapshot(description="Second")

        snapshots = await ctx_manager.list_snapshots()
        assert len(snapshots) == 2

    async def test_list_sessions_ordered_by_timestamp_desc(
        self, ctx_manager: ContextManager
    ) -> None:
        await ctx_manager.persist_message(role="user", content="A")
        await ctx_manager.save_snapshot(description="Older")
        await ctx_manager.persist_message(role="user", content="B")
        await ctx_manager.save_snapshot(description="Newer")

        snapshots = await ctx_manager.list_snapshots()
        assert snapshots[0].description == "Newer"
        assert snapshots[1].description == "Older"


class TestBotRestart:
    async def test_context_survives_restart_via_sqlite(
        self, context_db: Database, sessions_dir: Path
    ) -> None:
        now = datetime.now(UTC).isoformat()
        await context_db.register_agent("test-agent", 12345, 99999, None, "standard", now)

        cm1 = ContextManager("test-agent", context_db, sessions_dir)
        await cm1.persist_message(role="user", content="Survive restart")

        # Simulate restart: create a new ContextManager against same DB
        cm2 = ContextManager("test-agent", context_db, sessions_dir)
        msgs = await cm2.get_context()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Survive restart"

    async def test_clear_time_survives_restart(
        self, context_db: Database, sessions_dir: Path
    ) -> None:
        now = datetime.now(UTC).isoformat()
        await context_db.register_agent("test-agent", 12345, 99999, None, "standard", now)

        cm1 = ContextManager("test-agent", context_db, sessions_dir)
        await cm1.persist_message(role="user", content="Before clear")
        await cm1.clear()

        cm2 = ContextManager("test-agent", context_db, sessions_dir)
        msgs = await cm2.get_context()
        assert len(msgs) == 0


class TestSummaryGeneration:
    async def test_generate_summary_calls_llm(
        self, ctx_manager: ContextManager
    ) -> None:
        await ctx_manager.persist_message(role="user", content="Test")
        summarizer = AsyncMock(return_value="LLM-generated summary")

        meta = await ctx_manager.save_snapshot(description="Test", summarizer=summarizer)
        summarizer.assert_called_once()
        assert meta.summary == "LLM-generated summary"

    async def test_generate_summary_returns_short_text(
        self, ctx_manager: ContextManager
    ) -> None:
        await ctx_manager.persist_message(role="user", content="Test")

        async def short_summarizer(messages: list[dict]) -> str:
            return "Short."

        meta = await ctx_manager.save_snapshot(description="Test", summarizer=short_summarizer)
        assert len(meta.summary) < 100

    async def test_generate_summary_handles_llm_failure_gracefully(
        self, ctx_manager: ContextManager
    ) -> None:
        await ctx_manager.persist_message(role="user", content="Test")

        async def failing_summarizer(messages: list[dict]) -> str:
            raise RuntimeError("LLM API error")

        meta = await ctx_manager.save_snapshot(description="Test", summarizer=failing_summarizer)
        assert meta.summary == "(summary generation failed)"


# ── Token Budget ────────────────────────────────────────────────────────


class TestGetContextLimit:
    def test_known_model_returns_correct_limit(self) -> None:
        assert _get_context_limit("claude-opus-4-20250514") == 200_000
        assert _get_context_limit("gpt-4o") == 128_000
        assert _get_context_limit("gpt-4") == 8_192

    def test_unknown_model_returns_default(self) -> None:
        assert _get_context_limit("some-unknown-model") == 128_000

    def test_none_model_returns_default(self) -> None:
        assert _get_context_limit(None) == 128_000

    def test_prefix_match_for_dated_variants(self) -> None:
        # Models like "gpt-4o-2024-08-06" should match "gpt-4o"
        assert _get_context_limit("gpt-4o-2024-08-06") == 128_000


class TestEstimateTokens:
    def test_basic_estimate(self) -> None:
        # 100 chars / 4 = 25 tokens
        text = "a" * 100
        assert estimate_tokens(text) == 25

    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_short_string(self) -> None:
        # "hi" = 2 chars / 4 = 0 (integer division)
        assert estimate_tokens("hi") == 0

    def test_reasonable_estimate(self) -> None:
        # A typical paragraph should give a reasonable estimate
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert tokens < len(text)  # Should be less than char count


class TestTruncateToBudget:
    def test_empty_messages(self) -> None:
        assert _truncate_to_budget([], 1000) == []

    def test_preserves_system_messages(self) -> None:
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = _truncate_to_budget(msgs, 10000)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."

    def test_keeps_recent_messages_on_truncation(self) -> None:
        msgs = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Old message " * 100},
            {"role": "assistant", "content": "Old reply " * 100},
            {"role": "user", "content": "Recent message"},
            {"role": "assistant", "content": "Recent reply"},
        ]
        # Very small budget: system + only recent messages should fit
        result = _truncate_to_budget(msgs, 50)
        # System message preserved
        assert result[0]["role"] == "system"
        # Most recent conversation messages kept
        assert result[-1]["content"] == "Recent reply"

    def test_all_fit_within_budget(self) -> None:
        msgs = [
            {"role": "system", "content": "Short."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = _truncate_to_budget(msgs, 100000)
        assert len(result) == 3

    def test_system_only_when_budget_tiny(self) -> None:
        msgs = [
            {"role": "system", "content": "Prompt."},
            {"role": "user", "content": "A very long message " * 500},
        ]
        # Budget so small only system message fits (plus last message as fallback)
        result = _truncate_to_budget(msgs, 5)
        assert result[0]["role"] == "system"
        # Should have system + last conv msg
        assert len(result) == 2


class TestBuildLlmContextModelInfo:
    async def test_context_includes_model_name(
        self, ctx_manager: ContextManager
    ) -> None:
        agent = Agent(name="test-agent", channel_id=12345, system_prompt="You are helpful.")
        tm = ThreadManager("test-agent")

        result = await build_llm_context(
            agent, None, ctx_manager, tm, None, model="gpt-4o"
        )
        system_content = result[0]["content"]
        assert "gpt-4o" in system_content

    async def test_context_includes_available_models(
        self, ctx_manager: ContextManager
    ) -> None:
        agent = Agent(name="test-agent", channel_id=12345, system_prompt="You are helpful.")
        tm = ThreadManager("test-agent")

        result = await build_llm_context(
            agent, None, ctx_manager, tm, None,
            model="gpt-4o",
            available_models=["gpt-4o", "claude-sonnet-4-20250514"],
        )
        system_content = result[0]["content"]
        assert "gpt-4o" in system_content
        assert "claude-sonnet-4-20250514" in system_content

    async def test_context_uses_agent_model_as_fallback(
        self, ctx_manager: ContextManager
    ) -> None:
        agent = Agent(
            name="test-agent", channel_id=12345,
            system_prompt="You are helpful.",
            model="claude-opus-4-20250514",
        )
        tm = ThreadManager("test-agent")

        result = await build_llm_context(agent, None, ctx_manager, tm, None)
        system_content = result[0]["content"]
        assert "claude-opus-4-20250514" in system_content


class TestBuildLlmContextScopePath:
    async def test_context_includes_scope_path_when_set(
        self, ctx_manager: ContextManager
    ) -> None:
        agent = Agent(name="test-agent", channel_id=12345, system_prompt="You are helpful.")
        tm = ThreadManager("test-agent")

        result = await build_llm_context(
            agent, None, ctx_manager, tm, None,
            scope_path=Path("/mnt/host"),
        )
        system_content = result[0]["content"]
        assert "/mnt/host" in system_content
        assert "$SCOPE_PATH" in system_content
        assert "Host Filesystem Access" in system_content

    async def test_context_excludes_scope_path_when_not_set(
        self, ctx_manager: ContextManager
    ) -> None:
        agent = Agent(name="test-agent", channel_id=12345, system_prompt="You are helpful.")
        tm = ThreadManager("test-agent")

        result = await build_llm_context(agent, None, ctx_manager, tm, None)
        system_content = result[0]["content"]
        assert "Host Filesystem Access" not in system_content
        assert "SCOPE_PATH" not in system_content

    async def test_scope_path_none_by_default(
        self, ctx_manager: ContextManager
    ) -> None:
        agent = Agent(name="test-agent", channel_id=12345, system_prompt="You are helpful.")
        tm = ThreadManager("test-agent")

        # No scope_path kwarg — should be same as not set
        result = await build_llm_context(agent, None, ctx_manager, tm, None)
        system_content = result[0]["content"]
        assert "SCOPE_PATH" not in system_content


class TestBuildLlmContextPreviousBranchSummary:
    async def test_includes_previous_branch_summary(
        self, ctx_manager: ContextManager
    ) -> None:
        agent = Agent(name="test-agent", channel_id=12345, system_prompt="You are helpful.")
        tm = ThreadManager("test-agent")

        result = await build_llm_context(
            agent, None, ctx_manager, tm, None,
            previous_branch_summary="We refactored the auth module.",
            previous_branch_id=3,
        )
        # Should have a system message with the branch summary
        summary_msgs = [
            m for m in result
            if m["role"] == "system" and "refactored the auth module" in m.get("content", "")
        ]
        assert len(summary_msgs) == 1
        assert "branch #3" in summary_msgs[0]["content"]

    async def test_no_summary_when_not_provided(
        self, ctx_manager: ContextManager
    ) -> None:
        agent = Agent(name="test-agent", channel_id=12345, system_prompt="You are helpful.")
        tm = ThreadManager("test-agent")

        result = await build_llm_context(agent, None, ctx_manager, tm, None)
        summary_msgs = [
            m for m in result
            if m["role"] == "system" and "Previous conversation" in m.get("content", "")
        ]
        assert len(summary_msgs) == 0

    async def test_no_summary_when_only_summary_no_id(
        self, ctx_manager: ContextManager
    ) -> None:
        agent = Agent(name="test-agent", channel_id=12345, system_prompt="You are helpful.")
        tm = ThreadManager("test-agent")

        # Summary provided but no branch_id — should not include
        result = await build_llm_context(
            agent, None, ctx_manager, tm, None,
            previous_branch_summary="Some summary",
        )
        summary_msgs = [
            m for m in result
            if m["role"] == "system" and "Previous conversation" in m.get("content", "")
        ]
        assert len(summary_msgs) == 0


class TestMaxInputTokensCap:
    def test_max_input_tokens_value(self) -> None:
        assert MAX_INPUT_TOKENS == 200_000

    def test_get_context_limit_capped_at_max(self) -> None:
        """Even if a model has a higher limit, _get_context_limit caps it."""
        # All known models are <= 200K so they should be unchanged
        assert _get_context_limit("claude-opus-4-20250514") == 200_000
        assert _get_context_limit("gpt-4o") == 128_000

    def test_get_context_limit_caps_unknown_model(self) -> None:
        """Unknown models get the default, which is already below MAX."""
        assert _get_context_limit("future-model-1m") <= MAX_INPUT_TOKENS

    def test_get_context_limit_none_capped(self) -> None:
        assert _get_context_limit(None) <= MAX_INPUT_TOKENS
