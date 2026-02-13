"""Live test suite definitions for Chorus."""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from chorus.permissions.engine import get_preset
from chorus.tools.bash import bash_execute
from chorus.tools.file_ops import create_file, view

if TYPE_CHECKING:
    from chorus.testing.runner import TestRunner


# ---------------------------------------------------------------------------
# Basic suite
# ---------------------------------------------------------------------------


async def test_ping_latency(runner: TestRunner) -> str | None:
    """Check bot websocket latency."""
    latency_ms = runner.bot.latency * 1000
    if latency_ms < 0:
        raise RuntimeError("Negative latency — bot may not be connected")
    return f"{latency_ms:.0f}ms"


async def test_channel_send(runner: TestRunner) -> str | None:
    """Send a message to the test channel and verify it arrives."""
    msg = await runner.channel.send("_live test probe_")
    if msg.id is None:
        raise RuntimeError("Message ID is None after send")
    await msg.delete()
    return f"msg_id={msg.id}"


async def test_db_connectivity(runner: TestRunner) -> str | None:
    """Run a simple DB query to verify connectivity."""
    db = runner.bot.db  # type: ignore[attr-defined]
    agents = await db.list_agents()
    return f"{len(agents)} agent(s) in DB"


# ---------------------------------------------------------------------------
# Throughput suite
# ---------------------------------------------------------------------------

_BURST_COUNT = 10


async def test_message_burst(runner: TestRunner) -> str | None:
    """Send N messages rapidly and measure throughput."""
    msgs = []
    for i in range(_BURST_COUNT):
        m = await runner.channel.send(f"_burst {i}_")
        msgs.append(m)
    # Cleanup
    for m in msgs:
        await m.delete()
    return f"{_BURST_COUNT} messages sent"


async def test_concurrent_sends(runner: TestRunner) -> str | None:
    """Send messages from multiple asyncio tasks concurrently."""
    count = 5

    async def _send(i: int) -> Any:
        return await runner.channel.send(f"_concurrent {i}_")

    results = await asyncio.gather(*[_send(i) for i in range(count)])
    # Cleanup
    for m in results:
        await m.delete()
    return f"{count} concurrent sends OK"


# ---------------------------------------------------------------------------
# Agent suite
# ---------------------------------------------------------------------------


async def test_agent_create_destroy(runner: TestRunner) -> str | None:
    """Create a temporary agent, verify, then destroy it."""
    manager = runner.bot.agent_manager  # type: ignore[attr-defined]
    name = "_live_test_agent"
    channel_id = runner.channel.id
    guild_id = runner.channel.guild.id

    try:
        agent = await manager.create(
            name=name,
            guild_id=guild_id,
            channel_id=channel_id,
        )
        if agent.name != name:
            raise RuntimeError(f"Agent name mismatch: {agent.name!r} != {name!r}")

        # Verify DB entry
        db_agent = await manager._db.get_agent(name)
        if db_agent is None:
            raise RuntimeError("Agent not found in DB after creation")
    finally:
        # Always clean up
        with contextlib.suppress(Exception):
            await manager.destroy(name, keep_files=False)

    return "create + destroy OK"


async def test_agent_list(runner: TestRunner) -> str | None:
    """List agents and verify response format."""
    manager = runner.bot.agent_manager  # type: ignore[attr-defined]
    agents = await manager.list_agents()
    return f"{len(agents)} agent(s)"


# ---------------------------------------------------------------------------
# Tools suite
# ---------------------------------------------------------------------------


async def test_file_create_view(runner: TestRunner) -> str | None:
    """Create a file in a temp workspace, view it, verify content."""
    workspace = Path(tempfile.mkdtemp(prefix="chorus_test_"))
    try:
        result = await create_file(workspace, "test.txt", "hello from live test")
        if not result.success:
            raise RuntimeError(f"create_file failed: {result.error}")

        view_result = await view(workspace, "test.txt")
        if not view_result.success:
            raise RuntimeError(f"view failed: {view_result.error}")
        if (
            view_result.content_snippet
            and "hello from live test" not in view_result.content_snippet
        ):
            raise RuntimeError("Content mismatch")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    return "create + view OK"


async def test_bash_echo(runner: TestRunner) -> str | None:
    """Run echo hello via bash tool and verify output."""
    workspace = Path(tempfile.mkdtemp(prefix="chorus_test_"))
    try:
        profile = get_preset("open")
        result = await bash_execute(
            command="echo hello",
            workspace=workspace,
            profile=profile,
            timeout=10.0,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Non-zero exit: {result.exit_code}")
        if "hello" not in result.stdout:
            raise RuntimeError(f"Unexpected output: {result.stdout!r}")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    return f"exit={result.exit_code}, stdout={result.stdout.strip()!r}"


# ---------------------------------------------------------------------------
# Suite registry
# ---------------------------------------------------------------------------

SUITES: dict[str, list[Callable[..., Any]]] = {
    "basic": [test_ping_latency, test_channel_send, test_db_connectivity],
    "throughput": [test_message_burst, test_concurrent_sends],
    "agent": [test_agent_create_destroy, test_agent_list],
    "tools": [test_file_create_view, test_bash_echo],
}
# "all" is the union of every other suite
SUITES["all"] = [fn for suite in SUITES.values() for fn in suite]
