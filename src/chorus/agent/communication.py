"""Inter-agent communication — send messages, read docs, list agents.

Provides fire-and-forget messaging between agents. The target agent runs
the message as a new branch with its own permissions (not admin).
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("chorus.agent.communication")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_first_paragraph(markdown: str) -> str:
    """Extract the first meaningful text paragraph from markdown.

    Skips headings (#), emphasis-only lines (*bold*, **bold**, > quotes),
    and empty lines. Returns the first line that looks like prose, capped
    at 200 characters.
    """
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip headings
        if stripped.startswith("#"):
            continue
        # Skip emphasis-only lines (e.g. *Status: active*, **Bold**)
        if stripped.startswith("*") or stripped.startswith("**"):
            continue
        # Skip blockquotes
        if stripped.startswith(">"):
            continue
        # Found a prose line
        if len(stripped) > 200:
            return stripped[:197] + "..."
        return stripped
    return ""


# ---------------------------------------------------------------------------
# send_to_agent
# ---------------------------------------------------------------------------


async def send_to_agent(
    target_agent: str,
    message: str,
    *,
    agent_name: str,
    bot: Any = None,
    chorus_home: Path | None = None,
) -> str:
    """Send a fire-and-forget message to another agent.

    Spawns a new branch in the target agent's channel with an attributed
    message. The target runs with its own permissions (not admin).
    """
    # Validate: no self-send
    if target_agent == agent_name:
        return json.dumps({"error": "Cannot send a message to your own agent."})

    # Validate: target exists
    if chorus_home is not None:
        target_dir = chorus_home / "agents" / target_agent
        if not target_dir.is_dir():
            return json.dumps({"error": f"Agent '{target_agent}' not found."})

    # Validate: bot available
    if bot is None:
        return json.dumps({"error": "Bot not available — cannot deliver messages."})

    # Build attributed message
    attributed = f"Message from agent '{agent_name}': {message}"

    # Spawn branch in target agent
    try:
        await bot.spawn_agent_branch(
            agent_name=target_agent,
            message=attributed,
            sender_agent=agent_name,
        )
    except Exception as exc:
        logger.warning(
            "Failed to deliver message from %s to %s: %s",
            agent_name, target_agent, exc,
        )
        return json.dumps({
            "error": f"Failed to deliver message to '{target_agent}': {exc}",
        })

    logger.info("Agent %s sent message to %s", agent_name, target_agent)
    return json.dumps({"delivered": True, "target": target_agent})


# ---------------------------------------------------------------------------
# read_agent_docs
# ---------------------------------------------------------------------------


async def read_agent_docs(
    target_agent: str,
    *,
    agent_name: str,
    chorus_home: Path | None = None,
) -> str:
    """Read all .md files from another agent's docs/ directory."""
    if target_agent == agent_name:
        return json.dumps({"error": "Use your own docs/ directory directly."})

    if chorus_home is None:
        return json.dumps({"error": "chorus_home not configured."})

    target_dir = chorus_home / "agents" / target_agent
    if not target_dir.is_dir():
        return json.dumps({"error": f"Agent '{target_agent}' not found."})

    docs_dir = target_dir / "docs"
    docs: dict[str, str] = {}
    if docs_dir.is_dir():
        for md_file in sorted(docs_dir.rglob("*.md")):
            rel = md_file.relative_to(docs_dir)
            try:
                docs[str(rel)] = md_file.read_text(encoding="utf-8")
            except Exception:
                docs[str(rel)] = "(unreadable)"

    return json.dumps({"agent": target_agent, "docs": docs})


# ---------------------------------------------------------------------------
# list_agents
# ---------------------------------------------------------------------------


async def list_agents(
    *,
    agent_name: str,
    chorus_home: Path | None = None,
    db: Any = None,
) -> str:
    """List all available agents (excluding self) with descriptions and models."""
    if chorus_home is None:
        return json.dumps({"agents": []})

    agents_dir = chorus_home / "agents"
    if not agents_dir.is_dir():
        return json.dumps({"agents": []})

    result: list[dict[str, Any]] = []
    for agent_path in sorted(agents_dir.iterdir()):
        if not agent_path.is_dir():
            continue
        name = agent_path.name
        if name == agent_name:
            continue

        # Read model from agent.json
        model: str | None = None
        agent_json = agent_path / "agent.json"
        if agent_json.exists():
            try:
                data = json.loads(agent_json.read_text())
                model = data.get("model")
            except Exception:
                pass

        # Read description from docs/README.md
        description = ""
        readme = agent_path / "docs" / "README.md"
        if readme.exists():
            with contextlib.suppress(Exception):
                description = _extract_first_paragraph(
                    readme.read_text(encoding="utf-8")
                )

        result.append({
            "name": name,
            "model": model,
            "description": description,
        })

    return json.dumps({"agents": result})
