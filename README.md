# Chorus

**Multi-Agent Discord Workspace** — each channel is an autonomous AI agent.

---

> **THIS SOFTWARE IS DANGEROUS. READ THIS BEFORE USING IT.**
>
> Chorus gives LLM agents direct access to a real shell, a real filesystem, and real git repos running as your OS user. There is no sandbox, no VM boundary, and no undo button. A single bad tool call can delete files, leak secrets, push to production, or trash your system.
>
> **The threat surface is enormous** — both from a security standpoint and from the reality that LLMs are sometimes reckless, confidently wrong, or outright incompetent. An agent might `rm -rf` your project, commit garbage to main, or curl something it shouldn't. The permission engine helps, but it is not a security boundary. Docker is the only hard boundary, and even that only limits blast radius.
>
> **This trade-off is made deliberately for convenience and utility.** If you accept the risk:
>
> - **Only run Chorus on a Discord server with just you, or with people you would trust to sit at your computer unsupervised.** Every member with access to an agent channel can instruct that agent to do anything its permissions allow — which, on the `open` profile, is everything.
> - **Never run Chorus on a public or semi-public server.** There is no user-level isolation between agents. Any server member can talk to any agent channel they can see.
> - **Treat the host machine as compromised surface area.** If you wouldn't run `curl <random-url> | bash` on the machine, don't run Chorus on it without Docker.
> - **Review the permission profiles.** Start agents on `standard` or `locked`, not `open`. Understand what `ask` vs `allow` means before you grant bash access.
>
> You have been warned.

---

Chorus turns Discord channels into autonomous AI agents. Each channel maps to a dedicated agent with its own working directory, permission profile, system prompt, tools, and git repo. Agents can edit files, run commands, create merge requests, manage their own context, and edit their own configuration — all orchestrated through Discord slash commands.

## Features

- **General-purpose agents** — code, write, research, manage projects, maintain docs, orchestrate workflows
- **Isolated workspaces** — each agent gets its own directory, git repo, and session history
- **Permission profiles** — regex-based allow/ask/deny system controls what agents can do
- **Multi-provider LLM** — Anthropic and OpenAI, with automatic key validation and model discovery
- **Agent self-editing** — agents can update their own system prompt, docs, and configuration
- **Context management** — automatic idle timeout with NL summaries, manual save/restore
- **Task tracking** — agents maintain a list of running programs visible via `/tasks` and `/status`

## Quick Start

```bash
# Clone and set up
git clone <repo-url> chorus
cd chorus
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your Discord token and at least one LLM API key

# Run
python -m chorus
```

## Docker

```bash
cp .env.example .env
# Edit .env with your credentials
docker compose up -d
```

## Commands

| Command | Description |
|---------|-------------|
| `/agent init <name>` | Create a new agent with its own channel and workspace |
| `/agent destroy <name>` | Tear down an agent (optionally keep files) |
| `/agent list` | List all active agents |
| `/agent config <name> <key> <value>` | Update agent configuration |
| `/settings model <model>` | Set default model for new agents |
| `/settings permissions <profile>` | Set default permission profile |
| `/settings show` | Display current defaults and available models |
| `/settings validate-keys` | Re-test API keys and refresh model list |
| `/context save [description]` | Save current session |
| `/context clear` | Clear context (auto-saves first) |
| `/context history` | List saved sessions |
| `/context restore <id>` | Restore a previous session |
| `/tasks` | View agent's running programs |
| `/status` | Agent status including running tasks |

## Development

```bash
pytest                        # Run all tests
pytest tests/test_config.py   # Run single test file
mypy src/chorus/              # Type checking
ruff check src/ tests/        # Lint
ruff format src/ tests/       # Format
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system design.

## License

TBD
