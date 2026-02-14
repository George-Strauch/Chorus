# TODO 020 — Web Search Tool (Backlog)

> **Status:** PENDING

## Objective

Give agents the ability to search the web and retrieve content. Adds a `web_search` tool to the tool registry with dedicated permission integration (`tool:web_search:<query>`).

## Approach Options

Decide on a search API backend before implementation:

1. **Tavily** — Purpose-built for AI agents, returns clean text extracts (not just links). Free tier (1000 searches/mo). `pip install tavily-python`. Most popular in agent frameworks.
2. **Brave Search API** — Good free tier, structured JSON results. Requires parsing snippets.
3. **DuckDuckGo** — No API key needed (`duckduckgo-search` package). Rate-limited, less reliable.
4. **Bash/curl** — Agents already have bash. No new code, but no dedicated permissions and messy output.

Recommendation: Tavily for clean agent-friendly output with minimal parsing.

## Acceptance Criteria

- New `tools/web_search.py` with a `web_search(query, ...)` tool function.
- Tool registered in `registry.py`, permission category `web_search`.
- API key loaded from env var (e.g. `TAVILY_API_KEY`), added to `.env.example`.
- Results returned as structured text the LLM can reason over (not raw HTML).
- Optional `max_results` parameter (default 5).
- Permission action string: `tool:web_search:<query>`.
- Standard preset updated to include `web_search` in allow or ask as appropriate.
- Tests with mocked API client (no real API calls in tests).

## Dependencies

- TODO 004 (tool registry pattern)
- TODO 003 (permission engine)

## Implementation Notes

- Follow the same pattern as `tools/bash.py` and `tools/file_ops.py`: function with keyword-only injected args, registered in `create_default_registry()`.
- Add `SEARCH_API_KEY` (or provider-specific key) to `BotConfig.from_env()`.
- Consider adding a `web_fetch(url)` companion tool that retrieves and extracts text from a specific URL.
