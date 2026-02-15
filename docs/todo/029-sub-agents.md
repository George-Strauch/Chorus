# TODO: Sub-Agents System

## Overview

Sub-agents are lightweight, model-specific LLM workers that any agent can delegate tasks to. They inherit the calling agent's **permission profile** and **workspace context**, but run on cheaper/faster models (Haiku, gpt-4o-mini) to keep costs down. The calling agent (running on Opus, Sonnet, etc.) offloads well-defined, bounded tasks to sub-agents instead of burning expensive tokens on routine work.

## Architecture

### Core: `chorus/sub_agents/`

```
src/chorus/sub_agents/
├── __init__.py
├── runner.py          — SubAgentRunner: create provider, build messages, call LLM, return result
├── registry.py        — Registry of available sub-agent task types
└── tasks/
    ├── __init__.py
    ├── git_commit_msg.py  — Generate commit messages from diffs
    ├── git_status.py      — Advanced git status across workspace repos
    ├── file_search.py     — Search/grep files with LLM-guided filtering
    ├── summarize.py       — Summarize text, code, or conversation chunks
    ├── code_review.py     — Quick code review on diffs
    └── explain.py         — Explain code snippets or errors
```

### Permission Inheritance

Sub-agents **absorb the permission profile of the calling agent**:
- If the parent agent has `open` permissions → sub-agent can run any bash/file/git
- If the parent agent has `standard` → sub-agent follows the same allow/ask/deny rules
- If the parent agent has `locked` → sub-agent is view-only

This is achieved by passing the parent's `ToolExecutionContext` (which includes the `PermissionProfile`) through to the sub-agent runner. Sub-agents do NOT get their own permission profile — they always use the caller's.

### Model Selection

Sub-agents default to the cheapest available model:
1. `claude-haiku-4-5-20251001` (preferred — fast, cheap)
2. `gpt-4o-mini` (fallback if no Anthropic key)

The calling agent can override this per-call (e.g., use Sonnet for a harder sub-task).

### Integration Points

Sub-agents are exposed as **tools** in the main tool registry, so the calling agent invokes them the same way it invokes `bash` or `create_file`. The tool loop handles permission checks, the sub-agent runner handles LLM calls.

Example tool: `sub_agent_git_status` — the calling agent invokes it, the sub-agent (Haiku) figures out which repos to scan, runs the static git status script, and returns formatted results.

---

## Planned Sub-Agent Tasks

### 1. Git Commit Message Generator
- **Model**: Haiku
- **Input**: Staged diff (from `git diff --cached` or working tree diff)
- **Output**: Conventional commit message
- **How**: Sub-agent sees the diff, generates a concise commit message following conventional commits format

### 2. Git Status Reporter ✅ (IMPLEMENTED)
- **Model**: Haiku  
- **Input**: Agent's workspace context (docs, recent messages)
- **Output**: Formatted status report for all git repos in scope
- **How**: 
  1. Haiku analyzes the agent's context to identify relevant repo paths
  2. Static script runs `git` commands on each path (branch, last 3 commits, diffstat)
  3. If a path errors, Haiku gets the error and tries to find the correct path or reports the error
  4. Returns formatted Discord-friendly output

### 3. File Search / Grep
- **Model**: Haiku or Sonnet
- **Input**: Natural language query + workspace path
- **Output**: Relevant file paths and snippets
- **How**: Sub-agent uses `find`, `grep`, `rg` to search, then filters results with LLM judgment

### 4. Text/Code Summarizer
- **Model**: Haiku
- **Input**: Long text, code file, or conversation chunk
- **Output**: Concise summary
- **How**: Direct LLM call with summarization prompt

### 5. Quick Code Review
- **Model**: Sonnet or Haiku
- **Input**: Diff or code snippet
- **Output**: Issues, suggestions, style notes
- **How**: Sub-agent reviews code with a focused prompt

### 6. Error Explainer
- **Model**: Haiku
- **Input**: Error message + traceback + surrounding code
- **Output**: Plain-English explanation + suggested fix
- **How**: Direct LLM call with diagnostic prompt

---

## Implementation Plan

### Phase 1: Core Framework (Current)
- [x] `SubAgentRunner` — core runner that creates a provider, calls the LLM, returns structured result
- [x] Permission inheritance from calling agent's `ToolExecutionContext`
- [x] Model selection logic (cheapest available)
- [x] Tool registration pattern for sub-agent tasks

### Phase 2: Git Status (Current)
- [x] `git_status` sub-agent task
- [x] Static bash script for repo status collection
- [x] Haiku-driven path discovery from agent context
- [x] Error recovery (Haiku retries with corrected paths)
- [x] Formatted output for Discord

### Phase 3: More Tasks
- [ ] Git commit message generator
- [ ] File search with LLM filtering
- [ ] Text summarizer
- [ ] Code reviewer
- [ ] Error explainer

### Phase 4: Advanced Features
- [ ] Sub-agent tool access (let sub-agents use a restricted tool set)
- [ ] Cost tracking per sub-agent call
- [ ] Sub-agent result caching (same input → cached output)
- [ ] Configurable model preferences per task type
- [ ] Sub-agent chain-of-thought logging for debugging

---

## Design Decisions

### Why tools, not a separate agent system?
Sub-agents are registered as tools in the existing tool registry. This means:
- They go through the same permission engine
- They show up in the tool loop's event stream (status view)
- No new Discord channels or agent records needed
- The calling agent controls when and how to use them

### Why not just use Claude Code?
Claude Code is heavy — it spins up a full agentic loop with file editing capabilities. Sub-agents are for **quick, focused tasks** that need LLM judgment but not a full agent. A sub-agent call should complete in 1-3 seconds and cost < $0.01.

### Why Haiku specifically?
Haiku is ~60x cheaper than Opus and ~10x cheaper than Sonnet. For bounded tasks (generate a commit message, scan for repos, summarize text), Haiku is more than capable. The cost savings are enormous when these tasks happen frequently.
