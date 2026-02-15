# Sub-Agent System for Chorus

This module provides a lightweight sub-agent system for running specialized LLM tasks without tool access. Sub-agents are useful for filtering, formatting, error recovery, and other tasks that benefit from cheap LLM calls.

## Architecture

### Core Components

1. **`runner.py`** - Sub-agent execution engine
   - `SubAgentResult` - Result dataclass with success, output, model_used, usage, and error
   - `pick_cheap_model()` - Selects cheapest available model (Haiku preferred, gpt-4o-mini fallback)
   - `run_sub_agent()` - Executes a one-shot LLM call with system prompt and messages

2. **`tasks/`** - Task-specific sub-agent implementations
   - `git_status.py` - Advanced git status tool with multi-step workflow

## Usage

### Basic Sub-Agent Call

```python
from chorus.sub_agents.runner import run_sub_agent

result = await run_sub_agent(
    system_prompt="You are a helpful assistant.",
    messages=[
        {"role": "user", "content": "Process this data: ..."}
    ],
    timeout=15.0,
)

if result.success:
    print(result.output)
else:
    print(f"Error: {result.error}")
```

### Advanced Git Status Tool

The `git_status` tool demonstrates a sophisticated multi-step sub-agent workflow:

#### Step 1: Path Discovery (Haiku call)
- Scans workspace and common mount points for `.git` directories
- Sends discovered repos to Haiku with agent context
- Haiku filters to relevant repositories based on agent's workspace and name

#### Step 2: Static Script Execution
- For each relevant repo, runs git commands to collect:
  - Current branch name
  - Last 3 commit messages
  - Working tree changes (diffstat)
  - Staged changes (diffstat)
  - Untracked file count
- No LLM involved - pure bash execution

#### Step 3: Error Recovery (Haiku call, conditional)
- If any repo paths fail, sends errors to Haiku
- Haiku suggests corrected paths or confirms real errors
- Retries with corrected paths

#### Step 4: Format Output
- Combines all results into a formatted report with emoji indicators
- Shows branch, commits, changes, and untracked files per repo

### Example Output

```
📁 /mnt/host/PycharmProjects/Chorus
   Branch: main
   Recent commits:
     abc1234 feat: add sub-agent system
     def5678 fix: permission inheritance
     ghi9012 refactor: tool registry
   Changes: 4 files changed, 200 insertions(+), 10 deletions(-)
     src/chorus/sub_agents/runner.py   +85 -0
     src/chorus/sub_agents/git_status.py   +120 -0
   Staged: (none)
   Untracked: 2 files

📁 /mnt/host/Projects/other-repo
   Branch: develop
   Recent commits:
     ...
   Changes: (clean)
```

## Design Principles

1. **Cheap by default** - Uses Haiku or gpt-4o-mini to minimize costs
2. **No tool access** - Sub-agents make pure LLM calls without tool execution
3. **Graceful degradation** - Falls back to simpler behavior if API keys unavailable
4. **Timeout protection** - All operations have reasonable timeouts
5. **Error handling** - Returns structured results with success flags and error messages

## Permission Integration

The `git_status` tool is registered in the tool registry and uses the "git" permission category. It's allowed by the "standard" permission preset without requiring approval.

## Implementation Notes

- Sub-agents use environment variables for API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY)
- Model selection prefers Anthropic Haiku for cost efficiency
- All async operations have configurable timeouts
- Logging uses the `chorus.sub_agents.*` namespace
- Type hints and docstrings follow the Chorus codebase style
