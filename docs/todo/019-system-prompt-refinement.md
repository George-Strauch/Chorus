# TODO 019 — System Prompt Refinement on Init

> **Status:** COMPLETED

## Objective

When an agent is created via `/agent init`, use a small/fast LLM (Haiku or gpt-4o-mini) to refine the default system prompt based on the agent's name, any user-provided description, and the channel context. This gives each new agent a tailored starting prompt instead of a generic template, making agents useful immediately without manual prompt engineering.

## Acceptance Criteria

- On `/agent init <name> [system_prompt]`, after copying the template, a small LLM call refines the system prompt before writing it to `agent.json`.
- The refinement LLM receives: the agent name, any user-supplied system prompt or description, and the default template prompt as a starting point.
- The refined prompt preserves the structural conventions of the template prompt (tool usage instructions, workspace awareness, permission awareness) while adding personality, focus area, and domain-specific guidance inferred from the name and description.
- If no LLM keys are configured or the refinement call fails, fall back silently to the default template prompt (agent creation must never fail because of refinement).
- The refinement uses the cheapest available model (prefer Haiku/gpt-4o-mini) — not the agent's configured model.
- Refinement adds no more than ~2 seconds to agent creation in the common case.
- The refined prompt is visible via `/agent config <name> show` and editable via self-edit tools as usual.

## Tests to Write First

File: `tests/test_agent/test_prompt_refinement.py`
```
# Core refinement
test_refine_prompt_returns_refined_text
test_refine_prompt_includes_agent_name
test_refine_prompt_preserves_template_structure
test_refine_prompt_incorporates_user_description

# Fallback behavior
test_refine_prompt_returns_default_on_no_api_key
test_refine_prompt_returns_default_on_llm_error
test_refine_prompt_returns_default_on_timeout

# Model selection
test_refine_picks_cheapest_available_model
test_refine_prefers_haiku_over_sonnet
test_refine_falls_back_to_openai_mini_if_no_anthropic_key

# Integration with agent init
test_agent_init_calls_refinement
test_agent_init_stores_refined_prompt_in_agent_json
test_agent_init_succeeds_even_if_refinement_fails
```

## Implementation Notes

1. **Module location:** `src/chorus/agent/prompt_refinement.py`.

2. **Core function:**
   ```python
   async def refine_system_prompt(
       agent_name: str,
       user_description: str | None,
       template_prompt: str,
       config: BotConfig,
   ) -> str:
       """Refine the template system prompt using a small LLM.

       Returns the refined prompt, or the original template_prompt
       on any failure.
       """
   ```

3. **Meta-prompt for refinement:** The LLM call that generates the refined prompt should receive a meta-prompt like:
   ```
   You are configuring a new AI agent. Given the agent's name and description,
   refine the system prompt to be specific to this agent's role.

   Keep all structural elements (tool instructions, workspace rules, permission
   awareness) from the template. Add personality, domain expertise, and
   task-specific guidance based on the name and description.

   Agent name: {name}
   User description: {description or "none provided — infer from the name"}
   Template prompt:
   {template_prompt}

   Output ONLY the refined system prompt, nothing else.
   ```

4. **Model selection logic:**
   ```python
   def _pick_refinement_model(config: BotConfig) -> tuple[str, LLMProvider] | None:
       """Pick the cheapest available model for prompt refinement."""
       if config.anthropic_api_key:
           return ("claude-haiku-4-5-20251001", AnthropicProvider(...))
       if config.openai_api_key:
           return ("gpt-4o-mini", OpenAIProvider(...))
       return None
   ```

5. **Timeout:** Wrap the LLM call in `asyncio.wait_for(coro, timeout=10.0)`. If it times out, log a warning and return the template prompt unchanged.

6. **Integration point:** Call `refine_system_prompt()` in `AgentManager.create_agent()`, after copying the template but before writing the final `agent.json`. Pass the result as the system prompt.

7. **No tool use in refinement:** This is a single-shot LLM call with no tools — just text in, text out. Use the provider's `chat()` method directly.

## Dependencies

- **001-core-bot**: BotConfig for API keys.
- **002-agent-manager**: Agent creation flow where refinement is called.
- **008-llm-integration**: LLM providers for making the refinement call.
