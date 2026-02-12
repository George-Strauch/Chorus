# TODO 008 — LLM Integration

## Objective

Implement the multi-provider LLM client supporting Anthropic and OpenAI, API key validation, model discovery, and the agentic tool use loop. This is the brain of the system — it takes a conversation, available tools, and a system prompt, sends them to the LLM, executes any tool calls the LLM makes (with permission checks), feeds results back, and loops until the LLM produces a final text response.

## Acceptance Criteria

- `LLMProvider` protocol defines the interface: `chat(messages, tools, system_prompt, model) -> LLMResponse`.
- `AnthropicProvider` implements `LLMProvider` using the `anthropic` SDK (Messages API with tool use).
- `OpenAIProvider` implements `LLMProvider` using the `openai` SDK (Chat Completions API with function calling).
- Both providers normalize responses into a unified `LLMResponse` dataclass containing: `content` (text), `tool_calls` (list), `stop_reason`, `usage` (input/output tokens), `model`.
- `validate_key(provider, key) -> bool` tests whether an API key is valid by making a minimal API call.
- `discover_models(provider, key) -> list[str]` returns available models for a key. For Anthropic, returns a hardcoded list of known models (Anthropic doesn't have a models endpoint). For OpenAI, calls the models list API and filters to chat models.
- Results of key validation and model discovery are cached in `~/.chorus-agents/available_models.json`. Cache is refreshed on `/settings validate-keys`.
- **Tool loop** (`tool_loop.py`):
  1. Send messages + tools to LLM.
  2. If response contains tool calls, execute each tool (with permission checks).
  3. For `ASK` results, post a confirmation prompt to Discord and wait for user response.
  4. Append tool results to messages, loop back to step 1.
  5. If response is text only (no tool calls), return it as the final answer.
  6. Maximum loop iterations: configurable (default 25) to prevent infinite loops.
- Tool definitions are translated from `ToolRegistry` format to provider-specific format (Anthropic tool schema vs OpenAI function schema).
- Streaming is supported for the final text response (yields chunks for Discord message updates).
- Usage tracking: each loop iteration logs token usage for cost awareness.

## Tests to Write First

File: `tests/test_llm/test_providers.py`
```
# Anthropic provider
test_anthropic_provider_sends_correct_request_format
test_anthropic_provider_parses_text_response
test_anthropic_provider_parses_tool_use_response
test_anthropic_provider_includes_system_prompt
test_anthropic_provider_handles_rate_limit
test_anthropic_provider_handles_invalid_key
test_anthropic_provider_normalizes_to_llm_response

# OpenAI provider
test_openai_provider_sends_correct_request_format
test_openai_provider_parses_text_response
test_openai_provider_parses_function_call_response
test_openai_provider_includes_system_prompt_as_system_message
test_openai_provider_handles_rate_limit
test_openai_provider_handles_invalid_key
test_openai_provider_normalizes_to_llm_response

# Unified interface
test_provider_protocol_compatibility
test_both_providers_return_same_response_shape
```

File: `tests/test_llm/test_discovery.py`
```
test_validate_key_anthropic_valid
test_validate_key_anthropic_invalid
test_validate_key_openai_valid
test_validate_key_openai_invalid
test_discover_models_anthropic_returns_known_models
test_discover_models_openai_filters_chat_models
test_cache_writes_to_available_models_json
test_cache_reads_from_available_models_json
test_cache_refresh_overwrites_stale_data
```

File: `tests/test_llm/test_tool_loop.py`
```
# Core loop
test_tool_loop_returns_text_response_immediately
test_tool_loop_executes_single_tool_call
test_tool_loop_executes_multiple_tool_calls_in_sequence
test_tool_loop_feeds_tool_result_back_to_llm
test_tool_loop_multi_turn_conversation
test_tool_loop_max_iterations_stops_infinite_loop
test_tool_loop_max_iterations_returns_partial_result

# Permission integration
test_tool_loop_allowed_tool_executes_automatically
test_tool_loop_ask_tool_prompts_user
test_tool_loop_denied_tool_skipped_with_error
test_tool_loop_user_approves_ask_tool
test_tool_loop_user_rejects_ask_tool

# Error handling
test_tool_loop_tool_execution_error_fed_back_to_llm
test_tool_loop_llm_api_error_raised
test_tool_loop_malformed_tool_call_handled

# Schema translation
test_tool_schema_anthropic_format
test_tool_schema_openai_format
test_tool_schema_parameter_types_preserved

# Streaming
test_tool_loop_streams_final_response
test_tool_loop_no_streaming_during_tool_calls

# Usage tracking
test_tool_loop_tracks_total_token_usage
```

## Implementation Notes

1. **Module locations:**
   - `src/chorus/llm/providers.py` — `LLMProvider` protocol, `AnthropicProvider`, `OpenAIProvider`
   - `src/chorus/llm/discovery.py` — key validation, model discovery, caching
   - `src/chorus/llm/tool_loop.py` — agentic loop

2. **LLMProvider protocol:**
   ```python
   class LLMProvider(Protocol):
       async def chat(
           self,
           messages: list[dict[str, Any]],
           tools: list[dict[str, Any]] | None = None,
           system_prompt: str | None = None,
           model: str | None = None,
           stream: bool = False,
       ) -> LLMResponse | AsyncIterator[LLMChunk]: ...
   ```

3. **LLMResponse dataclass:**
   ```python
   @dataclass
   class LLMResponse:
       content: str | None
       tool_calls: list[ToolCall]
       stop_reason: str  # "end_turn", "tool_use", "max_tokens"
       usage: Usage
       model: str

   @dataclass
   class ToolCall:
       id: str
       name: str
       arguments: dict[str, Any]

   @dataclass
   class Usage:
       input_tokens: int
       output_tokens: int
   ```

4. **Anthropic message format** — the Anthropic SDK uses:
   ```python
   client.messages.create(
       model=model,
       system=system_prompt,
       messages=messages,
       tools=[{"name": ..., "description": ..., "input_schema": ...}],
       max_tokens=4096,
   )
   ```
   Tool results go back as `{"role": "user", "content": [{"type": "tool_result", "tool_use_id": ..., "content": ...}]}`.

5. **OpenAI function calling format** — the OpenAI SDK uses:
   ```python
   client.chat.completions.create(
       model=model,
       messages=[{"role": "system", "content": system_prompt}, ...],
       tools=[{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}],
   )
   ```
   Tool results go back as `{"role": "tool", "tool_call_id": ..., "content": ...}`.

6. **Tool loop core logic:**
   ```python
   async def run_tool_loop(
       provider: LLMProvider,
       messages: list[dict],
       tools: ToolRegistry,
       profile: PermissionProfile,
       system_prompt: str,
       model: str,
       max_iterations: int = 25,
       ask_callback: Callable[[str, str], Awaitable[bool]] | None = None,
   ) -> ToolLoopResult:
       for i in range(max_iterations):
           response = await provider.chat(messages, tools.get_for_provider(...), system_prompt, model)
           if not response.tool_calls:
               return ToolLoopResult(content=response.content, messages=messages, ...)
           for tc in response.tool_calls:
               tool = tools.get(tc.name)
               perm = check(format_action(tc.name, str(tc.arguments)), profile)
               if perm == DENY:
                   result = {"error": "Permission denied"}
               elif perm == ASK:
                   approved = await ask_callback(tc.name, str(tc.arguments)) if ask_callback else False
                   result = await tool.handler(**tc.arguments) if approved else {"error": "User declined"}
               else:
                   result = await tool.handler(**tc.arguments)
               messages.append(tool_result_message(tc.id, result))
       return ToolLoopResult(content="Max iterations reached", ...)
   ```

7. **Ask callback for Discord:** The `ask_callback` is provided by the Discord layer. It posts an embed with the tool name and arguments, adds approve/reject buttons, and awaits the user's response with a timeout.

8. **Key validation:** For Anthropic, send a minimal `messages.create` with a tiny prompt and `max_tokens=1`. For OpenAI, call `models.list()`. Catch `AuthenticationError` to detect invalid keys.

9. **Model discovery caching:**
   ```json
   {
     "last_updated": "2026-02-12T10:00:00Z",
     "providers": {
       "anthropic": {
         "valid": true,
         "models": ["claude-sonnet-4-20250514", "claude-haiku-35-20241022", ...]
       },
       "openai": {
         "valid": true,
         "models": ["gpt-4o", "gpt-4o-mini", ...]
       }
     }
   }
   ```

10. **Testing:** Mock both SDK clients entirely. Use `unittest.mock.AsyncMock` to mock `anthropic.AsyncAnthropic` and `openai.AsyncOpenAI`. The tool loop tests should use a fake provider that returns scripted responses. Never make real API calls in tests.

## Dependencies

- **003-permission-profiles**: Permission checks within the tool loop.
