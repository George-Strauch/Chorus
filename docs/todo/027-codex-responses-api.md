# TODO 027 — OpenAI Responses API Support

## Status: FUTURE

## Objective

Add support for OpenAI's Responses API to enable models that require it (e.g. `codex-mini`, `deep-research`). These models do not work through the Chat Completions API and need a separate provider implementation.

## Context

The current OpenAI integration uses the Chat Completions API exclusively. Some newer OpenAI models (codex-mini for code generation, deep-research for extended research tasks) only support the Responses API, which has a different request/response format and supports additional features like built-in tool use and web search.

## Scope

- New `OpenAIResponsesProvider` implementing the `LLMProvider` protocol
- Responses API tool format mapping (different from Chat Completions tool format)
- Model routing: detect which models need Responses API vs Chat Completions
- Streaming support for long-running deep-research queries
- Update model discovery to identify Responses API models

## Dependencies

- TODO 008 (LLM integration) — completed
- Model shortcut commands (implemented) — provides the command infrastructure
