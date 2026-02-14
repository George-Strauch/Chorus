# TODO 025 — Message Chunking for Long Responses

> **Status:** PENDING
> **Priority:** Medium — currently causes truncated agent responses
> **Dependencies:** 017 (live status feedback)

## Problem

Discord has a **2000-character message limit**. When the LLM produces a response longer than ~1900 characters, `_truncate_response()` in `ui/status.py` hard-cuts the text and appends `…`:

```python
def _truncate_response(content: str, max_len: int = 1900) -> str:
    if len(content) <= max_len:
        return content
    return content[:max_len] + "…"
```

This means the user **loses the end of the response** with no indication of what was cut. For detailed explanations, code blocks, or multi-step plans, this can cut off critical information.

## Solution

Split long responses into multiple Discord messages, each within the 2000-character limit. The last message gets the status footer.

### Chunking Strategy

1. **Split on natural boundaries** — prefer splitting at paragraph breaks (`\n\n`), then line breaks (`\n`), then sentence boundaries (`. `), then at the character limit as a last resort.
2. **Preserve code blocks** — never split inside a triple-backtick code block. If a code block pushes a chunk over the limit, put the entire code block in the next chunk.
3. **Footer on last chunk only** — the `branch #N · X steps · ...` footer goes on the final message.
4. **First message may be an edit** — the existing `LiveStatusView` edits the "Thinking..." message with the response. If the response needs multiple messages, edit the first message with the first chunk and send additional messages for the rest.

### Character Budget Per Chunk

- Discord limit: 2000 characters
- Footer reserve: ~100 characters (for the last chunk)
- Safe chunk size: **1900 characters** (non-final chunks), **1800 characters** (final chunk with footer)

## Implementation

### 1. New Helper: `chunk_response()`

```python
def chunk_response(content: str, max_chunk: int = 1900) -> list[str]:
    """Split content into Discord-safe chunks, respecting natural boundaries."""
    if len(content) <= max_chunk:
        return [content]

    chunks = []
    remaining = content
    while remaining:
        if len(remaining) <= max_chunk:
            chunks.append(remaining)
            break
        # Find best split point
        split_at = _find_split_point(remaining, max_chunk)
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    return chunks
```

### 2. Update `LiveStatusView.finalize()`

Instead of calling `_truncate_response()`, call `chunk_response()` and send multiple messages:

```python
chunks = chunk_response(response_content)
# First chunk: edit or send as before
# Subsequent chunks: send as new messages
# Footer appended to last chunk only
```

### 3. Track All Message IDs

All chunk messages should be registered with `tm.register_bot_message()` so reply-based routing works when a user replies to any chunk of a multi-message response.

## Acceptance Criteria

- [ ] Responses longer than 1900 chars are split into multiple Discord messages
- [ ] Splits prefer paragraph breaks > line breaks > sentence boundaries > hard cut
- [ ] Code blocks (triple-backtick) are never split mid-block
- [ ] Status footer appears only on the last message
- [ ] First chunk edits the "Thinking..." message (existing behavior preserved)
- [ ] All chunk message IDs are registered for reply-based routing
- [ ] Existing tests pass
- [ ] New tests: chunking logic, code block preservation, footer placement, multi-message finalize

## Files

| File | Action |
|------|--------|
| `src/chorus/ui/status.py` | Modify — replace `_truncate_response` with `chunk_response`, update `finalize()` |
| `src/chorus/bot.py` | Modify — register additional message IDs from chunks |
| `tests/test_ui/test_status.py` | Add — chunk_response tests |

## Notes

- Discord rate limits: sending multiple messages in quick succession may hit rate limits. Add a small delay (~0.1s) between chunk sends if needed.
- Consider a max chunk count (e.g., 10 messages = 19,000 chars) to prevent flooding the channel. If the response exceeds this, truncate and append "(response truncated — N characters omitted)".
- The `persist_message` call should store the full content, not individual chunks — chunking is a display concern only.
