# TODO 021 — File Editing Refinements

> **Status:** PLANNED

## Problem

The current file editing tools (`create_file`, `str_replace`, `view`) cover the basics but leave gaps that force agents into awkward workarounds:

1. **No line-based insertion.** To insert new content into an existing file, the agent must use `str_replace` with enough surrounding context to produce a unique match, then include the new content alongside the matched text. This is fragile — the match may not be unique, or the agent may get the surrounding context slightly wrong.

2. **No way to replace non-unique strings.** `str_replace` fails if the target appears more than once. This is a safety feature (prevents accidental mass-replacement), but sometimes you genuinely want to replace all occurrences, or target a specific occurrence by line number.

3. **No line-range replacement.** Sometimes you want to replace lines 45–60 with new content. Currently this requires crafting a unique `str_replace` match spanning those exact lines — easy to get wrong on large blocks.

## Proposed New Tools

### 1. `insert_at` — Insert text at a line position

```python
async def insert_at(
    workspace: Path,
    path: str,
    line: int,         # 1-based line number
    content: str,      # Text to insert
    position: str = "before",  # "before" or "after"
) -> FileResult:
```

**Behavior:**
- `line=5, position="before"` → inserts content before the current line 5 (new content becomes lines 5+, old line 5 shifts down).
- `line=5, position="after"` → inserts content after line 5 (new content starts at line 6).
- `line=0, position="after"` or `line=1, position="before"` → insert at the very beginning.
- `line=<last>, position="after"` → append to end of file.
- If `line` is beyond the file's length, append at end (or raise — TBD).
- Returns a context snippet showing the inserted content with surrounding lines.

**Why not just use `create_file`?** Because `create_file` overwrites the entire file. The agent would have to `view` the file, mentally reconstruct the content with its insertion, then write the whole thing back. That's token-expensive, error-prone, and defeats the purpose of structured editing tools.

### 2. `str_replace_all` — Replace all occurrences of a string

```python
async def str_replace_all(
    workspace: Path,
    path: str,
    old_str: str,
    new_str: str,
) -> FileResult:
```

**Behavior:**
- Replaces ALL occurrences of `old_str` with `new_str`.
- Returns the count of replacements made.
- Fails if `old_str` is not found at all (still an error — you expected something to replace).
- Returns a summary snippet rather than context around each replacement.

**Why separate from `str_replace`?** The existing `str_replace` uniqueness constraint is a valuable safety feature. Making it optional (via a flag) would weaken that default safety. A separate tool makes the intent explicit: "I know there are multiple occurrences and I want them all replaced."

### 3. `replace_lines` — Replace a range of lines

```python
async def replace_lines(
    workspace: Path,
    path: str,
    start_line: int,   # 1-based, inclusive
    end_line: int,      # 1-based, inclusive
    new_content: str,   # Replacement text (can be more or fewer lines)
) -> FileResult:
```

**Behavior:**
- Replaces lines `start_line` through `end_line` (inclusive) with `new_content`.
- `new_content` can be empty string → deletes those lines.
- `new_content` can have more lines than the replaced range → file grows.
- Returns context snippet around the replacement.
- Validates that `start_line` and `end_line` are within the file's bounds.

**Why?** This is the most natural way to express "replace this block of code" when you already know the line numbers from a `view` call. No need to construct a unique string match.

## Implementation Plan

### Phase 1: `insert_at`
1. Add `insert_at` function to `tools/file_ops.py`
2. Register in `tools/registry.py`
3. Add permission pattern: `tool:file:insert_at <path>` (same category as file edits)
4. Write tests

### Phase 2: `replace_lines`
1. Add `replace_lines` function to `tools/file_ops.py`
2. Register in `tools/registry.py`
3. Write tests

### Phase 3: `str_replace_all`
1. Add `str_replace_all` function to `tools/file_ops.py`
2. Register in `tools/registry.py`
3. Write tests

## Error Handling

All new tools should follow existing patterns:
- `FileNotFoundInWorkspaceError` if the file doesn't exist
- `PathTraversalError` if path escapes workspace (though current code allows absolute paths — follow that pattern)
- New: `LineOutOfRangeError` for invalid line numbers in `insert_at` and `replace_lines`
- Return `FileResult` with context snippet on success

## Tests to Write

File: `tests/test_tools/test_file_ops_insert.py`
```
# insert_at
test_insert_at_before_line
test_insert_at_after_line
test_insert_at_beginning_of_file
test_insert_at_end_of_file
test_insert_at_multiline_content
test_insert_at_empty_file
test_insert_at_line_beyond_file_appends  (or raises — decide)
test_insert_at_returns_context_snippet
test_insert_at_preserves_file_permissions
test_insert_at_nonexistent_file_raises
```

File: `tests/test_tools/test_file_ops_replace_lines.py`
```
# replace_lines
test_replace_lines_single_line
test_replace_lines_range
test_replace_lines_with_fewer_lines (shrink)
test_replace_lines_with_more_lines (grow)
test_replace_lines_delete (empty new_content)
test_replace_lines_invalid_range_raises
test_replace_lines_out_of_bounds_raises
test_replace_lines_returns_context_snippet
test_replace_lines_preserves_file_permissions
```

File: `tests/test_tools/test_file_ops_replace_all.py`
```
# str_replace_all
test_str_replace_all_multiple_occurrences
test_str_replace_all_single_occurrence
test_str_replace_all_no_match_raises
test_str_replace_all_returns_count
test_str_replace_all_preserves_file_permissions
```

## Open Questions

1. **Should `insert_at` with an out-of-range line silently append, or raise?** Leaning toward appending with a warning in the result — more forgiving for the LLM.
2. **Do we need `insert_at` at all if we have `replace_lines`?** You could simulate insertion with `replace_lines(start=5, end=4, content=...)` (zero-width range), but that's confusing. Explicit `insert_at` is clearer for the LLM.
3. **Should `replace_lines` support a `dry_run` mode?** Show what would change without changing it. Probably overkill for now.
4. **Permission categories:** Should these new tools use the same permission action as `create_file`/`str_replace` (i.e., auto-allow in standard profile), or should they have their own?  Probably same — they're all file edits.

## Dependencies

- **004-file-tools**: Extends existing implementation
- Existing `resolve_path`, `FileResult`, error classes are reused
