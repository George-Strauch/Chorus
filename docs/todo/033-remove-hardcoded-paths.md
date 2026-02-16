# 033 — Remove Hardcoded & Machine-Specific Paths

## Status: TODO

## Summary

The codebase and agent configuration contain paths that are specific to the current developer's machine (e.g. `/home/george/PycharmProjects/Chorus`, `/mnt/host/PycharmProjects/Chorus`). These should be replaced with environment variables, config lookups, or relative paths so the project is portable and works on any machine.

## Tasks

### 1. Audit hardcoded paths in source code

Grep the source tree for machine-specific paths and replace them:

- `src/chorus/tools/bash.py:184` — comment references `/home/george/X`
- `src/chorus/sub_agents/README.md` — example path `/mnt/host/PycharmProjects/Chorus`
- `src/chorus/sub_agents/tasks/git_status.py:33` — docstring example with `/mnt/host/PycharmProjects/Chorus`

Replace with references to `$SCOPE_PATH`, `$CHORUS_PROJECT_ROOT`, or similar env vars / config values.

### 2. Audit agent docs and system prompts

Agent documentation (self-edit docs, `CLAUDE.md`, etc.) likely contains absolute paths that are machine-specific. These should use placeholder tokens or be dynamically injected.

### 3. Remove `.claude/` from git history

The `.claude/` directory (Claude Code local settings) is already in `.gitignore`, so it's not currently tracked. However, it should be confirmed that:

- `.claude/` never gets accidentally committed (`.gitignore` entry is present ✓)
- If it was ever tracked in past commits, scrub it from git history using `git filter-branch` or `git filter-repo`
- The `.claude/settings.local.json` file contains local-only permissions and should **never** be in the repo — it's developer-specific

If `.claude/` is found in any branch's tracked files, remove it:
```bash
git rm -r --cached .claude/
git commit -m "chore: remove .claude/ from tracking"
```

### 4. Ensure `.gitignore` coverage

Verify `.gitignore` covers all machine-specific / local-only files:

- [x] `.claude/` — already ignored
- [ ] Any other IDE or tool-specific local config directories
- [ ] Agent workspace paths that might leak into commits

## Why This Matters

- **Portability** — Other developers (or CI) can't use paths like `/home/george/...`
- **Security** — Local settings shouldn't leak into version control
- **Clean history** — Machine-specific artifacts don't belong in the repo

## References

- Current `.gitignore`: already has `.claude/` entry
- `CHORUS_SCOPE_PATH` env var: already exists for Docker host mount detection
