# TODO 006 — Git Integration

## Objective

Implement Git operations as agent tools: init, commit, push, branch, checkout, diff, log, and merge_request. All operations execute within the agent's `workspace/` directory. Push and merge_request require explicit permission (via the ask flow for the `standard` profile). Git operations use the bash execution layer from TODO 005 rather than a Python git library — this keeps the implementation simple and gives agents the full power of the git CLI.

## Acceptance Criteria

- `git_init(workspace)` initializes a git repo in the workspace (already done during agent creation, but the tool is available for re-init or submodule init).
- `git_commit(workspace, message, files?)` stages specified files (or all changes if no files given) and commits. Returns the commit hash and summary.
- `git_push(workspace, remote, branch)` pushes to a remote. Permission action: `tool:git:push <remote> <branch>`.
- `git_branch(workspace, branch_name?, delete?)` creates, lists, or deletes branches.
- `git_checkout(workspace, ref)` checks out a branch, tag, or commit.
- `git_diff(workspace, ref1?, ref2?)` shows diff (default: working tree vs HEAD).
- `git_log(workspace, count?, oneline?)` shows commit log. Default count: 20.
- `git_merge_request(workspace, title, description, source_branch, target_branch)` creates a merge request. For GitHub, this calls `gh pr create`; for GitLab, `glab mr create`. Detection is based on remote URL.
- All git operations go through permission checks with action string `tool:git:<operation> <args>`.
- Git user config (`user.name`, `user.email`) is set per-repo during `git_init` using the agent name and a placeholder email.
- Errors from git commands (merge conflicts, detached HEAD, etc.) are returned as structured results, not swallowed.
- All operations are registered in the tool registry.

## Tests to Write First

File: `tests/test_tools/test_git.py`
```
# Init
test_git_init_creates_repo
test_git_init_sets_user_name_and_email
test_git_init_idempotent_on_existing_repo

# Commit
test_git_commit_stages_and_commits
test_git_commit_specific_files
test_git_commit_returns_hash
test_git_commit_empty_working_tree_fails
test_git_commit_permission_check

# Push
test_git_push_permission_action_string
test_git_push_asks_in_standard_profile
test_git_push_allowed_in_open_profile

# Branch
test_git_branch_create
test_git_branch_list
test_git_branch_delete
test_git_branch_already_exists_error

# Checkout
test_git_checkout_branch
test_git_checkout_creates_branch_with_b_flag
test_git_checkout_nonexistent_ref_error

# Diff
test_git_diff_working_tree_vs_head
test_git_diff_between_refs
test_git_diff_no_changes
test_git_diff_returns_structured_output

# Log
test_git_log_default_count
test_git_log_custom_count
test_git_log_oneline_format
test_git_log_empty_repo

# Merge request
test_merge_request_github_detection
test_merge_request_gitlab_detection
test_merge_request_no_remote_error
test_merge_request_permission_check
test_merge_request_constructs_correct_command

# General
test_git_operation_runs_in_workspace_dir
test_git_operation_returns_stderr_on_failure
```

File: `tests/conftest.py` (new fixtures)
```
git_workspace    — tmp workspace with initialized git repo and initial commit
git_workspace_with_remote — workspace with a mock remote (bare repo in tmp_path)
```

## Implementation Notes

1. **Module location:** `src/chorus/tools/git.py`.

2. **Implementation strategy — wrap bash execution:** Every git tool calls `bash_execute` from TODO 005 under the hood. This avoids a `gitpython` or `pygit2` dependency, gives agents the real git CLI, and reuses all the timeout/output handling from bash.
   ```python
   async def _git(workspace: Path, args: str, profile: PermissionProfile, operation: str) -> BashResult:
       action = format_action("git", f"{operation} {args}".strip())
       perm_result = check(action, profile)
       if perm_result == PermissionResult.DENY:
           raise PermissionDeniedError(action)
       if perm_result == PermissionResult.ASK:
           return BashResult(command=f"git {args}", exit_code=None, stdout="", stderr="",
                             timed_out=False, duration_ms=0, needs_confirmation=True)
       return await bash_execute(f"git {args}", workspace, profile=OPEN_PROFILE, timeout=60)
   ```
   Note: the inner `bash_execute` uses `OPEN_PROFILE` because we've already done the permission check at the git level — don't double-check at the bash level.

3. **git_commit implementation:**
   ```python
   async def git_commit(workspace: Path, message: str, profile: PermissionProfile,
                         files: list[str] | None = None) -> GitResult:
       if files:
           for f in files:
               await _git(workspace, f"add {shlex.quote(f)}", profile, "add")
       else:
           await _git(workspace, "add -A", profile, "add")
       result = await _git(workspace, f"commit -m {shlex.quote(message)}", profile, "commit")
       # Parse commit hash from output
       ...
   ```

4. **Merge request detection:** Check the remote URL:
   ```python
   async def _detect_forge(workspace: Path) -> str:
       result = await bash_execute("git remote get-url origin", workspace, ...)
       url = result.stdout.strip()
       if "github.com" in url:
           return "github"
       elif "gitlab" in url:
           return "gitlab"
       else:
           raise UnsupportedForgeError(url)
   ```
   Then dispatch to `gh pr create` or `glab mr create`.

5. **GitResult dataclass:**
   ```python
   @dataclass
   class GitResult:
       operation: str
       success: bool
       stdout: str
       stderr: str
       commit_hash: str | None = None
       needs_confirmation: bool = False
   ```

6. **User config on init:**
   ```python
   await _git(workspace, f'config user.name "{agent_name}"', profile, "config")
   await _git(workspace, f'config user.email "{agent_name}@chorus.local"', profile, "config")
   ```

7. **Error handling:** Don't swallow git errors. If `git commit` fails because of an empty tree or conflicts, return the stderr as the error message in `GitResult`. The LLM should see these errors and react accordingly.

8. **Testing:** Use real git repos in `tmp_path`. The `git_workspace` fixture should create a temp dir, run `git init`, create an initial file, and commit it. For push tests, create a bare repo as a "remote" in another tmp_path and add it as origin.

## Dependencies

- **002-agent-manager**: Agent workspace directory structure.
- **003-permission-profiles**: Permission checks for git operations.
- **005-bash-execution**: All git commands are executed through `bash_execute`.
