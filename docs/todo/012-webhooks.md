# TODO 012 — Webhooks (Future Phase)

## Objective

Implement a webhook system that allows external services to trigger agent actions and allows agents to notify external services of events. This turns agents from passive (respond to Discord messages) into reactive participants in broader workflows — agents can respond to GitHub push events, CI/CD results, monitoring alerts, scheduled cron triggers, and more.

## Acceptance Criteria

- Each agent can register incoming webhook URLs: `POST /webhook/<agent_name>/<hook_id>` triggers a message in the agent's channel with the webhook payload.
- Webhook payloads are validated (HMAC signature verification with a per-hook secret).
- Agents can create outgoing webhooks: register a URL that receives POST notifications when specific events occur (e.g., agent completes a task, file changed, commit pushed).
- `/agent webhook add <name> <type> [url]` registers a webhook. `type` is `incoming` or `outgoing`.
- `/agent webhook list <name>` shows all registered webhooks.
- `/agent webhook remove <name> <hook_id>` removes a webhook.
- Incoming webhooks translate the payload into a formatted message the agent can understand and act on.
- Outgoing webhooks fire on configurable events with JSON payloads.
- Rate limiting prevents webhook abuse (configurable per-hook, default 60 requests/minute).
- Webhook secrets are stored encrypted in SQLite, not in plain text.
- The webhook HTTP server runs alongside the Discord bot (shared event loop, separate port).

## Tests to Write First

File: `tests/test_webhooks/test_incoming.py` (future)
```
test_incoming_webhook_delivers_message_to_channel
test_incoming_webhook_validates_hmac_signature
test_incoming_webhook_rejects_invalid_signature
test_incoming_webhook_rate_limited
test_incoming_webhook_formats_github_push_payload
test_incoming_webhook_formats_generic_json_payload
test_incoming_webhook_nonexistent_agent_returns_404
test_incoming_webhook_disabled_agent_returns_503
```

File: `tests/test_webhooks/test_outgoing.py` (future)
```
test_outgoing_webhook_fires_on_event
test_outgoing_webhook_sends_correct_payload
test_outgoing_webhook_retries_on_failure
test_outgoing_webhook_respects_rate_limit
test_outgoing_webhook_configurable_events
```

## Implementation Notes

1. **HTTP server:** Use `aiohttp.web` running on a configurable port (default 8080). Start it in the bot's `setup_hook` alongside the Discord connection. Share the `asyncio` event loop.

2. **Incoming webhook flow:**
   ```
   External service → POST /webhook/<agent>/<hook_id>
       → Validate HMAC signature
       → Rate limit check
       → Format payload as Discord message
       → Send to agent's channel
       → Agent processes like any other message
   ```

3. **Payload formatters:** Define formatters for common services:
   - GitHub (push, PR, issue)
   - GitLab (push, MR, pipeline)
   - Generic JSON (pretty-printed key-value summary)

   Formatters are registered by name and selected based on a `source` field in the webhook config.

4. **Outgoing webhook events:**
   - `agent.task_complete` — agent finishes a tool loop
   - `agent.file_changed` — file created or modified in workspace
   - `agent.commit` — git commit in workspace
   - `agent.error` — agent encounters an error
   - `agent.session_saved` — context saved

5. **Security:** Webhook secrets generated via `secrets.token_urlsafe(32)`. HMAC uses SHA-256. The signature header follows the GitHub convention: `X-Chorus-Signature-256: sha256=<hex>`.

6. **This is a future phase.** Do not implement until TODOs 001-010 are complete and stable.

## Dependencies

- **001-010**: All core functionality must be complete and stable.
