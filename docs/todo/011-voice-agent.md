# TODO 011 — Voice Agent (Future Phase)

> **Status:** PENDING

## Objective

Extend Chorus to support voice channel agents: agents that can join voice channels, listen to speech (via speech-to-text), respond with synthesized speech (via text-to-speech), and maintain the same tool-use capabilities as text agents. A voice agent is an agent whose primary interface is audio instead of text, but the underlying agent architecture (workspace, tools, permissions, context) remains identical.

## Acceptance Criteria

- An agent can be created with `type: "voice"` that binds to a voice channel instead of a text channel.
- The agent joins its voice channel on activation and leaves on deactivation or idle timeout.
- Incoming audio is transcribed to text using a speech-to-text provider (Whisper API or equivalent).
- Agent text responses are synthesized to audio using a text-to-speech provider and played in the voice channel.
- Voice agents still have a companion text channel for tool output, logs, and confirmation prompts (the `ASK` flow).
- The permission system works identically — tool actions are checked before execution regardless of input modality.
- Voice activity detection determines when a user has finished speaking (pause-based endpoint detection).
- Context management works the same: conversations are stored as text transcriptions.
- `/agent init <name> --voice` creates a voice agent.
- Voice agents can be muted/deafened via slash commands.
- Concurrent speakers are handled gracefully (queue or interleave based on voice activity).

## Tests to Write First

File: `tests/test_agent/test_voice.py` (future)
```
test_voice_agent_creation_binds_to_voice_channel
test_voice_agent_joins_channel_on_activation
test_voice_agent_leaves_channel_on_deactivation
test_voice_agent_transcribes_audio_to_text
test_voice_agent_synthesizes_text_to_audio
test_voice_agent_companion_text_channel_created
test_voice_agent_permission_checks_apply
test_voice_agent_idle_timeout_triggers_leave
test_voice_agent_context_stores_transcriptions
test_voice_agent_handles_concurrent_speakers
test_voice_agent_mute_unmute_commands
```

## Implementation Notes

1. **discord.py voice support:** Use `discord.VoiceClient` and `discord.FFmpegPCMAudio`. The bot needs the `voice_states` intent and `connect`/`speak` permissions.

2. **Audio pipeline:**
   ```
   User speaks → discord.py PCM audio sink → buffer → Whisper API (STT)
       → text message into agent context → LLM response
       → TTS API → PCM audio source → discord.py playback
   ```

3. **STT/TTS providers:** Start with OpenAI Whisper for STT and OpenAI TTS for synthesis. Abstract behind provider interfaces so alternatives can be added.

4. **Voice activity detection:** Use `discord.py`'s `on_voice_state_update` and audio receive hooks. Buffer audio per user. Detect end-of-utterance by silence duration (configurable, default 1.5 seconds).

5. **Companion text channel:** Every voice agent gets a text channel named `<agent-name>-log` for tool output, confirmations, and debugging. Agents read/write their workspace the same way — the only difference is the I/O modality.

6. **Cost consideration:** Voice agents will consume significantly more API calls (continuous STT). Implement a "push-to-talk" mode as an alternative to voice activity detection.

7. **This is a future phase.** Do not implement until TODOs 001-010 are complete and stable.

## Dependencies

- **001-010**: All core functionality must be complete and stable.
