# CLAUDE.md — pipecat-agent

## Project

Human Virtual (hv.ai) — Voice agent for live rooms. Completely separate service from backend/frontend.

## Tech Stack

- Python 3.12, Pipecat framework
- Pipeline: VAD → STT (Deepgram) → LLM (OpenAI) → TTS (Cartesia) → WebRTC
- SmallWebRTCTransport for local dev, DailyTransport in production
- Deployed to Pipecat Cloud

## Transport Divergence (CRITICAL)

```python
# DailyTransport (production):
await transport.send_app_message({"type": "some_event"})  # dict

# SmallWebRTCTransport (local dev):
await transport.send_message(json.dumps({"type": "some_event"}))  # string
```

Always use the existing `_send_data_message` helper that abstracts this.

## Key Files

- `bot.py` — Main pipeline setup, event handlers (on_client_connected, etc.)
- `api_client.py` — Fetches data from backend API (persona prompt, scene snapshot, scene image)
- `scene_context.py` — Builds system prompt from scene data (instruction, elements, scripts)
- `canvas_actions.py` — 5 LLM tools (highlight, arrow, annotation, navigate, clear)

## Canvas Action Conventions

- `run_llm=False` for fire-and-forget tools (highlight, arrow, annotation, clear)
- `run_llm=True` only for `navigate_scene` (LLM describes the new scene)
- Element coordinate resolution uses keyword matching (not AI)

## Session 49 Changes

### What to do

1. **`api_client.py`** — Ensure the scene snapshot parsing preserves the `"scripts"` key. If missing from response, default to `[]`.

2. **`scene_context.py`** — When building the system prompt, if `scene_snapshot.get("scripts")` is non-empty, append:
   ```
   Scene Scripts (you will present these via TTS before conversation begins):
   1. {text}
   2. {text}
   ```

3. **`bot.py`** — In `on_client_connected` (or `on_first_participant_joined`):
   ```python
   from pipecat.frames.frames import TTSSpeakFrame, LLMRunFrame

   if scene_snapshot and scene_snapshot.get("scripts"):
       scripts = sorted(scene_snapshot["scripts"], key=lambda s: s.get("order", 0))
       for script in scripts:
           text = script.get("text", "").strip()
           if text:
               await task.queue_frames([TTSSpeakFrame(text=text)])

       await _send_data_message(transport, {"type": "script_complete"})

       context.add_message(
           "developer",
           "You just finished presenting the scene scripts to the visitor. "
           "They heard your full presentation. Now respond naturally to any "
           "questions or comments they have. Don't repeat what you already said.",
       )
       await task.queue_frames([LLMRunFrame()])
   else:
       # Existing greeting behavior
       context.add_message(
           "developer",
           "A visitor just joined. Greet them warmly and briefly introduce "
           "yourself and the scene you're presenting. Keep it to 1-2 sentences.",
       )
       await task.queue_frames([LLMRunFrame()])
   ```

### Key Points

- `TTSSpeakFrame` bypasses the LLM — direct text-to-speech
- Scripts queue sequentially; Pipecat processes them in order
- VAD handles visitor interruption natively during script playback
- The `developer` context message after scripts ensures LLM doesn't repeat content
- `script_complete` data channel message tells frontend to transition UI

## Environment Variables

```
DEEPGRAM_API_KEY=
OPENAI_API_KEY=
CARTESIA_API_KEY=
HV_API_URL=http://localhost:8000  # or https://api.hv.ai
```

## Commands

```bash
python bot.py  # local dev (SmallWebRTCTransport)
```
