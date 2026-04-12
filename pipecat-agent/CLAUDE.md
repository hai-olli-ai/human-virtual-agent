# CLAUDE.md — pipecat-agent

## Project

Human Virtual (hv.ai) — Voice agent for live rooms. Separate service.

## Tech Stack

- Python 3.12, Pipecat framework
- Pipeline: VAD → STT (Deepgram) → LLM (OpenAI) → TTS (Cartesia) → WebRTC
- SmallWebRTCTransport (local dev), DailyTransport (production)
- Deployed to Pipecat Cloud

## Completed Sessions (through 50)

Full pipeline deployed. Vision, canvas actions, avatar media awareness, scene script auto-speak, thinking state notifications. Data channel messages: `transcript`, `speaking_state`, `canvas_action`, `script_complete`, `llm_thinking`.

## Session 51a

**No pipecat-agent changes.** Session 51a is backend-only (Cartesia voice cloning + TTS in Celery tasks). The pipecat agent already uses Cartesia TTS in its real-time pipeline (via Pipecat's built-in Cartesia integration) — that's separate from the batch TTS being added in this session.

Note: In the future, when an avatar has a `voice_model_id` from Cartesia cloning, the pipecat agent could use that cloned voice instead of a stock voice. That wiring would happen in a future session by reading the avatar's `voice_model_id` and passing it to Pipecat's CartesiaTTSService.

## Environment Variables

```
DEEPGRAM_API_KEY=
OPENAI_API_KEY=
CARTESIA_API_KEY=
HV_API_URL=http://localhost:8000
```
