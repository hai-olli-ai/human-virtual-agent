# CLAUDE.md — pipecat-agent

## Project

Human Virtual (hv.ai) — Voice agent for live rooms. Separate service.

## Tech Stack

- Python 3.12, Pipecat framework
- Pipeline: VAD → STT (Deepgram) → LLM (OpenAI) → TTS (Cartesia) → WebRTC
- SmallWebRTCTransport (local dev), DailyTransport (production)

## Completed Sessions (through 51b)

Full pipeline deployed with vision, canvas actions, scene script auto-speak, thinking state, live room polish.

## Session 52a

**No pipecat-agent changes.** Session 52a is backend-only (Gemini-powered background/video/music generation). The agent doesn't interact with these generation tasks.

## Environment Variables

```
DEEPGRAM_API_KEY=
OPENAI_API_KEY=
CARTESIA_API_KEY=
HV_API_URL=http://localhost:8000
```
