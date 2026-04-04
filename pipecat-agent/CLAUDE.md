# CLAUDE.md — pipecat-agent

> Last updated: Session 45b (deployed to Pipecat Cloud with DailyTransport)

## Overview

Voice agent for Human Virtual's Avatar Live URL. Deployed to **Pipecat Cloud**. Pipeline: VAD → STT (Deepgram) → LLM (OpenAI) → TTS (Cartesia) → Daily WebRTC.

## Commands

```bash
uv sync              # Install
uv run bot.py        # Run locally → http://localhost:7860/client
```

## Deploy to Pipecat Cloud

```bash
docker build --platform=linux/arm64 -t human-virtual-agent:0.2 .
docker tag human-virtual-agent:0.2 YOUR_USER/human-virtual-agent:0.2
docker push YOUR_USER/human-virtual-agent:0.2
pcc secrets set human-virtual-agent-secrets --file .env
pcc deploy
pcc agent status human-virtual-agent
```

## Structure

```
pipecat-agent/
├── bot.py              # Pipeline + TranscriptForwarder + SpeakingStateNotifier
├── config.py           # Env vars
├── persona.py          # Prompt builder (uses backend persona-prompt endpoint)
├── scene_context.py    # Scene snapshot → agent-readable text
├── api_client.py       # HTTP client (public + auth endpoints)
├── pcc-deploy.toml     # Pipecat Cloud config
└── Dockerfile          # ARM64 image
```

## Key Decisions

- **Prompt:** Calls `GET /live-rooms/{room_id}/persona-prompt` (no auth). Falls back to local build from avatar+scene. Last resort: DEFAULT_PROMPT
- **Data channel messages (DailyTransport):** Uses `transport.send_app_message(dict)` not `send_message(string)`
  - `{ type: "transcript", speaker, text }` — real-time transcription
  - `{ type: "speaking_state", isSpeaking }` — avatar animation
  - `{ type: "canvas_action", action }` — overlays (Session 47)
- **Entry:** `bot(runner_args)` → extracts `room_id` from `runner_args.body` → `run_bot()`
- **Pipeline order:** input → STT → TranscriptForwarder → UserAggregator → LLM → SpeakingNotifier → TTS → output → AssistantAggregator
- **Secrets:** Managed via `pcc secrets set` — NOT baked into Docker image
- **All API calls:** try/except wrapped — agent never crashes on API failure
- **NEVER** imports from backend's `app/` package — completely separate service

## Environment Variables (via Pipecat Cloud secret set)

```bash
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
CARTESIA_API_KEY=...
HV_API_URL=https://api.hv.ai/api/v1    # Backend URL the agent calls
CARTESIA_VOICE_ID=71a7ad14-091c-4e8e-a314-022ece01c121
LLM_MODEL=gpt-4.1
```
