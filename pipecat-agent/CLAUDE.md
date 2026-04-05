# CLAUDE.md — pipecat-agent

> Last updated: Session 46 (vision context — agent can see the canvas)

## Overview

Voice agent for Human Virtual's Avatar Live URL. Deployed to **Pipecat Cloud**. Pipeline: VAD → STT (Deepgram) → LLM (OpenAI) → TTS (Cartesia) → Daily WebRTC. **Now with multimodal vision** — agent sees the rendered canvas via GPT-4o/4.1 image input.

## Commands

```bash
uv sync              # Install
uv run bot.py        # Run locally → http://localhost:7860/client
```

## Deploy to Pipecat Cloud

```bash
docker build --platform=linux/arm64 -t human-virtual-agent:0.3 .
docker tag human-virtual-agent:0.3 YOUR_USER/human-virtual-agent:0.3
docker push YOUR_USER/human-virtual-agent:0.3
pcc secrets set human-virtual-agent-secrets --file .env
pcc deploy
```

## Structure

```
pipecat-agent/
├── bot.py              # Pipeline + vision injection + TranscriptForwarder + SpeakingStateNotifier
├── config.py           # Env vars
├── persona.py          # Prompt builder (persona-prompt endpoint + local fallback)
├── scene_context.py    # Scene snapshot → text descriptions + vision message builder
├── api_client.py       # HTTP client (public + auth endpoints + scene image fetch)
├── pcc-deploy.toml     # Pipecat Cloud config
└── Dockerfile          # ARM64 image
```

## Key Decisions

### Vision (Session 46)
- On startup, agent calls `GET /live-rooms/{room_id}/scene-snapshot/image?format=base64`
- Base64 PNG injected as first message in LLMContext via `build_vision_message()`
- Uses OpenAI `image_url` content format with `detail: "high"` for text recognition
- Model: `gpt-4.1` (supports vision natively)
- **Graceful degradation:** if image fetch fails, agent works text-only (no vision)
- Vision is one-shot at startup; scene change re-fetch wired in Session 47

### Prompt Building
- Calls `GET /live-rooms/{room_id}/persona-prompt` (no auth). Falls back to local build
- Sections: Identity → Knowledge → Scene Context → Instruction → Guidelines → Canvas Tools

### Data Channel Messages (DailyTransport)
- `transport.send_app_message(dict)` for production
- `{ type: "transcript", speaker, text }` — real-time transcription
- `{ type: "speaking_state", isSpeaking }` — avatar animation
- `{ type: "canvas_action", action }` — overlays (Session 47)

### Pipeline
- input → STT → TranscriptForwarder → UserAggregator → LLM → SpeakingNotifier → TTS → output → AssistantAggregator
- `LLMContext(messages=[vision_message])` — canvas image is first context message

### Rules
- NEVER imports from backend's `app/` package
- All API calls try/except wrapped — never crashes
- `bot()` extracts `room_id` from `runner_args.body`
- Greeting hints at visual awareness if image was loaded

## Environment Variables (via Pipecat Cloud secret set)

```bash
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
CARTESIA_API_KEY=...
HV_API_URL=https://api.hv.ai/api/v1
CARTESIA_VOICE_ID=71a7ad14-091c-4e8e-a314-022ece01c121
LLM_MODEL=gpt-4.1
```
