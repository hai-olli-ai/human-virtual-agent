# CLAUDE.md — pipecat-agent

> Last updated: Session 45 (full voice pipeline with persona prompt + transcript forwarding)

## Overview

Voice agent for Human Virtual's Avatar Live URL feature. Powers real-time conversations between visitors and AI avatars. Deployed to Pipecat Cloud, runs locally via SmallWebRTCTransport.

## Tech Stack

- **Framework:** Pipecat (by Daily) — voice AI pipeline framework
- **STT:** Deepgram
- **LLM:** OpenAI (gpt-4.1)
- **TTS:** Cartesia
- **Transport:** SmallWebRTCTransport (local), DailyTransport (production)
- **VAD:** Silero (built into Pipecat)
- **HTTP Client:** httpx (calls api.hv.ai)
- **Package Manager:** `uv`
- **Deployment:** Pipecat Cloud (Docker image)

## Structure

```
pipecat-agent/
├── bot.py              # Main pipeline: VAD → STT → LLM → TTS → transport
├── config.py           # Environment variable configuration
├── persona.py          # System prompt builder (uses Session 43 endpoints)
├── scene_context.py    # Scene snapshot → agent-readable text
├── api_client.py       # HTTP client for api.hv.ai (auth + public endpoints)
├── pyproject.toml      # Dependencies
├── pcc-deploy.toml     # Pipecat Cloud deployment config
├── Dockerfile          # ARM64 image for Pipecat Cloud
├── env.example
└── .env                # Local dev API keys (git-ignored)
```

## Commands

```bash
uv sync                 # Install dependencies
uv run bot.py           # Run locally → http://localhost:7860/client
```

## Pipeline Architecture

```
Visitor → Mic → WebRTC → Pipecat Pipeline:
  transport.input()     # Audio in
  → STT (Deepgram)      # Speech → text
  → TranscriptForwarder # Forward user text to frontend via Daily data channel
  → UserAggregator      # Add to conversation history
  → LLM (OpenAI)        # Generate response
  → SpeakingNotifier    # Notify frontend avatar is speaking
  → TTS (Cartesia)      # Response → speech
  → transport.output()  # Audio out → WebRTC → visitor speaker
  → AssistantAggregator # Add bot response to history
```

## Key Decisions

### Prompt Building (Session 45)
- **Primary:** Calls `GET /live-rooms/{room_id}/persona-prompt` (Session 43, no auth)
- **Fallback:** Builds locally from `get_avatar()` + `get_scene()` (auth required)
- **Last resort:** Uses `DEFAULT_PROMPT` (friendly assistant)
- Prompt sections: Identity → Knowledge → Scene Context → Instruction → Guidelines → Canvas Tools

### Data Channel Messages (Session 45)
- `{ type: "transcript", speaker: "user"|"avatar", text: "..." }` — real-time transcription
- `{ type: "speaking_state", isSpeaking: true|false }` — avatar animation
- `{ type: "canvas_action", action: {...} }` — overlays (Session 47)

### API Client
- **Public endpoints (no auth):** get_persona_prompt, get_scene_snapshot, navigate_scene
- **Auth endpoints:** get_avatar, get_scene (uses HV_API_TOKEN)
- All calls wrapped in try/except — agent never crashes on API failure

### Entry Points
- `bot(runner_args)` — called by Pipecat runner
- `runner_args.body` contains `{ room_id, avatar_id, scene_id, flow_id, hv_api_url }`
- `if __name__ == "__main__": main()` — local dev via Pipecat's built-in runner

### Custom Frame Processors
- `TranscriptForwarder` — captures TranscriptionFrame (user) + TextFrame (bot), sends via Daily data channel
- `SpeakingStateNotifier` — detects when bot starts speaking, sends state to frontend

## Environment Variables

```bash
DEEPGRAM_API_KEY=...          # Speech-to-text
OPENAI_API_KEY=...            # LLM
CARTESIA_API_KEY=...          # Text-to-speech
HV_API_URL=http://localhost:3001/api/v1
HV_API_TOKEN=...              # JWT for auth endpoints
DEFAULT_AVATAR_ID=...         # Local dev fallback
DEFAULT_SCENE_ID=...          # Local dev fallback
DEFAULT_ROOM_ID=...           # Local dev fallback
CARTESIA_VOICE_ID=71a7ad14-091c-4e8e-a314-022ece01c121
LLM_MODEL=gpt-4.1
```

## Important Rules

- NEVER import from `app/` — the agent is a completely separate service
- All API calls must be try/except — never crash the agent
- Pipeline order matters: TranscriptForwarder AFTER stt, SpeakingNotifier AFTER llm
- `bot()` receives body from Pipecat Cloud, `run_bot()` does the actual work
- Greeting fires immediately on `on_client_connected` via developer message
