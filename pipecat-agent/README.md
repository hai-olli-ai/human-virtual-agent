# Human Virtual — Pipecat Voice Agent

Voice agent for Human Virtual's Avatar Live URL feature.
Powers real-time conversations between visitors and AI avatars.

## Local Development

```bash
# Install dependencies
uv sync

# Configure API keys
cp env.example .env
# Edit .env with your keys

# Run locally (opens http://localhost:7860/client)
uv run bot.py
```

## Architecture

```
Visitor → Mic → WebRTC → Pipecat Pipeline:
  VAD (Silero) → STT (Deepgram) → LLM (OpenAI) → TTS (Cartesia)
  → WebRTC → Speaker → Visitor
```

## Production Deployment

See Session 38 for Pipecat Cloud deployment instructions.
