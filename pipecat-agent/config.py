"""Configuration for the Human Virtual Pipecat agent."""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# AI Service keys
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")

# Human Virtual API
HV_API_URL = os.getenv("HV_API_URL", "http://localhost:3001/api/v1")
HV_API_TOKEN = os.getenv("HV_API_TOKEN", "")

# Daily (managed by Pipecat Cloud, but useful for local Daily testing)
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")

# Default IDs for testing
DEFAULT_AVATAR_ID = os.getenv("DEFAULT_AVATAR_ID", "")
DEFAULT_SCENE_ID = os.getenv("DEFAULT_SCENE_ID", "")
DEFAULT_ROOM_ID = os.getenv("DEFAULT_ROOM_ID", "")

# TTS voice configuration
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121")
# Default: "British Reading Lady" — will be customizable per avatar later

# ──────────────────────────────────────────────────────────────────────
# Narration audio cache (Block 1 — must equal the backend's renderer)
# ──────────────────────────────────────────────────────────────────────
# Cartesia returns 16-bit signed little-endian PCM regardless. These
# three values gate cache compatibility: a pre-rendered segment is only
# safe to replay if it was synthesized with the same model at the same
# sample_rate and channel count the live service is configured for.
# A mismatch would either (a) miss cleanly (safe — falls back to live)
# or (b) play mangled audio if the cache claims a match but the PCM was
# rendered against different settings. Keep these in lockstep with the
# backend's narration renderer.
NARRATION_TTS_MODEL_ID = os.getenv("NARRATION_TTS_MODEL_ID", "sonic-3")
NARRATION_AUDIO_SAMPLE_RATE = int(os.getenv("NARRATION_AUDIO_SAMPLE_RATE", "24000"))
NARRATION_AUDIO_NUM_CHANNELS = int(os.getenv("NARRATION_AUDIO_NUM_CHANNELS", "1"))


# ──────────────────────────────────────────────────────────────────────
# Vision-frame refresh policy (S66 Block 5a)
# ──────────────────────────────────────────────────────────────────────
# Pre-5a, every scene change re-fetched the Pillow-rendered scene image
# (a backend round-trip in the hot path of every navigation). Lazy mode
# (default) skips the per-scene fetch and defers it to the first
# canvas_analyze of the scene — the session-start bootstrap fetch is
# unchanged. Set to "eager" to restore the pre-5a behavior if a quality
# regression surfaces.
VISION_REFRESH_MODE = os.getenv("VISION_REFRESH_MODE", "lazy").strip().lower()

_VALID_VISION_REFRESH_MODES = {"lazy", "eager"}
if VISION_REFRESH_MODE not in _VALID_VISION_REFRESH_MODES:
    raise ValueError(
        f"VISION_REFRESH_MODE must be one of {sorted(_VALID_VISION_REFRESH_MODES)}, "
        f"got '{VISION_REFRESH_MODE}'"
    )

# LLM model — must support vision for scene understanding (Session 46)
# gpt-4.1 and gpt-4o both support vision
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.4")
#LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1")


# ──────────────────────────────────────────────────────────────────────
# Canvas-protocol LLM provider selection (S64c)
# ──────────────────────────────────────────────────────────────────────
#
# Selects which LLM service handles the Canvas Protocol tool surface.
# Vision (S46) stays hardcoded on OpenAI GPT-4.1 regardless — they're
# separate services. See CLAUDE.md "Vision (S46) — separate, hardcoded".
#
# Default is "openai" because that's the only LLM SDK extra currently
# installed (pipecat-ai[openai]). To use anthropic or gemini, install
# the corresponding pipecat extra (pipecat-ai[anthropic] /
# pipecat-ai[google]) and set this env var. CLAUDE.md targets anthropic
# as the eventual default; that flip happens once the extras land in
# pyproject.toml + uv.lock.
LLM_CANVAS_PROVIDER = os.getenv("LLM_CANVAS_PROVIDER", "openai").strip().lower()

_VALID_CANVAS_PROVIDERS = {"anthropic", "openai", "gemini"}
if LLM_CANVAS_PROVIDER not in _VALID_CANVAS_PROVIDERS:
    raise ValueError(
        f"LLM_CANVAS_PROVIDER must be one of {sorted(_VALID_CANVAS_PROVIDERS)}, "
        f"got '{LLM_CANVAS_PROVIDER}'"
    )

# Per-provider model overrides (defaults match CLAUDE.md's documented choices).
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


# ──────────────────────────────────────────────────────────────────────
# Deepgram language mapping (Session 61 — voice agent multi-language)
# ──────────────────────────────────────────────────────────────────────

# Maps live-room language codes (the backend's LiveRoomLanguage Literal)
# to the Deepgram `language` parameter. The 9 codes here mirror the
# nine-language enum the backend enforces via CHECK constraint, and
# Deepgram's nova-2 / nova-3 models support all of them directly.
DEEPGRAM_LANGUAGE_MAP: dict[str, str] = {
    "en": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "pt": "pt",
    "ja": "ja",
    "ko": "ko",
    "vi": "vi",
    "zh": "zh",
}

# Forward-compat: if the backend ever ships a code we haven't mapped
# (e.g. a 10th language added before this file is updated), fall back
# to Deepgram's multilingual auto-detect rather than crashing.
DEEPGRAM_FALLBACK_LANGUAGE: str = "multi"


def resolve_deepgram_language(snapshot_language: str | None) -> str:
    """Map a scene-snapshot language code to a Deepgram language parameter.

    - None / empty → "en" (matches the backend's default).
    - Mapped code  → its Deepgram value.
    - Unknown code → "multi" (auto-detect, slower but always works).
    """
    if not snapshot_language:
        return "en"
    return DEEPGRAM_LANGUAGE_MAP.get(snapshot_language, DEEPGRAM_FALLBACK_LANGUAGE)
