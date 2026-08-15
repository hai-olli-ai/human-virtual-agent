"""Human Virtual Pipecat voice agent.

Dual-pipeline architecture based on avatar display mode:

Classic pipeline (normal / invisible / 3dgs):
    Mic → STT (Deepgram) → LLM (OpenAI) → TTS (Cartesia) → Speaker
    Simple event handlers, direct greeting, standard duplex conversation.

Relay avatar pipeline (talking):
    Mic → AudioFilter → STT → LLM → text relay → SoulX avatar bot
    Complex participant management, relay protocol, no local TTS.
    SoulX server handles TTS + avatar video rendering in the same Daily room.

Pipeline selection is automatic based on the avatar's display_mode field
in the scene snapshot. Falls back to CLOUD_OUTPUT_MODE env var.

Local dev:  python bot.py  → opens http://localhost:7860/client
Production: Deployed to Pipecat Cloud with DailyTransport
"""

import asyncio
import json
import os
import time
import uuid
from urllib.parse import urlparse

import httpx

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.base_llm_adapter import LLMContext
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    InterruptionTaskFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    OutputTransportMessageFrame,
    StartFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
    TranscriptionFrame,
    TextFrame,
    UserAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import DailyRunnerArguments, RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams

load_dotenv(override=True)

import pipecat

logger.info(f"Pipecat SDK version: {pipecat.__version__}")

from config import (
    CARTESIA_API_KEY,
    CARTESIA_VOICE_ID,
    DEEPGRAM_API_KEY,
    DEEPGRAM_LANGUAGE_MAP,
    OPENAI_API_KEY,
    LLM_MODEL,
    DEFAULT_AVATAR_ID,
    DEFAULT_ROOM_ID,
    DEFAULT_SCENE_ID,
    LLM_CANVAS_PROVIDER,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    GOOGLE_AI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    MAIN_LLM_SUPPORTS_VISION,
    NARRATION_TTS_MODEL_ID,
    NARRATION_AUDIO_SAMPLE_RATE,
    NARRATION_AUDIO_NUM_CHANNELS,
    VISION_REFRESH_MODE,
    VISION_CAPTURE_TIMEOUT_MS,
    VISION_MAX_DIM,
    AGENT_ANNOTATE_TIMEOUT_MS,
    resolve_deepgram_language,
)
from persona import build_system_prompt

# S64c — Canvas Protocol generic tool surface (registered alongside V2.13 tools
# until Block 7 cutover). See CLAUDE.md "Coming in S64c".
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from tools.canvas_protocol_tools import (
    CanvasToolContext,
    PendingCommandRegistry,
    make_handlers as make_canvas_protocol_handlers,
    make_tool_schemas as make_canvas_protocol_schemas,
)
from context.canvas_manifest import CanvasManifestRegistry
from context.transcript_aggregator import WordBoundaryAggregator
from context.prompt_builder import (
    render_agent_playbook_section,
    render_canvas_page_section,
    render_voice_output_style_section,
)

# S64e — generate_quiz_from_knowledge tool + session-scoped slug/scene state.
import api_client
from tools.quiz_generation import (
    GENERATE_QUIZ_SCHEMA,
    SessionContext,
    make_handle_generate_quiz,
    request_quiz_ready,
    run_quiz_generation,
)

# S65 G3+G4 — per-scene narration helper with per-segment voice switching,
# post-narration invitation/cue branch, and the script_complete payload.
# run_scene_narration is the orchestrator wired into both
# on_client_connected (session start) AND refresh_agent_for_current_scene
# (scene change — S65 Bug #2 fix) so both moments narrate identically.
from narration import (
    NARRATION_INTERRUPTED,
    PLAYOUT_DRAIN_FALLBACK_S,
    NarrationCompletionGate,
    NarrationInterrupted,
    SceneNarrator,
    build_script_complete_payload,
    run_scene_narration,
)

# S79 — the animated-narration cue controller (shell plays cached MP4s;
# the agent cues, times out into TTS, pauses/resumes across barge-ins).
from services.narration_cue import NarrationCueController

# Block 12 — cache-first Cartesia TTS service (replay of pre-rendered
# narration PCM via single-shot prime). The narrator's prefetch+prime
# closures (~run_bot_classic) populate and consume the cache; on a miss
# the service falls through to live Cartesia synthesis.
from services.cached_first_tts import CachedFirstTTSService, CachedSegment

# S66 Block 5a — lazy vision-frame tracker. Owns "which scene_id's vision
# is currently in context"; the canvas_analyze handler calls ensure() to
# fetch+inject on demand instead of every scene change.
from vision_refresh import VisionFrameTracker
from services.vision_client import VisionClient
from services.vision_query import _fetch_live_bytes, run_vision_query
from tools.canvas_annotate import (
    AGENT_ANNOTATE_SCHEMA,
    make_handle_canvas_annotate,
)

# S66 Block 5b — per-session memoisation of the FLOW-scope knowledge
# block. Constructed once per pipeline; threaded through build_system_prompt
# on session start and every scene-change refresh.
from flow_knowledge_cache import FlowKnowledgeCache

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

GREETING_TRIGGER_PROMPT = (
    "A visitor just joined. Greet them warmly and briefly introduce yourself "
    "and what you can do. Do NOT use any canvas action tools for this greeting "
    "- just speak."
)

# Relay protocol
RELAY_PROTOCOL = "avatar-relay.v1"
RELAY_READY = "avatar_relay.ready"
RELAY_TURN_START = "avatar_relay.turn_start"
RELAY_TEXT = "avatar_relay.text"
RELAY_SENTENCE = "avatar_relay.sentence"
RELAY_TURN_END = "avatar_relay.turn_end"
RELAY_INTERRUPT = "avatar_relay.interrupt"

# Output mode fallback (env var used when scene snapshot unavailable)
VALID_OUTPUT_MODES = {"cartesia", "relay_avatar"}
CLOUD_OUTPUT_MODE = (
    os.getenv("CLOUD_OUTPUT_MODE", "cartesia").strip().lower() or "cartesia"
)
if CLOUD_OUTPUT_MODE not in VALID_OUTPUT_MODES:
    logger.warning(
        "Unknown CLOUD_OUTPUT_MODE={}, falling back to cartesia",
        CLOUD_OUTPUT_MODE,
    )
    CLOUD_OUTPUT_MODE = "cartesia"

# ── SoulX renderer, soulx-audio.v1 (avatar-production branch) ────────────────
# When SOULX_WS_URL is set, the relay pipeline runs TTS locally and streams PCM to the
# Modal renderer over a WebSocket instead of forwarding TEXT over Daily app-messages
# (avatar-relay.v1). Unset => the legacy text-relay path is untouched.
SOULX_WS_URL = os.getenv("SOULX_WS_URL", "").strip()
SOULX_AUTH_TOKEN = os.getenv("SOULX_AUTH_TOKEN", "").strip()
# The renderer must join the room within this long or the session degrades to voice-only.
# The pre-existing code waits on an UN-TIMED asyncio.Event here, which is the silent-bot
# deadlock: if the renderer never arrives the visitor gets a bot that never speaks.
#
# 300s because a COLD Modal container was measured at 134s once and 204s after a full
# scaledown (schedule + image pull + 15GB volume mount + Lite load + warmup) — the spread
# matters more than the median, so this is set above the worst observed case. At the
# original 60s default every cold session silently degraded to voice-only. This is a
# deliberate correctness-over-UX trade for the current phase: a visitor may wait a long
# time on a cold start. The real fix is warm containers (min_containers /
# buffer_containers / memory snapshots) in Week 3, after which this drops back down.
AVATAR_READY_TIMEOUT_S = float(os.getenv("AVATAR_READY_TIMEOUT_S", "300"))

# Bot names
CLOUD_BOT_NAME = (
    os.getenv("CLOUD_BOT_NAME", "Human Virtual Cloud").strip() or "Human Virtual Cloud"
)
AVATAR_BOT_NAME = (
    os.getenv("SOULX_AVATAR_BOT_NAME", "Digital Twin Avatar").strip()
    or "Digital Twin Avatar"
)


# ──────────────────────────────────────────────────────────────────────
# Participant helpers (relay pipeline)
# ──────────────────────────────────────────────────────────────────────


def _participant_id(participant: object) -> str:
    if not isinstance(participant, dict):
        return ""
    value = (
        participant.get("id")
        or participant.get("participant_id")
        or participant.get("participantId")
    )
    return str(value).strip() if value else ""


def _participant_info(participant: object) -> dict[str, object]:
    if not isinstance(participant, dict):
        return {}
    info = participant.get("info")
    return info if isinstance(info, dict) else {}


def _participant_name(participant: object) -> str:
    if not isinstance(participant, dict):
        return ""
    info = _participant_info(participant)
    for value in (
        participant.get("user_name"),
        participant.get("userName"),
        participant.get("name"),
        info.get("user_name"),
        info.get("userName"),
        info.get("name"),
    ):
        if value:
            return str(value).strip()
    return ""


def _participant_is_local(participant: object) -> bool:
    if not isinstance(participant, dict):
        return False
    pid = _participant_id(participant)
    if pid == "local":
        return True
    info = _participant_info(participant)
    return bool(participant.get("local") or info.get("isLocal"))


def _canonical_participant_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


AVATAR_BOT_NAME_ALIASES = {
    _canonical_participant_name(AVATAR_BOT_NAME),
    "digitaltwinavatar",
    "soulxavatar",
}
CLOUD_BOT_NAME_ALIASES = {
    _canonical_participant_name(CLOUD_BOT_NAME),
    "humanvirtualcloud",
    "pipecatbot",
}


def _participant_role(participant: object) -> str:
    if _participant_is_local(participant):
        return "cloud_bot"
    name = _canonical_participant_name(_participant_name(participant))
    if name in AVATAR_BOT_NAME_ALIASES:
        return "avatar_bot"
    if name in CLOUD_BOT_NAME_ALIASES:
        return "cloud_bot"
    return "human"


def _is_relay_ready_message(message: object) -> bool:
    return (
        isinstance(message, dict)
        and message.get("protocol") == RELAY_PROTOCOL
        and message.get("type") == RELAY_READY
    )


def _build_transport_message(
    message: dict[str, object], participant_id: str | None = None
):
    if participant_id:
        try:
            from pipecat.transports.daily.transport import (
                DailyOutputTransportMessageFrame,
            )
        except Exception:
            logger.debug(
                "Daily transport targeting unavailable, broadcasting relay message type={}",
                message.get("type"),
            )
        else:
            return DailyOutputTransportMessageFrame(
                message=message,
                participant_id=participant_id,
            )
    return OutputTransportMessageFrame(message=message)


# ──────────────────────────────────────────────────────────────────────
# Shared frame processors
# ──────────────────────────────────────────────────────────────────────


class TranscriptForwarder(FrameProcessor):
    """Forward user STT and bot text updates over the transport data channel.

    Avatar (LLM) text is aggregated to WORD boundaries before forwarding. The
    LLM streams sub-token deltas — for Vietnamese a single syllable arrives in
    pieces ("phân" → ["ph","ân"]) — and the shell's caption renders transcript
    messages separated, so forwarding raw deltas rendered as "ph ân đo ạn".
    Buffering until a whitespace boundary and emitting only whole (trimmed)
    words keeps every word intact (see context/transcript_aggregator). The
    frame stream is pushed through UNCHANGED, so TTS/audio is unaffected — only
    the transcript side-channel is reshaped. User STT (TranscriptionFrame) is
    already whole-utterance and is forwarded as-is.
    """

    def __init__(self, transport: BaseTransport):
        super().__init__()
        self._transport = transport
        self._avatar_agg = WordBoundaryAggregator()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text:
            await self._send_transcript("user", frame.text)
        elif (
            isinstance(frame, TextFrame)
            and not isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame))
            and frame.text
        ):
            ready = self._avatar_agg.feed(frame.text)
            if ready:
                await self._send_transcript("avatar", ready)
        elif isinstance(
            frame, (LLMFullResponseEndFrame, InterruptionFrame, EndFrame, CancelFrame)
        ):
            # Turn/stream boundary — release the final partial word so the
            # caption isn't left missing the tail of the utterance.
            ready = self._avatar_agg.flush()
            if ready:
                await self._send_transcript("avatar", ready)

        await self.push_frame(frame, direction)

    async def _send_transcript(self, speaker: str, text: str):
        try:
            payload = {
                "type": "transcript",
                "speaker": speaker,
                "text": text,
            }
            await self._transport.send_message(
                OutputTransportMessageFrame(message=payload)
            )
        except Exception as exc:
            logger.warning("Could not forward transcript: {}", exc)


class SpeakingStateNotifier(FrameProcessor):
    """Notify listeners when the bot starts and stops speaking."""

    def __init__(self, transport: BaseTransport):
        super().__init__()
        self._transport = transport
        self._is_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if (
            isinstance(frame, TextFrame)
            and not isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame))
            and frame.text
            and not self._is_speaking
        ):
            self._is_speaking = True
            await self._send_state(True)
        elif (
            isinstance(frame, (LLMFullResponseEndFrame, InterruptionFrame))
            and self._is_speaking
        ):
            self._is_speaking = False
            await self._send_state(False)

        await self.push_frame(frame, direction)

    async def _send_state(self, is_speaking: bool):
        try:
            payload = {
                "type": "speaking_state",
                "isSpeaking": is_speaking,
            }
            await self._transport.send_message(
                OutputTransportMessageFrame(message=payload)
            )
        except Exception as exc:
            logger.warning("Could not send speaking state: {}", exc)


class ThinkingNotifier(FrameProcessor):
    """Notify frontend when the LLM starts and finishes processing."""

    def __init__(self, transport: BaseTransport):
        super().__init__()
        self._transport = transport
        self._is_thinking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame) and not self._is_thinking:
            self._is_thinking = True
            await self._send_state(True)
        elif (
            isinstance(frame, (LLMFullResponseEndFrame, InterruptionFrame))
            and self._is_thinking
        ):
            self._is_thinking = False
            await self._send_state(False)

        await self.push_frame(frame, direction)

    async def _send_state(self, thinking: bool):
        try:
            payload = {
                "type": "llm_thinking",
                "thinking": thinking,
            }
            await self._transport.send_message(
                OutputTransportMessageFrame(message=payload)
            )
        except Exception as exc:
            logger.warning("Could not send thinking state: {}", exc)


# ──────────────────────────────────────────────────────────────────────
# Relay-only frame processors
# ──────────────────────────────────────────────────────────────────────


class HumanOnlyAudioInputFilter(FrameProcessor):
    """Drops SoulX/local bot audio before it reaches STT in relay mode."""

    def __init__(self, avatar_participant_id_getter, local_participant_id_getter):
        super().__init__()
        self._avatar_participant_id_getter = avatar_participant_id_getter
        self._local_participant_id_getter = local_participant_id_getter
        self._logged_drops: set[tuple[str, str]] = set()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserAudioRawFrame):
            user_id = str(frame.user_id or "").strip()
            avatar_participant_id = str(
                self._avatar_participant_id_getter() or ""
            ).strip()
            local_participant_id = str(
                self._local_participant_id_getter() or ""
            ).strip()

            if user_id and avatar_participant_id and user_id == avatar_participant_id:
                self._log_drop(user_id, "avatar_participant")
                return

            if user_id and user_id in {"local", local_participant_id}:
                self._log_drop(user_id, "local_participant")
                return

        await self.push_frame(frame, direction)

    def _log_drop(self, user_id: str, reason: str):
        key = (user_id, reason)
        if key in self._logged_drops:
            return
        self._logged_drops.add(key)
        logger.info(
            "Dropping audio before STT from participant_id={} reason={}",
            user_id,
            reason,
        )


class AvatarReadyGateProcessor(FrameProcessor):
    """Blocks relay-mode LLM traffic until the avatar bot reports ready.

    The wait is BOUNDED. It used to be an un-timed `Event.wait()`, which is the
    silent-bot deadlock: if the avatar never arrives — the box is down, the invite
    failed (the backend logs that as non-fatal and continues) — this blocks forever and
    the visitor sits with a bot that never speaks and no error anywhere. On timeout we
    open the gate and let the session run voice-only instead.
    """

    def __init__(self, ready_event: asyncio.Event, timeout_s: float | None = None):
        super().__init__()
        self._ready_event = ready_event
        self._timeout_s = timeout_s
        self._waiting_logged = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (StartFrame, EndFrame, CancelFrame)):
            await self.push_frame(frame, direction)
            return

        if not self._ready_event.is_set():
            if not self._waiting_logged:
                logger.info(
                    "Waiting for avatar relay bot readiness before processing {}",
                    frame.__class__.__name__,
                )
                self._waiting_logged = True
            try:
                if self._timeout_s:
                    await asyncio.wait_for(
                        self._ready_event.wait(), timeout=self._timeout_s
                    )
                else:
                    await self._ready_event.wait()
                logger.info("Avatar relay bot ready; resuming queued pipeline traffic")
            except asyncio.TimeoutError:
                logger.error(
                    "Avatar never became ready after {}s — opening the gate and "
                    "continuing WITHOUT the avatar. Never leave the visitor with a "
                    "silent bot.",
                    self._timeout_s,
                )
                self._ready_event.set()
            self._waiting_logged = False

        await self.push_frame(frame, direction)


class AvatarRelayProcessor(FrameProcessor):
    """Relays the same text/control frames that would normally feed TTS."""

    def __init__(self, transport: BaseTransport, participant_id_getter):
        super().__init__()
        self._transport = transport
        self._participant_id_getter = participant_id_getter
        self._turn_id: str | None = None
        self._seq = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            await self._start_turn()
        elif (
            isinstance(frame, TextFrame)
            and not isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame))
            and frame.text
            and not getattr(frame, "skip_tts", False)
        ):
            await self._send_text(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._end_turn()
        elif isinstance(frame, InterruptionFrame):
            await self._interrupt_turn()

        await self.push_frame(frame, direction)

    async def _start_turn(self):
        self._turn_id = str(uuid.uuid4())
        self._seq = 0
        logger.info("Avatar relay start turn_id={}", self._turn_id)
        await self._send_payload(RELAY_TURN_START, turn_id=self._turn_id)

    async def _ensure_turn(self):
        if self._turn_id is None:
            await self._start_turn()

    async def _send_text(self, text: str):
        if text == "":
            return
        await self._ensure_turn()
        assert self._turn_id is not None
        logger.info(
            "Avatar relay text turn_id={} seq={} text={!r}",
            self._turn_id,
            self._seq,
            text[:160],
        )
        await self._send_payload(
            RELAY_TEXT,
            turn_id=self._turn_id,
            seq=self._seq,
            text=text,
        )
        self._seq += 1

    async def _end_turn(self):
        if not self._turn_id:
            return
        turn_id = self._turn_id
        self._turn_id = None
        self._seq = 0
        logger.info("Avatar relay end turn_id={}", turn_id)
        await self._send_payload(RELAY_TURN_END, turn_id=turn_id)

    async def _interrupt_turn(self):
        if not self._turn_id:
            return
        turn_id = self._turn_id
        self._turn_id = None
        self._seq = 0
        logger.info("Avatar relay interrupt turn_id={}", turn_id)
        await self._send_payload(RELAY_INTERRUPT, turn_id=turn_id)

    async def _send_payload(self, message_type: str, **payload_fields):
        participant_id = self._participant_id_getter()
        if not participant_id:
            logger.warning(
                "Dropping avatar relay message type={} because no avatar participant is ready",
                message_type,
            )
            return

        payload = {
            "type": message_type,
            "protocol": RELAY_PROTOCOL,
            **payload_fields,
        }
        try:
            await self._transport.send_message(
                _build_transport_message(payload, participant_id=participant_id)
            )
            logger.info(
                "Sent avatar relay message type={} target_participant_id={} payload={}",
                message_type,
                participant_id,
                payload,
            )
        except Exception:
            logger.exception(
                "Failed to send avatar relay message type={}", message_type
            )


# ──────────────────────────────────────────────────────────────────────
# Output mode resolution
# ──────────────────────────────────────────────────────────────────────


async def _resolve_output_mode(room_id: str, api_url: str | None = None) -> str:
    """Determine output mode from the avatar's display mode in the scene.

    Mapping:
      "talking"   -> "relay_avatar" (SoulX avatar with lip-sync video)
      everything else (normal, invisible, 3dgs) -> "cartesia" (classic voice)

    Falls back to CLOUD_OUTPUT_MODE env var when the scene snapshot
    cannot be fetched (no room_id, API error, etc.).
    """
    if room_id:
        from api_client import get_scene_snapshot

        snapshot = await get_scene_snapshot(room_id, api_url)
        if snapshot:
            # S65 (Option B) — avatar_display_mode nested under current_scene.
            display_mode = (snapshot.get("current_scene") or {}).get(
                "avatar_display_mode", "normal"
            )
            if display_mode == "talking":
                logger.info(
                    "Avatar display_mode={} -> output_mode=relay_avatar",
                    display_mode,
                )
                return "relay_avatar"
            logger.info(
                "Avatar display_mode={} -> output_mode=cartesia",
                display_mode,
            )
            return "cartesia"
    logger.info(
        "Could not resolve display mode from scene; falling back to CLOUD_OUTPUT_MODE={}",
        CLOUD_OUTPUT_MODE,
    )
    return CLOUD_OUTPUT_MODE


# ──────────────────────────────────────────────────────────────────────
# LLM provider selection (S64c)
# ──────────────────────────────────────────────────────────────────────
#
# Per-provider branching with lazy imports — only the SDK for the
# selected provider needs to be installed. CLAUDE.md S64c documents
# anthropic as the eventual default; today the default is openai
# because that's the only LLM extra in pyproject.toml. Override via
# LLM_CANVAS_PROVIDER env var after installing the corresponding extra.


def _cartesia_language(code: str | None) -> Language | str:
    """S77 B5 — map an ISO 639-1 room/narration code to what pipecat
    0.0.108's Cartesia service accepts.

    Prefer the :class:`Language` enum member (so the service's
    ``language_to_cartesia_language`` conversion applies on both the
    constructor path and ``TTSUpdateSettingsFrame`` deltas); unknown
    codes fall back to the raw lowercase base code, which Cartesia
    accepts as a bare ISO 639-1 string.
    """
    base = (code or "en").split("-")[0].lower() or "en"
    try:
        return Language(base)
    except ValueError:
        return base


def _assemble_full_prompt(base: str, manifest: dict | None) -> str:
    """Concatenate base persona prompt, CANVAS PAGE, AGENT PLAYBOOK, VOICE OUTPUT.

    Single helper for the three sites that rebuild the prompt (session
    start, canvas.register, canvas.sceneChanged). The CANVAS PAGE section
    is driven by the active Page's manifest; AGENT PLAYBOOK is a stable
    string that documents cross-tool sequences the agent should follow
    (currently the quiz flow — S64e). VOICE OUTPUT STYLE is appended last
    (recency) to keep the model's reply plain-spoken — no Markdown/ellipsis
    noise reaching the TTS or the live caption.
    """
    return "\n\n".join(
        [
            base,
            render_canvas_page_section(manifest),
            render_agent_playbook_section(),
            render_voice_output_style_section(),
        ]
    )


def _build_llm_and_eager_hook(
    *,
    provider: str,
    system_prompt: str,
    canvas_pending: PendingCommandRegistry,
    send_canvas_message,
):
    """Construct the per-provider LLM service + eager-dispatch hook.

    Returns (llm_service, eager_hook). The hook is instantiated but not
    yet wired into the streaming loop — see the NOTE in run_bot_classic.
    """
    if provider == "openai":
        from services.eager_dispatch.openai_adapter import OpenAIEagerHook

        llm = OpenAILLMService(
            api_key=OPENAI_API_KEY,
            settings=OpenAILLMService.Settings(
                model=LLM_MODEL,
                system_instruction=system_prompt,
            ),
        )
        return llm, OpenAIEagerHook(canvas_pending, send_canvas_message)

    if provider == "groq":
        # Groq's GroqLLMService subclasses OpenAILLMService (OpenAI-compatible
        # wire format), so OpenAIEagerHook parses its streamed tool calls
        # verbatim — no groq-specific adapter needed. Import the .llm submodule
        # directly: the package __init__ eagerly pulls groq/tts.py, which needs
        # the native `groq` SDK shipped via the pipecat-ai[groq] extra.
        from pipecat.services.groq.llm import GroqLLMService
        from services.eager_dispatch.openai_adapter import OpenAIEagerHook

        llm = GroqLLMService(
            api_key=GROQ_API_KEY,
            settings=GroqLLMService.Settings(
                model=GROQ_MODEL,
                system_instruction=system_prompt,
            ),
        )
        return llm, OpenAIEagerHook(canvas_pending, send_canvas_message)

    if provider == "anthropic":
        from pipecat.services.anthropic.llm import AnthropicLLMService
        from services.eager_dispatch.anthropic_adapter import AnthropicEagerHook

        llm = AnthropicLLMService(
            api_key=ANTHROPIC_API_KEY,
            model=ANTHROPIC_MODEL,
        )
        # Anthropic services in Pipecat typically take system prompt via
        # context messages or a service-specific setter; until S64c Block 7
        # restructures the prompt assembly, we fall back to setting it on
        # an attribute the service exposes (matches the pattern V2.13 uses
        # for the OpenAI service).
        try:
            llm._settings.system_instruction = system_prompt  # type: ignore[attr-defined]
        except Exception:
            logger.warning(
                "Anthropic LLM service did not accept system_instruction setter"
            )
        return llm, AnthropicEagerHook(canvas_pending, send_canvas_message)

    if provider == "gemini":
        from pipecat.services.google.llm import GoogleLLMService
        from services.eager_dispatch.gemini_adapter import GeminiEagerHook

        llm = GoogleLLMService(
            api_key=GOOGLE_AI_API_KEY,
            model=GEMINI_MODEL,
            system_instruction=system_prompt,
        )
        return llm, GeminiEagerHook(canvas_pending, send_canvas_message)

    raise ValueError(f"unknown LLM_CANVAS_PROVIDER={provider!r}")


# ======================================================================
#
#  CLASSIC PIPELINE  (avatar display: normal / invisible / 3dgs)
#
#  Mic -> STT (Deepgram) -> LLM (OpenAI) -> TTS (Cartesia) -> Speaker
#
#  Simple event handlers.  Any connecting client triggers greeting;
#  any disconnect cancels the pipeline.
#
# ======================================================================


async def run_bot_classic(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    room_id: str = "",
    avatar_id: str = "",
    scene_id: str = "",
    flow_id: str | None = None,
    api_url: str | None = None,
    slug: str = "",
):
    """Classic voice agent pipeline with Cartesia TTS."""
    logger.info("Starting classic voice agent (room={}, avatar={})", room_id, avatar_id)

    # S64c — initial element alias map. Populated by build_system_prompt
    # below and passed into CanvasToolContext when it's constructed
    # further down. The same dict object is reused across the session;
    # post_scene_change clears + repopulates it on every scene nav so
    # the tool handlers always see the current scene's aliases.
    element_alias_map: dict[str, str] = {}
    # S66 Block 5b — single per-session FlowKnowledgeCache. Reused on
    # every build_system_prompt call (session start + each scene-change
    # refresh) so the FLOW block is rendered once per knowledge content
    # hash. SCENE-scope is always rebuilt (varies per navigation).
    flow_knowledge_cache = FlowKnowledgeCache()

    # ── Session-start backend reads (P3 2026-07-13) ──
    # Snapshot + avatar-config + scene-image are independent — fetch them
    # CONCURRENTLY, then thread the snapshot into build_system_prompt
    # (which used to fetch its own copy, and a fourth fetch below pulled
    # the same snapshot AGAIN for scripts). 5 serial round trips → one
    # gather + the persona fetch.
    scene_snapshot = None
    avatar_config = None
    scene_image_b64 = None
    if room_id:
        from api_client import (
            get_avatar_config,
            get_scene_image_base64,
            get_scene_snapshot,
        )

        scene_snapshot, avatar_config, scene_image_b64 = await asyncio.gather(
            get_scene_snapshot(room_id, api_url),
            get_avatar_config(room_id, api_url),
            get_scene_image_base64(room_id, api_url),
        )
        if avatar_config:
            logger.info(
                f"Avatar config: name={avatar_config.get('name')}, voiceModelId={avatar_config.get('voiceModelId')}"
            )
        else:
            logger.info("No avatar config available — using default voice")
        if scene_image_b64:
            logger.info(
                "Fetched scene canvas image ({} chars base64)", len(scene_image_b64)
            )
        else:
            logger.info("No scene image available; vision disabled for this session")
        if scene_snapshot:
            scripts_len = len(
                ((scene_snapshot.get("current_scene") or {}).get("scripts")) or []
            )
            logger.info("Scene snapshot loaded (scripts={})", scripts_len)

    system_prompt = await build_system_prompt(
        room_id=room_id,
        avatar_id=avatar_id,
        scene_id=scene_id,
        api_url=api_url,
        aliases_out=element_alias_map,
        flow_cache=flow_knowledge_cache,
        snapshot=scene_snapshot,
    )
    logger.info(f"System prompt length: {len(system_prompt)} chars")

    # ── Canvas Protocol substrate (S64c) ──
    # Manifest + pending command registries are per-session; an instance pair
    # is created here and threaded through the eager hook, the new tool
    # handlers, and the on_app_message router below.
    output_transport = transport.output()
    canvas_manifest = CanvasManifestRegistry()
    canvas_pending = PendingCommandRegistry()

    async def send_canvas_message(payload: dict) -> None:
        """Send a Canvas Protocol Daily app-message to the frontend."""
        try:
            await output_transport.send_message(
                OutputTransportMessageFrame(message=payload)
            )
        except Exception as exc:
            logger.warning("Failed to send canvas message: {}", exc)

    # ── S67b — vision capture round-trip (sibling of canvas_pending) ──
    # Keyed by a fresh captureId, NOT a canvas commandId: a separate dict so
    # the two correlation spaces can never collide. The shell screenshots the
    # visitor's canvas, uploads the JPEG to the backend ingest, and replies
    # with a tiny canvas_capture_result carrying {status,w,h} only — the bytes
    # are fetched separately by captureId (api_client.get_vision_capture).
    _pending_captures: dict[str, asyncio.Future] = {}
    # ── Block 8 — agent-annotate ack registry (sibling of _pending_captures) ──
    # Keyed by a fresh annotateId so annotate acks and capture acks never collide;
    # drained on disconnect alongside _pending_captures.
    _pending_annotates: dict[str, asyncio.Future] = {}

    async def request_canvas_capture(hint: str) -> tuple[str, dict | None]:
        """Ask the shell to capture the visitor's canvas; await the ack.

        Returns ``(capture_id, result)`` — result is the dict
        {captureId, status, w?, h?, error?} on reply, or None on timeout. The
        caller needs the capture_id to fetch the bytes by id (run_vision_query →
        _fetch_live_bytes), so we surface it alongside the result (B12). Rides send_canvas_message
        (the generic outbound helper — non-canvas payloads ride it too, e.g.
        quiz_generation_state); the reply lands in the canvas_capture_result
        on_app_message branch. The captured BYTES are not in the dict.
        """
        capture_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        _pending_captures[capture_id] = fut
        logger.info("[VISION] requesting canvas capture {} hint={!r}", capture_id, hint)
        try:
            await send_canvas_message(
                {
                    "type": "request_canvas_capture",
                    "captureId": capture_id,
                    "hint": hint,
                    "maxDim": VISION_MAX_DIM,  # advisory; the shell owns the encode
                }
            )
            result = await asyncio.wait_for(
                fut, timeout=VISION_CAPTURE_TIMEOUT_MS / 1000
            )
            logger.info(
                "[VISION] capture {} resolved status={!r}",
                capture_id,
                (result or {}).get("status"),
            )
            return capture_id, result
        except asyncio.TimeoutError:
            logger.warning(
                "[VISION] capture {} timed out after {}ms",
                capture_id,
                VISION_CAPTURE_TIMEOUT_MS,
            )
            return capture_id, None
        finally:
            _pending_captures.pop(capture_id, None)

    canvas_ctx = CanvasToolContext(
        manifest_registry=canvas_manifest,
        pending=canvas_pending,
        send_app_message=send_canvas_message,
        element_alias_map=element_alias_map,
        command_timeout_s=6.0,
    )

    # S64e — session-scoped state for non-canvas tools that need slug +
    # current scene id (currently generate_quiz_from_knowledge). slug
    # comes from the runner-args body (the backend's live-room start
    # endpoint passes it alongside room_id); current_scene_id starts at
    # the body's scene_id and is refreshed on every canvas.sceneChanged
    # from the post-nav snapshot. If slug isn't present in the body, the
    # quiz tool returns "no active live-room session" and the LLM
    # apologises gracefully — see make_handle_generate_quiz.
    session_context = SessionContext()
    session_context.set_slug(slug or None)
    # Seed scene id from the body so the very first quiz request lands
    # on the room's initial scene even before any sceneChanged event.
    # Prefer the live snapshot's scene_id over the body field, because
    # for flow-based rooms the body carries scene_id=None while the
    # snapshot resolves the flow's current scene.
    # S65 (Option B) — scene_id nested under current_scene.
    initial_scene_id = (
        ((scene_snapshot or {}).get("current_scene") or {}).get("scene_id")
        or scene_id
        or None
    )
    session_context.set_scene(str(initial_scene_id) if initial_scene_id else None)
    logger.info(
        "[SESSION_CONTEXT] slug={!r} initial_scene_id={!r}",
        session_context.get_slug(),
        session_context.get_current_scene_id(),
    )

    # S66 Block 5a — lazy vision-frame tracker. Marks the initial scene
    # as loaded so the first canvas_analyze doesn't fire a redundant
    # re-fetch when scene_image_b64 succeeded above. The closure
    # _ensure_vision_for_active_scene is wired into canvas_ctx after the
    # LLMContext is constructed (Python looks up `context` at call time).
    vision_tracker = VisionFrameTracker()
    if scene_image_b64 and initial_scene_id:
        vision_tracker.mark_loaded(str(initial_scene_id))

    # S67b — one dedicated Gemini vision client per session (reads VISION_MODEL
    # + GOOGLE_AI_API_KEY itself; stubs gracefully when the key is unset). Reused
    # across every visual question; see services/vision_query.run_vision_query.
    vision_client = VisionClient()

    # S64d — persona.build_system_prompt doesn't include the CANVAS PAGE
    # section (the migration to prompt_builder.build_system_prompt_split was
    # planned but never landed). Append it here so the LLM knows what verbs
    # the active Page supports. At session start the manifest is empty —
    # the section renders the "no page registered yet" guidance. The
    # on_app_message canvas.register branch below rebuilds the prompt with
    # the real manifest once the iframe registers.
    # S64e — also append AGENT PLAYBOOK (quiz flow guidance). The helper
    # is the single source of truth for the prompt's append-suffix shape.
    base_system_prompt = system_prompt
    system_prompt = _assemble_full_prompt(base_system_prompt, canvas_manifest.current())

    # ── AI Services ──
    canvas_tools = ToolsSchema(
        standard_tools=[
            *make_canvas_protocol_schemas(canvas_manifest.current()),
            # S64e — generate_quiz_from_knowledge sits alongside the 5
            # canvas protocol tools. It's not a canvas verb (it talks to
            # the backend directly, not through the iframe), but it
            # shares the same registration surface.
            GENERATE_QUIZ_SCHEMA,
            AGENT_ANNOTATE_SCHEMA,
        ],
    )

    # STT language driven by the live-room language (S61).
    # S65 (Option B) — language nested under live_room.
    snapshot_language = ((scene_snapshot or {}).get("live_room") or {}).get("language")
    deepgram_language = resolve_deepgram_language(snapshot_language)
    logger.info(
        "Deepgram language configured: snapshot_language={} deepgram_language={}",
        snapshot_language,
        deepgram_language,
    )
    if snapshot_language and snapshot_language not in DEEPGRAM_LANGUAGE_MAP:
        logger.warning(
            "Snapshot language not in Deepgram map; falling back to multi: snapshot_language={}",
            snapshot_language,
        )

    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        settings=DeepgramSTTService.Settings(language=deepgram_language),
    )
    voice_id = (avatar_config or {}).get("voiceModelId") or CARTESIA_VOICE_ID
    # Block 13 — Cartesia constructor pinned to the narration cache's
    # sample_rate + model so a primed segment can be replayed byte-for-
    # byte alongside live miss-path audio. The voice is the live-room
    # primary; per-segment voice switches still go through
    # _classic_set_voice as a TTSUpdateSettingsFrame delta.
    tts = CachedFirstTTSService(
        api_key=CARTESIA_API_KEY,
        sample_rate=NARRATION_AUDIO_SAMPLE_RATE,
        # #2 — strip any Markdown gpt-oss emits (**, *, `, #, lists) before
        # synthesis so it isn't spoken as literal symbols. Audio-side safety
        # net; the VOICE OUTPUT prompt directive curbs it at the source (and
        # also keeps the caption clean — the transcript forwarder sits
        # upstream of this filter). Flows through CachedFirstTTSService /
        # CartesiaTTSService **kwargs to the base TTSService.
        text_filters=[MarkdownTextFilter()],
        # S77 B5 — the service boots at the ROOM language; per-line
        # narration_language switches ride TTSUpdateSettingsFrame deltas
        # (_classic_set_language) and SceneNarrator restores the room
        # language after every narration run.
        settings=CartesiaTTSService.Settings(
            voice=voice_id,
            model=NARRATION_TTS_MODEL_ID,
            language=_cartesia_language(snapshot_language),
        ),
    )

    llm, eager_hook = _build_llm_and_eager_hook(
        provider=LLM_CANVAS_PROVIDER,
        system_prompt=system_prompt,
        canvas_pending=canvas_pending,
        send_canvas_message=send_canvas_message,
    )
    logger.info(
        "Canvas Protocol LLM provider={} eager_hook={}",
        LLM_CANVAS_PROVIDER,
        eager_hook.__class__.__name__,
    )
    # NOTE: eager_hook is instantiated but not yet wired into the LLM
    # service's streaming loop — that's a per-provider integration that
    # depends on Pipecat's hook surface and lands in a follow-up. Until
    # then, all canvas tool calls go through the regular tool-handler
    # path (no eager-dispatch latency win, but correctness is identical).

    # ── Conversation context ──
    # S46 injects the scene image straight into the MAIN LLM context. Gate on
    # MAIN_LLM_SUPPORTS_VISION — a text-only main model (e.g. Groq gpt-oss-120b)
    # 400s on image content ("content must be a string"). Visual questions are
    # still answered via the decoupled S67b Gemini path (canvas_analyze →
    # run_vision_query), which injects TEXT reasoning, not a raw image.
    initial_messages = []
    if scene_image_b64 and MAIN_LLM_SUPPORTS_VISION:
        from scene_context import build_vision_message

        initial_messages.append(build_vision_message(scene_image_b64))
    elif scene_image_b64:
        logger.info(
            "[VISION] not injecting scene image into main-LLM context "
            "(MAIN_LLM_SUPPORTS_VISION=false, provider={}); S67b Gemini path still handles visual Q&A",
            LLM_CANVAS_PROVIDER,
        )

    context = LLMContext(
        messages=initial_messages if initial_messages else None,
        tools=canvas_tools,
    )

    # S66 Block 5a — bridge vision_tracker + per-session state to the
    # canvas_analyze handler. The closure captures `context` (LLMContext),
    # `vision_tracker`, and `session_context` so it can fetch the current
    # scene's image and add it to context on demand. No-op when the
    # tracker already covers the active scene (cache hit).
    async def _ensure_vision_for_active_scene(question: str = "") -> str | None:
        # S67b — Design B: prefer a live capture of the visitor's annotated
        # canvas (reasoned by the dedicated Gemini client), fall back to the
        # base-scene Pillow PNG with a blind-spot note. The orchestration is
        # the testable run_vision_query core; this closure RETURNS its result
        # text (handle_analyze folds it into the canvas_analyze tool result —
        # in-band, S67b fix). Always fresh per
        # question — no VisionFrameTracker reuse, because live annotations
        # change WITHIN a scene (A-AG-3 gotcha #1).
        # S67b — deterministic visual-indicator signal: bracket the Gemini
        # analyze call so the shell can show "looking at your screen…" only
        # while the model is actually running. Non-canvas, session-level
        # (mirrors quiz_generation_state); state ∈ {"analyzing","idle"}.
        async def _emit_vision_state(state: str) -> None:
            await send_canvas_message({"type": "vision_state", "state": state})

        msg = await run_vision_query(
            question,
            request_capture=request_canvas_capture,
            vision_client=vision_client,
            backend_client=api_client,
            session_context=session_context,
            room_id=room_id,
            api_url=api_url,
            on_vision_state=_emit_vision_state,
        )
        # S67b fix — RETURN the vision text so handle_analyze folds it into the
        # canvas_analyze TOOL RESULT (in-band). Previously injected out-of-band
        # via context.add_message, which raced the function-call re-run and
        # intermittently left the spoken reply without the vision answer.
        return msg.get("content") if msg else None

    canvas_ctx.ensure_vision = _ensure_vision_for_active_scene

    # P3 (2026-07-13) — single-slot handle for the backgrounded
    # narration run. Auto Play Phase A widened the slot's tenants:
    # scene-change narration, session-start narration (A5), manual
    # request_narrate replays, and autoplay_control resume all run here.
    # A newer run cancels the previous one before starting its own, and
    # a cancelled run never emits script_complete.
    scene_narration_task: dict[str, asyncio.Task | None] = {"task": None}

    def _start_narration_task(coro, *, replace: bool = True) -> bool:
        """Put a narration coroutine into the single slot (Phase A).

        ``replace=True`` (scene change / manual replay / resume) cancels
        the active run first — newest wins. ``replace=False`` (session
        start) yields to an already-active run instead: a run that raced
        ahead already owns narration, and stomping it would
        double-narrate. Returns True iff the task was started.
        """
        prev = scene_narration_task["task"]
        if prev is not None and not prev.done():
            if not replace:
                coro.close()  # avoid a "never awaited" warning
                return False
            prev.cancel()
        scene_narration_task["task"] = asyncio.create_task(coro)
        return True

    # ── Scene-change refresh (S64c) ──
    # Single refresh entry point for both voice-initiated and
    # visitor-initiated scene navigation. The frontend's navigateToIndex
    # emits a `canvas.sceneChanged` Daily message after every successful
    # nav, regardless of trigger (voice via canvas_control, or visitor
    # rail-click). The on_app_message handler below routes that single
    # message to this closure, which:
    #   1. Rebuilds the system prompt (so CANVAS ELEMENTS + ids and
    #      knowledge.flow reflect the new scene).
    #   2. Refreshes the vision frame.
    # No verb-specific logic and no api_navigate call. Since P2
    # (2026-07-13) the frontend advances the backend cursor in the
    # BACKGROUND, so this closure must NOT rely on the cursor — every
    # fetch here targets the broadcast sceneId explicitly. aliases_out updates canvas_ctx.element_alias_map
    # in place so the next canvas_annotate / canvas_action resolves
    # aliases against the new scene's elements.
    async def refresh_agent_for_current_scene(
        target_scene_id: str | None = None,
    ) -> None:
        # S66 Block 5c — ``target_scene_id`` is the scene_id forwarded
        # from canvas.sceneChanged. When present, every snapshot fetch
        # in this closure targets THAT scene by id rather than relying
        # on the room's cursor — eliminates the cursor race where
        # canvas.sceneChanged arrives before the backend's cursor
        # commits. None preserves pre-5c cursor-based behavior.
        if not room_id:
            logger.warning("[CANVAS SCENECHANGED] refresh skipped: no room_id")
            return

        # S66 Block 0 — perf instrumentation. T_agent = entry → prompt
        # reassigned (the gate metric per the S66 plan). vision is logged
        # separately since vision is eager today but will become lazy in
        # Block 5a — the log shape stays stable so before/after deltas
        # are directly comparable.
        t0 = time.monotonic()

        nonlocal base_system_prompt
        # S66 Block 5c — only the broadcast scene_id drives the by-id
        # snapshot fetch. The body's ``scene_id`` is a Pipecat
        # runner-args hint that can be stale for flow rooms (the room
        # cursor moves; the body doesn't), so it stays Strategy-2-only
        # via the legacy ``scene_id=`` kwarg below — it is NOT forwarded
        # to the snapshot fetch. When the broadcast lacks sceneId
        # (target_scene_id is None) we fall through to a cursor-based
        # fetch — correct only eventually: since P2 the shell advances
        # the cursor with a background goto it does not await, so a
        # cursor read can trail the visual by one scene until it lands.
        # P3 (2026-07-13) — the post-nav snapshot is fetched ONCE here
        # and threaded into build_system_prompt (which used to fetch an
        # identical copy internally — the redundant fetch #2 on every
        # scene change).
        from api_client import get_scene_image_base64, get_scene_snapshot

        fresh_snapshot = await get_scene_snapshot(
            room_id, api_url, scene_id=target_scene_id or None
        )
        new_base = await build_system_prompt(
            room_id=room_id,
            avatar_id=avatar_id,
            scene_id=scene_id,
            api_url=api_url,
            aliases_out=canvas_ctx.element_alias_map,
            flow_cache=flow_knowledge_cache,
            snapshot_scene_id=target_scene_id or None,
            snapshot=fresh_snapshot,
        )
        # S64d — preserve CANVAS PAGE section across scene refreshes. Otherwise
        # the section is dropped on every navigation and the LLM loses its
        # verb list. Also update base_system_prompt so subsequent
        # canvas.register rebuilds use the post-navigation base.
        base_system_prompt = new_base
        new_prompt = _assemble_full_prompt(new_base, canvas_manifest.current())
        try:
            llm._settings.system_instruction = new_prompt  # type: ignore[attr-defined]
            logger.info(
                "[CANVAS SCENECHANGED] system prompt refreshed ({} chars)",
                len(new_prompt),
            )
        except Exception:
            logger.warning(
                "[CANVAS SCENECHANGED] could not set system_instruction on llm service"
            )
        t_prompt = time.monotonic()

        # S64e — refresh session_context.current_scene_id from the
        # post-nav snapshot so generate_quiz_from_knowledge targets the
        # scene the visitor is actually looking at.
        # S65 (Option B) — scene_id nested under current_scene.
        fresh_cs = (fresh_snapshot or {}).get("current_scene") or {}
        if fresh_snapshot and fresh_cs.get("scene_id"):
            new_scene_id = str(fresh_cs["scene_id"])
            if new_scene_id != session_context.get_current_scene_id():
                logger.info(
                    "[CANVAS SCENECHANGED] session_context scene_id {} -> {}",
                    session_context.get_current_scene_id(),
                    new_scene_id,
                )
            session_context.set_scene(new_scene_id)

        # S66 Block 5a — vision-frame refresh policy. Lazy (default)
        # invalidates the tracker so the next canvas_analyze refetches;
        # eager preserves pre-5a behavior (fetch + add to context here).
        # The session-start fetch is unconditional and lives outside this
        # closure — only the per-scene-change refetch is gated.
        if VISION_REFRESH_MODE == "eager" and MAIN_LLM_SUPPORTS_VISION:
            # P3 — render by broadcast scene_id: a cursor-relative render
            # races the shell's background cursor advance and can return
            # the OLD scene's image.
            new_image = await get_scene_image_base64(
                room_id, api_url, scene_id=target_scene_id or None
            )
            if new_image:
                from scene_context import build_vision_message

                context.add_message(build_vision_message(new_image))
                vision_tracker.mark_loaded(session_context.get_current_scene_id())
                logger.info(
                    "[CANVAS SCENECHANGED] vision context refreshed with new scene image"
                )
            else:
                logger.warning(
                    "[CANVAS SCENECHANGED] could not fetch new scene image after navigation"
                )
        else:
            # lazy mode, OR a text-only main LLM (MAIN_LLM_SUPPORTS_VISION=false)
            # where injecting the image would 400 — either way defer to the
            # S67b Gemini path (run_vision_query) on the next canvas_analyze.
            vision_tracker.invalidate()
            logger.info(
                "[CANVAS SCENECHANGED] vision-refresh deferred (mode={}, main_llm_vision={}) — "
                "scene image fetched on next canvas_analyze via S67b",
                VISION_REFRESH_MODE,
                MAIN_LLM_SUPPORTS_VISION,
            )

        # S66 Block 0 — vision spans prompt-reassigned → vision-in-context.
        # In lazy mode (Block 5a) the "vision" half measures the invalidate
        # tick (sub-ms) rather than a backend round-trip — the win shows
        # up as a smaller T_agent on subsequent scene changes.
        # pipeline=classic distinguishes this from the run_bot_relay
        # closure's [perf-refresh] line.
        # Block 5b: flow_cache.hits/misses report whether the FLOW-scope
        # render hit the per-session cache for this refresh.
        logger.info(
            "[perf-refresh] pipeline=classic T_agent={}ms vision={}ms mode={} flow_cache(hits={},misses={})",
            int((t_prompt - t0) * 1000),
            int((time.monotonic() - t_prompt) * 1000),
            VISION_REFRESH_MODE,
            flow_knowledge_cache.hits,
            flow_knowledge_cache.misses,
        )

        # ── S65 Bug #2 — narrate the new scene + emit script_complete ──
        # The shell's auto-advance keys off script_complete; if the agent
        # doesn't narrate + emit on canvas.sceneChanged, auto-advance
        # stalls at the first scene change. Cancel any in-flight
        # narration_gate futures first (leftover from a prior scene's
        # narration that may not have fully resolved), then orchestrate
        # narrate → followup speak → script_complete in spec'd order.
        # narrator + _classic_speak resolve from the enclosing
        # run_bot_classic scope (defined below; Python closures look up
        # names at call time, so this works even though they appear
        # textually later in the file).
        #
        # P3 (2026-07-13) — narration runs as a BACKGROUND task. It was
        # awaited inline, which held the sceneChanged handler (and with
        # it the Daily app-message dispatch for this session) hostage
        # for the full narration duration — tens of seconds for long
        # scripts. Only the prompt refresh needs to complete inline
        # (that's the ordering guarantee the on_app_message comment
        # documents); narration is playback. Single-slot: a newer scene
        # change cancels the previous scene's narration task, and a
        # cancelled task deliberately does NOT emit script_complete —
        # the superseding navigation already moved the shell forward.
        if fresh_snapshot:
            prev_task = scene_narration_task["task"]
            narration_active = prev_task is not None and not prev_task.done()
            narration_gate.cancel_all("scene_change")
            if narration_active:
                prev_task.cancel()
                # A3 — flush UNCONDITIONALLY when a narration run was
                # active. Gating on bot_is_speaking missed two windows
                # where old-scene TTS work is in flight but not yet
                # audible (Cartesia TTFB after a queued TTSSpeakFrame;
                # the inter-utterance boundary gap) — the old scene's
                # audio then played over the new scene AND its stale
                # TTSStoppedFrame could misalign the new run's FIFO.
                # The flush's InterruptionFrame also makes the TTS
                # service drop that in-flight context, closing both. A
                # completed (drained) prior run skips this — flushing
                # then would only clip unrelated conversation audio.
                await _flush_bot_audio("scene_change")
            _start_narration_task(_narrate_and_complete(fresh_snapshot))

    # ── Canvas Protocol generic tool handlers (S64c) ──
    canvas_protocol_handlers = make_canvas_protocol_handlers(canvas_ctx)
    for name, handler in canvas_protocol_handlers.items():
        llm.register_function(name, handler)

    # ── Quiz generation tool (S64e) ──
    # generate_quiz_from_knowledge talks to the backend for the quiz blob,
    # then dispatches a canvas set_page through canvas_ctx so the iframe
    # activates the quiz Page with the new blob in the same tool call
    # (S64e Option D — LLMs unreliably copy large structured args between
    # tool calls). The handler is factory-bound to the api_client module,
    # the session_context (slug + current scene id), and canvas_ctx (the
    # same context the 5 canvas tool handlers use).
    llm.register_function(
        "generate_quiz_from_knowledge",
        make_handle_generate_quiz(api_client, session_context, canvas_ctx),
    )

    # ── Block 8 — canvas_annotate (agent overlay annotations) ──
    # Standalone tool (like generate_quiz_from_knowledge), NOT a canvas protocol
    # tool: it emits a session-level agent_annotate message instead of dispatching
    # a canvas.command. Factory-bound to this session's S67b capture + vision
    # machinery; the agent_annotate_result ack resolves _pending_annotates futures.
    llm.register_function(
        "canvas_annotate",
        make_handle_canvas_annotate(
            send_message=send_canvas_message,
            pending=_pending_annotates,
            vision_client=vision_client,
            request_capture=request_canvas_capture,
            fetch_live_bytes=_fetch_live_bytes,
            backend_client=api_client,
            session_context=session_context,
            element_alias_map=element_alias_map,
            room_id=room_id,
            api_url=api_url,
            timeout_s=AGENT_ANNOTATE_TIMEOUT_MS / 1000,
        ),
    )

    # ── Aggregators with VAD ──
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # ── Transcript forwarding + speaking/thinking state ──
    user_transcript_fwd = TranscriptForwarder(output_transport)
    avatar_transcript_fwd = TranscriptForwarder(output_transport)
    speaking_notifier = SpeakingStateNotifier(output_transport)
    thinking_notifier = ThinkingNotifier(output_transport)

    # ── Narration completion gate (S65 G3) ──
    # Sits between TTS and output_transport so it observes every
    # TTSStoppedFrame; SceneNarrator awaits these to know when a script
    # segment has finished rendering before applying the next voice
    # update. See narration.NarrationCompletionGate for the FIFO + race
    # caveat.
    narration_gate = NarrationCompletionGate()

    # ── Pipeline ──
    pipeline = Pipeline(
        [
            transport.input(),  # Visitor's microphone audio (WebRTC)
            stt,  # Deepgram: speech -> text
            user_transcript_fwd,  # Forward user STT transcripts to frontend
            user_aggregator,  # Add user message to conversation history
            llm,  # OpenAI: generate response
            thinking_notifier,  # Notify frontend of LLM thinking state
            avatar_transcript_fwd,  # Forward avatar LLM text to frontend
            speaking_notifier,  # Notify frontend of speaking state
            tts,  # Cartesia: response -> speech audio
            narration_gate,  # S65 G3: observe TTSStoppedFrame for narration
            output_transport,  # Send audio back to visitor (WebRTC)
            assistant_aggregator,  # Add bot response to conversation history
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # ── Scene narrator (S65 G3 — classic pipeline, per-segment voice) ──
    # primary_voice is the Cartesia voice the TTS service was constructed
    # with at session start (avatar config first, CARTESIA_VOICE_ID
    # fallback). After per-segment voice switching, SceneNarrator resets
    # back to primary BEFORE returning so the closing "feel free to ask"
    # line + all subsequent conversation use the agent's own voice. See
    # narration.SceneNarrator for the loop + idempotency contract.
    primary_voice_id_classic = voice_id

    async def _classic_set_voice(target_voice_id: str) -> None:
        # Queue a TTSUpdateSettingsFrame; the TTS service applies the
        # delta inline before the next TTSSpeakFrame is processed
        # (pipecat 0.0.108 TTSService.process_frame). No await needed
        # for completion — the delta is synchronous within the TTS
        # processor's frame loop.
        delta = CartesiaTTSService.Settings(voice=target_voice_id)
        await task.queue_frames([TTSUpdateSettingsFrame(delta=delta)])

    async def _classic_set_language(language_code: str) -> None:
        # S77 B5 — same delta mechanism as the voice switch; language
        # and voice are independent Settings fields, so this never
        # disturbs the active voice (Q8: same voice, per-line language).
        delta = CartesiaTTSService.Settings(language=_cartesia_language(language_code))
        await task.queue_frames([TTSUpdateSettingsFrame(delta=delta)])

    async def _classic_speak(text: str) -> None:
        # Register the completion future BEFORE queuing the speak — a
        # short utterance can fire TTSStoppedFrame before we register
        # otherwise (race), leaving us waiting forever. 30 s upper
        # bound stays as a lost-frame backstop only (Phase A A2: an
        # interruption now resolves the future immediately instead of
        # stalling it out); it is NOT reduced because a long cached
        # segment legitimately holds its TTSStoppedFrame for the full
        # playback duration (Block 15 sleep). On timeout we log and
        # proceed so the rest of the narration plan still runs.
        fut = narration_gate.expect_next_stop()
        if fut.done() and fut.result() is NARRATION_INTERRUPTED:
            # A2 — an InterruptionFrame landed in the between-segments
            # window (no future registered). Abort BEFORE queuing the
            # speak so the interrupted run can't say another word. Drop
            # the armed prime too — the narrator primed this segment
            # right before calling us, and a stale prime would be
            # consumed by the NEXT run_tts (the LLM's reply to the
            # barge-in would play scene-script PCM instead).
            tts.prime_cached(None)
            logger.info("[NARRATION] segment skipped — run already interrupted")
            raise NarrationInterrupted()
        await task.queue_frames([TTSSpeakFrame(text=text)])
        try:
            result = await asyncio.wait_for(fut, timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("[NARRATION] segment TTSStoppedFrame timeout — continuing")
            return
        except asyncio.CancelledError:
            logger.warning("[NARRATION] segment future cancelled")
            raise
        if result is NARRATION_INTERRUPTED:
            # A2 — visitor speech interrupted this segment mid-playout.
            # The transport already flushed the audio; abort the run so
            # narration doesn't resume mid-scene over the conversation.
            # (Prime already consumed by this segment's run_tts, or
            # dropped by CachedFirstTTSService's InterruptionFrame
            # handler — nothing to clear here.)
            logger.info("[NARRATION] segment interrupted by visitor speech")
            raise NarrationInterrupted()

    # Block 13 — narration cache closures. Shared dict between prefetch
    # (fills it once at the top of each per-scene narration) and prime
    # (per-segment consume; primes the TTS service for the very next
    # run_tts call). On any HTTP / decode failure we fall back to live —
    # the prime returns False, the narrator runs the normal voice-switch
    # + live synthesis path for that segment.
    _narration_cache: dict[str, CachedSegment] = {}

    async def _narration_prefetch(plan):
        _narration_cache.clear()
        targets = [seg for seg in plan if seg.id and seg.audio and seg.audio.get("url")]
        if not targets:
            return
        # 10 s per-segment timeout — narration is on the critical path
        # of scene entry; a slow CDN should fail fast to live rather
        # than stall the visitor.
        async with httpx.AsyncClient(timeout=10.0) as client:
            for seg in targets:
                url = seg.audio["url"]
                try:
                    r = await client.get(url)
                    if r.status_code != 200:
                        logger.warning(
                            "[NARRATION] prefetch {} -> HTTP {}, live fallback",
                            seg.id,
                            r.status_code,
                        )
                        continue
                    sr = int(
                        seg.audio.get("sample_rate") or NARRATION_AUDIO_SAMPLE_RATE
                    )
                    if sr != NARRATION_AUDIO_SAMPLE_RATE:
                        logger.warning(
                            "[NARRATION] prefetch {} sr={} != configured {}, skip",
                            seg.id,
                            sr,
                            NARRATION_AUDIO_SAMPLE_RATE,
                        )
                        continue
                    _narration_cache[seg.id] = CachedSegment(
                        pcm=r.content,
                        sample_rate=sr,
                        num_channels=NARRATION_AUDIO_NUM_CHANNELS,
                    )
                except Exception as exc:
                    logger.warning("[NARRATION] prefetch {} failed: {!r}", seg.id, exc)

    def _narration_prime(seg) -> bool:
        cached = _narration_cache.get(seg.id) if seg.id else None
        tts.prime_cached(cached)
        return cached is not None

    # ── S79 — animated-narration cue path (§2.6, classic) ───────────────
    # The classic mirror of the relay wiring: the gate supplies both the
    # barge-in futures and the bot-speaking signal; the pre-cue drain rides
    # the existing _classic_wait_playout (the gate resolves immediately
    # when nothing is pending, so no dirty-tracking is needed here).
    async def _send_cue_message_classic(payload: dict) -> None:
        await output_transport.send_message(
            OutputTransportMessageFrame(message=payload)
        )

    narration_cue_classic = NarrationCueController(
        send_message=_send_cue_message_classic,
        speak_fallback=_classic_speak,
        expect_interruption=narration_gate.expect_interruption,
        bot_is_speaking=lambda: narration_gate.bot_is_speaking,
    )

    async def _classic_cue_line(idx: int, seg) -> None:
        # Drain any earlier TTS'd lines' playout before the clip's own
        # audio starts (the §3 one-audible-source law); immediate when
        # nothing is pending. NarrationInterrupted propagates — a barge-in
        # during the TTS tail keeps today's abort semantics.
        await _classic_wait_playout(PLAYOUT_DRAIN_FALLBACK_S)
        anim = seg.animation or {}
        await narration_cue_classic.cue(
            line_index=idx,
            url=anim.get("url") or "",
            duration_seconds=float(anim.get("duration_seconds") or 0.0),
            text=seg.text,
        )

    narrator = SceneNarrator(
        primary_voice_id=primary_voice_id_classic,
        set_voice=_classic_set_voice,
        speak=_classic_speak,
        prefetch=_narration_prefetch,
        prime=_narration_prime,
        set_language=_classic_set_language,  # S77 B5
        room_language=(snapshot_language or "en"),
        cue=_classic_cue_line,
    )

    # ── Auto Play Phase A — narration-run machinery (classic) ──

    async def _classic_wait_playout(timeout_s: float) -> None:
        """A1 — block until queued narration audio truly finishes playing.

        Resolves on BotStoppedSpeakingFrame (the transport's true
        audio-queue drain, broadcast upstream through the gate);
        immediately when nothing is playing (covers the cached-playback
        race where drain lands before registration, and the no-audio
        edge); with NARRATION_INTERRUPTED → NarrationInterrupted on a
        visitor barge-in during the playout tail. The timeout is the
        compute_playout_drain_timeout budget — a lost-frame backstop,
        not the normal exit path.
        """
        fut = narration_gate.expect_playout_drain()
        try:
            result = await asyncio.wait_for(fut, timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                "[NARRATION] playout-drain timeout after {}s — continuing",
                timeout_s,
            )
            return
        if result is NARRATION_INTERRUPTED:
            logger.info("[NARRATION] playout drain interrupted by visitor speech")
            raise NarrationInterrupted()

    async def _flush_bot_audio(reason: str) -> None:
        """Flush queued bot audio via a pipeline interruption (A3/A4).

        pipecat exposes no narrower transport-level bot-audio flush —
        MediaSender.handle_interruptions (the thing that must run) only
        fires off an InterruptionFrame. Queueing an InterruptionTaskFrame
        makes the PipelineTask broadcast one from the source; that also
        cancels any in-flight LLM turn, which is acceptable at both call
        sites (old-scene audio on scene change; everything on autoplay
        stop — stale output must die either way).

        Awaits the gate's expect_interruption so the flush has traversed
        the TTS + gate pipeline positions before returning — a narration
        run started right after this would otherwise race the in-flight
        InterruptionFrame and lose its first segment to it.
        """
        confirm = narration_gate.expect_interruption()
        await task.queue_frames([InterruptionTaskFrame()])
        try:
            await asyncio.wait_for(confirm, timeout=2.0)
            logger.info("[NARRATION] bot audio flushed ({})", reason)
        except asyncio.TimeoutError:
            logger.warning(
                "[NARRATION] flush ({}) not confirmed within 2s — continuing",
                reason,
            )
        except asyncio.CancelledError:
            # Distinguish "a concurrent handler ran narration_gate.
            # cancel_all (scene change / stop / teardown racing this
            # flush)" — the WAITER was cancelled, our task wasn't; the
            # interruption is still in flight, so just proceed — from a
            # genuine cancellation of the calling task, which must
            # propagate.
            if confirm.cancelled():
                logger.info(
                    "[NARRATION] flush ({}) confirm waiter cancelled — continuing",
                    reason,
                )
            else:
                raise

    # Phase A — session-start context seeding must survive slot
    # supersession. The greeting/presentation-done developer message
    # (and the LLMRunFrame wake for script-less scenes) used to live
    # exclusively in the session-start path; now that path is
    # cancellable (A5), so whichever run COMPLETES first performs the
    # one-time seeding instead. An interrupted run also marks it done —
    # the visitor's own speech drives the LLM from there, and a
    # belated greeting mid-conversation would be worse than none.
    session_seeded = {"done": False}

    async def _seed_session_context_once(spoke_script: bool) -> None:
        if session_seeded["done"]:
            return
        session_seeded["done"] = True
        if spoke_script:
            context.add_message(
                {
                    "role": "developer",
                    "content": (
                        "You just finished presenting the scene scripts to the visitor. "
                        "They heard your full presentation. Don't repeat what you already said."
                    ),
                }
            )
        else:
            context.add_message(
                {
                    "role": "developer",
                    "content": GREETING_TRIGGER_PROMPT,
                }
            )
            await task.queue_frames([LLMRunFrame()])

    async def _narrate_and_complete(
        snap: dict, *, force: bool = False, trigger: str = "auto"
    ) -> None:
        """One narration run: narrate → followup → drain → script_complete.

        Runs as the single-slot background task (P3). Emission rules per
        the frozen wire contract: script_complete goes out only after
        true playout drain (A1); NEVER for cancelled (superseded /
        stopped) or interrupted runs (A2 / rule 2b); a plain failure
        still emits with hadScript=false so the shell doesn't stall.
        """
        try:
            # Clear the gate's interruption latch + any stale futures
            # from a superseded run (see NarrationCompletionGate.begin_run).
            narration_gate.begin_run()
            # S79 — arm the cue path (scene id scopes stale completions).
            narration_cue_classic.begin_run(
                (snap.get("current_scene") or {}).get("scene_id")
            )
            try:
                spoke_script = await run_scene_narration(
                    snap,
                    narrator=narrator,
                    speak_followup=_classic_speak,
                    force=force,
                    wait_playout=_classic_wait_playout,
                )
            except NarrationInterrupted:
                session_seeded["done"] = True  # visitor spoke — no greeting
                logger.info(
                    "[NARRATION] run interrupted by visitor speech — "
                    "script_complete suppressed (trigger={})",
                    trigger,
                )
                return
            except Exception as exc:
                logger.warning("[NARRATION] narration failed: {!r}", exc)
                spoke_script = False
            await output_transport.send_message(
                OutputTransportMessageFrame(
                    message=build_script_complete_payload(
                        snap, spoke_script=spoke_script, trigger=trigger
                    )
                )
            )
            logger.info(
                "[NARRATION] run complete spoke_script={} trigger={}",
                spoke_script,
                trigger,
            )
            # If this run superseded the (cancelled) session-start run
            # before it could seed the LLM context, do it now with THIS
            # run's outcome — otherwise a script-less flow never wakes
            # the LLM and the avatar stays silent until spoken to.
            await _seed_session_context_once(spoke_script)
        except asyncio.CancelledError:
            logger.info(
                "[NARRATION] run superseded or stopped — script_complete suppressed"
            )
            raise

    async def _session_start_narration_run() -> None:
        """A5 — session-start narration in the single slot.

        Previously inlined in on_client_connected, which made the
        opening narration uncancellable — a scene change during it kept
        the old scene's script playing over the new scene. Same emission
        rules as _narrate_and_complete, plus the one-time session
        context seeding (presentation-done note vs greeting wake).
        """
        try:
            narration_gate.begin_run()
            # S79 — same cue arming as the scene-change site.
            narration_cue_classic.begin_run(
                (scene_snapshot.get("current_scene") or {}).get("scene_id")
            )
            try:
                spoke_script = await run_scene_narration(
                    scene_snapshot,
                    narrator=narrator,
                    speak_followup=_classic_speak,
                    wait_playout=_classic_wait_playout,
                )
            except NarrationInterrupted:
                session_seeded["done"] = True  # visitor spoke — no greeting
                logger.info(
                    "[NARRATION] session-start narration interrupted — "
                    "script_complete suppressed"
                )
                return
            except Exception as exc:
                logger.warning("[NARRATION] session-start narration failed: {!r}", exc)
                spoke_script = False
            await output_transport.send_message(
                OutputTransportMessageFrame(
                    message=build_script_complete_payload(
                        scene_snapshot, spoke_script=spoke_script
                    )
                )
            )
            await _seed_session_context_once(spoke_script)
        except asyncio.CancelledError:
            logger.info(
                "[NARRATION] session-start narration superseded — "
                "script_complete suppressed"
            )
            raise

    # ── Event handlers (simple — no participant role detection) ──

    @transport.event_handler("on_app_message")
    async def on_app_message(transport, message, sender):
        """Route Canvas Protocol messages from the frontend (S64c)."""
        # S64d defensive: some Daily SDK / Pipecat Cloud versions deliver
        # app-messages as JSON strings rather than parsed dicts. Parse
        # before the isinstance(dict) guard rejects everything (otherwise
        # canvas.register hits the silent-return path below and the bot's
        # manifest registry never updates).
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except (json.JSONDecodeError, ValueError):
                pass
        if not isinstance(message, dict):
            return
        msg_type = message.get("type")

        # ── S79 — shell→agent cue completion (Canvas Protocol v0.3) ──
        # The shell emits `script_complete` when a cued animation clip
        # ends; the controller resolves the active cue. Legacy shells
        # never send this — the branch is diff-zero.
        if msg_type == "script_complete":
            narration_cue_classic.on_script_complete(message)
            return

        # ── S65c Block 5 — manual visitor-action triggers ──
        # Sit BEFORE the canvas.* dispatch as early-return branches: an
        # inbound payload's `type` is the sole discriminator and we want
        # zero risk of canvas.* fallthrough accidentally also matching a
        # manual trigger (impossible with exact string match today, but
        # the ordering documents intent for future maintainers).
        if msg_type == "request_narrate":
            # E2 — re-narrate the visitor's current scene on demand.
            # Fetches a fresh snapshot so a click after navigation always
            # targets the scene the visitor is actually looking at (the
            # session-start ``scene_snapshot`` isn't refreshed across
            # scene changes; the equivalent fetch lives in
            # ``refresh_agent_for_current_scene`` and we mirror it here).
            # ``force=True`` bypasses the once-per-entry guard from
            # Block 4b. ``trigger="manual"`` propagates to the shell so
            # ``script_complete`` doesn't queue an auto-advance on a
            # visitor-initiated replay.
            if not room_id:
                logger.info("[REQUEST_NARRATE] skipped: no room_id")
                return
            from api_client import get_scene_snapshot

            manual_snapshot = await get_scene_snapshot(room_id, api_url)
            if not manual_snapshot:
                logger.info("[REQUEST_NARRATE] skipped: snapshot fetch failed")
                return
            # Phase A — manual replays now run through the single-slot
            # narration task instead of inline. Inline, the replay held
            # this session's app-message dispatch hostage for the whole
            # narration (the exact P3 lesson, worsened by the new A1
            # drain-wait), and a scene change couldn't cancel it. The
            # slot run applies the frozen-contract emission rules
            # (drain-gated, suppressed on interruption/supersede) with
            # trigger="manual" preserved on the wire.
            narration_gate.cancel_all("manual_replay")
            _start_narration_task(
                _narrate_and_complete(manual_snapshot, force=True, trigger="manual")
            )
            logger.info("[REQUEST_NARRATE] manual replay started")
            return

        if msg_type == "autoplay_control":
            # ── Auto Play Phase A (A4) — shell playback controls ──
            # Session-level early-return branch, same discipline as
            # request_* (exact string match; never rides DailyRelay).
            # Frozen wire contract v1: action ∈ {stop, resume}.
            action = message.get("action")
            if action == "stop":
                # Cancel the active narration run (a cancelled run never
                # emits script_complete) AND flush queued bot audio.
                # Unconditional flush: the pause control means "stop the
                # bot's voice now", so an in-flight LLM reply dies too.
                prev = scene_narration_task["task"]
                if prev is not None and not prev.done():
                    prev.cancel()
                narration_gate.cancel_all("autoplay_stop")
                await _flush_bot_audio("autoplay_stop")
                logger.info("[AUTOPLAY] stop: narration cancelled, audio flushed")
            elif action == "resume":
                # Fresh snapshot → re-narrate the CURRENT scene from
                # segment 0. Fetched by the session's tracked scene id
                # (refreshed on every canvas.sceneChanged) so the P2
                # background-cursor window can't serve the PREVIOUS
                # scene; cursor fallback when the id is unknown.
                # force=True bypasses the once-per-entry guard (the
                # stopped run already marked the scene); default
                # trigger="auto" keeps the completion advance-eligible
                # per the contract.
                if not room_id:
                    logger.info("[AUTOPLAY] resume skipped: no room_id")
                    return
                from api_client import get_scene_snapshot

                resume_snapshot = await get_scene_snapshot(
                    room_id,
                    api_url,
                    scene_id=session_context.get_current_scene_id() or None,
                )
                if not resume_snapshot:
                    logger.info("[AUTOPLAY] resume skipped: snapshot fetch failed")
                    return
                _start_narration_task(
                    _narrate_and_complete(resume_snapshot, force=True)
                )
                logger.info("[AUTOPLAY] resume: re-narration started")
            else:
                logger.warning("[AUTOPLAY] unknown action {!r} ignored", action)
            return

        if msg_type == "request_quiz":
            # E3 — if the agent hasn't fully initialized (no slug or no
            # current scene_id yet), silently ignore the click. The
            # frontend button is gated on agentJoined (Block 9) so this
            # should be rare; the silent path keeps test fixtures simple
            # by not emitting an error event for a not-ready agent.
            # The gate is extracted to ``request_quiz_ready(...)`` so a
            # unit test machine-verifies the predicate.
            if not request_quiz_ready(session_context):
                logger.info("[REQUEST_QUIZ] skipped: session_context not ready")
                return
            quiz_count = message.get("count", 3)
            quiz_language = message.get("language") or "en"

            async def _emit_quiz_state(state: str, err: str | None) -> None:
                payload = {"type": "quiz_generation_state", "state": state}
                if err:
                    payload["error"] = err
                await send_canvas_message(payload)

            logger.info(
                "[REQUEST_QUIZ] generating: count={} language={!r}",
                quiz_count,
                quiz_language,
            )
            try:
                result = await run_quiz_generation(
                    backend_client=api_client,
                    session_context=session_context,
                    canvas_ctx=canvas_ctx,
                    count=quiz_count,
                    language=quiz_language,
                    on_state=_emit_quiz_state,
                )
                logger.info(
                    "[REQUEST_QUIZ] complete ok={} error={!r}",
                    result.ok,
                    result.error,
                )
                if result.ok:
                    # Wake the LLM with the blob in context. The voice path
                    # naturally gets this via Pipecat's function-call
                    # aggregator (tool result ⇒ context); the button path
                    # bypasses the LLM turn entirely, so without this the
                    # quiz Page shows on screen but the agent has no idea
                    # a quiz was activated — it can't narrate the first
                    # question and can't answer "what's on the quiz?".
                    # Mirrors the no-script greeting wake pattern below
                    # (developer message + LLMRunFrame). On failure we
                    # skip both — the visitor sees the error state and
                    # the LLM stays in its prior context.
                    context.add_message(
                        {
                            "role": "developer",
                            "content": (
                                "A quiz has just been activated on the canvas by the visitor "
                                "clicking the Quiz action button (not by your tool call). The "
                                "quiz Page is already showing on screen. Here is the quiz blob "
                                "you need to drive the session:\n\n"
                                f"{json.dumps(result.blob)}\n\n"
                                "Now read the first question aloud and wait for the visitor's "
                                "answer, exactly as you would after calling "
                                "generate_quiz_from_knowledge yourself. Use canvas_action "
                                "verbs (submit_answer / skip_question) to record their answers "
                                "and let the Quiz Page own the pacing."
                            ),
                        }
                    )
                    await task.queue_frames([LLMRunFrame()])
            except Exception as exc:
                # run_quiz_generation already catches its own failures
                # and emits ("error", msg); any escape past it means
                # something on our orchestration side broke. Surface a
                # generic state so the button can flip out of spinner.
                logger.exception("[REQUEST_QUIZ] unexpected failure")
                await _emit_quiz_state(
                    "error", f"unexpected: {type(exc).__name__}: {str(exc)[:200]}"
                )
            return

        if msg_type == "canvas_capture_result":
            # ── S67b — vision capture ack (sibling of canvas.commandResult) ──
            # Non-canvas, session-level: resolve the captureId future opened by
            # request_canvas_capture. Carries {status,w,h} only — the JPEG bytes
            # live in the backend (fetched by captureId). Early-return alongside
            # request_*, BEFORE the canvas dispatch (A-AG-1).
            cid = message.get("captureId")
            fut = _pending_captures.get(cid)
            logger.info(
                "[VISION] canvas_capture_result reached on_app_message: captureId={} "
                "status={!r} pending_future={}",
                cid,
                message.get("status"),
                "found"
                if fut is not None
                else "MISSING (already timed out, unknown id, or this handler never received earlier captures)",
            )
            if fut is not None and not fut.done():
                fut.set_result(message)
            return

        if msg_type == "agent_annotate_result":
            # ── Block 8 — agent-annotate ack (sibling of canvas_capture_result) ──
            # Non-canvas, session-level: resolve the annotateId future opened by
            # make_handle_canvas_annotate's _emit. Early-return BEFORE the canvas
            # dispatch — same discipline as request_* / canvas_capture_result.
            aid = message.get("annotateId")
            ann_fut = _pending_annotates.get(aid)
            if ann_fut is not None and not ann_fut.done():
                ann_fut.set_result(message)
            return

        if msg_type == "canvas.register":
            logger.info(
                f"[CANVAS REGISTER] pageType={message.get('pageType')!r} version={message.get('version')!r}"
            )
            canvas_manifest.set_manifest(message)
            # S64d — rebuild the system prompt so the LLM sees the new
            # Page's verb list in the CANVAS PAGE section. Without this
            # the LLM keeps seeing "No page registered yet" guidance and
            # tries canvas_set_page (which v0.1 doesn't end-to-end-wire).
            try:
                new_prompt = _assemble_full_prompt(
                    base_system_prompt, canvas_manifest.current()
                )
                llm._settings.system_instruction = new_prompt  # type: ignore[attr-defined]
                logger.info(
                    "[CANVAS REGISTER] system prompt rebuilt with manifest section"
                )
            except Exception as exc:
                logger.warning("[CANVAS REGISTER] prompt rebuild failed: {!r}", exc)
        elif msg_type == "canvas.stateChange":
            logger.info(
                f"[CANVAS STATECHANGE] keys={list((message.get('semanticState') or {}).keys())}"
            )
            canvas_manifest.update_state(message.get("semanticState") or {})
        elif msg_type == "canvas.sceneChanged":
            # S64c — single refresh trigger for ALL scene navigations,
            # voice-initiated and visitor-initiated alike. The frontend
            # emits this from navigateToIndex, which is the canonical
            # scene-change function (both rail-click and voice-tool paths
            # bottom out there). Awaited inline so the refresh completes
            # before subsequent messages (in particular canvas.commandResult
            # for voice nav) are processed by the loop, keeping the prompt
            # fresh by the time the LLM speaks its tool result.
            scene_index = message.get("sceneIndex")
            scene_id_from_msg = message.get("sceneId")  # S66 Block 5c
            logger.info(
                f"[CANVAS SCENECHANGED] sceneIndex={scene_index!r} sceneId={scene_id_from_msg!r}"
            )
            await refresh_agent_for_current_scene(
                target_scene_id=scene_id_from_msg or None
            )
        elif msg_type == "canvas.commandResult":
            cid = message.get("commandId")
            logger.info(
                f"[CANVAS COMMANDRESULT] commandId={cid!r} result={message.get('result')!r}"
            )
            if cid:
                canvas_pending.resolve(cid, message.get("result") or {})
        elif msg_type == "canvas.commandError":
            cid = message.get("commandId")
            logger.warning(
                f"[CANVAS COMMANDERROR] commandId={cid!r} error={message.get('error')!r}"
            )
            if cid:
                canvas_pending.reject(cid, message.get("error") or {})

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Visitor connected to live room")

        # S65 G3+G4 — narrate scene scripts via SceneNarrator (per-segment
        # voice + S77 language switching, idempotent per scene_id), then
        # speak the localized invitation on the manual branch (S77 B6:
        # the auto-advance branch is SILENT — 600 ms pause, no cue),
        # then emit script_complete LAST.
        #
        # Auto Play Phase A (A5) — this used to run INLINE, outside the
        # single-slot task, so a scene change during the opening
        # narration could not cancel it and the old scene's script kept
        # playing over the new scene. It now runs in the same slot as
        # scene-change narration. replace=False: if a run already owns
        # the slot (a scene change raced ahead of this handler, or a
        # second client connected mid-narration), yield to it instead of
        # double-narrating.
        _start_narration_task(_session_start_narration_run(), replace=False)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Visitor disconnected")
        canvas_pending.cancel_all("session_end")
        # S67b — drain any in-flight capture futures (sibling registry).
        for _cap_fut in _pending_captures.values():
            if not _cap_fut.done():
                _cap_fut.cancel()
        _pending_captures.clear()
        for _ann_fut in _pending_annotates.values():
            if not _ann_fut.done():
                _ann_fut.cancel()
        _pending_annotates.clear()
        # S65 G3 — surface any narration segments still awaiting their
        # TTSStoppedFrame so the disconnect doesn't leave coroutines
        # hung on futures that will never resolve.
        narration_gate.cancel_all("session_end")
        # P3 — cancel any backgrounded scene narration so it can't emit
        # script_complete into a torn-down session.
        _nar_task = scene_narration_task["task"]
        if _nar_task is not None and not _nar_task.done():
            _nar_task.cancel()
        await task.cancel()

    # ── Run ──
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


# ======================================================================
#
#  RELAY AVATAR PIPELINE  (avatar display: talking)
#
#  Mic -> AudioFilter -> STT -> LLM -> text relay -> SoulX avatar bot
#
#  No local TTS.  Cloud bot is silent; SoulX handles speech + video.
#  Complex participant management: role detection, audio filtering,
#  avatar readiness gating, relay protocol.
#
# ======================================================================


async def run_bot_relay(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    room_id: str = "",
    avatar_id: str = "",
    scene_id: str = "",
    flow_id: str | None = None,
    api_url: str | None = None,
    slug: str = "",
):
    """Relay avatar pipeline — forwards LLM text to SoulX for speech + video."""
    logger.info("Starting relay avatar agent (room={}, avatar={})", room_id, avatar_id)

    # S64c — initial element alias map. Populated by build_system_prompt
    # below and passed into CanvasToolContext when it's constructed
    # further down. The same dict object is reused across the session;
    # post_scene_change clears + repopulates it on every scene nav so
    # the tool handlers always see the current scene's aliases.
    element_alias_map: dict[str, str] = {}
    # S66 Block 5b — see run_bot_classic. One cache per session.
    flow_knowledge_cache = FlowKnowledgeCache()

    # ── Session-start backend reads (P3 2026-07-13) ──
    # Snapshot + scene-image are independent — fetch them CONCURRENTLY,
    # then thread the snapshot into build_system_prompt (which used to
    # fetch its own copy, and a third fetch below pulled the same
    # snapshot AGAIN for scripts). See run_bot_classic's mirror.
    scene_snapshot = None
    scene_image_b64 = None
    # avatar_config joins the gather on this branch too (classic already fetched it —
    # bot.py:851). The relay pipeline now runs Cartesia locally, so it needs the
    # per-user cloned `voiceModelId`; without it every avatar would speak in the
    # default stock voice.
    avatar_config = None
    if room_id:
        from api_client import (
            get_avatar_config,
            get_scene_image_base64,
            get_scene_snapshot,
        )

        scene_snapshot, avatar_config, scene_image_b64 = await asyncio.gather(
            get_scene_snapshot(room_id, api_url),
            get_avatar_config(room_id, api_url),
            get_scene_image_base64(room_id, api_url),
        )
        if avatar_config:
            logger.info(
                "Avatar config: name={} voiceModelId={}",
                avatar_config.get("name"),
                avatar_config.get("voiceModelId"),
            )
        if scene_image_b64:
            logger.info(
                "Fetched scene canvas image ({} chars base64)", len(scene_image_b64)
            )
        else:
            logger.info("No scene image available; vision disabled for this session")
        if scene_snapshot:
            scripts_len = len(
                ((scene_snapshot.get("current_scene") or {}).get("scripts")) or []
            )
            logger.info("Scene snapshot loaded (scripts={})", scripts_len)

    system_prompt = await build_system_prompt(
        room_id=room_id,
        avatar_id=avatar_id,
        scene_id=scene_id,
        api_url=api_url,
        aliases_out=element_alias_map,
        flow_cache=flow_knowledge_cache,
        snapshot=scene_snapshot,
    )
    logger.info(f"System prompt length: {len(system_prompt)} chars")

    # ── Canvas Protocol substrate (S64c) ──
    output_transport = transport.output()
    canvas_manifest = CanvasManifestRegistry()
    canvas_pending = PendingCommandRegistry()

    async def send_canvas_message(payload: dict) -> None:
        """Send a Canvas Protocol Daily app-message to the frontend."""
        try:
            await output_transport.send_message(
                OutputTransportMessageFrame(message=payload)
            )
        except Exception as exc:
            logger.warning("Failed to send canvas message: {}", exc)

    # ── S67b — vision capture round-trip (sibling of canvas_pending) ──
    # Keyed by a fresh captureId, NOT a canvas commandId: a separate dict so
    # the two correlation spaces can never collide. The shell screenshots the
    # visitor's canvas, uploads the JPEG to the backend ingest, and replies
    # with a tiny canvas_capture_result carrying {status,w,h} only — the bytes
    # are fetched separately by captureId (api_client.get_vision_capture).
    _pending_captures: dict[str, asyncio.Future] = {}
    # ── Block 8 — agent-annotate ack registry (sibling of _pending_captures) ──
    # Keyed by a fresh annotateId so annotate acks and capture acks never collide;
    # drained on disconnect alongside _pending_captures.
    _pending_annotates: dict[str, asyncio.Future] = {}

    async def request_canvas_capture(hint: str) -> tuple[str, dict | None]:
        """Ask the shell to capture the visitor's canvas; await the ack.

        Returns ``(capture_id, result)`` — result is the dict
        {captureId, status, w?, h?, error?} on reply, or None on timeout. The
        caller needs the capture_id to fetch the bytes by id (run_vision_query →
        _fetch_live_bytes), so we surface it alongside the result (B12). Rides send_canvas_message
        (the generic outbound helper — non-canvas payloads ride it too, e.g.
        quiz_generation_state); the reply lands in the canvas_capture_result
        on_app_message branch. The captured BYTES are not in the dict.
        """
        capture_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        _pending_captures[capture_id] = fut
        logger.info("[VISION] requesting canvas capture {} hint={!r}", capture_id, hint)
        try:
            await send_canvas_message(
                {
                    "type": "request_canvas_capture",
                    "captureId": capture_id,
                    "hint": hint,
                    "maxDim": VISION_MAX_DIM,  # advisory; the shell owns the encode
                }
            )
            result = await asyncio.wait_for(
                fut, timeout=VISION_CAPTURE_TIMEOUT_MS / 1000
            )
            logger.info(
                "[VISION] capture {} resolved status={!r}",
                capture_id,
                (result or {}).get("status"),
            )
            return capture_id, result
        except asyncio.TimeoutError:
            logger.warning(
                "[VISION] capture {} timed out after {}ms",
                capture_id,
                VISION_CAPTURE_TIMEOUT_MS,
            )
            return capture_id, None
        finally:
            _pending_captures.pop(capture_id, None)

    canvas_ctx = CanvasToolContext(
        manifest_registry=canvas_manifest,
        pending=canvas_pending,
        send_app_message=send_canvas_message,
        element_alias_map=element_alias_map,
        command_timeout_s=6.0,
    )

    # S64e — session-scoped state for non-canvas tools that need slug +
    # current scene id (currently generate_quiz_from_knowledge). slug
    # comes from the runner-args body (the backend's live-room start
    # endpoint passes it alongside room_id); current_scene_id starts at
    # the body's scene_id and is refreshed on every canvas.sceneChanged
    # from the post-nav snapshot. If slug isn't present in the body, the
    # quiz tool returns "no active live-room session" and the LLM
    # apologises gracefully — see make_handle_generate_quiz.
    session_context = SessionContext()
    session_context.set_slug(slug or None)
    # Seed scene id from the body so the very first quiz request lands
    # on the room's initial scene even before any sceneChanged event.
    # Prefer the live snapshot's scene_id over the body field, because
    # for flow-based rooms the body carries scene_id=None while the
    # snapshot resolves the flow's current scene.
    # S65 (Option B) — scene_id nested under current_scene.
    initial_scene_id = (
        ((scene_snapshot or {}).get("current_scene") or {}).get("scene_id")
        or scene_id
        or None
    )
    session_context.set_scene(str(initial_scene_id) if initial_scene_id else None)
    logger.info(
        "[SESSION_CONTEXT] slug={!r} initial_scene_id={!r}",
        session_context.get_slug(),
        session_context.get_current_scene_id(),
    )

    # S66 Block 5a — see run_bot_classic for rationale. Mirror tracker
    # init so lazy mode invalidates on scene change here too.
    vision_tracker = VisionFrameTracker()
    if scene_image_b64 and initial_scene_id:
        vision_tracker.mark_loaded(str(initial_scene_id))

    # S67b — one dedicated Gemini vision client per session (reads VISION_MODEL
    # + GOOGLE_AI_API_KEY itself; stubs gracefully when the key is unset). Reused
    # across every visual question; see services/vision_query.run_vision_query.
    vision_client = VisionClient()

    # S64d — see run_bot_classic for rationale. Append CANVAS PAGE section
    # to system_prompt so the LLM knows the active Page's verbs; on
    # canvas.register, rebuild via base_system_prompt + new section.
    # S64e — _assemble_full_prompt also tacks on the AGENT PLAYBOOK
    # section (cross-tool sequences, currently the quiz flow).
    base_system_prompt = system_prompt
    system_prompt = _assemble_full_prompt(base_system_prompt, canvas_manifest.current())

    # ── AI Services (no TTS — SoulX handles speech) ──
    canvas_tools = ToolsSchema(
        standard_tools=[
            *make_canvas_protocol_schemas(canvas_manifest.current()),
            # S64e — generate_quiz_from_knowledge alongside canvas tools.
            GENERATE_QUIZ_SCHEMA,
            AGENT_ANNOTATE_SCHEMA,
        ],
    )

    # STT language driven by the live-room language (S61).
    # S65 (Option B) — language nested under live_room.
    snapshot_language = ((scene_snapshot or {}).get("live_room") or {}).get("language")
    deepgram_language = resolve_deepgram_language(snapshot_language)
    logger.info(
        "Deepgram language configured: snapshot_language={} deepgram_language={}",
        snapshot_language,
        deepgram_language,
    )
    if snapshot_language and snapshot_language not in DEEPGRAM_LANGUAGE_MAP:
        logger.warning(
            "Snapshot language not in Deepgram map; falling back to multi: snapshot_language={}",
            snapshot_language,
        )

    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        settings=DeepgramSTTService.Settings(language=deepgram_language),
    )

    llm, eager_hook = _build_llm_and_eager_hook(
        provider=LLM_CANVAS_PROVIDER,
        system_prompt=system_prompt,
        canvas_pending=canvas_pending,
        send_canvas_message=send_canvas_message,
    )
    logger.info(
        "Canvas Protocol LLM provider={} eager_hook={}",
        LLM_CANVAS_PROVIDER,
        eager_hook.__class__.__name__,
    )
    # NOTE: see run_bot_classic — eager_hook is instantiated but not wired
    # into the streaming loop yet.

    # ── Conversation context ──
    # S46 injects the scene image straight into the MAIN LLM context. Gate on
    # MAIN_LLM_SUPPORTS_VISION — a text-only main model (e.g. Groq gpt-oss-120b)
    # 400s on image content ("content must be a string"). Visual questions are
    # still answered via the decoupled S67b Gemini path (canvas_analyze →
    # run_vision_query), which injects TEXT reasoning, not a raw image.
    initial_messages = []
    if scene_image_b64 and MAIN_LLM_SUPPORTS_VISION:
        from scene_context import build_vision_message

        initial_messages.append(build_vision_message(scene_image_b64))
    elif scene_image_b64:
        logger.info(
            "[VISION] not injecting scene image into main-LLM context "
            "(MAIN_LLM_SUPPORTS_VISION=false, provider={}); S67b Gemini path still handles visual Q&A",
            LLM_CANVAS_PROVIDER,
        )

    context = LLMContext(
        messages=initial_messages if initial_messages else None,
        tools=canvas_tools,
    )

    # S66 Block 5a — mirror run_bot_classic. See there for rationale.
    async def _ensure_vision_for_active_scene(question: str = "") -> str | None:
        # S67b — Design B: prefer a live capture of the visitor's annotated
        # canvas (reasoned by the dedicated Gemini client), fall back to the
        # base-scene Pillow PNG with a blind-spot note. The orchestration is
        # the testable run_vision_query core; this closure RETURNS its result
        # text (handle_analyze folds it into the canvas_analyze tool result —
        # in-band, S67b fix). Always fresh per
        # question — no VisionFrameTracker reuse, because live annotations
        # change WITHIN a scene (A-AG-3 gotcha #1).
        # S67b — deterministic visual-indicator signal: bracket the Gemini
        # analyze call so the shell can show "looking at your screen…" only
        # while the model is actually running. Non-canvas, session-level
        # (mirrors quiz_generation_state); state ∈ {"analyzing","idle"}.
        async def _emit_vision_state(state: str) -> None:
            await send_canvas_message({"type": "vision_state", "state": state})

        msg = await run_vision_query(
            question,
            request_capture=request_canvas_capture,
            vision_client=vision_client,
            backend_client=api_client,
            session_context=session_context,
            room_id=room_id,
            api_url=api_url,
            on_vision_state=_emit_vision_state,
        )
        # S67b fix — RETURN the vision text so handle_analyze folds it into the
        # canvas_analyze TOOL RESULT (in-band). Previously injected out-of-band
        # via context.add_message, which raced the function-call re-run and
        # intermittently left the spoken reply without the vision answer.
        return msg.get("content") if msg else None

    canvas_ctx.ensure_vision = _ensure_vision_for_active_scene

    # P3 (2026-07-13) — single-slot handle for the backgrounded
    # narration run. Auto Play Phase A widened the slot's tenants:
    # scene-change narration, session-start narration (A5), manual
    # request_narrate replays, and autoplay_control resume all run here.
    # A newer run cancels the previous one before starting its own, and
    # a cancelled run never emits script_complete.
    scene_narration_task: dict[str, asyncio.Task | None] = {"task": None}

    def _start_narration_task(coro, *, replace: bool = True) -> bool:
        """Put a narration coroutine into the single slot (Phase A).

        ``replace=True`` (scene change / manual replay / resume) cancels
        the active run first — newest wins. ``replace=False`` (session
        start) yields to an already-active run instead: a run that raced
        ahead already owns narration, and stomping it would
        double-narrate. Returns True iff the task was started.
        """
        prev = scene_narration_task["task"]
        if prev is not None and not prev.done():
            if not replace:
                coro.close()  # avoid a "never awaited" warning
                return False
            prev.cancel()
        scene_narration_task["task"] = asyncio.create_task(coro)
        return True

    # ── Scene-change refresh (S64c) ──
    # Single refresh entry point for both voice-initiated and
    # visitor-initiated scene navigation. The frontend's navigateToIndex
    # emits a `canvas.sceneChanged` Daily message after every successful
    # nav, regardless of trigger (voice via canvas_control, or visitor
    # rail-click). The on_app_message handler below routes that single
    # message to this closure, which:
    #   1. Rebuilds the system prompt (so CANVAS ELEMENTS + ids and
    #      knowledge.flow reflect the new scene).
    #   2. Refreshes the vision frame.
    # No verb-specific logic and no api_navigate call. Since P2
    # (2026-07-13) the frontend advances the backend cursor in the
    # BACKGROUND, so this closure must NOT rely on the cursor — every
    # fetch here targets the broadcast sceneId explicitly. aliases_out updates canvas_ctx.element_alias_map
    # in place so the next canvas_annotate / canvas_action resolves
    # aliases against the new scene's elements.
    async def refresh_agent_for_current_scene(
        target_scene_id: str | None = None,
    ) -> None:
        # S66 Block 5c — ``target_scene_id`` is the scene_id forwarded
        # from canvas.sceneChanged. When present, every snapshot fetch
        # in this closure targets THAT scene by id rather than relying
        # on the room's cursor — eliminates the cursor race where
        # canvas.sceneChanged arrives before the backend's cursor
        # commits. None preserves pre-5c cursor-based behavior.
        if not room_id:
            logger.warning("[CANVAS SCENECHANGED] refresh skipped: no room_id")
            return

        # S66 Block 0 — perf instrumentation. T_agent = entry → prompt
        # reassigned (the gate metric per the S66 plan). vision is logged
        # separately since vision is eager today but will become lazy in
        # Block 5a — the log shape stays stable so before/after deltas
        # are directly comparable.
        t0 = time.monotonic()

        nonlocal base_system_prompt
        # S66 Block 5c — only the broadcast scene_id drives the by-id
        # snapshot fetch. The body's ``scene_id`` is a Pipecat
        # runner-args hint that can be stale for flow rooms (the room
        # cursor moves; the body doesn't), so it stays Strategy-2-only
        # via the legacy ``scene_id=`` kwarg below — it is NOT forwarded
        # to the snapshot fetch. When the broadcast lacks sceneId
        # (target_scene_id is None) we fall through to a cursor-based
        # fetch — correct only eventually: since P2 the shell advances
        # the cursor with a background goto it does not await, so a
        # cursor read can trail the visual by one scene until it lands.
        # P3 (2026-07-13) — the post-nav snapshot is fetched ONCE here
        # and threaded into build_system_prompt (which used to fetch an
        # identical copy internally — the redundant fetch #2 on every
        # scene change).
        from api_client import get_scene_image_base64, get_scene_snapshot

        fresh_snapshot = await get_scene_snapshot(
            room_id, api_url, scene_id=target_scene_id or None
        )
        new_base = await build_system_prompt(
            room_id=room_id,
            avatar_id=avatar_id,
            scene_id=scene_id,
            api_url=api_url,
            aliases_out=canvas_ctx.element_alias_map,
            flow_cache=flow_knowledge_cache,
            snapshot_scene_id=target_scene_id or None,
            snapshot=fresh_snapshot,
        )
        # S64d — preserve CANVAS PAGE section across scene refreshes. Otherwise
        # the section is dropped on every navigation and the LLM loses its
        # verb list. Also update base_system_prompt so subsequent
        # canvas.register rebuilds use the post-navigation base.
        base_system_prompt = new_base
        new_prompt = _assemble_full_prompt(new_base, canvas_manifest.current())
        try:
            llm._settings.system_instruction = new_prompt  # type: ignore[attr-defined]
            logger.info(
                "[CANVAS SCENECHANGED] system prompt refreshed ({} chars)",
                len(new_prompt),
            )
        except Exception:
            logger.warning(
                "[CANVAS SCENECHANGED] could not set system_instruction on llm service"
            )
        t_prompt = time.monotonic()

        # S64e — refresh session_context.current_scene_id from the
        # post-nav snapshot so generate_quiz_from_knowledge targets the
        # scene the visitor is actually looking at.
        # S65 (Option B) — scene_id nested under current_scene.
        fresh_cs = (fresh_snapshot or {}).get("current_scene") or {}
        if fresh_snapshot and fresh_cs.get("scene_id"):
            new_scene_id = str(fresh_cs["scene_id"])
            if new_scene_id != session_context.get_current_scene_id():
                logger.info(
                    "[CANVAS SCENECHANGED] session_context scene_id {} -> {}",
                    session_context.get_current_scene_id(),
                    new_scene_id,
                )
            session_context.set_scene(new_scene_id)

        # S66 Block 5a — see run_bot_classic for rationale.
        if VISION_REFRESH_MODE == "eager" and MAIN_LLM_SUPPORTS_VISION:
            # P3 — render by broadcast scene_id: a cursor-relative render
            # races the shell's background cursor advance and can return
            # the OLD scene's image.
            new_image = await get_scene_image_base64(
                room_id, api_url, scene_id=target_scene_id or None
            )
            if new_image:
                from scene_context import build_vision_message

                context.add_message(build_vision_message(new_image))
                vision_tracker.mark_loaded(session_context.get_current_scene_id())
                logger.info(
                    "[CANVAS SCENECHANGED] vision context refreshed with new scene image"
                )
            else:
                logger.warning(
                    "[CANVAS SCENECHANGED] could not fetch new scene image after navigation"
                )
        else:
            # lazy mode, OR a text-only main LLM (MAIN_LLM_SUPPORTS_VISION=false)
            # where injecting the image would 400 — either way defer to the
            # S67b Gemini path (run_vision_query) on the next canvas_analyze.
            vision_tracker.invalidate()
            logger.info(
                "[CANVAS SCENECHANGED] vision-refresh deferred (mode={}, main_llm_vision={}) — "
                "scene image fetched on next canvas_analyze via S67b",
                VISION_REFRESH_MODE,
                MAIN_LLM_SUPPORTS_VISION,
            )

        # S66 Block 0 — vision spans prompt-reassigned → vision-in-context.
        # In lazy mode (Block 5a) "vision" measures the invalidate tick.
        # pipeline=relay distinguishes this from the run_bot_classic
        # closure's [perf-refresh] line.
        logger.info(
            "[perf-refresh] pipeline=relay T_agent={}ms vision={}ms mode={} flow_cache(hits={},misses={})",
            int((t_prompt - t0) * 1000),
            int((time.monotonic() - t_prompt) * 1000),
            VISION_REFRESH_MODE,
            flow_knowledge_cache.hits,
            flow_knowledge_cache.misses,
        )

        # ── S65 Bug #2 — narrate the new scene + emit script_complete ──
        # The shell's auto-advance keys off script_complete; if the agent
        # doesn't narrate + emit on canvas.sceneChanged, auto-advance
        # stalls at the first scene change. Mirrors the classic
        # pipeline's refresh-time narration. Closes the RELAY_TURN
        # BEFORE emitting script_complete so SoulX sees TURN_END before
        # the shell potentially auto-advances. narrator + _relay_speak +
        # _relay_close_turn resolve from the enclosing run_bot_relay
        # scope (defined below; Python closures look up names at call
        # time, so this works even though they appear textually later).
        #
        # P3 (2026-07-13) — narration runs as a BACKGROUND task (see the
        # classic pipeline's mirror for the full rationale). Single-slot:
        # a newer scene change cancels the previous scene's narration;
        # the finally still closes the RELAY_TURN so SoulX never sees a
        # dangling turn, but a cancelled task does NOT emit
        # script_complete — the superseding navigation already moved the
        # shell forward.
        if fresh_snapshot:
            prev_task = scene_narration_task["task"]
            if prev_task is not None and not prev_task.done():
                prev_task.cancel()
                # A3 (relay half) — best-effort: RELAY_INTERRUPT the open
                # narration turn so SoulX stops rendering the old scene's
                # text over the new scene. No local audio to flush here.
                await _relay_interrupt_narration_turn()
            _start_narration_task(_narrate_and_complete(fresh_snapshot))

    # ── Canvas Protocol generic tool handlers (S64c) ──
    canvas_protocol_handlers = make_canvas_protocol_handlers(canvas_ctx)
    for name, handler in canvas_protocol_handlers.items():
        llm.register_function(name, handler)

    # ── Quiz generation tool (S64e) ──
    # generate_quiz_from_knowledge talks to the backend for the quiz blob,
    # then dispatches a canvas set_page through canvas_ctx so the iframe
    # activates the quiz Page with the new blob in the same tool call
    # (S64e Option D — LLMs unreliably copy large structured args between
    # tool calls). The handler is factory-bound to the api_client module,
    # the session_context (slug + current scene id), and canvas_ctx (the
    # same context the 5 canvas tool handlers use).
    llm.register_function(
        "generate_quiz_from_knowledge",
        make_handle_generate_quiz(api_client, session_context, canvas_ctx),
    )

    # ── Block 8 — canvas_annotate (agent overlay annotations) ──
    # Standalone tool (like generate_quiz_from_knowledge), NOT a canvas protocol
    # tool: it emits a session-level agent_annotate message instead of dispatching
    # a canvas.command. Factory-bound to this session's S67b capture + vision
    # machinery; the agent_annotate_result ack resolves _pending_annotates futures.
    llm.register_function(
        "canvas_annotate",
        make_handle_canvas_annotate(
            send_message=send_canvas_message,
            pending=_pending_annotates,
            vision_client=vision_client,
            request_capture=request_canvas_capture,
            fetch_live_bytes=_fetch_live_bytes,
            backend_client=api_client,
            session_context=session_context,
            element_alias_map=element_alias_map,
            room_id=room_id,
            api_url=api_url,
            timeout_s=AGENT_ANNOTATE_TIMEOUT_MS / 1000,
        ),
    )

    # ── Aggregators with VAD ──
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # ── Transcript forwarding + speaking/thinking state ──
    user_transcript_fwd = TranscriptForwarder(output_transport)
    avatar_transcript_fwd = TranscriptForwarder(output_transport)
    speaking_notifier = SpeakingStateNotifier(output_transport)
    thinking_notifier = ThinkingNotifier(output_transport)

    # ── Participant tracking ──
    avatar_participant_id: str | None = None
    active_human_id: str | None = None
    avatar_ready_event = asyncio.Event()
    captured_audio_participant_id: str | None = None
    greeting_sent = False

    def get_avatar_participant_id() -> str | None:
        return avatar_participant_id

    def get_local_participant_id() -> str | None:
        pid = str(getattr(transport, "participant_id", "")).strip()
        return pid or None

    def _remove_pending_audio_capture(pid: str | None):
        if not pid:
            return
        input_transport = getattr(transport, "_input", None)
        pending = getattr(input_transport, "_capture_participant_audio", None)
        if not isinstance(pending, list):
            return
        filtered = [item for item in pending if not item or str(item[0]) != pid]
        if len(filtered) != len(pending):
            pending[:] = filtered
            logger.info("Removed pending audio capture for participant_id={}", pid)

    async def _ensure_avatar_participant_ignored(pid: str | None):
        nonlocal captured_audio_participant_id, active_human_id
        if not pid:
            return

        _remove_pending_audio_capture(pid)

        update_subscriptions = getattr(transport, "update_subscriptions", None)
        if callable(update_subscriptions):
            await update_subscriptions(
                participant_settings={
                    pid: {
                        "media": {
                            "microphone": "unsubscribed",
                            "screenAudio": "unsubscribed",
                        }
                    }
                }
            )

        if captured_audio_participant_id == pid:
            captured_audio_participant_id = None
        if active_human_id == pid:
            active_human_id = None

        logger.info(
            "Ensured SoulX avatar participant is ignored participant_id={}", pid
        )

    async def _start_human_audio_capture(pid: str | None):
        nonlocal captured_audio_participant_id
        if not pid or pid == avatar_participant_id:
            return
        if captured_audio_participant_id == pid:
            return

        _remove_pending_audio_capture(avatar_participant_id)

        capture_participant_audio = getattr(
            transport, "capture_participant_audio", None
        )
        if callable(capture_participant_audio):
            await capture_participant_audio(pid, "microphone")

        input_transport = getattr(transport, "_input", None)
        start_audio_in_streaming = getattr(
            input_transport, "start_audio_in_streaming", None
        )
        if callable(start_audio_in_streaming) and not getattr(
            input_transport, "_streaming_started", False
        ):
            await start_audio_in_streaming()
            logger.info("Started Daily audio input streaming")

        captured_audio_participant_id = pid
        logger.info("Started human-only audio capture for participant_id={}", pid)

    # ── Relay-mode processors ──
    human_audio_filter = HumanOnlyAudioInputFilter(
        get_avatar_participant_id,
        get_local_participant_id,
    )
    avatar_ready_gate = AvatarReadyGateProcessor(
        avatar_ready_event, timeout_s=AVATAR_READY_TIMEOUT_S
    )
    relay_processor = AvatarRelayProcessor(output_transport, get_avatar_participant_id)

    # ── soulx-audio.v1: TTS moves INTO the agent ──────────────────────────────
    # Legacy path (SOULX_WS_URL unset) is untouched: forward TEXT over Daily
    # app-messages and let SoulX synthesize. New path: synthesize here with Cartesia
    # (so per-user cloned voices work exactly as in the classic pipeline) and stream
    # PCM to the Modal renderer, which returns synced video.
    soulx_client = None
    soulx_sink = None
    soulx_tail: list = [relay_processor]

    if SOULX_WS_URL:
        from services.soulx_audio import SoulXAudioClient, SoulXAudioSink

        relay_voice_id = (avatar_config or {}).get("voiceModelId") or CARTESIA_VOICE_ID
        # Same construction as the classic pipeline (bot.py:1046) — pinned to the
        # narration cache's sample_rate + model so cached segments stay replayable.
        relay_tts = CachedFirstTTSService(
            api_key=CARTESIA_API_KEY,
            sample_rate=NARRATION_AUDIO_SAMPLE_RATE,
            text_filters=[MarkdownTextFilter()],
            settings=CartesiaTTSService.Settings(
                voice=relay_voice_id,
                model=NARRATION_TTS_MODEL_ID,
            ),
        )

        avatar_token = await _mint_avatar_token(getattr(runner_args, "room_url", ""))
        # The renderer REQUIRES the room avatar's photo (no-fallback policy,
        # 2026-08-06): it conditions the model on this image per session, so
        # the room shows the SELECTED avatar or errors — never a default face.
        avatar_photo_url = (avatar_config or {}).get("profilePhotoUrl") or ""
        soulx_client = SoulXAudioClient(
            ws_url=SOULX_WS_URL,
            auth_token=SOULX_AUTH_TOKEN,
            room_url=getattr(runner_args, "room_url", ""),
            room_token=avatar_token or "",
            sample_rate=NARRATION_AUDIO_SAMPLE_RATE,
            ready_timeout_s=AVATAR_READY_TIMEOUT_S,
            avatar_ref=avatar_photo_url,
        )
        if avatar_photo_url:
            connected = await soulx_client.connect()
        else:
            # Don't burn a GPU cold start on a session the renderer will
            # refuse anyway — same outcome (voice-only + loud error), decided
            # here instead of after a 2-minute container spin-up.
            logger.error(
                "SoulX: avatar config has no profilePhotoUrl — no-fallback "
                "policy forbids a default identity; running voice-only"
            )
            connected = False
        # Release the gate either way. Connected => the avatar is in the room. Failed =>
        # voice-only fallback, which is still infinitely better than a silent bot.
        avatar_ready_event.set()
        logger.info(
            "SoulX renderer {} — relay pipeline running in {} mode",
            "connected" if connected else "UNAVAILABLE",
            "avatar" if connected else "voice-only fallback",
        )
        soulx_sink = SoulXAudioSink(soulx_client)
        soulx_tail = [relay_tts, soulx_sink]

    # ── Pipeline ──
    pipeline = Pipeline(
        [
            transport.input(),  # Participant audio (per-track)
            human_audio_filter,  # Drop avatar/local bot audio
            stt,  # Deepgram: speech -> text
            user_transcript_fwd,  # Forward user STT transcripts to frontend
            user_aggregator,  # Add user message to conversation history
            avatar_ready_gate,  # Block until avatar bot is ready
            llm,  # OpenAI: generate response
            thinking_notifier,  # Notify frontend of LLM thinking state
            avatar_transcript_fwd,  # Forward avatar LLM text to frontend
            speaking_notifier,  # Notify frontend of speaking state
            *soulx_tail,  # TTS -> WS to renderer, or the legacy text relay
            assistant_aggregator,  # Add bot response to conversation history
            output_transport,  # Data channel (+ audio only on fallback)
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # ── Relay helpers ──

    async def _send_relay(msg_type: str, **fields):
        """Send a relay protocol message directly to the avatar bot."""
        pid = get_avatar_participant_id()
        if not pid:
            logger.warning("Cannot send relay {}: no avatar participant", msg_type)
            return
        payload = {"type": msg_type, "protocol": RELAY_PROTOCOL, **fields}
        try:
            await output_transport.send_message(
                _build_transport_message(payload, participant_id=pid)
            )
        except Exception:
            logger.exception("Failed to send relay message type={}", msg_type)

    # ── Scene narrator (S65 G3 — relay pipeline) ──
    # LEGACY path (SOULX_WS_URL unset): the pipeline drives no local TTS, so
    # narration text is forwarded over `avatar-relay.v1` and SoulX synthesizes it
    # in its single configured voice (per-script-avatar voice = the v0.2 punt per
    # CLAUDE.md S65). The narrator owns scene-script iteration + idempotency so
    # the relay and classic paths share the same loop.
    #
    # `relay_turn_state` is the single mutable holder for the
    # currently-open RELAY_TURN. _relay_speak lazily opens the turn on
    # the first segment so a narrate() call that yields no segments
    # leaves no orphan TURN_START on the wire; _relay_close_turn
    # finalises the turn after the closing line is sent. Keeping the
    # turn lifecycle here (and not inside SceneNarrator) lets the
    # narrator stay relay-agnostic.
    relay_turn_state: dict[str, object] = {"turn_id": None, "seq": 0}

    async def _relay_speak(text: str) -> None:
        if relay_turn_state["turn_id"] is None:
            new_turn_id = str(uuid.uuid4())
            relay_turn_state["turn_id"] = new_turn_id
            relay_turn_state["seq"] = 0
            await _send_relay(RELAY_TURN_START, turn_id=new_turn_id)
        await _send_relay(
            RELAY_TEXT,
            turn_id=relay_turn_state["turn_id"],
            seq=relay_turn_state["seq"],
            text=text,
        )
        relay_turn_state["seq"] = int(relay_turn_state["seq"]) + 1

    async def _relay_close_turn() -> None:
        turn_id = relay_turn_state["turn_id"]
        if turn_id is None:
            return
        await _send_relay(RELAY_TURN_END, turn_id=turn_id)
        relay_turn_state["turn_id"] = None
        relay_turn_state["seq"] = 0

    async def _soulx_narration_speak(text: str) -> None:
        """Narrate one segment through the pipeline's own TTS (soulx-audio.v1).

        The legacy `_relay_speak` forwards TEXT to a SoulX bot that ran its own
        TTS. The Modal renderer consumes PCM, not text, so on this path those
        app-messages go nowhere and a scripted scene is SILENT — worse, the
        run still reports success, so `_seed_session_context_once` takes the
        `spoke_script` branch and suppresses the greeting too.

        Queuing a TTSSpeakFrame is exactly what the classic pipeline does
        (`_classic_speak`); the audio then takes the identical route a
        conversational turn already takes — relay_tts -> SoulXAudioSink -> WS
        -> renderer -> synced video.

        No per-segment await (the classic path's NarrationCompletionGate):
        `plan_narration_segments` pins every relay segment to voice_id=None and
        the narrator is built with set_voice/prefetch/prime all None, so the
        gate's two jobs — voice switching and cache priming — are both inert
        here. Ordering is already guaranteed: the segments are frames in one
        pipeline. `script_complete` therefore still fires at queue-time rather
        than playout, which is the pre-existing relay behaviour (CLAUDE.md
        "Known v1 relay limitations"), not a new regression.
        """
        await task.queue_frames([TTSSpeakFrame(text=text)])

    # In soulx-audio mode the pipeline owns a real Cartesia service, so narration
    # speaks in the SAME (per-user cloned) voice as conversation. Legacy text
    # relay keeps its old behaviour untouched.
    _narration_speak = (
        _soulx_narration_speak if soulx_client is not None else _relay_speak
    )

    async def _soulx_wait_playout(timeout_s: float) -> None:
        """Relay drain-wait: script_complete must mean "the visitor finished
        HEARING it", or an auto-advance shell navigates ~1s after each scene
        loads and every script is cut off at its first syllable. The classic
        NarrationCompletionGate can't serve here — narration audio plays out on
        the RENDERER (the sink swallows it), so BotStoppedSpeakingFrame never
        fires agent-side. The renderer's playout_completed is the drain signal;
        the sink also falls back to a sent-audio estimate for renderer builds
        that predate the emission.
        """
        await soulx_sink.wait_group_playout(timeout_s)

    # Legacy text relay (SOULX_WS_URL unset) keeps queue-time emission — its
    # renderer consumed text and offered no playout signal at all.
    _relay_wait_playout = _soulx_wait_playout if soulx_client is not None else None

    # ── S79 — animated-narration cue path (§2.6, relay) ─────────────────
    # Lines whose snapshot entry carries `animation.url` are PLAYED BY THE
    # SHELL (the MP4 carries its own audio); the agent cues, times out into
    # TTS (never-block), and pauses/resumes across barge-ins. Rooms without
    # animated lines never construct a cue — the off-path is diff-zero.
    #
    # `cue_tts_dirty` tracks whether any TTS was queued since the last
    # drain: a cue must never start while a TTS'd line's tail still plays
    # (the §3 one-audible-source law), and the run-level wait_playout must
    # SKIP entirely on an all-cued run (nothing was ever queued — waiting
    # on the sink's empty watch would stall to its budget).
    cue_tts_dirty = {"spoken": False}

    async def _narration_speak_tracked(text: str) -> None:
        cue_tts_dirty["spoken"] = True
        await _narration_speak(text)

    async def _relay_wait_playout_s79(timeout_s: float) -> None:
        if _relay_wait_playout is None:
            return
        if not cue_tts_dirty["spoken"]:
            return  # all-cued run: nothing queued, nothing to drain
        await _relay_wait_playout(timeout_s)
        cue_tts_dirty["spoken"] = False

    async def _send_cue_message(payload: dict) -> None:
        await output_transport.send_message(
            OutputTransportMessageFrame(message=payload)
        )

    if soulx_sink is not None:
        narration_cue = NarrationCueController(
            send_message=_send_cue_message,
            speak_fallback=_narration_speak_tracked,
            expect_interruption=soulx_sink.expect_interruption,
            bot_is_speaking=lambda: soulx_sink._turn_id is not None,  # noqa: SLF001
        )

        async def _relay_cue_line(idx: int, seg) -> None:
            # Drain any TTS'd lines queued before this cue, then re-arm the
            # sink watch + turn group for the TTS lines that may follow —
            # the debounced close that fired during the drain ended the
            # group, and ungrouped consecutive TTS lines would re-create
            # the 2026-08-05 idle-race desync.
            if cue_tts_dirty["spoken"]:
                await _relay_wait_playout_s79(PLAYOUT_DRAIN_FALLBACK_S)
                soulx_sink.begin_narration_watch()
                soulx_sink.begin_turn_group()
            anim = seg.animation or {}
            await narration_cue.cue(
                line_index=idx,
                url=anim.get("url") or "",
                duration_seconds=float(anim.get("duration_seconds") or 0.0),
                text=seg.text,
            )
    else:
        # Legacy text relay: no renderer, no cue surface — animated lines
        # fall through to the (inert) text relay exactly as any line does.
        narration_cue = None
        _relay_cue_line = None

    narrator = SceneNarrator(
        primary_voice_id=None,
        set_voice=None,
        speak=_narration_speak_tracked,
        cue=_relay_cue_line,
    )

    # ── Auto Play Phase A — narration-run machinery (relay) ──
    # Known v1 limitations (recorded per the Phase A brief, alongside the
    # existing relay narration punts): narration speak is fire-and-forget,
    # so there is NO drain-wait (script_complete fires when the segment is
    # handed off, not at playout-complete) and NO speech-interruption
    # awareness for narration turns (the narration turn bypasses
    # AvatarRelayProcessor, whose InterruptionFrame handling only covers
    # LLM turns). "stop" is a best-effort cancel: the run task is
    # cancelled and the open narration turn is RELAY_INTERRUPTed so SoulX
    # stops rendering what it already received.
    #
    # soulx-audio.v1 narrows both punts without closing them. Narration now
    # rides the pipeline's own TTS, so an InterruptionFrame DOES flush the
    # queued narration audio (it never reached the legacy text relay at
    # all). The drain-wait still needs the classic path's
    # NarrationCompletionGate, which is deliberately not wired here — see
    # _soulx_narration_speak.

    async def _relay_interrupt_narration_turn() -> None:
        """Best-effort SoulX-side flush of the open narration turn.

        Clears the turn state FIRST (synchronously) so the cancelled
        run's finally-block close_turn no-ops instead of racing this
        with a TURN_END, then sends RELAY_INTERRUPT — the same primitive
        AvatarRelayProcessor uses for LLM-turn interruptions.

        ORDERING RULE (all call sites): cancel the superseded run's task
        BEFORE calling this. Interrupting first leaves the still-live
        old run free to reopen a fresh turn between the interrupt and
        the cancel; and cancelling first means the old task's
        shield-deferred close_turn unwinds during THIS function's await
        (while turn state is already zeroed) instead of after the new
        run opens its turn — which would close the NEW run's turn and
        eat its first segment.
        """
        turn_id = relay_turn_state["turn_id"]
        if turn_id is None:
            return
        relay_turn_state["turn_id"] = None
        relay_turn_state["seq"] = 0
        await _send_relay(RELAY_INTERRUPT, turn_id=turn_id)
        logger.info("[NARRATION] relay narration turn {} interrupted", turn_id)

    def _cancel_active_narration_run() -> None:
        """Cancel the single-slot run if active (see ordering rule above)."""
        prev = scene_narration_task["task"]
        if prev is not None and not prev.done():
            prev.cancel()

    # Phase A — session-start context seeding must survive slot
    # supersession (mirror of the classic pipeline's helper; see there
    # for the full rationale). _queue_greeting seeds on completion;
    # if it was cancelled by a superseding run, that run seeds instead.
    session_seeded = {"done": False}

    async def _seed_session_context_once(spoke_script: bool) -> None:
        if session_seeded["done"]:
            return
        session_seeded["done"] = True
        if spoke_script:
            context.add_message(
                {
                    "role": "developer",
                    "content": (
                        "You just finished presenting the scene scripts to the visitor. "
                        "They heard your full presentation. Don't repeat what you already said."
                    ),
                }
            )
        else:
            context.add_message(
                {
                    "role": "developer",
                    "content": GREETING_TRIGGER_PROMPT,
                }
            )
            await task.queue_frames([LLMRunFrame()])

    async def _narrate_and_complete(
        snap: dict, *, force: bool = False, trigger: str = "auto"
    ) -> None:
        """One relay narration run: narrate → close turn → script_complete.

        Runs as the single-slot background task (P3). The RELAY_TURN is
        closed in a shielded finally so a cancellation mid-narration
        never strands SoulX waiting for TURN_END; a cancelled run does
        NOT emit script_complete (frozen wire rule 2b).
        """
        spoke_script = False
        try:
            # Turn grouping: the renderer must see the whole script (all
            # segments + invitation) as ONE turn, or its finalize/idle race
            # injects silent video between segments (the 2026-08-05 desync).
            # The group is CLOSED by the frames themselves (debounced in the
            # sink) — NOT here: narration speak is fire-and-forget, so this
            # coroutine returns while envelopes are still in flight, and an
            # eager close would re-create the per-segment boundaries. No
            # cancellation cleanup either: a superseding run re-arms the
            # group (merging turns, which is boundary-free and safe), and
            # barge-in clears the open turn via the sink's interrupt path.
            if soulx_sink is not None:
                soulx_sink.begin_narration_watch()
                if (snap.get("current_scene") or {}).get("has_script"):
                    soulx_sink.begin_turn_group()
            # S79 — arm the cue path for this run (scene id scopes stale
            # completions; the dirty flag resets so an all-cued run skips
            # the sink drain entirely).
            cue_tts_dirty["spoken"] = False
            if narration_cue is not None:
                narration_cue.begin_run(
                    (snap.get("current_scene") or {}).get("scene_id")
                )
            try:
                spoke_script = await run_scene_narration(
                    snap,
                    narrator=narrator,
                    speak_followup=_narration_speak_tracked,
                    force=force,
                    wait_playout=_relay_wait_playout_s79,
                )
            except NarrationInterrupted:
                # Mirror of the classic call site (wire rule 2b): the visitor
                # barged in — no script_complete, no greeting.
                session_seeded["done"] = True
                logger.info(
                    "[NARRATION] run interrupted by visitor speech — "
                    "script_complete suppressed (trigger={})",
                    trigger,
                )
                return
            except Exception as exc:
                logger.warning("[NARRATION] narration failed: {!r}", exc)
            finally:
                # Shielded so the RELAY_TURN still closes even when this
                # task is cancelled mid-narration (an await in a cancelled
                # task would otherwise re-raise before the close completes).
                await asyncio.shield(_relay_close_turn())
            await output_transport.send_message(
                OutputTransportMessageFrame(
                    message=build_script_complete_payload(
                        snap, spoke_script=spoke_script, trigger=trigger
                    )
                )
            )
            logger.info(
                "[NARRATION] run complete spoke_script={} trigger={}",
                spoke_script,
                trigger,
            )
            # Seed the session context if the cancelled greeting never
            # got to (script-less flows must still wake the LLM — see
            # the classic pipeline's mirror).
            await _seed_session_context_once(spoke_script)
        except asyncio.CancelledError:
            logger.info(
                "[NARRATION] run superseded or stopped — script_complete suppressed"
            )
            raise

    # ── Greeting (waits for avatar readiness) ──

    async def _queue_greeting():
        nonlocal greeting_sent
        if greeting_sent:
            return
        if not avatar_ready_event.is_set():
            logger.info(
                "Waiting for avatar relay bot to become ready before greeting visitor"
            )
            # Bounded, for the same reason as AvatarReadyGateProcessor: an un-timed
            # wait here means the greeting never fires and the visitor is met by
            # silence with nothing surfaced.
            try:
                await asyncio.wait_for(
                    avatar_ready_event.wait(), timeout=AVATAR_READY_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Avatar not ready after {}s — greeting the visitor anyway",
                    AVATAR_READY_TIMEOUT_S,
                )
                avatar_ready_event.set()
        if greeting_sent:
            return

        greeting_sent = True

        spoke_script = False
        if scene_snapshot:
            # S65 G3+G4 — narrate + followup via the shared orchestrator,
            # wrapped in try/finally so an in-flight RELAY_TURN is
            # always closed BEFORE script_complete is emitted: otherwise
            # SoulX waits forever for TURN_END AND the shell might have
            # already auto-advanced. The orchestrator returns
            # spoke_script for the script_complete payload + the
            # developer-context message branch below.
            #
            # Phase A (A5) — this coroutine now runs in the single
            # narration slot, so a scene change can cancel it. The close
            # is shielded (mirrors _narrate_and_complete) so cancellation
            # mid-narration never strands SoulX on a dangling turn; the
            # propagating CancelledError then suppresses the
            # script_complete emit below (wire rule 2b).
            # Same turn-grouping as _narrate_and_complete: session-start
            # narration is THE path a scripted room takes, and the whole
            # script must reach the renderer as one boundary-free turn.
            if soulx_sink is not None:
                soulx_sink.begin_narration_watch()
                if (scene_snapshot.get("current_scene") or {}).get("has_script"):
                    soulx_sink.begin_turn_group()
            # S79 — same cue arming as the scene-change site.
            cue_tts_dirty["spoken"] = False
            if narration_cue is not None:
                narration_cue.begin_run(
                    (scene_snapshot.get("current_scene") or {}).get("scene_id")
                )
            try:
                spoke_script = await run_scene_narration(
                    scene_snapshot,
                    narrator=narrator,
                    speak_followup=_narration_speak_tracked,
                    wait_playout=_relay_wait_playout_s79,
                )
            except NarrationInterrupted:
                # Mirror of the classic session-start site (wire rule 2b).
                session_seeded["done"] = True
                logger.info(
                    "[NARRATION] session-start narration interrupted — "
                    "script_complete suppressed"
                )
                return
            finally:
                await asyncio.shield(_relay_close_turn())

        # S65 G4 — script_complete fires for BOTH branches; payload
        # carries {sceneIndex, hadScript} so the shell knows whether
        # to wait for narration (hadScript=True) or short-circuit
        # auto-advance immediately (hadScript=False).
        await output_transport.send_message(
            OutputTransportMessageFrame(
                message=build_script_complete_payload(
                    scene_snapshot, spoke_script=spoke_script
                )
            )
        )

        # Phase A — one-time seeding via the shared helper: if this
        # greeting gets cancelled by a superseding run before reaching
        # here, that run performs the seeding instead (script-less
        # flows must still wake the LLM).
        await _seed_session_context_once(spoke_script)

    async def _cancel_for_human_leave(reason: str, pid: str | None):
        nonlocal active_human_id, captured_audio_participant_id, greeting_sent
        logger.info(
            "Human participant left reason={} participant_id={}; cancelling relay bot",
            reason,
            pid,
        )
        active_human_id = None
        captured_audio_participant_id = None
        greeting_sent = False
        canvas_pending.cancel_all("session_end")
        # S67b — drain any in-flight capture futures (sibling registry).
        for _cap_fut in _pending_captures.values():
            if not _cap_fut.done():
                _cap_fut.cancel()
        _pending_captures.clear()
        for _ann_fut in _pending_annotates.values():
            if not _ann_fut.done():
                _ann_fut.cancel()
        _pending_annotates.clear()
        # P3 — cancel any backgrounded scene narration so it can't emit
        # script_complete into a torn-down session.
        _nar_task = scene_narration_task["task"]
        if _nar_task is not None and not _nar_task.done():
            _nar_task.cancel()
        await task.cancel()

    # ── Event handlers (complex — participant role detection) ──

    @transport.event_handler("on_app_message")
    async def on_app_message(transport, message, sender):
        nonlocal avatar_participant_id

        # S64d defensive: parse JSON-string payloads so canvas.register
        # reaches the manifest registry even if Daily delivers as string.
        # See classic handler above for full rationale.
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except (json.JSONDecodeError, ValueError):
                pass

        # Canvas Protocol routing (S64c) — handled before relay-ready check
        # so the canvas message types short-circuit cleanly.
        if isinstance(message, dict):
            msg_type = message.get("type")

            # ── S79 — shell→agent cue completion (Canvas Protocol v0.3) ──
            # The shell emits `script_complete` when a cued animation clip
            # ends; the controller resolves the active cue (stale lineIndex
            # mismatches are ignored). Never one of our own run-level
            # emissions — Daily doesn't loop app-messages back to sender.
            # Legacy shells never send this, so the branch is diff-zero.
            if msg_type == "script_complete":
                if narration_cue is not None:
                    narration_cue.on_script_complete(message)
                return

            # ── S65c Block 5 — manual visitor-action triggers ──
            # See classic pipeline for full rationale. Relay-pipeline
            # specifics: ``request_narrate`` speaks via ``_narration_speak``
            # and MUST close any open RELAY_TURN before emitting
            # ``script_complete``, otherwise SoulX waits forever for
            # TURN_END (S65 Bug #2 lesson, applied to manual replay). On
            # soulx-audio.v1 no RELAY_TURN is opened, so that close is a
            # no-op — kept because the legacy text path still needs it.
            # ``request_quiz`` is identical to classic — quiz generation
            # doesn't touch the SoulX turn lifecycle.
            if msg_type == "request_narrate":
                if not room_id:
                    logger.info("[REQUEST_NARRATE] skipped: no room_id")
                    return
                from api_client import get_scene_snapshot

                manual_snapshot = await get_scene_snapshot(room_id, api_url)
                if not manual_snapshot:
                    logger.info("[REQUEST_NARRATE] skipped: snapshot fetch failed")
                    return
                # Phase A — slot-run like the classic pipeline: a replay
                # racing an active narration run now supersedes it
                # (newest wins) instead of interleaving RELAY_TEXT into
                # its open turn. Cancel FIRST, then interrupt the turn
                # (see _relay_interrupt_narration_turn's ordering rule),
                # then start the new run.
                _cancel_active_narration_run()
                await _relay_interrupt_narration_turn()
                _start_narration_task(
                    _narrate_and_complete(manual_snapshot, force=True, trigger="manual")
                )
                logger.info("[REQUEST_NARRATE] manual replay started")
                return

            if msg_type == "autoplay_control":
                # ── Auto Play Phase A (A4) — relay half. Known v1
                # limitation (see the machinery comment above): no local
                # audio, so "stop" is a best-effort cancel + turn
                # interrupt, and completions fire at text-forwarded.
                action = message.get("action")
                if action == "stop":
                    _cancel_active_narration_run()
                    await _relay_interrupt_narration_turn()
                    logger.info(
                        "[AUTOPLAY] stop: narration cancelled, relay turn interrupted"
                    )
                elif action == "resume":
                    if not room_id:
                        logger.info("[AUTOPLAY] resume skipped: no room_id")
                        return
                    from api_client import get_scene_snapshot

                    # By tracked scene id (cursor fallback) — see the
                    # classic resume branch for the P2 rationale.
                    resume_snapshot = await get_scene_snapshot(
                        room_id,
                        api_url,
                        scene_id=session_context.get_current_scene_id() or None,
                    )
                    if not resume_snapshot:
                        logger.info("[AUTOPLAY] resume skipped: snapshot fetch failed")
                        return
                    # Cancel + interrupt BEFORE starting (ordering rule in
                    # _relay_interrupt_narration_turn): a resume racing an
                    # active run must not let the superseded run's
                    # shield-deferred close_turn eat the new run's turn.
                    _cancel_active_narration_run()
                    await _relay_interrupt_narration_turn()
                    _start_narration_task(
                        _narrate_and_complete(resume_snapshot, force=True)
                    )
                    logger.info("[AUTOPLAY] resume: re-narration started")
                else:
                    logger.warning("[AUTOPLAY] unknown action {!r} ignored", action)
                return

            if msg_type == "request_quiz":
                # See classic pipeline for the E3 silent-ignore rationale.
                if not request_quiz_ready(session_context):
                    logger.info("[REQUEST_QUIZ] skipped: session_context not ready")
                    return
                quiz_count = message.get("count", 3)
                quiz_language = message.get("language") or "en"

                async def _emit_quiz_state(state: str, err: str | None) -> None:
                    payload = {"type": "quiz_generation_state", "state": state}
                    if err:
                        payload["error"] = err
                    await send_canvas_message(payload)

                logger.info(
                    "[REQUEST_QUIZ] generating: count={} language={!r}",
                    quiz_count,
                    quiz_language,
                )
                try:
                    result = await run_quiz_generation(
                        backend_client=api_client,
                        session_context=session_context,
                        canvas_ctx=canvas_ctx,
                        count=quiz_count,
                        language=quiz_language,
                        on_state=_emit_quiz_state,
                    )
                    logger.info(
                        "[REQUEST_QUIZ] complete ok={} error={!r}",
                        result.ok,
                        result.error,
                    )
                    if result.ok:
                        # See classic pipeline for the LLM-wake rationale.
                        # In the relay pipeline the LLM still drives the
                        # text turn (SoulX renders the speech), so the same
                        # context.add_message + LLMRunFrame pattern works
                        # unchanged — no RELAY_TURN bookkeeping needed here
                        # (the LLM's output text flows through the relay
                        # forwarder downstream of the assistant aggregator).
                        context.add_message(
                            {
                                "role": "developer",
                                "content": (
                                    "A quiz has just been activated on the canvas by the visitor "
                                    "clicking the Quiz action button (not by your tool call). The "
                                    "quiz Page is already showing on screen. Here is the quiz blob "
                                    "you need to drive the session:\n\n"
                                    f"{json.dumps(result.blob)}\n\n"
                                    "Now read the first question aloud and wait for the visitor's "
                                    "answer, exactly as you would after calling "
                                    "generate_quiz_from_knowledge yourself. Use canvas_action "
                                    "verbs (submit_answer / skip_question) to record their answers "
                                    "and let the Quiz Page own the pacing."
                                ),
                            }
                        )
                        await task.queue_frames([LLMRunFrame()])
                except Exception as exc:
                    logger.exception("[REQUEST_QUIZ] unexpected failure")
                    await _emit_quiz_state(
                        "error",
                        f"unexpected: {type(exc).__name__}: {str(exc)[:200]}",
                    )
                return

            if msg_type == "canvas_capture_result":
                # S67b — see classic handler for rationale. Non-canvas ack:
                # resolve the captureId future; {status,w,h} only, bytes in the
                # backend. Early-return before the canvas dispatch.
                cid = message.get("captureId")
                fut = _pending_captures.get(cid)
                logger.info(
                    "[VISION] canvas_capture_result reached on_app_message: captureId={} "
                    "status={!r} pending_future={}",
                    cid,
                    message.get("status"),
                    "found"
                    if fut is not None
                    else "MISSING (already timed out, unknown id, or this handler never received earlier captures)",
                )
                if fut is not None and not fut.done():
                    fut.set_result(message)
                return

            if msg_type == "agent_annotate_result":
                # ── Block 8 — agent-annotate ack (see classic handler) ──
                aid = message.get("annotateId")
                ann_fut = _pending_annotates.get(aid)
                if ann_fut is not None and not ann_fut.done():
                    ann_fut.set_result(message)
                return

            if msg_type == "canvas.register":
                logger.info(
                    f"[CANVAS REGISTER] pageType={message.get('pageType')!r} version={message.get('version')!r}"
                )
                canvas_manifest.set_manifest(message)
                # S64d — rebuild the system prompt so the LLM learns the new
                # Page's verbs (see classic pipeline for full rationale).
                try:
                    new_prompt = _assemble_full_prompt(
                        base_system_prompt, canvas_manifest.current()
                    )
                    llm._settings.system_instruction = new_prompt  # type: ignore[attr-defined]
                    logger.info(
                        "[CANVAS REGISTER] system prompt rebuilt with manifest section"
                    )
                except Exception as exc:
                    logger.warning("[CANVAS REGISTER] prompt rebuild failed: {!r}", exc)
                return
            if msg_type == "canvas.stateChange":
                logger.info(
                    f"[CANVAS STATECHANGE] keys={list((message.get('semanticState') or {}).keys())}"
                )
                canvas_manifest.update_state(message.get("semanticState") or {})
                return
            if msg_type == "canvas.sceneChanged":
                # S64c — see classic pipeline's on_app_message for the
                # rationale. Single refresh trigger for both voice and
                # visitor-initiated nav; awaited inline so the refresh
                # finishes before canvas.commandResult is processed.
                scene_index = message.get("sceneIndex")
                scene_id_from_msg = message.get("sceneId")  # S66 Block 5c
                logger.info(
                    f"[CANVAS SCENECHANGED] sceneIndex={scene_index!r} sceneId={scene_id_from_msg!r}"
                )
                await refresh_agent_for_current_scene(
                    target_scene_id=scene_id_from_msg or None
                )
                return
            if msg_type == "canvas.commandResult":
                cid = message.get("commandId")
                logger.info(
                    f"[CANVAS COMMANDRESULT] commandId={cid!r} result={message.get('result')!r}"
                )
                if cid:
                    canvas_pending.resolve(cid, message.get("result") or {})
                return
            if msg_type == "canvas.commandError":
                cid = message.get("commandId")
                logger.warning(
                    f"[CANVAS COMMANDERROR] commandId={cid!r} error={message.get('error')!r}"
                )
                if cid:
                    canvas_pending.reject(cid, message.get("error") or {})
                return

        if not _is_relay_ready_message(message):
            return
        avatar_participant_id = str(sender or "").strip() or avatar_participant_id
        avatar_ready_event.set()
        await _ensure_avatar_participant_ignored(avatar_participant_id)
        logger.info(
            "Avatar relay bot is ready: participant_id={}", avatar_participant_id
        )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        nonlocal active_human_id, avatar_participant_id

        role = _participant_role(client)
        pid = _participant_id(client)
        pname = _participant_name(client)
        if avatar_participant_id and pid and pid == avatar_participant_id:
            role = "avatar_bot"
        logger.info("Participant connected role={} id={} name={}", role, pid, pname)

        if role == "avatar_bot":
            avatar_participant_id = pid or avatar_participant_id
            avatar_ready_event.set()
            await _ensure_avatar_participant_ignored(pid)
            return

        if role != "human":
            return

        active_human_id = pid or active_human_id
        await _start_human_audio_capture(active_human_id)
        if not avatar_ready_event.is_set():
            logger.info(
                "Human joined before avatar relay bot was ready; cloud bot will wait"
            )
        # Phase A (A5) — the greeting (session-start narration) runs in
        # the single narration slot so a scene change during the opening
        # narration cancels it instead of letting the old scene's script
        # keep playing. replace=False: never stomp a run that already
        # owns the slot (e.g. a raced scene change, or a second client
        # connecting mid-narration).
        _start_narration_task(_queue_greeting(), replace=False)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        nonlocal avatar_participant_id

        role = _participant_role(client)
        pid = _participant_id(client)
        if avatar_participant_id and pid and pid == avatar_participant_id:
            role = "avatar_bot"
        logger.info(
            "Participant disconnected role={} id={} name={}",
            role,
            pid,
            _participant_name(client),
        )

        if role == "avatar_bot":
            if pid and pid == avatar_participant_id:
                avatar_participant_id = None
                avatar_ready_event.clear()
            return

        if role != "human":
            return

        await _cancel_for_human_leave("client_disconnected", pid)

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        nonlocal avatar_participant_id

        role = _participant_role(participant)
        pid = _participant_id(participant)
        if avatar_participant_id and pid and pid == avatar_participant_id:
            role = "avatar_bot"
        logger.info(
            "Participant left role={} id={} name={} reason={}",
            role,
            pid,
            _participant_name(participant),
            reason,
        )

        if role == "avatar_bot":
            if pid and pid == avatar_participant_id:
                avatar_participant_id = None
                avatar_ready_event.clear()
            return

        if role != "human":
            return

        await _cancel_for_human_leave("participant_left", pid)

    # ── Run ──
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────


async def bot(runner_args: RunnerArguments):
    """Entry point called by Pipecat runner.

    Resolves the output mode from the avatar's display_mode in the scene,
    then dispatches to either the classic or relay pipeline.
    """
    body = getattr(runner_args, "body", {}) or {}
    room_id = body.get("room_id") or DEFAULT_ROOM_ID
    avatar_id = body.get("avatar_id") or DEFAULT_AVATAR_ID
    scene_id = body.get("scene_id") or DEFAULT_SCENE_ID
    flow_id = body.get("flow_id")
    api_url = body.get("hv_api_url")
    # S64e — slug threads through to SessionContext for the
    # generate_quiz_from_knowledge tool, whose backend endpoint is
    # scoped by-slug (POST /live-rooms/by-slug/{slug}/scenes/{scene_id}/
    # generate-quiz). Backend live-room start endpoint must pass `slug`
    # in the body for the quiz tool to function; until it does, the
    # quiz handler returns "no active live-room session" gracefully.
    slug = body.get("slug") or ""

    output_mode = await _resolve_output_mode(room_id, api_url)

    if isinstance(runner_args, DailyRunnerArguments):
        from pipecat.transports.daily.transport import DailyTransport

        transport = DailyTransport(
            runner_args.room_url,
            runner_args.token,
            CLOUD_BOT_NAME,
            params=_daily_params(output_mode),
        )
    else:
        transport_params = {
            "daily": lambda: _daily_params(output_mode),
            "webrtc": lambda: TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=output_mode == "cartesia",
            ),
        }

        transport = await create_transport(runner_args, transport_params)

    if output_mode == "relay_avatar":
        await run_bot_relay(
            transport,
            runner_args,
            room_id=room_id,
            avatar_id=avatar_id,
            scene_id=scene_id,
            flow_id=flow_id,
            api_url=api_url,
            slug=slug,
        )
    else:
        await run_bot_classic(
            transport,
            runner_args,
            room_id=room_id,
            avatar_id=avatar_id,
            scene_id=scene_id,
            flow_id=flow_id,
            api_url=api_url,
            slug=slug,
        )


async def _mint_avatar_token(room_url: str) -> str | None:
    """Mint an owner Daily token so the renderer can join this room as the avatar.

    The agent mints it rather than the backend so this whole feature needs ZERO backend
    changes — the renderer receives it in `session_init`. When the token source moves
    server-side later, only where it comes from changes; the renderer is unaffected.
    """
    daily_api_key = os.getenv("DAILY_API_KEY", "").strip()
    if not daily_api_key or not room_url:
        logger.warning("Cannot mint avatar token: DAILY_API_KEY or room_url missing")
        return None

    room_name = urlparse(room_url).path.strip("/")
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.daily.co/v1/meeting-tokens",
                headers={"Authorization": f"Bearer {daily_api_key}"},
                json={
                    "properties": {
                        "room_name": room_name,
                        "is_owner": True,
                        "exp": int(time.time()) + 7200,
                        "user_name": "Digital Twin Avatar",
                    }
                },
            )
            resp.raise_for_status()
            return resp.json().get("token")
    except Exception as exc:  # noqa: BLE001 — failure means fall back, not crash
        logger.error("Avatar token mint failed ({}): {}", type(exc).__name__, exc)
        return None


def _daily_params(output_mode: str = "cartesia"):
    """Lazy import DailyParams so the daily extra isn't required for local dev."""
    from pipecat.transports.daily.transport import DailyParams

    if output_mode == "relay_avatar":
        return DailyParams(
            audio_in_enabled=True,
            audio_in_user_tracks=True,
            audio_in_stream_on_start=False,
            # audio_out was False because SoulX owned all audio. It must be True now so
            # the VOICE-ONLY FALLBACK can actually speak when the renderer is
            # unavailable. Nothing is published in the happy path: SoulXAudioSink
            # swallows audio frames while the renderer is healthy (it publishes them
            # itself, already synced to the video) and only lets them through on
            # fallback.
            audio_out_enabled=True,
            video_out_enabled=False,
        )

    return DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    )


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
