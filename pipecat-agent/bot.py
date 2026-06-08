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
from context.prompt_builder import (
    render_agent_playbook_section,
    render_canvas_page_section,
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
    NarrationCompletionGate,
    SceneNarrator,
    build_script_complete_payload,
    run_scene_narration,
)
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
CLOUD_OUTPUT_MODE = os.getenv("CLOUD_OUTPUT_MODE", "cartesia").strip().lower() or "cartesia"
if CLOUD_OUTPUT_MODE not in VALID_OUTPUT_MODES:
    logger.warning(
        "Unknown CLOUD_OUTPUT_MODE={}, falling back to cartesia",
        CLOUD_OUTPUT_MODE,
    )
    CLOUD_OUTPUT_MODE = "cartesia"

# Bot names
CLOUD_BOT_NAME = os.getenv("CLOUD_BOT_NAME", "Human Virtual Cloud").strip() or "Human Virtual Cloud"
AVATAR_BOT_NAME = os.getenv("SOULX_AVATAR_BOT_NAME", "Digital Twin Avatar").strip() or "Digital Twin Avatar"


# ──────────────────────────────────────────────────────────────────────
# Participant helpers (relay pipeline)
# ──────────────────────────────────────────────────────────────────────

def _participant_id(participant: object) -> str:
    if not isinstance(participant, dict):
        return ""
    value = participant.get("id") or participant.get("participant_id") or participant.get("participantId")
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


def _build_transport_message(message: dict[str, object], participant_id: str | None = None):
    if participant_id:
        try:
            from pipecat.transports.daily.transport import DailyOutputTransportMessageFrame
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
    """Forward user STT and bot text updates over the transport data channel."""

    def __init__(self, transport: BaseTransport):
        super().__init__()
        self._transport = transport

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text:
            await self._send_transcript("user", frame.text)

        if (
            isinstance(frame, TextFrame)
            and not isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame))
            and frame.text
        ):
            await self._send_transcript("avatar", frame.text)

        await self.push_frame(frame, direction)

    async def _send_transcript(self, speaker: str, text: str):
        try:
            payload = {
                "type": "transcript",
                "speaker": speaker,
                "text": text,
            }
            await self._transport.send_message(OutputTransportMessageFrame(message=payload))
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
        elif isinstance(frame, (LLMFullResponseEndFrame, InterruptionFrame)) and self._is_speaking:
            self._is_speaking = False
            await self._send_state(False)

        await self.push_frame(frame, direction)

    async def _send_state(self, is_speaking: bool):
        try:
            payload = {
                "type": "speaking_state",
                "isSpeaking": is_speaking,
            }
            await self._transport.send_message(OutputTransportMessageFrame(message=payload))
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
        elif isinstance(frame, (LLMFullResponseEndFrame, InterruptionFrame)) and self._is_thinking:
            self._is_thinking = False
            await self._send_state(False)

        await self.push_frame(frame, direction)

    async def _send_state(self, thinking: bool):
        try:
            payload = {
                "type": "llm_thinking",
                "thinking": thinking,
            }
            await self._transport.send_message(OutputTransportMessageFrame(message=payload))
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
            avatar_participant_id = str(self._avatar_participant_id_getter() or "").strip()
            local_participant_id = str(self._local_participant_id_getter() or "").strip()

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
    """Blocks relay-mode LLM traffic until the avatar bot reports ready."""

    def __init__(self, ready_event: asyncio.Event):
        super().__init__()
        self._ready_event = ready_event
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
            await self._ready_event.wait()
            logger.info("Avatar relay bot ready; resuming queued pipeline traffic")
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
            logger.exception("Failed to send avatar relay message type={}", message_type)


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

def _assemble_full_prompt(base: str, manifest: dict | None) -> str:
    """Concatenate the base persona prompt, CANVAS PAGE, and AGENT PLAYBOOK.

    Single helper for the three sites that rebuild the prompt (session
    start, canvas.register, canvas.sceneChanged). The CANVAS PAGE section
    is driven by the active Page's manifest; AGENT PLAYBOOK is a stable
    string that documents cross-tool sequences the agent should follow
    (currently the quiz flow — S64e).
    """
    return "\n\n".join([
        base,
        render_canvas_page_section(manifest),
        render_agent_playbook_section(),
    ])


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
            logger.warning("Anthropic LLM service did not accept system_instruction setter")
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
    system_prompt = await build_system_prompt(
        room_id=room_id,
        avatar_id=avatar_id,
        scene_id=scene_id,
        api_url=api_url,
        aliases_out=element_alias_map,
        flow_cache=flow_knowledge_cache,
    )
    logger.info(f"System prompt length: {len(system_prompt)} chars")

    # ── Fetch avatar config for TTS voice ──
    avatar_config = None
    if room_id:
        from api_client import get_avatar_config
        avatar_config = await get_avatar_config(room_id, api_url)
        if avatar_config:
            logger.info(f"Avatar config: name={avatar_config.get('name')}, voiceModelId={avatar_config.get('voiceModelId')}")
        else:
            logger.info("No avatar config available — using default voice")

    # ── Fetch canvas image for vision ──
    scene_image_b64 = None
    if room_id:
        from api_client import get_scene_image_base64
        scene_image_b64 = await get_scene_image_base64(room_id, api_url)
        if scene_image_b64:
            logger.info("Fetched scene canvas image ({} chars base64)", len(scene_image_b64))
        else:
            logger.info("No scene image available; vision disabled for this session")

    # ── Fetch scene snapshot for scripts ──
    # S65 (Option B) — snapshot is nested under {live_room, flow_state,
    # current_scene, knowledge, survey}. Pull the per-scene block once
    # so subsequent reads stay terse.
    scene_snapshot = None
    if room_id:
        from api_client import get_scene_snapshot
        scene_snapshot = await get_scene_snapshot(room_id, api_url)
        if scene_snapshot:
            scripts_len = len(
                ((scene_snapshot.get("current_scene") or {}).get("scripts")) or []
            )
            logger.info("Scene snapshot loaded (scripts={})", scripts_len)

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
            await output_transport.send_message(OutputTransportMessageFrame(message=payload))
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
            await send_canvas_message({
                "type": "request_canvas_capture",
                "captureId": capture_id,
                "hint": hint,
                "maxDim": VISION_MAX_DIM,  # advisory; the shell owns the encode
            })
            result = await asyncio.wait_for(
                fut, timeout=VISION_CAPTURE_TIMEOUT_MS / 1000
            )
            logger.info(
                "[VISION] capture {} resolved status={!r}",
                capture_id, (result or {}).get("status"),
            )
            return capture_id, result
        except asyncio.TimeoutError:
            logger.warning(
                "[VISION] capture {} timed out after {}ms",
                capture_id, VISION_CAPTURE_TIMEOUT_MS,
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
        settings=CartesiaTTSService.Settings(
            voice=voice_id,
            model=NARRATION_TTS_MODEL_ID,
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
        LLM_CANVAS_PROVIDER, eager_hook.__class__.__name__,
    )
    # NOTE: eager_hook is instantiated but not yet wired into the LLM
    # service's streaming loop — that's a per-provider integration that
    # depends on Pipecat's hook surface and lands in a follow-up. Until
    # then, all canvas tool calls go through the regular tool-handler
    # path (no eager-dispatch latency win, but correctness is identical).

    # ── Conversation context ──
    initial_messages = []
    if scene_image_b64:
        from scene_context import build_vision_message
        initial_messages.append(build_vision_message(scene_image_b64))

    context = LLMContext(
        messages=initial_messages if initial_messages else None,
        tools=canvas_tools,
    )

    # S66 Block 5a — bridge vision_tracker + per-session state to the
    # canvas_analyze handler. The closure captures `context` (LLMContext),
    # `vision_tracker`, and `session_context` so it can fetch the current
    # scene's image and add it to context on demand. No-op when the
    # tracker already covers the active scene (cache hit).
    async def _ensure_vision_for_active_scene(question: str = "") -> None:
        # S67b — Design B: prefer a live capture of the visitor's annotated
        # canvas (reasoned by the dedicated Gemini client), fall back to the
        # base-scene Pillow PNG with a blind-spot note. The orchestration is
        # the testable run_vision_query core; this closure just injects its
        # result (a developer message) into the LLM context. Always fresh per
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
        if msg:
            context.add_message(msg)

    canvas_ctx.ensure_vision = _ensure_vision_for_active_scene

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
    # No verb-specific logic and no api_navigate call — the frontend has
    # already advanced the backend cursor by the time this fires, so
    # build_system_prompt's internal /scene-snapshot fetch returns the
    # post-nav scene. aliases_out updates canvas_ctx.element_alias_map
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
        # fetch, which is also correct.
        new_base = await build_system_prompt(
            room_id=room_id, avatar_id=avatar_id, scene_id=scene_id, api_url=api_url,
            aliases_out=canvas_ctx.element_alias_map,
            flow_cache=flow_knowledge_cache,
            snapshot_scene_id=target_scene_id or None,
        )
        # S64d — preserve CANVAS PAGE section across scene refreshes. Otherwise
        # the section is dropped on every navigation and the LLM loses its
        # verb list. Also update base_system_prompt so subsequent
        # canvas.register rebuilds use the post-navigation base.
        base_system_prompt = new_base
        new_prompt = _assemble_full_prompt(new_base, canvas_manifest.current())
        try:
            llm._settings.system_instruction = new_prompt  # type: ignore[attr-defined]
            logger.info("[CANVAS SCENECHANGED] system prompt refreshed ({} chars)", len(new_prompt))
        except Exception:
            logger.warning("[CANVAS SCENECHANGED] could not set system_instruction on llm service")
        t_prompt = time.monotonic()

        # S64e — refresh session_context.current_scene_id from the
        # post-nav snapshot so generate_quiz_from_knowledge targets the
        # scene the visitor is actually looking at. The snapshot was
        # fetched inside build_system_prompt; we re-fetch here (cheap,
        # cached on the backend) rather than threading the snapshot
        # through every prompt-builder caller.
        # S65 (Option B) — scene_id nested under current_scene.
        from api_client import get_scene_snapshot, get_scene_image_base64
        # S66 Block 5c — second fetch also takes the broadcast scene_id
        # so both refresh paths agree on which scene is "current".
        fresh_snapshot = await get_scene_snapshot(
            room_id, api_url, scene_id=target_scene_id or None
        )
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
        if VISION_REFRESH_MODE == "eager":
            new_image = await get_scene_image_base64(room_id, api_url)
            if new_image:
                from scene_context import build_vision_message
                context.add_message(build_vision_message(new_image))
                vision_tracker.mark_loaded(session_context.get_current_scene_id())
                logger.info("[CANVAS SCENECHANGED] vision context refreshed with new scene image")
            else:
                logger.warning("[CANVAS SCENECHANGED] could not fetch new scene image after navigation")
        else:
            vision_tracker.invalidate()
            logger.info(
                "[CANVAS SCENECHANGED] vision-refresh mode=lazy — scene image will be fetched on next canvas_analyze"
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
        if fresh_snapshot:
            narration_gate.cancel_all("scene_change")
            try:
                spoke_script = await run_scene_narration(
                    fresh_snapshot,
                    narrator=narrator,
                    speak_followup=_classic_speak,
                )
            except Exception as exc:
                logger.warning(
                    "[CANVAS SCENECHANGED] narration failed: {!r}", exc
                )
                spoke_script = False
            await output_transport.send_message(
                OutputTransportMessageFrame(
                    message=build_script_complete_payload(
                        fresh_snapshot, spoke_script=spoke_script
                    )
                )
            )
            logger.info(
                "[CANVAS SCENECHANGED] narration complete spoke_script={}",
                spoke_script,
            )

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
    pipeline = Pipeline([
        transport.input(),       # Visitor's microphone audio (WebRTC)
        stt,                     # Deepgram: speech -> text
        user_transcript_fwd,     # Forward user STT transcripts to frontend
        user_aggregator,         # Add user message to conversation history
        llm,                     # OpenAI: generate response
        thinking_notifier,       # Notify frontend of LLM thinking state
        avatar_transcript_fwd,   # Forward avatar LLM text to frontend
        speaking_notifier,       # Notify frontend of speaking state
        tts,                     # Cartesia: response -> speech audio
        narration_gate,          # S65 G3: observe TTSStoppedFrame for narration
        output_transport,        # Send audio back to visitor (WebRTC)
        assistant_aggregator,    # Add bot response to conversation history
    ])

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

    async def _classic_speak(text: str) -> None:
        # Register the completion future BEFORE queuing the speak — a
        # short utterance can fire TTSStoppedFrame before we register
        # otherwise (race), leaving us waiting forever. 30 s upper
        # bound is generous for any single narration segment; on
        # timeout we log and proceed so the rest of the narration plan
        # still runs.
        fut = narration_gate.expect_next_stop()
        await task.queue_frames([TTSSpeakFrame(text=text)])
        try:
            await asyncio.wait_for(fut, timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(
                "[NARRATION] segment TTSStoppedFrame timeout — continuing"
            )
        except asyncio.CancelledError:
            logger.warning("[NARRATION] segment future cancelled")
            raise

    # Block 13 — narration cache closures. Shared dict between prefetch
    # (fills it once at the top of each per-scene narration) and prime
    # (per-segment consume; primes the TTS service for the very next
    # run_tts call). On any HTTP / decode failure we fall back to live —
    # the prime returns False, the narrator runs the normal voice-switch
    # + live synthesis path for that segment.
    _narration_cache: dict[str, CachedSegment] = {}

    async def _narration_prefetch(plan):
        _narration_cache.clear()
        targets = [
            seg for seg in plan
            if seg.id and seg.audio and seg.audio.get("url")
        ]
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
                            seg.id, r.status_code,
                        )
                        continue
                    sr = int(seg.audio.get("sample_rate") or NARRATION_AUDIO_SAMPLE_RATE)
                    if sr != NARRATION_AUDIO_SAMPLE_RATE:
                        logger.warning(
                            "[NARRATION] prefetch {} sr={} != configured {}, skip",
                            seg.id, sr, NARRATION_AUDIO_SAMPLE_RATE,
                        )
                        continue
                    _narration_cache[seg.id] = CachedSegment(
                        pcm=r.content,
                        sample_rate=sr,
                        num_channels=NARRATION_AUDIO_NUM_CHANNELS,
                    )
                except Exception as exc:
                    logger.warning(
                        "[NARRATION] prefetch {} failed: {!r}", seg.id, exc
                    )

    def _narration_prime(seg) -> bool:
        cached = _narration_cache.get(seg.id) if seg.id else None
        tts.prime_cached(cached)
        return cached is not None

    narrator = SceneNarrator(
        primary_voice_id=primary_voice_id_classic,
        set_voice=_classic_set_voice,
        speak=_classic_speak,
        prefetch=_narration_prefetch,
        prime=_narration_prime,
    )

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
            narration_gate.cancel_all("manual_replay")
            spoke_script = False
            try:
                spoke_script = await run_scene_narration(
                    manual_snapshot,
                    narrator=narrator,
                    speak_followup=_classic_speak,
                    force=True,
                )
            except Exception as exc:
                logger.warning("[REQUEST_NARRATE] narration failed: {!r}", exc)
            await output_transport.send_message(
                OutputTransportMessageFrame(
                    message=build_script_complete_payload(
                        manual_snapshot,
                        spoke_script=spoke_script,
                        trigger="manual",
                    )
                )
            )
            logger.info(
                "[REQUEST_NARRATE] manual replay complete spoke_script={}",
                spoke_script,
            )
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
                quiz_count, quiz_language,
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
                    result.ok, result.error,
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
                    context.add_message({
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
                    })
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
                cid, message.get("status"),
                "found" if fut is not None
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
            logger.info(f"[CANVAS REGISTER] pageType={message.get('pageType')!r} version={message.get('version')!r}")
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
                logger.info("[CANVAS REGISTER] system prompt rebuilt with manifest section")
            except Exception as exc:
                logger.warning("[CANVAS REGISTER] prompt rebuild failed: {!r}", exc)
        elif msg_type == "canvas.stateChange":
            logger.info(f"[CANVAS STATECHANGE] keys={list((message.get('semanticState') or {}).keys())}")
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
            logger.info(f"[CANVAS COMMANDRESULT] commandId={cid!r} result={message.get('result')!r}")
            if cid:
                canvas_pending.resolve(cid, message.get("result") or {})
        elif msg_type == "canvas.commandError":
            cid = message.get("commandId")
            logger.warning(f"[CANVAS COMMANDERROR] commandId={cid!r} error={message.get('error')!r}")
            if cid:
                canvas_pending.reject(cid, message.get("error") or {})

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Visitor connected to live room")

        # S65 G3+G4 — narrate scene scripts via SceneNarrator (per-segment
        # voice switching, idempotent per scene_id), then speak the
        # localized invitation OR transition_cue, then emit
        # script_complete LAST. run_scene_narration composes the
        # narrate + followup steps; we emit script_complete separately
        # because in the relay pipeline the RELAY_TURN must close
        # between followup and script_complete — keeping the emit at
        # the call site lets each pipeline own its turn lifecycle.
        spoke_script = await run_scene_narration(
            scene_snapshot,
            narrator=narrator,
            speak_followup=_classic_speak,
        )
        await output_transport.send_message(
            OutputTransportMessageFrame(
                message=build_script_complete_payload(
                    scene_snapshot, spoke_script=spoke_script
                )
            )
        )

        if spoke_script:
            context.add_message({
                "role": "developer",
                "content": (
                    "You just finished presenting the scene scripts to the visitor. "
                    "They heard your full presentation. Don't repeat what you already said."
                ),
            })
        else:
            context.add_message({
                "role": "developer",
                "content": GREETING_TRIGGER_PROMPT,
            })
            await task.queue_frames([LLMRunFrame()])

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
    system_prompt = await build_system_prompt(
        room_id=room_id,
        avatar_id=avatar_id,
        scene_id=scene_id,
        api_url=api_url,
        aliases_out=element_alias_map,
        flow_cache=flow_knowledge_cache,
    )
    logger.info(f"System prompt length: {len(system_prompt)} chars")

    # ── Fetch canvas image for vision ──
    scene_image_b64 = None
    if room_id:
        from api_client import get_scene_image_base64
        scene_image_b64 = await get_scene_image_base64(room_id, api_url)
        if scene_image_b64:
            logger.info("Fetched scene canvas image ({} chars base64)", len(scene_image_b64))
        else:
            logger.info("No scene image available; vision disabled for this session")

    # ── Fetch scene snapshot for scripts ──
    # S65 (Option B) — scripts nested under current_scene.
    scene_snapshot = None
    if room_id:
        from api_client import get_scene_snapshot
        scene_snapshot = await get_scene_snapshot(room_id, api_url)
        if scene_snapshot:
            scripts_len = len(
                ((scene_snapshot.get("current_scene") or {}).get("scripts")) or []
            )
            logger.info("Scene snapshot loaded (scripts={})", scripts_len)

    # ── Canvas Protocol substrate (S64c) ──
    output_transport = transport.output()
    canvas_manifest = CanvasManifestRegistry()
    canvas_pending = PendingCommandRegistry()

    async def send_canvas_message(payload: dict) -> None:
        """Send a Canvas Protocol Daily app-message to the frontend."""
        try:
            await output_transport.send_message(OutputTransportMessageFrame(message=payload))
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
            await send_canvas_message({
                "type": "request_canvas_capture",
                "captureId": capture_id,
                "hint": hint,
                "maxDim": VISION_MAX_DIM,  # advisory; the shell owns the encode
            })
            result = await asyncio.wait_for(
                fut, timeout=VISION_CAPTURE_TIMEOUT_MS / 1000
            )
            logger.info(
                "[VISION] capture {} resolved status={!r}",
                capture_id, (result or {}).get("status"),
            )
            return capture_id, result
        except asyncio.TimeoutError:
            logger.warning(
                "[VISION] capture {} timed out after {}ms",
                capture_id, VISION_CAPTURE_TIMEOUT_MS,
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
        LLM_CANVAS_PROVIDER, eager_hook.__class__.__name__,
    )
    # NOTE: see run_bot_classic — eager_hook is instantiated but not wired
    # into the streaming loop yet.

    # ── Conversation context ──
    initial_messages = []
    if scene_image_b64:
        from scene_context import build_vision_message
        initial_messages.append(build_vision_message(scene_image_b64))

    context = LLMContext(
        messages=initial_messages if initial_messages else None,
        tools=canvas_tools,
    )

    # S66 Block 5a — mirror run_bot_classic. See there for rationale.
    async def _ensure_vision_for_active_scene(question: str = "") -> None:
        # S67b — Design B: prefer a live capture of the visitor's annotated
        # canvas (reasoned by the dedicated Gemini client), fall back to the
        # base-scene Pillow PNG with a blind-spot note. The orchestration is
        # the testable run_vision_query core; this closure just injects its
        # result (a developer message) into the LLM context. Always fresh per
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
        if msg:
            context.add_message(msg)

    canvas_ctx.ensure_vision = _ensure_vision_for_active_scene

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
    # No verb-specific logic and no api_navigate call — the frontend has
    # already advanced the backend cursor by the time this fires, so
    # build_system_prompt's internal /scene-snapshot fetch returns the
    # post-nav scene. aliases_out updates canvas_ctx.element_alias_map
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
        # fetch, which is also correct.
        new_base = await build_system_prompt(
            room_id=room_id, avatar_id=avatar_id, scene_id=scene_id, api_url=api_url,
            aliases_out=canvas_ctx.element_alias_map,
            flow_cache=flow_knowledge_cache,
            snapshot_scene_id=target_scene_id or None,
        )
        # S64d — preserve CANVAS PAGE section across scene refreshes. Otherwise
        # the section is dropped on every navigation and the LLM loses its
        # verb list. Also update base_system_prompt so subsequent
        # canvas.register rebuilds use the post-navigation base.
        base_system_prompt = new_base
        new_prompt = _assemble_full_prompt(new_base, canvas_manifest.current())
        try:
            llm._settings.system_instruction = new_prompt  # type: ignore[attr-defined]
            logger.info("[CANVAS SCENECHANGED] system prompt refreshed ({} chars)", len(new_prompt))
        except Exception:
            logger.warning("[CANVAS SCENECHANGED] could not set system_instruction on llm service")
        t_prompt = time.monotonic()

        # S64e — refresh session_context.current_scene_id from the
        # post-nav snapshot so generate_quiz_from_knowledge targets the
        # scene the visitor is actually looking at. The snapshot was
        # fetched inside build_system_prompt; we re-fetch here (cheap,
        # cached on the backend) rather than threading the snapshot
        # through every prompt-builder caller.
        # S65 (Option B) — scene_id nested under current_scene.
        from api_client import get_scene_snapshot, get_scene_image_base64
        # S66 Block 5c — second fetch also takes the broadcast scene_id
        # so both refresh paths agree on which scene is "current".
        fresh_snapshot = await get_scene_snapshot(
            room_id, api_url, scene_id=target_scene_id or None
        )
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
        if VISION_REFRESH_MODE == "eager":
            new_image = await get_scene_image_base64(room_id, api_url)
            if new_image:
                from scene_context import build_vision_message
                context.add_message(build_vision_message(new_image))
                vision_tracker.mark_loaded(session_context.get_current_scene_id())
                logger.info("[CANVAS SCENECHANGED] vision context refreshed with new scene image")
            else:
                logger.warning("[CANVAS SCENECHANGED] could not fetch new scene image after navigation")
        else:
            vision_tracker.invalidate()
            logger.info(
                "[CANVAS SCENECHANGED] vision-refresh mode=lazy — scene image will be fetched on next canvas_analyze"
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
        if fresh_snapshot:
            spoke_script = False
            try:
                spoke_script = await run_scene_narration(
                    fresh_snapshot,
                    narrator=narrator,
                    speak_followup=_relay_speak,
                )
            except Exception as exc:
                logger.warning(
                    "[CANVAS SCENECHANGED] narration failed: {!r}", exc
                )
            finally:
                await _relay_close_turn()
            await output_transport.send_message(
                OutputTransportMessageFrame(
                    message=build_script_complete_payload(
                        fresh_snapshot, spoke_script=spoke_script
                    )
                )
            )
            logger.info(
                "[CANVAS SCENECHANGED] narration complete spoke_script={}",
                spoke_script,
            )

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

        logger.info("Ensured SoulX avatar participant is ignored participant_id={}", pid)

    async def _start_human_audio_capture(pid: str | None):
        nonlocal captured_audio_participant_id
        if not pid or pid == avatar_participant_id:
            return
        if captured_audio_participant_id == pid:
            return

        _remove_pending_audio_capture(avatar_participant_id)

        capture_participant_audio = getattr(transport, "capture_participant_audio", None)
        if callable(capture_participant_audio):
            await capture_participant_audio(pid, "microphone")

        input_transport = getattr(transport, "_input", None)
        start_audio_in_streaming = getattr(input_transport, "start_audio_in_streaming", None)
        if callable(start_audio_in_streaming) and not getattr(input_transport, "_streaming_started", False):
            await start_audio_in_streaming()
            logger.info("Started Daily audio input streaming")

        captured_audio_participant_id = pid
        logger.info("Started human-only audio capture for participant_id={}", pid)

    # ── Relay-mode processors ──
    human_audio_filter = HumanOnlyAudioInputFilter(
        get_avatar_participant_id,
        get_local_participant_id,
    )
    avatar_ready_gate = AvatarReadyGateProcessor(avatar_ready_event)
    relay_processor = AvatarRelayProcessor(output_transport, get_avatar_participant_id)

    # ── Pipeline ──
    pipeline = Pipeline([
        transport.input(),       # Participant audio (per-track)
        human_audio_filter,      # Drop avatar/local bot audio
        stt,                     # Deepgram: speech -> text
        user_transcript_fwd,     # Forward user STT transcripts to frontend
        user_aggregator,         # Add user message to conversation history
        avatar_ready_gate,       # Block until avatar bot is ready
        llm,                     # OpenAI: generate response
        thinking_notifier,       # Notify frontend of LLM thinking state
        avatar_transcript_fwd,   # Forward avatar LLM text to frontend
        speaking_notifier,       # Notify frontend of speaking state
        relay_processor,         # Relay text to SoulX avatar bot
        assistant_aggregator,    # Add bot response to conversation history
        output_transport,        # Data channel (no audio out)
    ])

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

    # ── Scene narrator (S65 G3 — relay pipeline, primary SoulX voice only) ──
    # The relay pipeline doesn't drive a local TTS, so per-segment voice
    # switching is not exposed — SoulX renders narration in its single
    # configured voice (the script-avatar voice clone is a v0.2 punt per
    # CLAUDE.md S65). The narrator still owns scene-script iteration +
    # idempotency so the relay and classic paths share the same loop.
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

    narrator = SceneNarrator(
        primary_voice_id=None,
        set_voice=None,
        speak=_relay_speak,
    )

    # ── Greeting (waits for avatar readiness) ──

    async def _queue_greeting():
        nonlocal greeting_sent
        if greeting_sent:
            return
        if not avatar_ready_event.is_set():
            logger.info("Waiting for avatar relay bot to become ready before greeting visitor")
            await avatar_ready_event.wait()
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
            try:
                spoke_script = await run_scene_narration(
                    scene_snapshot,
                    narrator=narrator,
                    speak_followup=_relay_speak,
                )
            finally:
                await _relay_close_turn()

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

        if spoke_script:
            context.add_message({
                "role": "developer",
                "content": (
                    "You just finished presenting the scene scripts to the visitor. "
                    "They heard your full presentation. Don't repeat what you already said."
                ),
            })
        else:
            context.add_message({
                "role": "developer",
                "content": GREETING_TRIGGER_PROMPT,
            })
            await task.queue_frames([LLMRunFrame()])

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

            # ── S65c Block 5 — manual visitor-action triggers ──
            # See classic pipeline for full rationale. Relay-pipeline
            # specifics: ``request_narrate`` uses ``_relay_speak`` and
            # MUST close the open RELAY_TURN before emitting
            # ``script_complete``, otherwise SoulX waits forever for
            # TURN_END (S65 Bug #2 lesson, applied to manual replay).
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
                spoke_script = False
                try:
                    spoke_script = await run_scene_narration(
                        manual_snapshot,
                        narrator=narrator,
                        speak_followup=_relay_speak,
                        force=True,
                    )
                except Exception as exc:
                    logger.warning("[REQUEST_NARRATE] narration failed: {!r}", exc)
                finally:
                    await _relay_close_turn()
                await output_transport.send_message(
                    OutputTransportMessageFrame(
                        message=build_script_complete_payload(
                            manual_snapshot,
                            spoke_script=spoke_script,
                            trigger="manual",
                        )
                    )
                )
                logger.info(
                    "[REQUEST_NARRATE] manual replay complete spoke_script={}",
                    spoke_script,
                )
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
                    quiz_count, quiz_language,
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
                        result.ok, result.error,
                    )
                    if result.ok:
                        # See classic pipeline for the LLM-wake rationale.
                        # In the relay pipeline the LLM still drives the
                        # text turn (SoulX renders the speech), so the same
                        # context.add_message + LLMRunFrame pattern works
                        # unchanged — no RELAY_TURN bookkeeping needed here
                        # (the LLM's output text flows through the relay
                        # forwarder downstream of the assistant aggregator).
                        context.add_message({
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
                        })
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
                    cid, message.get("status"),
                    "found" if fut is not None
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
                logger.info(f"[CANVAS REGISTER] pageType={message.get('pageType')!r} version={message.get('version')!r}")
                canvas_manifest.set_manifest(message)
                # S64d — rebuild the system prompt so the LLM learns the new
                # Page's verbs (see classic pipeline for full rationale).
                try:
                    new_prompt = _assemble_full_prompt(
                        base_system_prompt, canvas_manifest.current()
                    )
                    llm._settings.system_instruction = new_prompt  # type: ignore[attr-defined]
                    logger.info("[CANVAS REGISTER] system prompt rebuilt with manifest section")
                except Exception as exc:
                    logger.warning("[CANVAS REGISTER] prompt rebuild failed: {!r}", exc)
                return
            if msg_type == "canvas.stateChange":
                logger.info(f"[CANVAS STATECHANGE] keys={list((message.get('semanticState') or {}).keys())}")
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
                logger.info(f"[CANVAS COMMANDRESULT] commandId={cid!r} result={message.get('result')!r}")
                if cid:
                    canvas_pending.resolve(cid, message.get("result") or {})
                return
            if msg_type == "canvas.commandError":
                cid = message.get("commandId")
                logger.warning(f"[CANVAS COMMANDERROR] commandId={cid!r} error={message.get('error')!r}")
                if cid:
                    canvas_pending.reject(cid, message.get("error") or {})
                return

        if not _is_relay_ready_message(message):
            return
        avatar_participant_id = str(sender or "").strip() or avatar_participant_id
        avatar_ready_event.set()
        await _ensure_avatar_participant_ignored(avatar_participant_id)
        logger.info("Avatar relay bot is ready: participant_id={}", avatar_participant_id)

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
            logger.info("Human joined before avatar relay bot was ready; cloud bot will wait")
        asyncio.create_task(_queue_greeting())

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


def _daily_params(output_mode: str = "cartesia"):
    """Lazy import DailyParams so the daily extra isn't required for local dev."""
    from pipecat.transports.daily.transport import DailyParams

    if output_mode == "relay_avatar":
        return DailyParams(
            audio_in_enabled=True,
            audio_in_user_tracks=True,
            audio_in_stream_on_start=False,
            audio_out_enabled=False,
            video_out_enabled=False,
        )

    return DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    )


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
