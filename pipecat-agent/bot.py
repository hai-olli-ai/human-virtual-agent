"""Human Virtual Pipecat voice agent.

Pipeline: Mic → VAD → STT (Deepgram) → LLM (OpenAI) → TTS (Cartesia) → Speaker

Session 47: Canvas action tools — agent can highlight, draw arrows, annotate,
navigate scenes, and clear overlays via LLM function calling.
- 5 tools registered via FunctionSchema + ToolsSchema + llm.register_function()
- Element resolution maps LLM descriptions to canvas coordinates
- Actions dispatched as data channel messages to the frontend

Local dev:  python bot.py  → opens http://localhost:7860/client
Production: Deployed to Pipecat Cloud with DailyTransport

Default mode:
    Mic -> STT (Deepgram) -> LLM (OpenAI) -> TTS (Cartesia) -> Speaker

Relay mode:
    Mic -> STT (Deepgram) -> LLM (OpenAI) -> raw TTS-bound text relay over Daily app messages
    The remote SoulX/TTS bot handles speech and avatar rendering.
"""

import asyncio
import os
import uuid

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
    OPENAI_API_KEY,
    LLM_MODEL,
    DEFAULT_AVATAR_ID,
    DEFAULT_ROOM_ID,
    DEFAULT_SCENE_ID,
)
from canvas_actions import create_canvas_action_handlers, get_canvas_tools
from persona import build_system_prompt

GREETING_TRIGGER_PROMPT = (
    "A visitor just joined. Greet them warmly and briefly introduce yourself and what you can do. "
    "Do NOT use any canvas action tools for this greeting - just speak."
)
RELAY_PROTOCOL = "avatar-relay.v1"
RELAY_READY = "avatar_relay.ready"
RELAY_TURN_START = "avatar_relay.turn_start"
RELAY_TEXT = "avatar_relay.text"
RELAY_SENTENCE = "avatar_relay.sentence"
RELAY_TURN_END = "avatar_relay.turn_end"
RELAY_INTERRUPT = "avatar_relay.interrupt"
VALID_OUTPUT_MODES = {"cartesia", "relay_avatar"}
CLOUD_OUTPUT_MODE = os.getenv("CLOUD_OUTPUT_MODE", "cartesia").strip().lower() or "cartesia"
if CLOUD_OUTPUT_MODE not in VALID_OUTPUT_MODES:
    logger.warning(
        "Unknown CLOUD_OUTPUT_MODE={}, falling back to cartesia",
        CLOUD_OUTPUT_MODE,
    )
    CLOUD_OUTPUT_MODE = "cartesia"
CLOUD_BOT_NAME = os.getenv("CLOUD_BOT_NAME", "Human Virtual Cloud").strip() or "Human Virtual Cloud"
AVATAR_BOT_NAME = os.getenv("SOULX_AVATAR_BOT_NAME", "Digital Twin Avatar").strip() or "Digital Twin Avatar"


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
    participant_id = _participant_id(participant)
    if participant_id == "local":
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


async def run_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    room_id: str = "",
    avatar_id: str = "",
    scene_id: str = "",
    flow_id: str | None = None,
    api_url: str | None = None,
):
    logger.info(
        "Starting Human Virtual voice agent (room={}, avatar={}, output_mode={})",
        room_id,
        avatar_id,
        CLOUD_OUTPUT_MODE,
    )

    system_prompt = await build_system_prompt(
        room_id=room_id,
        avatar_id=avatar_id,
        scene_id=scene_id,
        api_url=api_url,
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

    canvas_tools = get_canvas_tools()
    stt = DeepgramSTTService(api_key=DEEPGRAM_API_KEY)
    voice_id = (avatar_config or {}).get("voiceModelId") or CARTESIA_VOICE_ID
    llm = OpenAILLMService(
        api_key=OPENAI_API_KEY,
        settings=OpenAILLMService.Settings(
            model=LLM_MODEL,
            system_instruction=system_prompt,
        ),
    )
    tts = None
    if CLOUD_OUTPUT_MODE == "cartesia":
        tts = CartesiaTTSService(
            api_key=CARTESIA_API_KEY,
            settings=CartesiaTTSService.Settings(
                voice=voice_id,
            ),
        )

    initial_messages = []
    if scene_image_b64:
        from scene_context import build_vision_message

        initial_messages.append(build_vision_message(scene_image_b64))

    context = LLMContext(
        messages=initial_messages if initial_messages else None,
        tools=canvas_tools,
    )

    output_transport = transport.output()
    action_handlers = create_canvas_action_handlers(
        output_transport=output_transport,
        context=context,
        llm=llm,
        room_id=room_id,
        api_url=api_url,
    )
    for func_name, handler in action_handlers.items():
        llm.register_function(func_name, handler)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    user_transcript_fwd = TranscriptForwarder(output_transport)
    avatar_transcript_fwd = TranscriptForwarder(output_transport)
    speaking_notifier = SpeakingStateNotifier(output_transport)

    avatar_participant_id: str | None = None
    active_human_id: str | None = None
    avatar_ready_event = asyncio.Event()
    captured_audio_participant_id: str | None = None
    greeting_sent = False

    def get_avatar_participant_id() -> str | None:
        return avatar_participant_id

    def get_local_participant_id() -> str | None:
        participant_id = str(getattr(transport, "participant_id", "")).strip()
        return participant_id or None

    def get_active_human_id() -> str | None:
        return active_human_id

    def _remove_pending_audio_capture(participant_id: str | None):
        if not participant_id:
            return
        input_transport = getattr(transport, "_input", None)
        pending = getattr(input_transport, "_capture_participant_audio", None)
        if not isinstance(pending, list):
            return
        filtered = [item for item in pending if not item or str(item[0]) != participant_id]
        if len(filtered) != len(pending):
            pending[:] = filtered
            logger.info("Removed pending audio capture for participant_id={}", participant_id)

    async def _ensure_avatar_participant_ignored(participant_id: str | None):
        nonlocal captured_audio_participant_id, active_human_id
        if CLOUD_OUTPUT_MODE != "relay_avatar" or not participant_id:
            return

        _remove_pending_audio_capture(participant_id)

        update_subscriptions = getattr(transport, "update_subscriptions", None)
        if callable(update_subscriptions):
            await update_subscriptions(
                participant_settings={
                    participant_id: {
                        "media": {
                            "microphone": "unsubscribed",
                            "screenAudio": "unsubscribed",
                        }
                    }
                }
            )

        if captured_audio_participant_id == participant_id:
            captured_audio_participant_id = None
        if active_human_id == participant_id:
            active_human_id = None

        logger.info("Ensured SoulX avatar participant is ignored participant_id={}", participant_id)

    async def _start_human_audio_capture(participant_id: str | None):
        nonlocal captured_audio_participant_id
        if CLOUD_OUTPUT_MODE != "relay_avatar":
            return
        if not participant_id or participant_id == avatar_participant_id:
            return
        if captured_audio_participant_id == participant_id:
            return

        _remove_pending_audio_capture(avatar_participant_id)

        capture_participant_audio = getattr(transport, "capture_participant_audio", None)
        if callable(capture_participant_audio):
            await capture_participant_audio(participant_id, "microphone")

        input_transport = getattr(transport, "_input", None)
        start_audio_in_streaming = getattr(input_transport, "start_audio_in_streaming", None)
        if callable(start_audio_in_streaming) and not getattr(input_transport, "_streaming_started", False):
            await start_audio_in_streaming()
            logger.info("Started Daily audio input streaming")

        captured_audio_participant_id = participant_id
        logger.info("Started human-only audio capture for participant_id={}", participant_id)

    human_audio_filter = None
    relay_processor = None
    avatar_ready_gate = None
    if CLOUD_OUTPUT_MODE == "relay_avatar":
        human_audio_filter = HumanOnlyAudioInputFilter(
            get_avatar_participant_id,
            get_local_participant_id,
        )
        avatar_ready_gate = AvatarReadyGateProcessor(avatar_ready_event)
        relay_processor = AvatarRelayProcessor(output_transport, get_avatar_participant_id)

    if CLOUD_OUTPUT_MODE == "cartesia":
        assert tts is not None
        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                user_transcript_fwd,
                user_aggregator,
                llm,
                avatar_transcript_fwd,
                speaking_notifier,
                tts,
                output_transport,
                assistant_aggregator,
            ]
        )
    else:
        assert human_audio_filter is not None
        assert avatar_ready_gate is not None
        assert relay_processor is not None
        pipeline = Pipeline(
            [
                transport.input(),
                human_audio_filter,
                stt,
                user_transcript_fwd,
                user_aggregator,
                avatar_ready_gate,
                llm,
                avatar_transcript_fwd,
                speaking_notifier,
                relay_processor,
                assistant_aggregator,
                output_transport,
            ]
        )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    async def _queue_greeting():
        nonlocal greeting_sent
        if greeting_sent:
            return
        if CLOUD_OUTPUT_MODE == "relay_avatar" and not avatar_ready_event.is_set():
            logger.info("Waiting for avatar relay bot to become ready before greeting visitor")
            await avatar_ready_event.wait()
        if greeting_sent:
            return

        greeting_sent = True
        context.add_message(
            {
                "role": "developer",
                "content": GREETING_TRIGGER_PROMPT,
            }
        )
        await task.queue_frames([LLMRunFrame()])

    async def _cancel_for_human_leave(reason: str, participant_id: str | None):
        nonlocal active_human_id, captured_audio_participant_id, greeting_sent
        logger.info(
            "Human participant left reason={} participant_id={}; cancelling cloud bot",
            reason,
            participant_id,
        )
        active_human_id = None
        captured_audio_participant_id = None
        greeting_sent = False
        await task.cancel()

    if CLOUD_OUTPUT_MODE == "relay_avatar":
        @transport.event_handler("on_app_message")
        async def on_app_message(transport, message, sender):
            nonlocal avatar_participant_id
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
        participant_id = _participant_id(client)
        participant_name = _participant_name(client)
        if avatar_participant_id and participant_id and participant_id == avatar_participant_id:
            role = "avatar_bot"
        logger.info(
            "Participant connected role={} id={} name={}",
            role,
            participant_id,
            participant_name,
        )

        if role == "avatar_bot":
            avatar_participant_id = participant_id or avatar_participant_id
            if CLOUD_OUTPUT_MODE == "relay_avatar":
                avatar_ready_event.set()
                await _ensure_avatar_participant_ignored(participant_id)
            return

        if role != "human":
            return

        active_human_id = participant_id or active_human_id
        if CLOUD_OUTPUT_MODE == "relay_avatar":
            await _start_human_audio_capture(active_human_id)
            if not avatar_ready_event.is_set():
                logger.info("Human joined before avatar relay bot was ready; cloud bot will wait")
        asyncio.create_task(_queue_greeting())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        nonlocal avatar_participant_id

        role = _participant_role(client)
        participant_id = _participant_id(client)
        if avatar_participant_id and participant_id and participant_id == avatar_participant_id:
            role = "avatar_bot"
        logger.info(
            "Participant disconnected role={} id={} name={}",
            role,
            participant_id,
            _participant_name(client),
        )

        if role == "avatar_bot":
            if participant_id and participant_id == avatar_participant_id:
                avatar_participant_id = None
                avatar_ready_event.clear()
            return

        if role != "human":
            return

        await _cancel_for_human_leave("client_disconnected", participant_id)

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        nonlocal avatar_participant_id

        role = _participant_role(participant)
        participant_id = _participant_id(participant)
        if avatar_participant_id and participant_id and participant_id == avatar_participant_id:
            role = "avatar_bot"
        logger.info(
            "Participant left role={} id={} name={} reason={}",
            role,
            participant_id,
            _participant_name(participant),
            reason,
        )

        if role == "avatar_bot":
            if participant_id and participant_id == avatar_participant_id:
                avatar_participant_id = None
                avatar_ready_event.clear()
            return

        if role != "human":
            return

        await _cancel_for_human_leave("participant_left", participant_id)

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Entry point called by Pipecat runner."""

    if isinstance(runner_args, DailyRunnerArguments):
        from pipecat.transports.daily.transport import DailyTransport

        transport = DailyTransport(
            runner_args.room_url,
            runner_args.token,
            CLOUD_BOT_NAME,
            params=_daily_params(),
        )
    else:
        transport_params = {
            "daily": lambda: _daily_params(),
            "webrtc": lambda: TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=CLOUD_OUTPUT_MODE == "cartesia",
            ),
        }

        transport = await create_transport(runner_args, transport_params)

    body = getattr(runner_args, "body", {}) or {}
    room_id = body.get("room_id") or DEFAULT_ROOM_ID
    avatar_id = body.get("avatar_id") or DEFAULT_AVATAR_ID
    scene_id = body.get("scene_id") or DEFAULT_SCENE_ID
    flow_id = body.get("flow_id")
    api_url = body.get("hv_api_url")

    await run_bot(
        transport,
        runner_args,
        room_id=room_id,
        avatar_id=avatar_id,
        scene_id=scene_id,
        flow_id=flow_id,
        api_url=api_url,
    )


def _daily_params():
    """Lazy import DailyParams so the daily extra isn't required for local dev."""
    from pipecat.transports.daily.transport import DailyParams

    if CLOUD_OUTPUT_MODE == "relay_avatar":
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
