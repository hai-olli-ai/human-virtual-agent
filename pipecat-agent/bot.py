"""Human Virtual Pipecat voice agent.

Pipeline: Mic → VAD → STT (Deepgram) → LLM (OpenAI) → TTS (Cartesia) → Speaker

Session 45: Full integration with Session 43 backend endpoints.
- Persona prompt fetched from backend
- Scene instruction + display mode awareness
- Real-time transcript forwarding to frontend via Daily data channel

Local dev:  python bot.py  → opens http://localhost:7860/client
Production: Deployed to Pipecat Cloud with DailyTransport
"""
import json

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMRunFrame,
    TranscriptionFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.adapters.base_llm_adapter import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams

load_dotenv(override=True)

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
from persona import build_system_prompt


class TranscriptForwarder(FrameProcessor):
    """Captures transcription and LLM text frames, forwards them
    to the frontend via the Daily data channel.

    The frontend listens for 'app-message' events with type 'transcript'.
    """

    def __init__(self, transport: BaseTransport):
        super().__init__()
        self._transport = transport

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # User speech transcription (from STT)
        if isinstance(frame, TranscriptionFrame) and frame.text:
            await self._send_transcript("user", frame.text)

        # Bot response text (from LLM, before TTS)
        if isinstance(frame, TextFrame) and frame.text:
            await self._send_transcript("avatar", frame.text)

        # Always pass the frame through
        await self.push_frame(frame, direction)

    async def _send_transcript(self, speaker: str, text: str):
        """Send transcript message via Daily data channel."""
        try:
            message = json.dumps({
                "type": "transcript",
                "speaker": speaker,
                "text": text,
            })
            if hasattr(self._transport, "send_message"):
                await self._transport.send_message(message)
            elif hasattr(self._transport, "output") and hasattr(self._transport.output(), "send_app_message"):
                await self._transport.output().send_app_message(message)
        except Exception as e:
            logger.debug(f"Could not forward transcript: {e}")


class SpeakingStateNotifier(FrameProcessor):
    """Notifies the frontend when the avatar starts/stops speaking.

    Sends 'speaking_state' messages via Daily data channel.
    """

    def __init__(self, transport: BaseTransport):
        super().__init__()
        self._transport = transport
        self._is_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # TextFrame from LLM means bot is about to speak
        if isinstance(frame, TextFrame) and frame.text and not self._is_speaking:
            self._is_speaking = True
            await self._send_state(True)

        await self.push_frame(frame, direction)

    async def _send_state(self, is_speaking: bool):
        try:
            message = json.dumps({
                "type": "speaking_state",
                "isSpeaking": is_speaking,
            })
            if hasattr(self._transport, "send_message"):
                await self._transport.send_message(message)
        except Exception as e:
            logger.debug(f"Could not send speaking state: {e}")


async def run_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    room_id: str = "",
    avatar_id: str = "",
    scene_id: str = "",
    flow_id: str | None = None,
    api_url: str | None = None,
):
    """Main bot pipeline."""
    logger.info(f"Starting Human Virtual voice agent (room={room_id}, avatar={avatar_id})")

    # Build system prompt using Session 43 endpoints
    system_prompt = await build_system_prompt(
        room_id=room_id,
        avatar_id=avatar_id,
        scene_id=scene_id,
        api_url=api_url,
    )
    logger.info(f"System prompt length: {len(system_prompt)} chars")

    # ── AI Services ──
    stt = DeepgramSTTService(api_key=DEEPGRAM_API_KEY)

    tts = CartesiaTTSService(
        api_key=CARTESIA_API_KEY,
        settings=CartesiaTTSService.Settings(
            voice=CARTESIA_VOICE_ID,
        ),
    )

    llm = OpenAILLMService(
        api_key=OPENAI_API_KEY,
        settings=OpenAILLMService.Settings(
            model=LLM_MODEL,
            system_instruction=system_prompt,
        ),
    )

    # ── Conversation Context ──
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # ── Transcript forwarding ──
    transcript_forwarder = TranscriptForwarder(transport)
    speaking_notifier = SpeakingStateNotifier(transport)

    # ── Pipeline ──
    pipeline = Pipeline([
        transport.input(),       # Visitor's microphone audio (WebRTC)
        stt,                     # Deepgram: speech → text
        transcript_forwarder,    # Forward user transcription to frontend
        user_aggregator,         # Add user message to conversation history
        llm,                     # OpenAI: generate response
        speaking_notifier,       # Notify frontend of speaking state
        tts,                     # Cartesia: response → speech audio
        transport.output(),      # Send audio back to visitor (WebRTC)
        assistant_aggregator,    # Add bot response to conversation history
    ])

    # ── Task ──
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # ── Event Handlers ──
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Visitor connected to live room")
        # Greet the visitor with the avatar's personality
        context.add_message({
            "role": "developer",
            "content": "A visitor just joined. Greet them warmly and briefly introduce yourself and the scene you're presenting. Keep it to 1-2 sentences.",
        })
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Visitor disconnected")
        await task.cancel()

    # ── Run ──
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Entry point called by Pipecat runner.

    Extracts room_id, avatar_id, scene_id, flow_id from runner_args.body
    (passed by the backend's start-session endpoint via Pipecat Cloud API).
    """
    transport_params = {
        "daily": lambda: _daily_params(),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    # Extract custom data passed via Pipecat Cloud start API body
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
    return DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    )


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
