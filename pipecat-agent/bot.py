"""Human Virtual Pipecat voice agent.

Pipeline: Mic + Camera → VAD → STT (Deepgram) → LLM (OpenAI) → TTS (Cartesia) → Speaker

Local dev:  python bot.py  → opens http://localhost:7860/client
Production: Deployed to Pipecat Cloud with DailyTransport (Session 38)
"""
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.base_llm_adapter import LLMContext
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, InputImageRawFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
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

from api_client import get_avatar_safe, get_scene_safe
from config import (
    CARTESIA_API_KEY,
    CARTESIA_VOICE_ID,
    DEEPGRAM_API_KEY,
    OPENAI_API_KEY,
    LLM_MODEL,
    DEFAULT_AVATAR_ID,
    DEFAULT_SCENE_ID,
)
from persona import build_system_prompt


# ── Vision: store the latest video frame for on-demand capture ──


class LatestImageCapture(FrameProcessor):
    """Passes all frames through, but stores the most recent video frame."""

    def __init__(self):
        super().__init__()
        self.latest_frame: InputImageRawFrame | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputImageRawFrame):
            self.latest_frame = frame
        await self.push_frame(frame, direction)


# ── Vision: tool definition for the LLM ──

CAPTURE_TOOL = FunctionSchema(
    name="capture_what_i_see",
    description=(
        "Capture and analyze the current live video frame. "
        "Use this when the visitor asks what you can see, asks you to look at something, "
        "or asks you to describe what is currently on screen."
    ),
    properties={},
    required=[],
)


async def run_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    avatar_id: str = "",
    scene_id: str = "",
    flow_id: str | None = None,
):
    """Main bot pipeline."""
    logger.info("Starting Human Virtual voice agent")

    # Fetch avatar and scene data from the API
    avatar = await get_avatar_safe(avatar_id)
    scene = await get_scene_safe(scene_id)

    # Build system prompt from avatar persona + scene context
    system_prompt = build_system_prompt(avatar, scene)

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
    context = LLMContext(
        tools=ToolsSchema(standard_tools=[CAPTURE_TOOL]),
    )
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # ── Vision: inject static background image at boot ──
    bg_url = None
    if scene:
        bg_url = scene.get("canvasState", {}).get("background", {}).get("url")
    if bg_url:
        bg_message = LLMContext.create_image_url_message(
            url=bg_url,
            text="This is the background image of the scene you are presenting.",
        )
        context.add_message(bg_message)
        logger.info(f"Injected background image into context: {bg_url}")

    # ── Vision: on-demand video frame capture ──
    image_capture = LatestImageCapture()

    async def handle_capture_what_i_see(params):
        frame = image_capture.latest_frame
        if frame:
            # format is a PIL mode like "RGB"/"RGBA", or None
            fmt = frame.format or "RGB"
            await params.context.add_image_frame_message(
                format=fmt,
                size=frame.size,
                image=frame.image,
                text="This is what you currently see in the live video feed. Describe it to the visitor.",
            )
            logger.info(f"Captured live video frame: {frame.size}, format={fmt}")
            await params.result_callback("Live video frame captured. Describe what you see.")
        else:
            logger.warning("No video frame available for capture")
            await params.result_callback(
                "No video feed is available right now. Describe the scene based on what you already know."
            )

    llm.register_function("capture_what_i_see", handle_capture_what_i_see)

    # ── Pipeline ──
    pipeline = Pipeline([
        transport.input(),                # Visitor's mic + camera (WebRTC)
        image_capture,                    # Store latest video frame
        stt,                              # Deepgram: speech → text
        context_aggregator.user(),        # Add user message to conversation history
        llm,                              # OpenAI: generate response (with tools)
        tts,                              # Cartesia: response → speech audio
        transport.output(),               # Send audio back to visitor (WebRTC)
        context_aggregator.assistant(),   # Add bot response to conversation history
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
    """Entry point called by Pipecat runner."""
    transport_params = {
        "daily": lambda: _daily_params(),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_in_enabled=True,
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    # Extract custom data passed via Pipecat Cloud start API body
    body = getattr(runner_args, "body", {}) or {}
    avatar_id = body.get("avatar_id") or DEFAULT_AVATAR_ID
    scene_id = body.get("scene_id") or DEFAULT_SCENE_ID
    flow_id = body.get("flow_id")

    await run_bot(transport, runner_args, avatar_id=avatar_id, scene_id=scene_id, flow_id=flow_id)


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
