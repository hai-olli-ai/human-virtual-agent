"""Human Virtual Pipecat voice agent.

Pipeline: Mic → VAD → STT (Deepgram) → LLM (OpenAI) → TTS (Cartesia) → Speaker

Local dev:  python bot.py  → opens http://localhost:7860/client
Production: Deployed to Pipecat Cloud with DailyTransport (Session 38)
"""
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import (
    LLMContext,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
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
    DEFAULT_SCENE_ID,
)
from persona import build_system_prompt


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Main bot pipeline."""
    logger.info("Starting Human Virtual voice agent")

    # Get avatar/scene IDs from runner args or defaults
    avatar_id = getattr(runner_args, "avatar_id", None) or DEFAULT_AVATAR_ID
    scene_id = getattr(runner_args, "scene_id", None) or DEFAULT_SCENE_ID

    # Build system prompt from avatar persona + scene context
    system_prompt = await build_system_prompt(avatar_id, scene_id)

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

    # ── Pipeline ──
    pipeline = Pipeline([
        transport.input(),       # Visitor's microphone audio (WebRTC)
        stt,                     # Deepgram: speech → text
        user_aggregator,         # Add user message to conversation history
        llm,                     # OpenAI: generate response
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
        context.add_message(
            "developer",
            "A visitor just joined. Greet them warmly and briefly introduce yourself and the scene you're presenting. Keep it to 1-2 sentences.",
        )
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
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


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
