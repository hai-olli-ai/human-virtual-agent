"""Cache-first Cartesia TTS service (Block 12).

If a cached PCM segment has been primed for the NEXT ``run_tts`` call,
replay it as the canonical TTS frame envelope (``TTSStartedFrame`` →
``TTSAudioRawFrame``\\ s → ``TTSStoppedFrame``); otherwise fall through
to live Cartesia synthesis via ``super().run_tts``.

Pre-rendered PCM must be 16-bit signed little-endian (Cartesia's default
``encoding='pcm_s16le'``) at the service's negotiated ``sample_rate``
and ``num_channels``. The caller keys the cache on (text, voice, model,
sample_rate, num_channels) — this class trusts the primed segment
matches the live Cartesia ``Settings``.

Composes with ``narration.NarrationCompletionGate``: the gate observes
``TTSStoppedFrame``, which this class emits on every hit (including the
empty-PCM edge case), so per-segment narration unblocks correctly
whether the segment was served from cache or live.
"""

import asyncio
import logging
from typing import AsyncGenerator, Optional

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.cartesia.tts import CartesiaTTSService

log = logging.getLogger(__name__)


class CachedSegment:
    __slots__ = ("pcm", "sample_rate", "num_channels")

    def __init__(self, pcm: bytes, sample_rate: int, num_channels: int = 1):
        self.pcm = pcm
        self.sample_rate = sample_rate
        self.num_channels = num_channels


class CachedFirstTTSService(CartesiaTTSService):
    """Cartesia TTS service with a single-shot pre-rendered PCM cache.

    Lifecycle: ``prime_cached(segment)`` stashes a segment to be played
    by the *next* ``run_tts`` call; ``_consume()`` clears it. A second
    ``run_tts`` with no re-prime falls through to live synthesis. This
    "prime → speak" pacing matches the narrator's per-segment loop: the
    narrator primes immediately before queuing the ``TTSSpeakFrame``
    with no intervening awaits that could trigger another ``run_tts``.
    """

    CHUNK_MS = 20
    _BYTES_PER_SAMPLE = 2  # pcm_s16le
    # Block 15 — extra wall-clock time waited between the last
    # ``TTSAudioRawFrame`` and ``TTSStoppedFrame`` on a cache hit, on
    # top of the PCM's exact playback duration. Covers
    # ``BaseTransportOutput``'s audio scheduler warmup (~CHUNK_MS at
    # first tick) plus a small safety margin so the visitor finishes
    # hearing the audio before ``script_complete`` fires and the shell
    # auto-advances. 50 ms is well under human "noticeable pause"
    # threshold while comfortably covering the scheduler.
    _PLAYBACK_DRAIN_PADDING_S = 0.05

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._primed: Optional[CachedSegment] = None

    def prime_cached(self, segment: Optional[CachedSegment]) -> None:
        """Single-shot prime for the next ``run_tts`` call.

        Passing ``None`` clears any previously primed segment so the
        next ``run_tts`` falls through to live. Calling ``prime_cached``
        twice replaces the previous prime without warning — the caller
        owns the pacing.
        """
        self._primed = segment

    def _consume(self) -> Optional[CachedSegment]:
        seg, self._primed = self._primed, None
        return seg

    async def process_frame(self, frame, direction):
        # Auto Play Phase A (A2) — drop an armed prime on interruption.
        # The narrator primes BEFORE each speak; if the run aborts
        # between prime and run_tts (barge-in latch, task cancellation,
        # or the TTSSpeakFrame being dropped by the interruption
        # itself), a stale prime would otherwise be consumed by the
        # NEXT run_tts — i.e. the LLM's reply to the barge-in would
        # play the scene-script PCM in the script avatar's voice
        # instead of the reply text.
        if isinstance(frame, InterruptionFrame):
            self._primed = None
        await super().process_frame(frame, direction)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        seg = self._consume()
        if seg is None:
            # Cache miss — live path. Cartesia's ``run_tts`` itself only
            # sends the websocket request and yields ``None`` /
            # ``ErrorFrame``; the actual TTSStartedFrame /
            # TTSAudioRawFrame / TTSStoppedFrame stream emerges from the
            # websocket-receiver coroutine. We just propagate whatever
            # super yields so the pipeline sees identical behavior to
            # the unwrapped Cartesia service.
            async for frame in super().run_tts(text, context_id):
                yield frame
            return

        log.info(
            "[narration-cache] HIT bytes=%d sr=%d nch=%d ctx=%s",
            len(seg.pcm),
            seg.sample_rate,
            seg.num_channels,
            context_id,
        )
        # Metric parity with the live path: without these, Grafana TTFB
        # graphs would step-function at the cache boundary (no TTFB
        # recorded on hits) and Cartesia usage metrics would
        # under-report the visible text the visitor actually heard.
        await self.start_ttfb_metrics()
        await self.start_tts_usage_metrics(text)

        yield TTSStartedFrame(context_id=context_id)
        await self.stop_ttfb_metrics()  # first chunk is "instant" on a hit

        sample_frame = self._BYTES_PER_SAMPLE * seg.num_channels
        chunk_bytes = int(seg.sample_rate * sample_frame * self.CHUNK_MS / 1000)
        chunk_bytes -= chunk_bytes % sample_frame  # whole-sample aligned

        for i in range(0, len(seg.pcm), chunk_bytes):
            yield TTSAudioRawFrame(
                audio=seg.pcm[i : i + chunk_bytes],
                sample_rate=seg.sample_rate,
                num_channels=seg.num_channels,
                context_id=context_id,
            )

        # Block 15 — defer ``TTSStoppedFrame`` until the audio has had
        # time to play out. The transport pulls ``TTSAudioRawFrame``\\ s
        # into its audio queue immediately and plays them at real-time
        # sample-rate, but ``BaseTransportOutput`` emits
        # ``BotStoppedSpeakingFrame`` synchronously off
        # ``TTSStoppedFrame`` (base_output.py routes
        # ``TTSStoppedFrame`` → ``_bot_stopped_speaking`` directly), and
        # ``NarrationCompletionGate`` releases its pending future on
        # ``TTSStoppedFrame``. Without this sleep, the gate releases
        # microseconds after the bytes-handoff, ``run_scene_narration``
        # returns, the agent emits ``script_complete``, and the shell
        # auto-advances while the visitor is still hearing the
        # narration — severe clipping for cached segments.
        bytes_per_second = seg.sample_rate * sample_frame
        if bytes_per_second:
            duration_s = len(seg.pcm) / bytes_per_second
            await asyncio.sleep(duration_s + self._PLAYBACK_DRAIN_PADDING_S)

        yield TTSStoppedFrame(context_id=context_id)
