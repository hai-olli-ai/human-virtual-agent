"""`soulx-audio.v1` client — streams TTS audio to the SoulX renderer on Modal.

Replaces the `avatar-relay.v1` text leg (bot.py:177-183 + `_relay_speak`). Previously the
agent forwarded TEXT over Daily app-messages and SoulX ran its own TTS; now the agent owns
TTS and streams PCM over a WebSocket. Two consequences worth stating:

  * the renderer becomes TTS-agnostic — Cartesia today, any streaming provider tomorrow,
    and a fully pre-rendered WAV is just a fast burst of the same chunks;
  * the coupling is transport-independent, so the Week-2 LiveKit migration never touches
    it — only room join/tokens/identities move.

Failure policy: a renderer that never becomes ready, or a socket that drops, must NEVER
produce a silent bot (today's `AvatarReadyGateProcessor` waits on an un-timed
asyncio.Event and blocks forever). On failure the sink flips to voice-only fallback and
passes audio downstream so the agent's own transport publishes it.

Protocol source of truth: soulx-modal/protocol.py. Constants are duplicated here rather
than imported because the two services are separate repos and deploy independently.
"""

from __future__ import annotations

import asyncio
import json
import struct
from typing import Any, Optional

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

PROTOCOL = "soulx-audio.v1"

SESSION_INIT = "session_init"
TURN_START = "turn_start"
TURN_END = "turn_end"
INTERRUPT = "interrupt"
END_SESSION = "end_session"

READY = "ready"
ROOM_JOINED = "room_joined"
AUDIO_ACK = "audio_ack"
PLAYOUT_STARTED = "playout_started"
PLAYOUT_COMPLETED = "playout_completed"
ERROR = "error"

SEQ_STRUCT = ">I"

# At most this much audio may be unacked. TTS emits far faster than real time, so without
# a window one long turn would shove the whole utterance at the renderer, making barge-in
# useless and wasting render work that an interruption throws away.
FLOW_WINDOW_SECONDS = 2.0

# Turn-group debounce (narration only): after a segment's TTSStoppedFrame, hold the
# renderer turn open this long for the next segment's TTSStartedFrame. Cartesia TTFB on
# the follow-up request is ~0.3-1.0s, so 1.5s coalesces adjacent narration segments into
# ONE renderer turn while a genuinely final stop still closes promptly.
TURN_GROUP_DEBOUNCE_S = 1.5


class SoulXAudioClient:
    """WebSocket client for one renderer session.

    Deliberately reconnect-free: a dropped socket mid-session fails the session over to
    voice-only rather than silently stalling. Reconnect would need renderer-side session
    resumption, which is not in the Week-1 protocol.
    """

    def __init__(
        self,
        ws_url: str,
        auth_token: str,
        room_url: str,
        room_token: str,
        sample_rate: int = 24000,
        ready_timeout_s: float = 300.0,
        avatar_ref: str = "",
    ):
        self._ws_url = ws_url
        self._auth_token = auth_token
        self._room_url = room_url
        self._room_token = room_token
        self._sample_rate = sample_rate
        self._ready_timeout_s = ready_timeout_s
        # https URL of the avatar photo. REQUIRED by the renderer since
        # 2026-08-06 (no-fallback policy) — a session without it is refused
        # with a fatal error rather than rendering a default face.
        self._avatar_ref = avatar_ref

        self._ws: Any = None
        self._reader: Optional[asyncio.Task] = None
        self._room_joined = asyncio.Event()
        self._healthy = False
        self._seq = 0
        self._sent_bytes = 0
        self._acked_bytes = 0
        self._ack_cond = asyncio.Condition()
        self._bytes_per_second = sample_rate * 2  # s16le mono

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def connect(self) -> bool:
        """Open the socket and wait for the renderer to join the room.

        Returns False (never raises) on any failure — the caller's job is to fall back,
        not to crash the session.
        """
        try:
            import websockets

            headers = {"Authorization": f"Bearer {self._auth_token}"} if self._auth_token else {}
            # open_timeout is load-bearing and easy to miss: it defaults to 10s, but a
            # COLD Modal container takes ~134s to accept the handshake (schedule + image
            # pull + 15GB volume + model load + warmup). Without this, raising the ready
            # timeout achieves nothing — the socket dies during the handshake, long
            # before anyone waits on room_joined.
            self._ws = await websockets.connect(
                self._ws_url,
                additional_headers=headers,
                max_size=None,
                ping_interval=20,
                open_timeout=self._ready_timeout_s,
            )

            hello = json.loads(await asyncio.wait_for(self._ws.recv(), timeout=30))
            if hello.get("type") != READY:
                logger.error("SoulX: expected ready, got {}", hello)
                return False
            logger.info("SoulX renderer ready: gpu={} model={}", hello.get("gpu"), hello.get("model_type"))

            await self._ws.send(
                json.dumps(
                    {
                        "type": SESSION_INIT,
                        "protocol": PROTOCOL,
                        "room_url": self._room_url,
                        "token": self._room_token,
                        "sample_rate": self._sample_rate,
                        "avatar_ref": self._avatar_ref,
                    }
                )
            )

            self._reader = asyncio.create_task(self._read_loop())

            # THE fix for the silent bot: bounded wait, never an un-timed Event.
            await asyncio.wait_for(self._room_joined.wait(), timeout=self._ready_timeout_s)
            self._healthy = True
            return True
        except Exception as exc:  # noqa: BLE001 — every failure means "fall back"
            logger.error("SoulX connect failed ({}): {}", type(exc).__name__, exc)
            self._healthy = False
            return False

    async def _read_loop(self):
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == ROOM_JOINED:
                    logger.info("SoulX avatar joined the room as {}", msg.get("participant_id"))
                    self._room_joined.set()
                elif kind == AUDIO_ACK:
                    async with self._ack_cond:
                        # acked_bytes is CUMULATIVE bytes the renderer has taken in. The
                        # `seq` field is diagnostic only — an earlier version defaulted
                        # to self._sent_bytes when the field was absent, which silently
                        # made the window a no-op instead of failing loudly.
                        acked = msg.get("acked_bytes")
                        if acked is None:
                            logger.warning("SoulX ack without acked_bytes: {}", msg)
                        else:
                            self._acked_bytes = max(self._acked_bytes, int(acked))
                        self._ack_cond.notify_all()
                elif kind == PLAYOUT_COMPLETED:
                    logger.debug("SoulX playout completed turn={}", msg.get("turn_id"))
                elif kind == ERROR:
                    logger.error("SoulX renderer error: {}", msg.get("error"))
                    if msg.get("fatal"):
                        self._healthy = False
                        self._room_joined.set()
        except Exception as exc:  # noqa: BLE001
            logger.warning("SoulX read loop ended ({}): {}", type(exc).__name__, exc)
        finally:
            self._healthy = False
            async with self._ack_cond:
                self._ack_cond.notify_all()

    async def turn_start(self, turn_id: str):
        await self._send_json({"type": TURN_START, "turn_id": turn_id})

    async def turn_end(self, turn_id: Optional[str]):
        await self._send_json({"type": TURN_END, "turn_id": turn_id})

    async def interrupt(self, turn_id: Optional[str]):
        # Reset the flow window: the renderer drops what it had queued, so holding the
        # agent back on stale unacked bytes would stall the NEXT turn.
        async with self._ack_cond:
            self._acked_bytes = self._sent_bytes
            self._ack_cond.notify_all()
        await self._send_json({"type": INTERRUPT, "turn_id": turn_id})

    async def send_pcm(self, pcm: bytes):
        if not self._healthy or self._ws is None:
            return
        window = int(FLOW_WINDOW_SECONDS * self._bytes_per_second)

        # Backpressure is an OPTIMISATION; the avatar is the product. These must not
        # share a failure path. An earlier version treated a stalled window as a dead
        # transport and flipped the whole session to voice-only permanently — one slow
        # ack cost every remaining turn its video.
        try:
            async with self._ack_cond:
                while self._healthy and (self._sent_bytes - self._acked_bytes) > window:
                    await asyncio.wait_for(self._ack_cond.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            # Proceed anyway. The renderer has been measured absorbing a whole utterance
            # burst without trouble, so overrunning the window is far cheaper than
            # losing the avatar.
            logger.warning(
                "SoulX flow window stalled ({} bytes unacked) — sending anyway",
                self._sent_bytes - self._acked_bytes,
            )
            self._acked_bytes = self._sent_bytes  # don't re-stall on every subsequent chunk

        # Only a genuine socket failure means the transport is dead.
        try:
            self._seq += 1
            await self._ws.send(struct.pack(SEQ_STRUCT, self._seq & 0xFFFFFFFF) + pcm)
            self._sent_bytes += len(pcm)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SoulX send failed ({}): {}", type(exc).__name__, exc)
            self._healthy = False

    async def _send_json(self, payload: dict):
        if not self._healthy or self._ws is None:
            return
        try:
            await self._ws.send(json.dumps(payload))
        except Exception as exc:  # noqa: BLE001
            logger.warning("SoulX control send failed ({}): {}", type(exc).__name__, exc)
            self._healthy = False

    async def close(self):
        self._healthy = False
        try:
            if self._ws is not None:
                await self._send_json({"type": END_SESSION})
                await self._ws.close()
        except Exception:  # noqa: BLE001 — teardown is best-effort
            pass
        if self._reader:
            self._reader.cancel()


class SoulXAudioSink(FrameProcessor):
    """Forwards the TTS frame envelope to the renderer; owns the fallback decision.

    Sits where `AvatarRelayProcessor` used to. When the renderer is healthy it SWALLOWS
    audio frames (the renderer publishes the audio alongside its video, already synced —
    letting them through too would double the audio in the room). When the renderer is
    unhealthy it PASSES THEM THROUGH so the agent's own transport speaks: voice-only,
    never a silent bot.
    """

    def __init__(self, client: SoulXAudioClient):
        super().__init__(name="soulx_audio_sink")
        self._client = client
        self._turn = 0
        self._turn_id: Optional[str] = None
        self._fallback_logged = False
        # Turn grouping (narration). Scene narration speaks one TTSSpeakFrame
        # PER SEGMENT, so the renderer used to see N turns with ~0.5s gaps —
        # and its 0.35s finalize raced Cartesia's TTFB in every gap, letting
        # its idle mode inject silent video frames between the segments. Those
        # frames displace all later lip video while the audio plays on time:
        # the measured scripted-scene A/V desync (2026-08-05, proven on demand
        # with soulx-modal/bisect_client.py run 3). Inside a group the sink
        # holds ONE renderer turn open across the segment envelopes, so the
        # renderer never sees an inter-segment boundary at all.
        #
        # The group must be closed by the FRAMES, not the narration coroutine:
        # narration speak is fire-and-forget (queue_frames returns before the
        # TTS envelopes reach this sink), so an eager end_turn_group() would
        # close the turn before the first segment even started. Hence the
        # debounced close on TTSStoppedFrame; end_turn_group() is the
        # cancellation-path cleanup, not the normal close.
        self._grouping = False
        self._group_close_task: Optional[asyncio.Task] = None

    @property
    def fallback(self) -> bool:
        return not self._client.healthy

    def begin_turn_group(self) -> None:
        """All TTS envelopes until the group closes ride ONE renderer turn."""
        self._grouping = True
        self._cancel_group_close()

    async def end_turn_group(self) -> None:
        """Cleanup path: close any open group turn NOW (cancellation/teardown)."""
        self._grouping = False
        self._cancel_group_close()
        if self._turn_id is not None:
            await self._client.turn_end(self._turn_id)
            self._turn_id = None

    def _cancel_group_close(self) -> None:
        if self._group_close_task is not None:
            self._group_close_task.cancel()
            self._group_close_task = None

    def _schedule_group_close(self) -> None:
        self._cancel_group_close()
        self._group_close_task = asyncio.create_task(self._group_close_after())

    async def _group_close_after(self) -> None:
        try:
            await asyncio.sleep(TURN_GROUP_DEBOUNCE_S)
        except asyncio.CancelledError:
            return  # a new segment arrived — the turn stays open
        # The debounced close ends the whole GROUP, not just the turn: after
        # narration finishes, conversation turns must go back to prompt
        # per-envelope turn_ends, or every short reply's tail render would be
        # delayed by the debounce.
        self._grouping = False
        if self._turn_id is not None:
            logger.debug("SoulX turn group closing turn={} (debounce)", self._turn_id)
            await self._client.turn_end(self._turn_id)
            self._turn_id = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.UPSTREAM or isinstance(
            frame, (StartFrame, EndFrame, CancelFrame)
        ):
            if isinstance(frame, (EndFrame, CancelFrame)):
                self._cancel_group_close()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InterruptionFrame):
            # The interrupt already ends the renderer turn; a pending debounced
            # close would otherwise fire turn_end for a dead turn id.
            self._cancel_group_close()
            await self._client.interrupt(self._turn_id)
            self._turn_id = None
            await self.push_frame(frame, direction)
            return

        if self.fallback:
            if not self._fallback_logged:
                logger.warning("SoulX unavailable — voice-only fallback for this session")
                self._fallback_logged = True
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TTSStartedFrame):
            self._cancel_group_close()
            if self._turn_id is None:
                self._turn += 1
                self._turn_id = f"turn-{self._turn}"
                await self._client.turn_start(self._turn_id)
            # else: grouped — the previous segment's stop was held back, so
            # this envelope continues the SAME renderer turn (no boundary).
        elif isinstance(frame, TTSAudioRawFrame):
            await self._client.send_pcm(bytes(frame.audio))
            return  # swallowed on purpose — the renderer publishes this audio
        elif isinstance(frame, TTSStoppedFrame):
            if self._grouping:
                self._schedule_group_close()
            else:
                await self._client.turn_end(self._turn_id)
                self._turn_id = None

        await self.push_frame(frame, direction)
