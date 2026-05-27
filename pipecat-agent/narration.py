"""Per-scene narration for the voice agent (S65 G3).

Replaces the inline script-loop in ``bot.py``'s session-start handlers
with a single helper that:

  * iterates the snapshot's scene scripts in order
    (``snapshot["scripts"]`` — flat at the top level of the snapshot;
    semantically the *current scene's* scripts, per the S65 backend
    payload — see ``LiveRoomService.build_scene_snapshot``)
  * switches the Cartesia TTS voice per segment in the classic pipeline
  * resets the voice back to the live-room primary before returning
  * narrates in the primary SoulX voice in the relay pipeline (no
    per-segment voice switching — v0.2 punt; see CLAUDE.md S65 plan)
  * is idempotent per ``scene_id`` (so ``canvas.register``-only prompt
    rebuilds don't re-narrate, while session start + ``canvas.sceneChanged``
    each get their one shot)

The decision logic lives in :func:`plan_narration_segments` (pure,
unit-testable). The runtime wrapper :class:`SceneNarrator` takes the
plan and drives the side effects through two injected callables (``set_voice``
and ``speak``). Keeping the side-effects behind callables means both
pipelines share the same loop without dragging FrameProcessor or
Daily-transport surface into unit tests — see ``test_scene_narration.py``.

The classic pipeline also needs a way to *await* per-segment TTS
completion so the voice for segment N+1 isn't applied while Cartesia is
still mid-render on segment N's WebSocket context.
:class:`NarrationCompletionGate` is the small FrameProcessor that
captures :class:`TTSStoppedFrame` events and resolves caller-supplied
futures in FIFO order.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable

from loguru import logger
from pipecat.frames.frames import Frame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


SpeakFn = Callable[[str], Awaitable[None]]
SetVoiceFn = Callable[[str], Awaitable[None]]


# ──────────────────────────────────────────────────────────────────────
# Pure helpers (no I/O — unit-testable directly)
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NarrationSegment:
    """One narration segment.

    ``voice_id`` is the resolved Cartesia voice for the segment in the
    classic pipeline, or ``None`` in the relay pipeline (where SoulX
    handles its own voice and we never switch).
    """

    text: str
    voice_id: str | None


def plan_post_narration_followup(
    snapshot: dict,
    *,
    spoke_script: bool,
) -> str | None:
    """Decide the line to speak after narration finishes (S65 G4).

    Pure: no I/O, no asyncio. Returns the resolved string to speak
    after the per-segment narration loop, or ``None`` if nothing
    should be spoken (no-script scene, or auto-advance with no cue).

    Decision tree (matches the S65 plan):

      * ``spoke_script=False`` ⇒ ``None`` — no-script scenes use the
        existing conversational greeting trigger; no invitation, no cue.
      * ``spoke_script=True`` AND ``live_room.auto_advance=True`` AND
        not last scene ⇒ optional ``current_scene.narration.transition_cue``
        (e.g. "Let's move on" in the room's language). The shell
        auto-advances on ``script_complete`` immediately after — no
        invitation here because the visitor isn't asked to take a turn.
      * Otherwise (``spoke_script=True`` AND (no auto_advance OR last
        scene)) ⇒ ``current_scene.narration.invitation_line`` (e.g.
        "What questions do you have?" in the room's language). If the
        snapshot is from a pre-S65 backend that lacks the localized
        field, fall back to the legacy hardcoded English line so the
        visitor still hears an invitation.

    Snapshot shape: S65 (Option B) nests these under three blocks —
    ``live_room`` (room config), ``flow_state`` (cursor), and
    ``current_scene`` (per-nav payload). See
    ``LiveRoomService.build_scene_snapshot``. We read defensively
    (``.get`` with ``{}`` fallback) so a degraded snapshot still
    returns a sensible follow-up rather than crashing.
    """
    if not spoke_script:
        return None

    live_room = snapshot.get("live_room") or {}
    flow_state = snapshot.get("flow_state") or {}
    current_scene = snapshot.get("current_scene") or {}

    auto = bool(live_room.get("auto_advance"))
    idx = int(flow_state.get("scene_index", 0) or 0)
    total = int(flow_state.get("total_scenes", 1) or 1)
    is_last = idx >= total - 1
    narr = current_scene.get("narration") or {}

    if auto and not is_last:
        # Auto-advance branch: optional cue, never an invitation.
        cue = (narr.get("transition_cue") or "").strip()
        return cue or None

    # Manual branch (last scene, or auto_advance disabled): invitation.
    line = (narr.get("invitation_line") or "").strip()
    if line:
        return line
    # Pre-S65 backend snapshot ⇒ legacy hardcoded English fallback so
    # the visitor isn't left in silence after the presentation.
    return "Please feel free to ask me if you have any questions."


def build_script_complete_payload(
    snapshot: dict | None,
    *,
    spoke_script: bool,
) -> dict:
    """Build the ``script_complete`` Daily app-message payload (S65 G4).

    Pure: no I/O. The shell uses this payload (specifically
    ``hadScript`` + ``sceneIndex``) to drive auto-advance — it advances
    on ``script_complete`` only when the agent reports it actually
    narrated something for the current scene. ``sceneIndex`` is the
    snapshot's ``flow_state.scene_index`` (0-based), defaulting to
    ``0`` for the degenerate no-snapshot case.
    """
    snap = snapshot or {}
    flow_state = snap.get("flow_state") or {}
    return {
        "type": "script_complete",
        "sceneIndex": int(flow_state.get("scene_index", 0) or 0),
        "hadScript": bool(spoke_script),
    }


def plan_narration_segments(
    snapshot: dict,
    *,
    primary_voice_id: str | None,
    is_relay: bool,
) -> list[NarrationSegment]:
    """Translate the snapshot's current-scene scripts into the narration plan.

    Pure: no I/O, no asyncio, no logging. Mirrors the backend's
    ``_resolve_segment_voice`` fallback rule — ``segment.voice_id`` wins
    if present (backend already resolved it to either the segment
    avatar's Cartesia clone or the live-room primary), else we use the
    agent-side ``primary_voice_id``. The redundancy guards against
    snapshots from older backend versions that don't populate the S65
    voice fields.

    Filters out segments whose ``text`` is empty/whitespace-only — that
    matches the original S49 loop, which checked ``text.strip()`` before
    queueing a ``TTSSpeakFrame``. Empty plan ⇒ caller treats it as
    "nothing to narrate" and returns ``spoken_any == False``.

    The ``scripts`` array is consumed in its server-provided order
    (sorted by ``order`` on the backend in ``build_scene_snapshot``).
    We don't re-sort here — re-sorting on every narrate would mask a
    backend ordering regression. Trust the snapshot.

    Snapshot shape: S65 (Option B) nests scripts under
    ``current_scene.scripts``. Defensive ``.get`` chain so a degenerate
    snapshot returns an empty plan rather than crashing.
    """
    current_scene = snapshot.get("current_scene") or {}
    raw = current_scene.get("scripts") or []
    plan: list[NarrationSegment] = []
    for seg in raw:
        if not isinstance(seg, dict):
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if is_relay:
            # Relay pipeline narrates in the primary SoulX voice; the
            # per-segment voice clone is a v0.2 punt (CLAUDE.md S65).
            plan.append(NarrationSegment(text=text, voice_id=None))
        else:
            voice = seg.get("voice_id") or primary_voice_id
            plan.append(NarrationSegment(text=text, voice_id=voice))
    return plan


# ──────────────────────────────────────────────────────────────────────
# Runtime narrator (drives the injected callables; tracks idempotency)
# ──────────────────────────────────────────────────────────────────────


class SceneNarrator:
    """Per-session narration runner.

    One instance is constructed in each pipeline (``run_bot_classic``,
    ``run_bot_relay``) and reused for the lifetime of the session. The
    runner caches ``_current_voice`` so a switch to the voice that's
    already active is a no-op (saves a Cartesia WS settings round-trip),
    and ``_narrated_scene_id`` for the once-per-scene-entry guard.

    For the relay pipeline pass ``set_voice=None`` and
    ``primary_voice_id=None``; the loop will skip voice handling entirely
    while still iterating scripts and calling ``speak``.

    ``narrate(snapshot)`` returns ``True`` iff at least one non-empty
    segment was narrated. Callers use the return value to decide whether
    to follow up with the post-narration closing line (and, in S65,
    whether to suppress the invitation line under ``auto_advance``).
    """

    def __init__(
        self,
        *,
        primary_voice_id: str | None,
        set_voice: SetVoiceFn | None,
        speak: SpeakFn,
    ):
        self._primary_voice_id = primary_voice_id
        self._set_voice = set_voice
        self._speak = speak
        # Track what voice the TTS service is currently configured with so
        # we can short-circuit no-op switches (segment.voice_id ==
        # primary, which is the common "fallback" case from the backend's
        # _resolve_segment_voice).
        self._current_voice = primary_voice_id
        self._narrated_scene_id: str | None = None

    @property
    def narrated_scene_id(self) -> str | None:
        return self._narrated_scene_id

    @property
    def current_voice(self) -> str | None:
        return self._current_voice

    async def narrate(self, snapshot: dict) -> bool:
        """Narrate the snapshot's scripts.

        Returns ``True`` if at least one non-empty segment was spoken,
        ``False`` otherwise (empty/missing scripts, or already-narrated
        scene). The "already-narrated" check is what keeps
        ``canvas.register``-driven prompt rebuilds from re-narrating —
        ``current_scene.scene_id`` only changes on a real navigation,
        so successive canvas.register messages within the same scene
        return ``False`` and the caller can no-op.

        Snapshot shape: S65 (Option B) nests scene_id under
        ``current_scene.scene_id``.
        """
        current_scene = snapshot.get("current_scene") or {}
        scene_id = current_scene.get("scene_id")
        scene_id_str = str(scene_id) if scene_id else None

        if scene_id_str is not None and scene_id_str == self._narrated_scene_id:
            logger.info(
                "[NARRATION] skip: scene_id={!r} already narrated this session",
                scene_id_str,
            )
            return False

        is_relay = self._set_voice is None
        plan = plan_narration_segments(
            snapshot,
            primary_voice_id=self._primary_voice_id,
            is_relay=is_relay,
        )

        # Mark BEFORE iterating so a mid-narration cancellation still
        # blocks re-entry for this scene (matches S49 once-per-entry
        # semantics — a partial repeat would be worse than partial
        # silence, e.g. if the visitor disconnects mid-narration).
        if scene_id_str is not None:
            self._narrated_scene_id = scene_id_str

        if not plan:
            logger.info(
                "[NARRATION] no scripts to narrate for scene_id={!r}",
                scene_id_str,
            )
            return False

        logger.info(
            "[NARRATION] narrating scene_id={!r} segments={} primary_voice={!r} is_relay={}",
            scene_id_str,
            len(plan),
            self._primary_voice_id,
            is_relay,
        )

        for idx, seg in enumerate(plan):
            if (
                self._set_voice is not None
                and seg.voice_id
                and seg.voice_id != self._current_voice
            ):
                logger.info(
                    "[NARRATION] segment={} voice switch {!r} -> {!r}",
                    idx,
                    self._current_voice,
                    seg.voice_id,
                )
                await self._set_voice(seg.voice_id)
                self._current_voice = seg.voice_id
            logger.info(
                "[NARRATION] segment={} speak (voice={!r}, chars={})",
                idx,
                self._current_voice,
                len(seg.text),
            )
            await self._speak(seg.text)

        # Reset to primary BEFORE returning so the post-narration closing
        # line (and any subsequent conversational TTS) uses the agent
        # voice. Skipped when there's no primary configured (defensive)
        # or when we already happen to be on it.
        if (
            self._set_voice is not None
            and self._primary_voice_id
            and self._current_voice != self._primary_voice_id
        ):
            logger.info(
                "[NARRATION] resetting voice to primary {!r}",
                self._primary_voice_id,
            )
            await self._set_voice(self._primary_voice_id)
            self._current_voice = self._primary_voice_id

        return True


# ──────────────────────────────────────────────────────────────────────
# Scene-entry orchestrator (S65 G3 + G4)
# ──────────────────────────────────────────────────────────────────────


async def run_scene_narration(
    snapshot: dict | None,
    *,
    narrator: SceneNarrator,
    speak_followup: SpeakFn,
) -> bool:
    """Compose narrate + post-narration follow-up into one call.

    Spec'd order: per-segment narration → optional invitation/cue speak.
    The follow-up uses the SAME speak callable the narrator was wired
    with, so for the classic pipeline it awaits ``TTSStoppedFrame`` via
    ``NarrationCompletionGate`` (the invitation finishes rendering
    before this function returns), and for the relay pipeline it sends
    a ``RELAY_TEXT`` inside the open RELAY_TURN (caller closes the
    turn AFTER this function returns).

    Returns ``spoke_script`` so the caller can decide its
    developer-context message (presentation-done vs greeting-trigger)
    AND emit the ``script_complete`` Daily message itself. Emission is
    intentionally NOT part of this orchestrator because the relay
    pipeline must close its ``RELAY_TURN`` between the follow-up speak
    and the script_complete emit (otherwise SoulX receives turn-end
    after the shell has already auto-advanced — race). Letting the
    caller emit keeps that ordering owned by the caller, which is the
    one who knows about turn lifecycle.

    Used by ``bot.py`` in four places: classic + relay
    ``on_client_connected`` (session-start narration), and classic +
    relay ``refresh_agent_for_current_scene`` (scene-change narration,
    S65 Bug #2 fix — the agent-side companion to the shell's
    script_complete-driven auto-advance).
    """
    spoke_script = False
    if snapshot:
        spoke_script = await narrator.narrate(snapshot)
    followup = plan_post_narration_followup(snapshot or {}, spoke_script=spoke_script)
    if followup:
        await speak_followup(followup)
    return spoke_script


# ──────────────────────────────────────────────────────────────────────
# Per-segment completion gate (classic pipeline only)
# ──────────────────────────────────────────────────────────────────────


class NarrationCompletionGate(FrameProcessor):
    """FrameProcessor that lets callers await per-segment TTS completion.

    Sits downstream of the TTS service (between ``tts`` and
    ``output_transport``) so it observes every :class:`TTSStoppedFrame`
    the Cartesia service emits. Each :meth:`expect_next_stop` returns a
    Future that resolves on the **next** ``TTSStoppedFrame``, in
    registration order (FIFO).

    The classic narration callable queues ``TTSSpeakFrame`` and then
    awaits the future this gate hands back — so segment N's
    ``TTSUpdateSettingsFrame(voice=...)`` for segment N+1 only ships
    after segment N's audio has fully rendered. Without this, queueing
    all narration frames at once would still apply voice changes in the
    correct order (the TTS service processes ``TTSUpdateSettingsFrame``
    inline before the next ``TTSSpeakFrame``), BUT it would also emit
    ``script_complete`` before the visitor finished hearing the
    narration — which breaks S65's auto-advance plan, where
    ``script_complete`` is the trigger.

    Limitation: a future awaiting completion can be satisfied by the
    next ``TTSStoppedFrame`` from any source — including a stray
    LLM-driven TTS that finishes mid-narration. The session-start call
    site is quiescent (no concurrent TTS), so this isn't a problem
    today. Scene-change narration (S65+) lands while the visitor's
    conversation is paused on the navigation; if that ever races, we'll
    add a "wait for BotStoppedSpeakingFrame quiescence" gate before
    entering narration.
    """

    def __init__(self):
        super().__init__()
        self._pending: deque[asyncio.Future] = deque()

    def expect_next_stop(self) -> asyncio.Future:
        """Register a Future that will resolve on the next ``TTSStoppedFrame``.

        Call this BEFORE queuing the ``TTSSpeakFrame``, otherwise a fast
        TTS could complete and emit ``TTSStoppedFrame`` before the
        future is registered, leading to a stuck wait.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending.append(fut)
        return fut

    def cancel_all(self, reason: str = "narration_cancelled") -> None:
        """Cancel all pending futures (called on session teardown)."""
        while self._pending:
            fut = self._pending.popleft()
            if not fut.done():
                fut.cancel(reason)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSStoppedFrame) and self._pending:
            fut = self._pending.popleft()
            if not fut.done():
                fut.set_result(getattr(frame, "context_id", None))
        await self.push_frame(frame, direction)
