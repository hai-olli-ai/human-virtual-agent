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

Auto Play Phase A extended the gate beyond per-segment sequencing:

  * **Playout drain (A1).** ``TTSStoppedFrame`` fires at *synthesis*
    complete (live Cartesia renders several× faster than realtime), so
    gating ``script_complete`` on it alone told the shell "done" while
    seconds of audio were still queued in the output transport.
    :meth:`NarrationCompletionGate.expect_playout_drain` resolves on the
    next :class:`BotStoppedSpeakingFrame` — which the transport
    broadcasts upstream through this gate's pipeline position only when
    its audio queue actually drains. ``run_scene_narration`` awaits it
    (via the injected ``wait_playout``) after the final speak, so the
    emission callers perform afterwards is truthful.
  * **Interruption awareness (A2).** Visitor speech pushes an
    :class:`InterruptionFrame` through the pipeline; the transport
    flushes its audio but Cartesia drops the context's ``done`` message,
    orphaning the pending gate future (a 30 s stall, then narration
    would resume mid-scene). The gate now observes ``InterruptionFrame``
    and resolves ALL pending futures with :data:`NARRATION_INTERRUPTED`;
    the speak/drain callables raise :class:`NarrationInterrupted`, the
    narrator aborts the segment loop, and callers suppress
    ``script_complete`` (frozen wire rule 2b: interrupted runs never
    emit). An interruption landing in the between-segments window (no
    future registered) is latched in ``_interrupted_since_run_start``
    and kills the run's *next* expect call; :meth:`~NarrationCompletionGate.begin_run`
    clears the latch at the start of each narration run.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class _InterruptedSentinel:
    """Result value the gate resolves futures with on ``InterruptionFrame``.

    A dedicated object (not ``None``/a string) so it can never collide
    with the ``context_id`` payload a normal ``TTSStoppedFrame``
    resolution carries.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<NARRATION_INTERRUPTED>"


NARRATION_INTERRUPTED = _InterruptedSentinel()


class NarrationInterrupted(Exception):
    """Visitor speech (or a deliberate flush) interrupted the narration run.

    Raised by the pipeline-side speak / wait-playout callables when a
    gate future resolves with :data:`NARRATION_INTERRUPTED`. Propagates
    through :meth:`SceneNarrator.narrate` (which resets the voice to
    primary on the way out) and :func:`run_scene_narration`; callers
    catch it and suppress the ``script_complete`` emission — an
    interrupted run must never advance the flow (wire rule 2b).
    """


SpeakFn = Callable[[str], Awaitable[None]]
SetVoiceFn = Callable[[str], Awaitable[None]]
# Block 13 — narration cache callables, both optional.
#  * PrefetchFn runs ONCE at the top of the per-scene narration loop
#    with the full plan; implementation stashes any cacheable PCM bytes
#    in a side-channel keyed on segment id.
#  * PrimeFn runs per-segment immediately before ``speak`` (sync —
#    no awaits between prime and the TTSSpeakFrame queue); returns
#    True iff the segment will be played from cache, in which case the
#    narrator skips the per-segment voice switch.
PrefetchFn = Callable[[list["NarrationSegment"]], Awaitable[None]]
PrimeFn = Callable[["NarrationSegment"], bool]
# Auto Play Phase A (A1) — final playout-drain callable, optional. Runs
# ONCE at the end of run_scene_narration (only when something was
# actually spoken) with the timeout budget from
# compute_playout_drain_timeout. The classic pipeline passes a closure
# awaiting NarrationCompletionGate.expect_playout_drain; the relay
# pipeline passes None (SoulX owns its own playout — v1 punt, see
# CLAUDE.md).
WaitPlayoutFn = Callable[[float], Awaitable[None]]


# ──────────────────────────────────────────────────────────────────────
# Pure helpers (no I/O — unit-testable directly)
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NarrationSegment:
    """One narration segment.

    ``voice_id`` is the resolved Cartesia voice for the segment in the
    classic pipeline, or ``None`` in the relay pipeline (where SoulX
    handles its own voice and we never switch).

    ``id`` and ``audio`` come from the Block 8 snapshot contract — when
    present, they let :class:`services.cached_first_tts.CachedFirstTTSService`
    replay pre-rendered PCM instead of synthesizing. Both are ``None``
    for pre-Block-8 snapshots and for the relay pipeline (SoulX renders
    speech itself, so the bytes wouldn't be useful).
    """

    text: str
    voice_id: str | None
    id: str | None = None
    audio: dict | None = None


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
    trigger: str = "auto",
) -> dict:
    """Build the ``script_complete`` Daily app-message payload (S65 G4).

    Pure: no I/O. The shell uses this payload (specifically
    ``hadScript`` + ``sceneIndex``) to drive auto-advance — it advances
    on ``script_complete`` only when the agent reports it actually
    narrated something for the current scene. ``sceneIndex`` is the
    snapshot's ``flow_state.scene_index`` (0-based), defaulting to
    ``0`` for the degenerate no-snapshot case.

    S65c Block 4 — ``trigger`` is ALWAYS present on the wire (default
    ``"auto"``). The shell's auto-advance handler short-circuits when it
    sees ``trigger=="manual"``, so a Script-button click on an
    auto-advance room never queues a scene jump. Auto-path emits keep
    the explicit ``trigger:"auto"`` for self-describing wire semantics —
    the field rides along on every ``script_complete`` so future
    additions (e.g. ``"keyboard_shortcut"``) don't require a wire
    versioning bump.
    """
    snap = snapshot or {}
    flow_state = snap.get("flow_state") or {}
    return {
        "type": "script_complete",
        "sceneIndex": int(flow_state.get("scene_index", 0) or 0),
        "hadScript": bool(spoke_script),
        "trigger": trigger,
    }


# Auto Play Phase A (A1) — playout-drain timeout budget. The drain wait
# resolves on BotStoppedSpeakingFrame; these bounds only matter when that
# frame is lost (the transport's own queue-empty fallback fires one within
# ~3 s of true drain, so in practice they never trip). MARGIN also covers
# the always-live followup line's playout tail on all-cached scenes.
PLAYOUT_DRAIN_MARGIN_S = 15.0
PLAYOUT_DRAIN_FALLBACK_S = 60.0


def compute_playout_drain_timeout(
    snapshot: dict | None,
    *,
    margin_s: float = PLAYOUT_DRAIN_MARGIN_S,
    fallback_s: float = PLAYOUT_DRAIN_FALLBACK_S,
) -> float:
    """Upper bound (seconds) for the final playout-drain wait (A1).

    Pure: no I/O. Prefers the sum of the scene's known cached-audio
    durations plus a margin; any narratable segment with a missing,
    zero, or malformed ``audio.duration_ms`` makes the total unknown
    (a backend dedup edge can legitimately serve 0 — treat it as
    unknown, per the Phase A brief) and the fixed fallback cap applies.
    Blank segments are skipped entirely — they're never narrated, so
    they neither add time nor force the fallback.
    """
    current_scene = (snapshot or {}).get("current_scene") or {}
    raw = current_scene.get("scripts") or []
    total_ms = 0.0
    saw_segment = False
    for seg in raw:
        if not isinstance(seg, dict):
            continue
        if not (seg.get("text") or "").strip():
            continue
        saw_segment = True
        audio = seg.get("audio") if isinstance(seg.get("audio"), dict) else None
        duration = (audio or {}).get("duration_ms")
        try:
            duration_ms = float(duration) if duration is not None else 0.0
        except (TypeError, ValueError):
            duration_ms = 0.0
        if duration_ms <= 0:
            return fallback_s
        total_ms += duration_ms
    if not saw_segment:
        return fallback_s
    return total_ms / 1000.0 + margin_s


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
        # Block 13 — propagate id + audio for the cache layer. Both stay
        # None on pre-Block-8 snapshots and on relay (where SoulX renders
        # speech itself, so cached PCM is moot).
        seg_id = seg.get("id")
        audio = seg.get("audio") if isinstance(seg.get("audio"), dict) else None
        if is_relay:
            # Relay pipeline narrates in the primary SoulX voice; the
            # per-segment voice clone is a v0.2 punt (CLAUDE.md S65).
            plan.append(
                NarrationSegment(text=text, voice_id=None, id=seg_id, audio=None)
            )
        else:
            voice = seg.get("voice_id") or primary_voice_id
            plan.append(
                NarrationSegment(text=text, voice_id=voice, id=seg_id, audio=audio)
            )
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
        prefetch: PrefetchFn | None = None,
        prime: PrimeFn | None = None,
    ):
        self._primary_voice_id = primary_voice_id
        self._set_voice = set_voice
        self._speak = speak
        # Block 13 — narration cache. Both None ⇒ behaves exactly like
        # the pre-cache narrator; the relay pipeline always passes None
        # because SoulX renders speech itself.
        self._prefetch = prefetch
        self._prime = prime
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

    async def narrate(self, snapshot: dict, *, force: bool = False) -> bool:
        """Narrate the snapshot's scripts.

        Returns ``True`` if at least one non-empty segment was spoken,
        ``False`` otherwise (empty/missing scripts, or already-narrated
        scene). The "already-narrated" check is what keeps
        ``canvas.register``-driven prompt rebuilds from re-narrating —
        ``current_scene.scene_id`` only changes on a real navigation,
        so successive canvas.register messages within the same scene
        return ``False`` and the caller can no-op.

        S65c Block 4 — ``force=True`` bypasses the once-per-entry guard.
        The S65c Script button (manual replay via ``request_narrate``)
        passes ``force=True`` so a visitor can re-narrate the same scene
        on demand. The guard still SETS ``_narrated_scene_id`` after a
        forced narrate so a subsequent unforced auto-trigger
        (e.g. ``canvas.register``) still no-ops as designed.

        Snapshot shape: S65 (Option B) nests scene_id under
        ``current_scene.scene_id``.
        """
        current_scene = snapshot.get("current_scene") or {}
        scene_id = current_scene.get("scene_id")
        scene_id_str = str(scene_id) if scene_id else None

        if (
            not force
            and scene_id_str is not None
            and scene_id_str == self._narrated_scene_id
        ):
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

        # Block 13 — single batched prefetch BEFORE the loop. Failures
        # downgrade to live: a logged warning + every segment misses
        # cleanly. Placed after the empty-plan + already-narrated guards
        # so we never fetch bytes for a scene we won't narrate.
        if self._prefetch is not None:
            try:
                await self._prefetch(plan)
            except Exception as exc:
                logger.warning("[NARRATION] prefetch failed; live fallback: {!r}", exc)

        try:
            for idx, seg in enumerate(plan):
                # Prime is sync — keeps zero awaits between the prime call
                # and the TTSSpeakFrame queue so no other run_tts can sneak
                # in and consume the primed segment.
                is_hit = bool(self._prime(seg)) if self._prime is not None else False
                if (
                    not is_hit
                    and self._set_voice is not None
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
                    "[NARRATION] segment={} speak (voice={!r}, chars={}, hit={})",
                    idx,
                    self._current_voice,
                    len(seg.text),
                    is_hit,
                )
                await self._speak(seg.text)
        except NarrationInterrupted:
            # Auto Play Phase A (A2) — the visitor barged in mid-scene.
            # Abort the segment loop, but STILL reset the voice to
            # primary: the LLM's conversational reply to the barge-in
            # would otherwise render in the script avatar's voice (the
            # TTSUpdateSettingsFrame delta is a control frame — safe to
            # queue during/after an interruption flush).
            logger.info("[NARRATION] interrupted mid-scene — aborting segment loop")
            await self._reset_to_primary()
            raise
        except asyncio.CancelledError:
            # Phase A — a cancelled run (autoplay stop / scene-change
            # supersede / disconnect) needs the same voice reset, or the
            # conversation stays stuck in the script avatar's clone
            # indefinitely (stop has no follow-up run to realign it).
            # Shielded so a second cancellation can't kill the cleanup;
            # the reset only queues one control frame, so it's cheap.
            logger.info("[NARRATION] cancelled mid-scene — resetting voice to primary")
            await asyncio.shield(self._reset_to_primary())
            raise

        # Reset to primary BEFORE returning so the post-narration closing
        # line (and any subsequent conversational TTS) uses the agent
        # voice. Skipped when there's no primary configured (defensive)
        # or when we already happen to be on it.
        await self._reset_to_primary()

        return True

    async def _reset_to_primary(self) -> None:
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


# ──────────────────────────────────────────────────────────────────────
# Scene-entry orchestrator (S65 G3 + G4)
# ──────────────────────────────────────────────────────────────────────


async def run_scene_narration(
    snapshot: dict | None,
    *,
    narrator: SceneNarrator,
    speak_followup: SpeakFn,
    force: bool = False,
    wait_playout: WaitPlayoutFn | None = None,
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

    S65c Block 4 — ``force`` is threaded through to
    :meth:`SceneNarrator.narrate` so the S65c Script button can re-narrate
    a scene the visitor already heard. ``trigger`` is NOT a parameter
    here because the orchestrator doesn't emit; the manual handler in
    ``bot.py`` constructs the ``script_complete`` payload with
    ``build_script_complete_payload(..., trigger="manual")`` directly.

    Auto Play Phase A (A1) — ``wait_playout``, when provided and when
    something was actually spoken, is awaited LAST (after the followup)
    with the drain-timeout budget from
    :func:`compute_playout_drain_timeout`. It returns only once the
    output transport's audio queue has truly drained
    (``BotStoppedSpeakingFrame``), so the caller's ``script_complete``
    emission means "the visitor finished hearing it" — not "synthesis
    finished". Script-less scenes never call it: the ``hadScript:false``
    emission stays immediate (wire rule 4). May raise
    :class:`NarrationInterrupted` (like the speak callables), which
    callers translate into a suppressed emission.

    Used by ``bot.py`` in four places: classic + relay
    ``on_client_connected`` (session-start narration), and classic +
    relay ``refresh_agent_for_current_scene`` (scene-change narration,
    S65 Bug #2 fix — the agent-side companion to the shell's
    script_complete-driven auto-advance).
    """
    spoke_script = False
    if snapshot:
        spoke_script = await narrator.narrate(snapshot, force=force)
    followup = plan_post_narration_followup(snapshot or {}, spoke_script=spoke_script)
    if followup:
        await speak_followup(followup)
    if wait_playout is not None and (spoke_script or followup):
        await wait_playout(compute_playout_drain_timeout(snapshot))
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

    Auto Play Phase A additions (see module docstring for the why):

      * :meth:`expect_playout_drain` — a Future resolving when every
        utterance synthesized this run has finished playing out, or
        immediately when that's already true (covers the cached-playback
        race where the drain lands before the caller registers, and the
        no-audio edge). Resolves with :data:`NARRATION_INTERRUPTED` on
        interruption. **Per-utterance accounting is load-bearing:** the
        transport emits ``BotStoppedSpeakingFrame`` at EVERY utterance
        boundary (``MediaSender._handle_frame`` fires
        ``_bot_stopped_speaking`` each time a ``TTSStoppedFrame`` is
        dequeued from its audio queue), and because per-segment gating
        releases at *synthesis*-complete, a multi-segment run stacks
        several utterances in the transport queue — resolving on the
        FIRST ``BotStoppedSpeakingFrame`` would re-open Bug 1 for every
        multi-utterance scene (segment 1's boundary, segments 2..N +
        followup still queued). The gate therefore counts synthesized
        utterances (``TTSStoppedFrame`` downstream, only when audio
        frames were seen — mirroring the transport's
        ``_tts_audio_received`` guard) against played-out utterances
        (``BotStoppedSpeakingFrame`` upstream, only on a true
        speaking→quiet transition — a post-flush stray is ignored) and
        releases the drain only when played ≥ synthesized.
      * :meth:`expect_next_stop` / :meth:`expect_playout_drain` resolve
        immediately with :data:`NARRATION_INTERRUPTED` when an
        ``InterruptionFrame`` has been observed since the last
        :meth:`begin_run` — closes the between-segments window where no
        future is registered.
      * :meth:`expect_interruption` — a fresh Future resolving on the
        NEXT ``InterruptionFrame`` regardless of the latch. Used by
        bot.py's ``_flush_bot_audio`` to confirm that the flush it just
        queued has traversed the TTS+gate positions (so the next
        narration run's frames can't be swallowed by it).
      * :meth:`begin_run` — per-run reset: clears the interruption latch
        and cancels stale stop/drain futures so a leftover (e.g. a
        timed-out, already-cancelled future) can't consume the new run's
        first ``TTSStoppedFrame`` and shift the FIFO off by one.
      * :attr:`bot_is_speaking` — mirror of the transport's speaking
        state (``BotStartedSpeakingFrame`` / ``BotStoppedSpeakingFrame``),
        used by bot.py to decide whether a scene change needs an audio
        flush at all.
    """

    def __init__(self):
        super().__init__()
        self._pending: deque[asyncio.Future] = deque()
        self._drain_pending: deque[asyncio.Future] = deque()
        self._interrupt_waiters: deque[asyncio.Future] = deque()
        self._bot_speaking = False
        self._interrupted_since_run_start = False
        # Per-run utterance accounting (see class docstring): synthesized
        # counts TTSStoppedFrames whose utterance produced audio;
        # played counts true speaking→quiet transitions. Drain resolves
        # only when played >= synthesized.
        self._synthesized_utterances = 0
        self._played_utterances = 0
        self._utterance_saw_audio = False

    @property
    def bot_is_speaking(self) -> bool:
        """Mirror of the output transport's bot-speaking state."""
        return self._bot_speaking

    def _all_playouts_observed(self) -> bool:
        return (
            not self._bot_speaking
            and self._played_utterances >= self._synthesized_utterances
        )

    def begin_run(self) -> None:
        """Reset per-run state at the start of a narration run (Phase A).

        Clears the interruption latch (a conversational barge-in from
        *before* this run must not abort it), resets the utterance
        counters, and drops any stale stop/drain futures left over from
        a superseded or timed-out run (a stale entry at the head of the
        FIFO would otherwise consume this run's first
        ``TTSStoppedFrame``). Interrupt waiters are NOT touched — they
        belong to an in-flight flush, not to a run.
        """
        self._interrupted_since_run_start = False
        self._synthesized_utterances = 0
        self._played_utterances = 0
        self._utterance_saw_audio = False
        self._cancel_deque(self._pending, "superseded_by_new_run")
        self._cancel_deque(self._drain_pending, "superseded_by_new_run")

    def expect_next_stop(self) -> asyncio.Future:
        """Register a Future that will resolve on the next ``TTSStoppedFrame``.

        Call this BEFORE queuing the ``TTSSpeakFrame``, otherwise a fast
        TTS could complete and emit ``TTSStoppedFrame`` before the
        future is registered, leading to a stuck wait. Resolves
        immediately with :data:`NARRATION_INTERRUPTED` when an
        interruption already landed since :meth:`begin_run` — callers
        must check ``fut.done()`` before queuing the speak frame.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        if self._interrupted_since_run_start:
            fut.set_result(NARRATION_INTERRUPTED)
            return fut
        self._pending.append(fut)
        return fut

    def expect_playout_drain(self) -> asyncio.Future:
        """Register a Future resolving when queued bot audio truly drains.

        Resolution order of precedence:

          * interruption already latched this run ⇒ immediate
            :data:`NARRATION_INTERRUPTED`;
          * every synthesized utterance already played out (and the bot
            is quiet) ⇒ immediate ``None`` — covers the cached-playback
            race (drain landed before registration) and the no-audio
            edge. The counter comparison, not just ``bot_is_speaking``,
            is what keeps a registration landing in the microsecond gap
            BETWEEN utterance boundaries pending correctly;
          * otherwise pends until played ≥ synthesized (``None``) or an
            ``InterruptionFrame`` (:data:`NARRATION_INTERRUPTED`).
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        if self._interrupted_since_run_start:
            fut.set_result(NARRATION_INTERRUPTED)
        elif self._all_playouts_observed():
            fut.set_result(None)
        else:
            self._drain_pending.append(fut)
        return fut

    def expect_interruption(self) -> asyncio.Future:
        """Register a Future resolving on the NEXT ``InterruptionFrame``.

        Deliberately ignores the latch: the caller (bot.py's audio
        flush) is about to CAUSE an interruption and needs to observe
        that specific one passing this pipeline position, not a stale
        latch from an earlier barge-in.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._interrupt_waiters.append(fut)
        return fut

    def cancel_all(self, reason: str = "narration_cancelled") -> None:
        """Cancel all pending futures (called on session teardown)."""
        self._cancel_deque(self._pending, reason)
        self._cancel_deque(self._drain_pending, reason)
        self._cancel_deque(self._interrupt_waiters, reason)

    @staticmethod
    def _cancel_deque(pending: deque, reason: str) -> None:
        while pending:
            fut = pending.popleft()
            if not fut.done():
                fut.cancel(reason)

    @staticmethod
    def _resolve_deque(pending: deque, result) -> None:
        while pending:
            fut = pending.popleft()
            if not fut.done():
                fut.set_result(result)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSStoppedFrame):
            if self._pending:
                fut = self._pending.popleft()
                if not fut.done():
                    fut.set_result(getattr(frame, "context_id", None))
            # Count the utterance as synthesized only if it produced
            # audio — mirrors the transport's _tts_audio_received guard,
            # which suppresses the per-utterance BotStoppedSpeakingFrame
            # for audio-less utterances. Counting those here would leave
            # played < synthesized forever (drain waits ride the timeout
            # backstop instead of hanging, but why pay it).
            if self._utterance_saw_audio:
                self._synthesized_utterances += 1
                self._utterance_saw_audio = False
        elif isinstance(frame, TTSAudioRawFrame):
            self._utterance_saw_audio = True
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # Guarded on a true speaking→quiet transition: the transport
            # pushes one extra BotStoppedSpeakingFrame from
            # handle_interruptions after a flush, which arrives when the
            # gate already marked itself quiet (InterruptionFrame branch
            # below) — counting it would skew a subsequent run's drain
            # accounting one utterance early.
            if self._bot_speaking:
                self._played_utterances += 1
            self._bot_speaking = False
            if self._played_utterances >= self._synthesized_utterances:
                self._resolve_deque(self._drain_pending, None)
        elif isinstance(frame, InterruptionFrame):
            self._bot_speaking = False
            self._interrupted_since_run_start = True
            self._resolve_deque(self._pending, NARRATION_INTERRUPTED)
            self._resolve_deque(self._drain_pending, NARRATION_INTERRUPTED)
            self._resolve_deque(self._interrupt_waiters, None)
        await self.push_frame(frame, direction)
