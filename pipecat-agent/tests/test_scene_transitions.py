"""S77 B6 — seamless scene transitions: zero spoken filler + the 600 ms gap.

The transcript probe runs the narration path over a 2-scene fixture and
asserts the produced utterance stream contains no transition token; the
gap tests pin SCENE_TRANSITION_PAUSE_MS=600 at the scene-advance site
(the pause runs after final-line playout, BEFORE script_complete — the
D-9 disposition; the shell's advance follows the emission).
"""

from __future__ import annotations

import asyncio
import re

from unittest.mock import AsyncMock

import narration as narration_mod
from narration import (
    SCENE_TRANSITION_PAUSE_MS,
    SceneNarrator,
    build_script_complete_payload,
    plan_scene_transition_pause_s,
    run_scene_narration,
)

TRANSITION_TOKENS = re.compile(
    r"let's continue|let us continue|moving on|next scene|continuing|let's move|shall we",
    re.IGNORECASE,
)


def _run(coro):
    return asyncio.run(coro)


def _snap(
    *, scene_id, scripts, scene_index, total_scenes, auto_advance=True, language="en"
):
    return {
        "live_room": {"auto_advance": auto_advance, "language": language},
        "flow_state": {"scene_index": scene_index, "total_scenes": total_scenes},
        "current_scene": {
            "scene_id": scene_id,
            "scripts": scripts,
            # Deliberately include a legacy cue — it must never be spoken.
            "narration": {
                "invitation_line": "Any questions?",
                "transition_cue": "Let's continue.",
            },
        },
    }


# ── The pure gap rule ─────────────────────────────────────────────────


def test_pause_constant_is_600ms():
    assert SCENE_TRANSITION_PAUSE_MS == 600


def test_pause_applies_only_on_auto_advance_not_last_spoken():
    auto_mid = _snap(scene_id="s1", scripts=[], scene_index=0, total_scenes=3)
    auto_last = _snap(scene_id="s3", scripts=[], scene_index=2, total_scenes=3)
    manual_mid = _snap(
        scene_id="s1", scripts=[], scene_index=0, total_scenes=3, auto_advance=False
    )

    assert (
        plan_scene_transition_pause_s(auto_mid, spoke_script=True)
        == SCENE_TRANSITION_PAUSE_MS / 1000.0
    )
    assert plan_scene_transition_pause_s(auto_last, spoke_script=True) == 0.0
    assert plan_scene_transition_pause_s(manual_mid, spoke_script=True) == 0.0
    assert plan_scene_transition_pause_s(auto_mid, spoke_script=False) == 0.0


# ── The transcript probe (2-scene auto-advance run) ───────────────────


def _drive(snapshot, narrator, utterances, sleeps, sends):
    async def _speak_followup(text: str) -> None:
        utterances.append(text)

    async def _go():
        spoke = await run_scene_narration(
            snapshot,
            narrator=narrator,
            speak_followup=_speak_followup,
        )
        await _send(build_script_complete_payload(snapshot, spoke_script=spoke))
        return spoke

    async def _send(payload) -> None:
        sends.append(payload)

    return _go()


def test_transcript_probe_two_scene_run_zero_transition_tokens(monkeypatch):
    """Both scenes narrate; the utterance stream between scene-boundary
    lines contains ZERO transition tokens; the inter-scene gap uses the
    constant (recorded via a patched narration-module sleep)."""
    utterances: list[str] = []
    sleeps: list[float] = []
    sends: list[dict] = []

    real_sleep = asyncio.sleep

    async def _recording_sleep(seconds, *args, **kwargs):
        sleeps.append(seconds)
        await real_sleep(0)  # keep the test instant

    monkeypatch.setattr(narration_mod.asyncio, "sleep", _recording_sleep)

    async def _speak(text: str) -> None:
        utterances.append(text)

    narrator = SceneNarrator(
        primary_voice_id="primary",
        set_voice=AsyncMock(),
        speak=_speak,
        set_language=AsyncMock(),
        room_language="en",
    )

    snap1 = _snap(
        scene_id="s1",
        scripts=[
            {
                "text": "welcome to scene one",
                "narration_text": "welcome to scene one",
                "narration_language": "en",
            },
            {
                "text": "the final line of scene one",
                "narration_text": "the final line of scene one",
                "narration_language": "en",
            },
        ],
        scene_index=0,
        total_scenes=2,
    )
    snap2 = _snap(
        scene_id="s2",
        scripts=[
            {
                "text": "scene two begins here",
                "narration_text": "scene two begins here",
                "narration_language": "en",
            }
        ],
        scene_index=1,
        total_scenes=2,
    )

    assert _run(_drive(snap1, narrator, utterances, sleeps, sends)) is True
    assert _run(_drive(snap2, narrator, utterances, sleeps, sends)) is True

    # ZERO transition tokens anywhere in the produced utterance stream.
    joined = "\n".join(utterances)
    assert not TRANSITION_TOKENS.search(joined), joined

    # The inter-scene gap fired exactly once (scene 1 → auto + not last),
    # using the constant; the LAST scene gets no gap (its followup is
    # the invitation — spoken, but carrying no transition token).
    assert sleeps == [SCENE_TRANSITION_PAUSE_MS / 1000.0]

    # script_complete still fired for both scenes, in order.
    assert [payload["sceneIndex"] for payload in sends] == [0, 1]
    assert all(payload["hadScript"] for payload in sends)
