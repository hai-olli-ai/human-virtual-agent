"""Tests for ``narration.py`` (S65 G3+G4 + Option B nested snapshot).

Layers under test:

  * :func:`narration.plan_narration_segments` — pure decision logic.
    Reads from ``snapshot["current_scene"]["scripts"]``.
  * :func:`narration.plan_post_narration_followup` — invitation OR cue
    branch decision. Reads ``live_room.auto_advance``,
    ``flow_state.scene_index`` / ``flow_state.total_scenes``, and
    ``current_scene.narration``.
  * :func:`narration.build_script_complete_payload` — shape contract.
    Reads ``flow_state.scene_index``.
  * :class:`narration.SceneNarrator` — runtime loop driving injected
    ``set_voice`` / ``speak`` callables. Idempotency keyed on
    ``current_scene.scene_id``.
  * :func:`narration.run_scene_narration` — orchestrator wiring narrate
    + followup. Used in bot.py for BOTH session-start and
    scene-change narration (S65 Bug #2 fix).
  * :func:`scene_context.build_scripts_section` — prompt directive
    contract (G5).

The :class:`narration.NarrationCompletionGate` FrameProcessor isn't
unit-tested directly here (it's a thin shim around a deque); its
correctness is exercised end-to-end by the SceneNarrator tests via a
stub ``speak`` callable that doesn't need the gate.

Snapshot shape: S65 (Option B) nests fields under ``live_room``,
``flow_state``, and ``current_scene``. The ``_snap`` helper below
builds well-formed nested fixtures so the per-test snapshots stay
terse. Pre-S65 (flat) snapshots are no longer used — backend cut over
in the same PR.

Follows the existing tests/ convention: no pytest-asyncio (not in the
dependency closure), so each async test goes through ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, call

from narration import (
    NarrationSegment,
    SceneNarrator,
    build_script_complete_payload,
    plan_narration_segments,
    plan_post_narration_followup,
    run_scene_narration,
)
from scene_context import build_scripts_section


def _run(coro):
    return asyncio.run(coro)


def _snap(
    *,
    scene_id: str | None = "scene-1",
    scripts: list | None = None,
    narration: dict | None = None,
    auto_advance: bool = False,
    scene_index: int = 0,
    total_scenes: int = 1,
) -> dict:
    """Build a nested S65 (Option B) snapshot fixture.

    Keeps per-test fixture noise low — only the fields a test cares
    about are passed in; everything else gets sensible defaults. The
    shape mirrors ``LiveRoomService.build_scene_snapshot``.
    """
    return {
        "live_room": {"auto_advance": auto_advance, "language": "en"},
        "flow_state": {"scene_index": scene_index, "total_scenes": total_scenes},
        "current_scene": {
            "scene_id": scene_id,
            "scripts": scripts if scripts is not None else [],
            "narration": narration if narration is not None else {},
        },
    }


# ──────────────────────────────────────────────────────────────────────
# plan_narration_segments — pure decision logic
# ──────────────────────────────────────────────────────────────────────


def test_plan_classic_resolves_per_segment_voice_with_primary_fallback():
    """voice_id present ⇒ used verbatim; voice_id missing ⇒ primary."""
    snapshot = _snap(scripts=[
        {"text": "intro", "order": 0, "voice_id": "voice-clone-A"},
        {"text": "middle", "order": 1, "voice_id": None},
        {"text": "outro", "order": 2},  # voice_id key absent entirely
    ])
    plan = plan_narration_segments(
        snapshot, primary_voice_id="primary-V", is_relay=False
    )
    assert plan == [
        NarrationSegment(text="intro", voice_id="voice-clone-A"),
        NarrationSegment(text="middle", voice_id="primary-V"),
        NarrationSegment(text="outro", voice_id="primary-V"),
    ]


def test_plan_relay_always_returns_none_voice_id():
    """Relay pipeline narrates in primary SoulX voice — no per-seg switching."""
    snapshot = _snap(scripts=[
        {"text": "a", "voice_id": "voice-clone-A"},
        {"text": "b", "voice_id": "voice-clone-B"},
    ])
    plan = plan_narration_segments(
        snapshot, primary_voice_id="primary-V", is_relay=True
    )
    assert all(seg.voice_id is None for seg in plan)
    assert [seg.text for seg in plan] == ["a", "b"]


def test_plan_filters_empty_and_whitespace_text():
    """Blank / whitespace-only segments are dropped (mirrors S49 loop)."""
    snapshot = _snap(scripts=[
        {"text": "kept", "voice_id": "v1"},
        {"text": "", "voice_id": "v2"},
        {"text": "   ", "voice_id": "v3"},
        {"text": "\n\t", "voice_id": "v4"},
        {"text": "also-kept", "voice_id": "v5"},
    ])
    plan = plan_narration_segments(
        snapshot, primary_voice_id="primary-V", is_relay=False
    )
    assert [seg.text for seg in plan] == ["kept", "also-kept"]


def test_plan_empty_or_missing_scripts_returns_empty():
    """Defensive: empty/missing current_scene OR scripts ⇒ empty plan."""
    assert plan_narration_segments({}, primary_voice_id="p", is_relay=False) == []
    assert plan_narration_segments(_snap(scripts=[]), primary_voice_id="p", is_relay=False) == []
    # `current_scene.scripts` explicitly None should also return empty.
    snap_none = _snap()
    snap_none["current_scene"]["scripts"] = None
    assert plan_narration_segments(snap_none, primary_voice_id="p", is_relay=False) == []
    # Snapshot without current_scene key at all.
    assert plan_narration_segments({"live_room": {}}, primary_voice_id="p", is_relay=False) == []


def test_plan_skips_non_dict_entries():
    """Defensive: tolerate the occasional malformed script row."""
    snapshot = _snap(scripts=[
        {"text": "ok", "voice_id": "v"},
        "not-a-dict",  # malformed entry
        None,
        {"text": "also-ok", "voice_id": "v"},
    ])
    plan = plan_narration_segments(
        snapshot, primary_voice_id="p", is_relay=False
    )
    assert [seg.text for seg in plan] == ["ok", "also-ok"]


def test_plan_preserves_server_order_does_not_resort():
    """The backend already sorts by ``order``; we trust that order."""
    # Note the deliberately reversed ``order`` field — if the helper
    # re-sorted, the output would come back as B, A, C; we want the
    # snapshot order preserved.
    snapshot = _snap(scripts=[
        {"text": "A", "order": 5, "voice_id": "v"},
        {"text": "B", "order": 1, "voice_id": "v"},
        {"text": "C", "order": 3, "voice_id": "v"},
    ])
    plan = plan_narration_segments(
        snapshot, primary_voice_id="p", is_relay=False
    )
    assert [seg.text for seg in plan] == ["A", "B", "C"]


# ──────────────────────────────────────────────────────────────────────
# SceneNarrator — per-segment voice switching (classic pipeline)
# ──────────────────────────────────────────────────────────────────────


def _make_classic_narrator(primary_voice_id: str | None = "primary-V"):
    """Construct a classic-mode narrator with mock set_voice + speak."""
    set_voice = AsyncMock()
    speak = AsyncMock()
    narrator = SceneNarrator(
        primary_voice_id=primary_voice_id,
        set_voice=set_voice,
        speak=speak,
    )
    return narrator, set_voice, speak


def test_narrate_classic_switches_voice_per_segment_then_resets():
    """Per-segment switch + reset-to-primary before returning."""
    narrator, set_voice, speak = _make_classic_narrator()
    snapshot = _snap(scripts=[
        {"text": "intro", "voice_id": "voice-A"},
        {"text": "middle", "voice_id": "voice-B"},
        {"text": "outro", "voice_id": "voice-A"},  # back to A
    ])
    spoken = _run(narrator.narrate(snapshot))
    assert spoken is True
    # Voice calls: primary→A, A→B, B→A, A→primary (reset at the end).
    assert set_voice.await_args_list == [
        call("voice-A"),
        call("voice-B"),
        call("voice-A"),
        call("primary-V"),
    ]
    # Speak calls in segment order.
    assert speak.await_args_list == [call("intro"), call("middle"), call("outro")]
    # Final voice is primary, ready for the closing line + conversation.
    assert narrator.current_voice == "primary-V"


def test_narrate_classic_skips_voice_call_when_already_active():
    """Adjacent segments on the same voice ⇒ only one switch call."""
    narrator, set_voice, speak = _make_classic_narrator(primary_voice_id="primary-V")
    snapshot = _snap(scripts=[
        {"text": "first", "voice_id": "voice-A"},
        {"text": "second", "voice_id": "voice-A"},  # same as previous
        {"text": "third", "voice_id": "voice-A"},
    ])
    _run(narrator.narrate(snapshot))
    # One switch primary→A at the start, then one reset A→primary at end.
    assert set_voice.await_args_list == [call("voice-A"), call("primary-V")]
    assert speak.await_count == 3


def test_narrate_classic_no_reset_when_already_on_primary():
    """All segments use primary voice ⇒ zero set_voice calls."""
    narrator, set_voice, speak = _make_classic_narrator(primary_voice_id="primary-V")
    snapshot = _snap(scripts=[
        {"text": "a", "voice_id": "primary-V"},  # matches primary
        {"text": "b", "voice_id": None},  # falls back to primary
    ])
    _run(narrator.narrate(snapshot))
    set_voice.assert_not_awaited()
    assert speak.await_count == 2
    assert narrator.current_voice == "primary-V"


def test_narrate_returns_false_for_empty_scripts():
    """No scripts ⇒ no speaks, no voice changes, return False."""
    narrator, set_voice, speak = _make_classic_narrator()
    spoken = _run(narrator.narrate(_snap(scripts=[])))
    assert spoken is False
    speak.assert_not_awaited()
    set_voice.assert_not_awaited()


def test_narrate_returns_false_when_scripts_key_absent():
    """Defensive: snapshot's current_scene missing the scripts key entirely."""
    narrator, set_voice, speak = _make_classic_narrator()
    snap = _snap()
    snap["current_scene"].pop("scripts")  # remove key explicitly
    spoken = _run(narrator.narrate(snap))
    assert spoken is False


def test_narrate_returns_false_when_all_segments_whitespace():
    """Scripts present but all blank ⇒ no speak, return False."""
    narrator, set_voice, speak = _make_classic_narrator()
    snapshot = _snap(scripts=[{"text": "  "}, {"text": ""}, {"text": "\t\n"}])
    spoken = _run(narrator.narrate(snapshot))
    assert spoken is False
    speak.assert_not_awaited()


# ──────────────────────────────────────────────────────────────────────
# SceneNarrator — idempotency per scene entry
# ──────────────────────────────────────────────────────────────────────


def test_narrate_idempotent_same_scene_id_second_call_is_noop():
    """Second narrate() with the same scene_id ⇒ skip (canvas.register safe)."""
    narrator, set_voice, speak = _make_classic_narrator()
    snapshot = _snap(scripts=[{"text": "intro", "voice_id": "voice-A"}])
    first = _run(narrator.narrate(snapshot))
    second = _run(narrator.narrate(snapshot))
    assert first is True
    assert second is False
    assert speak.await_count == 1  # second call did not re-speak


def test_narrate_re_narrates_when_scene_id_changes():
    """canvas.sceneChanged → new scene_id ⇒ narrate again.

    This is the test for Bug #2's wiring contract: when the agent
    re-enters the narrate path with a new scene_id (e.g. on
    canvas.sceneChanged), narration fires for that scene. The
    idempotency guard only short-circuits within the SAME scene.
    """
    narrator, set_voice, speak = _make_classic_narrator()
    snap1 = _snap(scene_id="scene-1", scripts=[{"text": "intro-1"}])
    snap2 = _snap(scene_id="scene-2", scripts=[{"text": "intro-2"}])
    assert _run(narrator.narrate(snap1)) is True
    assert _run(narrator.narrate(snap2)) is True
    assert speak.await_args_list == [call("intro-1"), call("intro-2")]


def test_narrate_idempotency_marks_scene_even_with_no_segments():
    """Empty scene marks the scene id so re-entry on canvas.register no-ops."""
    narrator, set_voice, speak = _make_classic_narrator()
    snapshot = _snap(scene_id="scene-empty", scripts=[])
    _run(narrator.narrate(snapshot))
    assert narrator.narrated_scene_id == "scene-empty"
    # A second call with the same scene_id still returns False (cached).
    assert _run(narrator.narrate(snapshot)) is False
    speak.assert_not_awaited()


def test_narrate_no_scene_id_does_not_persist_idempotency():
    """A snapshot whose current_scene has no scene_id ⇒ narrate, no mark."""
    narrator, set_voice, speak = _make_classic_narrator()
    snapshot = _snap(scene_id=None, scripts=[{"text": "free-floating"}])
    spoken = _run(narrator.narrate(snapshot))
    assert spoken is True
    # No scene_id → narrated_scene_id stays unset, so a repeat call
    # would re-narrate. This is the fail-open path used by tests.
    assert narrator.narrated_scene_id is None


# ──────────────────────────────────────────────────────────────────────
# plan_post_narration_followup — invitation OR cue branching (S65 G4)
# ──────────────────────────────────────────────────────────────────────


def test_followup_returns_none_when_no_script_was_spoken():
    """No-script scenes: existing conversational greeting handles intro."""
    snapshot = _snap(
        auto_advance=False, scene_index=0, total_scenes=3,
        narration={
            "invitation_line": "Any questions?",
            "transition_cue": "Onward.",
        },
    )
    assert plan_post_narration_followup(snapshot, spoke_script=False) is None


def test_followup_returns_invitation_when_manual_flow():
    """auto_advance=False ⇒ invitation regardless of scene index."""
    snapshot = _snap(
        auto_advance=False, scene_index=1, total_scenes=4,
        narration={
            "invitation_line": "What would you like to know?",
            "transition_cue": "Onward.",
        },
    )
    assert (
        plan_post_narration_followup(snapshot, spoke_script=True)
        == "What would you like to know?"
    )


def test_followup_returns_cue_when_auto_advance_and_not_last():
    """auto_advance=True + not last ⇒ cue, never invitation."""
    snapshot = _snap(
        auto_advance=True, scene_index=1, total_scenes=4,
        narration={
            "invitation_line": "Any questions?",
            "transition_cue": "Let's continue.",
        },
    )
    assert (
        plan_post_narration_followup(snapshot, spoke_script=True)
        == "Let's continue."
    )


def test_followup_returns_none_when_auto_advance_and_cue_blank():
    """auto_advance branch with blank cue ⇒ None (NOT invitation)."""
    snapshot = _snap(
        auto_advance=True, scene_index=0, total_scenes=3,
        narration={
            "invitation_line": "Any questions?",
            "transition_cue": "",  # blank — caller should skip the cue speak
        },
    )
    # NOT the invitation — auto-advance flow always skips the invitation.
    assert plan_post_narration_followup(snapshot, spoke_script=True) is None


def test_followup_returns_invitation_on_last_scene_even_with_auto_advance():
    """Last scene with auto_advance ⇒ invitation (nothing to advance TO)."""
    snapshot = _snap(
        auto_advance=True, scene_index=2,  # last (0-indexed)
        total_scenes=3,
        narration={
            "invitation_line": "What questions do you have?",
            "transition_cue": "Onward.",
        },
    )
    assert (
        plan_post_narration_followup(snapshot, spoke_script=True)
        == "What questions do you have?"
    )


def test_followup_falls_back_to_legacy_line_when_no_invitation_provided():
    """Pre-S65 backend snapshot ⇒ hardcoded English invitation fallback."""
    snapshot = _snap(auto_advance=False, scene_index=0, total_scenes=1)
    # no narration block populated
    snapshot["current_scene"].pop("narration")
    assert (
        plan_post_narration_followup(snapshot, spoke_script=True)
        == "Please feel free to ask me if you have any questions."
    )


def test_followup_treats_missing_flow_state_fields_as_single_scene():
    """No scene_index / total_scenes ⇒ single-scene defaults, invitation branch."""
    # Snapshot with current_scene.narration set but flow_state empty.
    snapshot = {
        "live_room": {},
        "flow_state": {},
        "current_scene": {"narration": {"invitation_line": "Ask me anything."}},
    }
    assert (
        plan_post_narration_followup(snapshot, spoke_script=True)
        == "Ask me anything."
    )


# ──────────────────────────────────────────────────────────────────────
# build_script_complete_payload — shape contract (S65 G4)
# ──────────────────────────────────────────────────────────────────────


def test_script_complete_payload_with_spoken_script():
    """sceneIndex sourced from flow_state.scene_index; hadScript reflects arg."""
    snapshot = _snap(scene_index=2, total_scenes=5)
    payload = build_script_complete_payload(snapshot, spoke_script=True)
    assert payload == {"type": "script_complete", "sceneIndex": 2, "hadScript": True, "trigger": "auto"}


def test_script_complete_payload_with_no_script():
    snapshot = _snap(scene_index=0, total_scenes=1)
    payload = build_script_complete_payload(snapshot, spoke_script=False)
    assert payload == {"type": "script_complete", "sceneIndex": 0, "hadScript": False, "trigger": "auto"}


def test_script_complete_payload_defaults_to_index_zero_for_no_snapshot():
    """Defensive: degraded session with no snapshot still produces a valid msg."""
    payload = build_script_complete_payload(None, spoke_script=False)
    assert payload == {"type": "script_complete", "sceneIndex": 0, "hadScript": False, "trigger": "auto"}


def test_script_complete_payload_defaults_when_flow_state_missing():
    """Snapshot present but no flow_state block ⇒ sceneIndex defaults to 0."""
    snapshot = {"live_room": {}, "current_scene": {"scene_id": "s1"}}
    payload = build_script_complete_payload(snapshot, spoke_script=True)
    assert payload == {"type": "script_complete", "sceneIndex": 0, "hadScript": True, "trigger": "auto"}


# ──────────────────────────────────────────────────────────────────────
# build_scripts_section — prompt directive (S65 G5) — reads nested shape
# ──────────────────────────────────────────────────────────────────────


def test_scripts_section_tells_llm_not_to_read_or_paraphrase():
    """S65 G5: directive must explicitly forbid the LLM from speaking the script."""
    snapshot = _snap(scripts=[{"text": "Hello world.", "order": 0}])
    section = build_scripts_section(snapshot)
    # The exact wording can drift; assert on the key contractual phrases
    # that distinguish the new directive from the legacy S49 wording.
    assert "narrated automatically" in section
    assert "DO NOT read or paraphrase" in section
    assert "stay silent" in section.lower()
    # The script content itself still appears so the LLM can reference it.
    assert "Hello world." in section
    # Sanity: the old misleading "you will present these to the visitor"
    # phrasing must NOT be present.
    assert "you will present these to the visitor" not in section


def test_scripts_section_empty_when_no_scripts():
    """Unchanged: no scripts ⇒ no section at all (and thus no directive)."""
    assert build_scripts_section({}) == ""
    assert build_scripts_section(_snap(scripts=[])) == ""
    assert build_scripts_section(_snap(scripts=[{"text": ""}])) == ""


# ──────────────────────────────────────────────────────────────────────
# End-to-end scenarios — compose the helpers the way bot.py does and
# assert call ordering across set_voice / speak / send_app_message
# (verifies items 3, 6, 7 of the S65 G3/G4 test plan). Uses the
# production orchestrator (``run_scene_narration``) directly, so a
# regression in the orchestrator surfaces here too.
# ──────────────────────────────────────────────────────────────────────


async def _drive_classic_scene_entry(snapshot: dict | None, narrator: SceneNarrator,
                                     speak: AsyncMock, send: AsyncMock) -> bool:
    """Mirror the orchestration in bot.py's run_bot_classic.on_client_connected.

    Calls the production ``run_scene_narration`` orchestrator so a
    regression in the orchestrator (not just in the helpers) surfaces
    in these scenario tests.
    """
    spoke_script = await run_scene_narration(
        snapshot,
        narrator=narrator,
        speak_followup=speak,
    )
    await send(build_script_complete_payload(snapshot, spoke_script=spoke_script))
    return spoke_script


def _make_event_log_narrator(primary_voice_id: str | None = "primary"):
    """Wire all three side-effect callables through a single events list so
    tests can assert on the cross-callable ordering (e.g. script_complete
    fires AFTER the followup speak, not before)."""
    events: list[tuple] = []

    set_voice = AsyncMock(side_effect=lambda v: events.append(("set_voice", v)))
    speak = AsyncMock(side_effect=lambda t: events.append(("speak", t)))
    send = AsyncMock(side_effect=lambda m: events.append(("send", m)))

    narrator = SceneNarrator(
        primary_voice_id=primary_voice_id,
        set_voice=set_voice,
        speak=speak,
    )
    return narrator, speak, send, events


def test_scenario_auto_advance_not_last_suppresses_invitation_speaks_cue_only():
    """Item 3: auto_advance + not last ⇒ cue spoken, invitation NOT spoken.

    This is the end-to-end version of
    ``test_followup_returns_cue_when_auto_advance_and_not_last``: that
    one verifies the decision helper; this one verifies the
    bot.py-style composition actually skips the invitation speak.
    """
    narrator, speak, send, events = _make_event_log_narrator()
    snapshot = _snap(
        scene_id="s1",
        scripts=[{"text": "hello world", "voice_id": "voice-A"}],
        auto_advance=True, scene_index=0, total_scenes=3,
        narration={
            "invitation_line": "Any questions?",
            "transition_cue": "Let's keep going.",
        },
    )
    spoke = _run(_drive_classic_scene_entry(snapshot, narrator, speak, send))
    assert spoke is True

    # Expected ordering: voice→A, speak script, voice→primary (reset),
    # speak cue (NOT the invitation), then script_complete.
    assert events == [
        ("set_voice", "voice-A"),
        ("speak", "hello world"),
        ("set_voice", "primary"),
        ("speak", "Let's keep going."),
        ("send", {"type": "script_complete", "sceneIndex": 0, "hadScript": True, "trigger": "auto"}),
    ]
    # Explicit: the invitation_line MUST NOT appear in the speak history.
    spoken_texts = [evt[1] for evt in events if evt[0] == "speak"]
    assert "Any questions?" not in spoken_texts


def test_scenario_script_complete_emitted_once_after_invitation_with_payload():
    """Item 6: script_complete fires exactly once, AFTER the invitation, with payload."""
    narrator, speak, send, events = _make_event_log_narrator()
    snapshot = _snap(
        scene_id="s1",
        scripts=[{"text": "intro", "voice_id": "voice-A"}],
        auto_advance=False,  # manual ⇒ invitation branch
        scene_index=1, total_scenes=4,
        narration={
            "invitation_line": "What would you like to explore?",
            "transition_cue": "Onward.",
        },
    )
    spoke = _run(_drive_classic_scene_entry(snapshot, narrator, speak, send))
    assert spoke is True

    # script_complete should be exactly one event and it should be LAST.
    send_events = [evt for evt in events if evt[0] == "send"]
    assert len(send_events) == 1
    assert events[-1] == send_events[0]
    assert events[-1] == (
        "send",
        {"type": "script_complete", "sceneIndex": 1, "hadScript": True, "trigger": "auto"},
    )

    # The invitation was spoken BEFORE script_complete (find both, compare indices).
    invitation_idx = events.index(("speak", "What would you like to explore?"))
    send_idx = events.index(send_events[0])
    assert invitation_idx < send_idx


def test_scenario_no_script_scene_skips_narration_and_invitation_emits_payload():
    """Item 7: no scripts ⇒ no narration, no invitation; script_complete fires once
    with hadScript=False so the shell can short-circuit auto-advance.

    Documents the no-script contract: script_complete IS emitted (with
    hadScript=False), NOT suppressed. The shell relies on the single
    signal regardless of script presence so it doesn't have to also
    inspect snapshot.current_scene.has_script.
    """
    narrator, speak, send, events = _make_event_log_narrator()
    snapshot = _snap(
        scene_id="s-empty",
        scripts=[],
        auto_advance=True, scene_index=0, total_scenes=3,
        narration={
            "invitation_line": "Any questions?",
            "transition_cue": "Onward.",
        },
    )
    spoke = _run(_drive_classic_scene_entry(snapshot, narrator, speak, send))
    assert spoke is False

    # No narration (no set_voice, no script speaks). No followup speak
    # (invitation suppressed because spoke_script is False — the
    # existing conversational greeting trigger covers no-script
    # scenes). Exactly one send for script_complete with
    # hadScript=False.
    assert events == [
        ("send", {"type": "script_complete", "sceneIndex": 0, "hadScript": False, "trigger": "auto"}),
    ]


# ──────────────────────────────────────────────────────────────────────
# Bug #2 — scene-change narration semantics via run_scene_narration
# ──────────────────────────────────────────────────────────────────────


def test_scenario_scene_change_rerruns_narration_with_new_scene_id():
    """S65 Bug #2: a second run_scene_narration call with a NEW scene_id
    narrates that scene's scripts (without idempotency skipping).

    This is the scenario the agent's refresh_agent_for_current_scene
    now triggers — without it, the shell auto-advances scene 0 → 1 but
    scene 1 never narrates, so scene_complete never fires for scene 1
    and auto-advance stalls. The narrator's idempotency guards only
    fire within the SAME scene_id.
    """
    narrator, speak, send, events = _make_event_log_narrator()

    snap1 = _snap(
        scene_id="s1",
        scripts=[{"text": "scene one"}],
        auto_advance=True, scene_index=0, total_scenes=3,
        narration={
            "invitation_line": "Q?",
            "transition_cue": "moving on",
        },
    )
    snap2 = _snap(
        scene_id="s2",
        scripts=[{"text": "scene two"}],
        auto_advance=True, scene_index=1, total_scenes=3,
        narration={
            "invitation_line": "Q?",
            "transition_cue": "moving on again",
        },
    )

    spoke1 = _run(_drive_classic_scene_entry(snap1, narrator, speak, send))
    spoke2 = _run(_drive_classic_scene_entry(snap2, narrator, speak, send))

    assert spoke1 is True
    assert spoke2 is True
    # Both scenes' scripts were spoken; the transition_cue from each
    # also fired; two script_complete events with their own sceneIndex.
    speaks = [evt[1] for evt in events if evt[0] == "speak"]
    sends = [evt[1] for evt in events if evt[0] == "send"]
    assert "scene one" in speaks
    assert "scene two" in speaks
    assert "moving on" in speaks
    assert "moving on again" in speaks
    assert sends == [
        {"type": "script_complete", "sceneIndex": 0, "hadScript": True, "trigger": "auto"},
        {"type": "script_complete", "sceneIndex": 1, "hadScript": True, "trigger": "auto"},
    ]


def test_scenario_scene_change_idempotency_skips_repeat_for_same_scene():
    """Repeated narrate() calls within the SAME scene_id ⇒ second is a no-op.

    Guards canvas.register-driven prompt rebuilds: those don't change
    scene_id, so they must not re-trigger narration.
    """
    narrator, speak, send, events = _make_event_log_narrator()
    snap = _snap(
        scene_id="s1",
        scripts=[{"text": "intro"}],
        auto_advance=False, scene_index=0, total_scenes=1,
        narration={"invitation_line": "Q?", "transition_cue": "—"},
    )
    spoke1 = _run(_drive_classic_scene_entry(snap, narrator, speak, send))
    spoke2 = _run(_drive_classic_scene_entry(snap, narrator, speak, send))

    assert spoke1 is True
    assert spoke2 is False
    # Only one script-speak fired across both calls; the second
    # invitation-speak was suppressed by spoke_script=False; script_complete
    # still fires twice (once per orchestration call, with hadScript
    # reflecting the actual narration outcome).
    speaks = [evt[1] for evt in events if evt[0] == "speak"]
    sends = [evt[1] for evt in events if evt[0] == "send"]
    assert speaks.count("intro") == 1
    assert sends == [
        {"type": "script_complete", "sceneIndex": 0, "hadScript": True, "trigger": "auto"},
        {"type": "script_complete", "sceneIndex": 0, "hadScript": False, "trigger": "auto"},
    ]


# ──────────────────────────────────────────────────────────────────────
# Relay narration — no per-segment voice switching
# ──────────────────────────────────────────────────────────────────────


def test_narrate_relay_never_calls_set_voice():
    """Relay narrator passes ``set_voice=None`` ⇒ voice never changes."""
    speak = AsyncMock()
    narrator = SceneNarrator(
        primary_voice_id=None,
        set_voice=None,
        speak=speak,
    )
    snapshot = _snap(scripts=[
        {"text": "a", "voice_id": "voice-A"},
        {"text": "b", "voice_id": "voice-B"},
    ])
    spoken = _run(narrator.narrate(snapshot))
    assert spoken is True
    # No set_voice callable to assert against — but verify the speak
    # calls fired in order and current_voice remained None throughout.
    assert speak.await_args_list == [call("a"), call("b")]
    assert narrator.current_voice is None
    assert narrator.narrated_scene_id == "scene-1"
