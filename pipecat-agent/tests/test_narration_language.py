"""S77 B5 — per-line TTS language.

Payload parsing with/without the new backend fields, the cached-audio
drop for translated lines, and the SceneNarrator language-switch loop:
switch exactly when narration_language changes, voice ID constant
across switches, room language restored for conversational turns.
"""

from __future__ import annotations

import asyncio

from unittest.mock import AsyncMock

from narration import (
    NarrationSegment,
    SceneNarrator,
    plan_narration_segments,
)


def _run(coro):
    return asyncio.run(coro)


def _snap(*, scripts: list, language: str = "en", scene_id: str = "scene-1") -> dict:
    return {
        "live_room": {"auto_advance": False, "language": language},
        "flow_state": {"scene_index": 0, "total_scenes": 1},
        "current_scene": {"scene_id": scene_id, "scripts": scripts, "narration": {}},
    }


# ── plan_narration_segments — payload parsing ─────────────────────────


def test_plan_reads_narration_text_and_language():
    """B4 fields present ⇒ the SPOKEN text is narration_text and the
    segment carries narration_language."""
    snapshot = _snap(
        language="vi",
        scripts=[
            {
                "text": "hello there",
                "narration_text": "xin chào quý vị",
                "narration_language": "vi",
                "voice_id": "voice-A",
            },
            {
                "text": "một dòng cố định",
                "narration_text": "một dòng cố định",
                "narration_language": "vi",
                "voice_id": "voice-A",
            },
        ],
    )
    plan = plan_narration_segments(
        snapshot, primary_voice_id="primary-V", is_relay=False
    )
    assert [seg.text for seg in plan] == ["xin chào quý vị", "một dòng cố định"]
    assert [seg.language for seg in plan] == ["vi", "vi"]


def test_plan_legacy_snapshot_falls_back_to_text_and_room_language():
    """Pre-B4 backend tolerance (rollout law): no narration_* fields ⇒
    speak the legacy text field in the ROOM language."""
    snapshot = _snap(
        language="vi",
        scripts=[{"text": "legacy line", "voice_id": "voice-A"}],
    )
    plan = plan_narration_segments(
        snapshot, primary_voice_id="primary-V", is_relay=False
    )
    assert plan == [
        NarrationSegment(text="legacy line", voice_id="voice-A", language="vi"),
    ]


def test_plan_drops_cached_audio_for_translated_lines():
    """The S65b cache holds BASE-text renders (no language in its key).
    A line served as a translation must NOT replay cached bytes — the
    visitor would hear the wrong language. Base-text lines keep theirs."""
    audio = {
        "url": "https://media.hv.ai/narration-cache/x.pcm",
        "format": "pcm_s16le",
        "sample_rate": 24000,
        "duration_ms": 900,
    }
    snapshot = _snap(
        language="vi",
        scripts=[
            {
                "id": "seg-translated",
                "text": "hello",
                "narration_text": "xin chào",
                "narration_language": "vi",
                "audio": dict(audio),
            },
            {
                "id": "seg-base",
                "text": "hello again",
                "narration_text": "hello again",
                "narration_language": "en",
                "audio": dict(audio),
            },
        ],
    )
    plan = plan_narration_segments(
        snapshot, primary_voice_id="primary-V", is_relay=False
    )
    assert plan[0].audio is None  # translated ⇒ cached base-language PCM dropped
    assert plan[1].audio is not None  # base text spoken ⇒ cache still valid


# ── SceneNarrator — the language-switch loop ──────────────────────────


def _make_language_narrator(room_language: str = "en"):
    events: list[tuple] = []
    set_voice = AsyncMock(side_effect=lambda v: events.append(("set_voice", v)))
    set_language = AsyncMock(side_effect=lambda c: events.append(("set_language", c)))
    speak = AsyncMock(side_effect=lambda t: events.append(("speak", t)))
    narrator = SceneNarrator(
        primary_voice_id="primary",
        set_voice=set_voice,
        speak=speak,
        set_language=set_language,
        room_language=room_language,
    )
    return narrator, events


def test_language_switch_called_exactly_on_change_voice_constant():
    """3-line mixed fixture (en → vi → vi) in an 'en' room: exactly one
    switch to 'vi' (line 2; line 3 is a no-op), then the room-language
    restore. The voice never changes across the switches (Q8)."""
    narrator, events = _make_language_narrator(room_language="en")
    snapshot = _snap(
        language="en",
        scripts=[
            {
                "text": "line one",
                "narration_text": "line one",
                "narration_language": "en",
                "voice_id": "primary",
            },
            {
                "text": "line two",
                "narration_text": "dòng hai",
                "narration_language": "vi",
                "voice_id": "primary",
            },
            {
                "text": "line three",
                "narration_text": "dòng ba",
                "narration_language": "vi",
                "voice_id": "primary",
            },
        ],
    )
    spoke = _run(narrator.narrate(snapshot))
    assert spoke is True

    language_calls = [evt[1] for evt in events if evt[0] == "set_language"]
    assert language_calls == ["vi", "en"]  # exactly one switch + the restore
    # Voice ID constant: every segment is on the primary voice, so no
    # voice delta ever ships.
    assert [evt for evt in events if evt[0] == "set_voice"] == []
    assert narrator.current_language == "en"  # conversational turns keep room lang


def test_no_language_switch_when_all_lines_match_room():
    narrator, events = _make_language_narrator(room_language="vi")
    snapshot = _snap(
        language="vi",
        scripts=[
            {"text": "a", "narration_text": "một", "narration_language": "vi"},
            {"text": "b", "narration_text": "hai", "narration_language": "vi"},
        ],
    )
    assert _run(narrator.narrate(snapshot)) is True
    assert [evt for evt in events if evt[0] == "set_language"] == []


def test_language_restored_after_interruption():
    """The barge-in path resets language along with voice — the LLM's
    conversational reply must not render in a script line's language."""
    from narration import NarrationInterrupted

    events: list[tuple] = []
    set_language = AsyncMock(side_effect=lambda c: events.append(("set_language", c)))

    async def _speak(text: str) -> None:
        if text == "dòng nổ":
            raise NarrationInterrupted()
        events.append(("speak", text))

    narrator = SceneNarrator(
        primary_voice_id="primary",
        set_voice=AsyncMock(),
        speak=_speak,
        set_language=set_language,
        room_language="en",
    )
    snapshot = _snap(
        language="en",
        scripts=[
            {"text": "boom", "narration_text": "dòng nổ", "narration_language": "vi"}
        ],
    )

    async def _go():
        try:
            await narrator.narrate(snapshot)
        except NarrationInterrupted:
            pass

    _run(_go())
    language_calls = [evt[1] for evt in events if evt[0] == "set_language"]
    assert language_calls == ["vi", "en"]  # switched in, restored on interrupt
    assert narrator.current_language == "en"


def test_relay_pipeline_skips_language_switching():
    """No Cartesia TTS on relay — set_language stays None and the loop
    must not blow up on language-bearing segments."""
    speak = AsyncMock()
    narrator = SceneNarrator(
        primary_voice_id=None,
        set_voice=None,
        speak=speak,
        room_language="vi",
    )
    snapshot = _snap(
        language="vi",
        scripts=[
            {"text": "hello", "narration_text": "xin chào", "narration_language": "vi"}
        ],
    )
    assert _run(narrator.narrate(snapshot)) is True
    speak.assert_awaited_once_with("xin chào")
