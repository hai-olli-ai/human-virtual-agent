"""S83 PR-13 — contract-mirror tests for the `user_text` inbound.

The router queues the VAD trio (Started → Transcription → Stopped) so a
typed question is byte-equivalent to speech; the validation is the
importable helper (the `request_quiz_ready` precedent)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from tools.user_text import USER_TEXT_MAX_CHARS, clean_user_text


def _run(coro):
    return asyncio.run(coro)


def test_clean_user_text_strips_and_caps():
    assert clean_user_text({"text": "  hello  "}) == "hello"
    long = "x" * (USER_TEXT_MAX_CHARS + 50)
    assert clean_user_text({"text": long}) == "x" * USER_TEXT_MAX_CHARS


def test_clean_user_text_rejects_garbage():
    assert clean_user_text({"text": ""}) is None
    assert clean_user_text({"text": "   "}) is None
    assert clean_user_text({"text": 42}) is None
    assert clean_user_text({}) is None
    assert clean_user_text(None) is None
    assert clean_user_text("just a string") is None


async def _drive_router(message: dict, queue_frames: AsyncMock):
    """Mirror of BOTH routers' user_text branch (identical composition)."""
    from datetime import UTC, datetime

    from pipecat.frames.frames import (
        TranscriptionFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
    )

    typed = clean_user_text(message)
    if not typed:
        return
    await queue_frames(
        [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(
                text=typed, user_id="visitor", timestamp=datetime.now(UTC).isoformat()
            ),
            UserStoppedSpeakingFrame(),
        ]
    )


def test_router_queues_the_vad_trio():
    from pipecat.frames.frames import (
        TranscriptionFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
    )

    queue = AsyncMock()
    _run(_drive_router({"type": "user_text", "text": "How tall is the roof?"}, queue))
    queue.assert_awaited_once()
    frames = queue.await_args.args[0]
    assert [type(f) for f in frames] == [
        UserStartedSpeakingFrame,
        TranscriptionFrame,
        UserStoppedSpeakingFrame,
    ]
    assert frames[1].text == "How tall is the roof?"
    assert frames[1].user_id == "visitor"


def test_router_ignores_empty_silently():
    queue = AsyncMock()
    _run(_drive_router({"type": "user_text", "text": "   "}, queue))
    queue.assert_not_awaited()
