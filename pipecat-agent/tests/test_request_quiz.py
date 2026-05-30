"""Tests for the ``request_quiz`` inbound-message handler (S65c Blocks 5+6).

The handler in ``bot.py`` is a nested closure that:
  1. Pre-gates on ``session_context`` readiness (slug + scene_id present).
  2. Constructs a local ``_emit_quiz_state(state, err)`` closure that
     wraps the wire payload defined by Block 6.
  3. Invokes ``run_quiz_generation(..., on_state=_emit_quiz_state)``.

The handler itself isn't importable, but the Block 6 wire-format
contract and the Block 3 ``on_state`` sequence ARE directly testable
through ``run_quiz_generation``. This file exercises:

  * The successful state sequence: ``generating → ready``.
  * The error sequence: ``generating → error`` with the correct
    diagnostic string format.
  * The session-not-ready error emitted by the core (the handler
    pre-gates this in production so the wire never sees it, but the
    core path stays valid as a defense-in-depth guarantee).
  * The Block 6 wire-shape correspondence: each ``on_state`` call maps
    1:1 to the ``quiz_generation_state`` payload the shell receives.

Convention: no pytest-asyncio (per project), so async work goes
through ``asyncio.run`` via ``_run``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from tools.quiz_generation import request_quiz_ready, run_quiz_generation


def _run(coro):
    return asyncio.run(coro)


def _ready_session_context():
    """Mock SessionContext with both slug and scene_id populated —
    mirrors a healthy live-room session."""
    ctx = MagicMock()
    ctx.get_slug.return_value = "test-slug"
    ctx.get_current_scene_id.return_value = "scene-1"
    return ctx


def _canvas_ctx_wired():
    """Canvas context with ``send_app_message`` set so the bundled
    set_page dispatch path runs."""
    ctx = MagicMock()
    ctx.send_app_message = AsyncMock()
    return ctx


def _record_state_emits():
    """Collect ``(state, err)`` tuples as ``on_state`` fires. Returns
    (callable, list-of-tuples) so tests can assert on the sequence."""
    emits: list[tuple] = []

    async def on_state(state: str, err):
        emits.append((state, err))

    return on_state, emits


# ──────────────────────────────────────────────────────────────────────
# Happy path — generating → ready
# ──────────────────────────────────────────────────────────────────────


def test_request_quiz_emits_generating_then_ready_on_success():
    """The Block 6 wire-shape spec maps to two emits: ``generating``
    immediately, then ``ready`` once the bundled set_page resolves.

    Asserting the SEQUENCE (not just the set of states) matters: the
    shell's button driver assumes ``generating`` arrives first so it
    can transition the button from idle ⇒ spinner before any
    ``ready`` flips it to the active-quiz state.
    """
    blob = {"questions": [{"id": "q1", "text": "?", "choices": []}], "language": "en"}
    backend_client = MagicMock()
    backend_client.generate_quiz = AsyncMock(return_value=blob)

    on_state, emits = _record_state_emits()

    with patch(
        "tools.quiz_generation.dispatch_canvas_command",
        new=AsyncMock(return_value={"pageType": "quiz", "version": "0.1"}),
    ):
        result = _run(
            run_quiz_generation(
                backend_client=backend_client,
                session_context=_ready_session_context(),
                canvas_ctx=_canvas_ctx_wired(),
                count=3,
                language="en",
                on_state=on_state,
            )
        )

    assert result.ok is True
    assert result.blob is blob
    # Block 6 wire-shape: (state, err) tuples map 1:1 to the dict
    # ``_emit_quiz_state`` builds for the shell.
    assert emits == [("generating", None), ("ready", None)]


# ──────────────────────────────────────────────────────────────────────
# Backend failure — generating → error
# ──────────────────────────────────────────────────────────────────────


def test_request_quiz_emits_error_on_backend_failure():
    """Backend raises ⇒ ``generating`` then ``error`` with the
    diagnostic string format that ``_emit_quiz_state`` surfaces on the
    wire as ``{"state":"error","error":"<msg>"}``.

    The error string format is asserted (not just "an error fired") so
    that any future change to the format is a deliberate decision —
    the shell may grow to log/display it.
    """
    backend_client = MagicMock()
    backend_client.generate_quiz = AsyncMock(side_effect=RuntimeError("backend down"))

    on_state, emits = _record_state_emits()

    with patch(
        "tools.quiz_generation.dispatch_canvas_command",
        new=AsyncMock(),
    ) as mock_dispatch:
        result = _run(
            run_quiz_generation(
                backend_client=backend_client,
                session_context=_ready_session_context(),
                canvas_ctx=_canvas_ctx_wired(),
                count=3,
                language="en",
                on_state=on_state,
            )
        )

    assert result.ok is False
    assert result.blob is None  # backend raised before producing a blob
    # No set_page dispatch because there's no blob to dispatch.
    mock_dispatch.assert_not_awaited()
    # State sequence + error format.
    assert len(emits) == 2
    assert emits[0] == ("generating", None)
    err_state, err_msg = emits[1]
    assert err_state == "error"
    assert "quiz generation failed" in err_msg
    assert "RuntimeError" in err_msg
    assert "backend down" in err_msg


# ──────────────────────────────────────────────────────────────────────
# set_page failure — generating → error (canvas swap failed)
# ──────────────────────────────────────────────────────────────────────


def test_request_quiz_emits_error_when_canvas_set_page_fails():
    """Backend produced a blob but the bundled set_page raised
    ``CanvasCommandError`` ⇒ emits ``generating`` then ``error`` with
    the canvas error code/message format. Documents that the shell
    sees a distinct error string for this failure mode (so the visitor
    can be told the quiz couldn't be displayed even though it was
    generated)."""
    from tools.canvas_protocol_tools import CanvasCommandError

    blob = {"questions": [{"id": "q1"}], "language": "en"}
    backend_client = MagicMock()
    backend_client.generate_quiz = AsyncMock(return_value=blob)

    swap_error = CanvasCommandError(
        {"code": "TIMEOUT", "message": "canvas set_page timed out"}
    )

    on_state, emits = _record_state_emits()

    with patch(
        "tools.quiz_generation.dispatch_canvas_command",
        new=AsyncMock(side_effect=swap_error),
    ):
        result = _run(
            run_quiz_generation(
                backend_client=backend_client,
                session_context=_ready_session_context(),
                canvas_ctx=_canvas_ctx_wired(),
                count=3,
                language="en",
                on_state=on_state,
            )
        )

    assert result.ok is False
    # blob IS retained on the result (creator-recoverable) even though
    # ok=False — see QuizGenerationResult docstring.
    assert result.blob is blob
    assert len(emits) == 2
    assert emits[0] == ("generating", None)
    err_state, err_msg = emits[1]
    assert err_state == "error"
    assert "quiz page swap failed" in err_msg
    assert "TIMEOUT" in err_msg


# ──────────────────────────────────────────────────────────────────────
# Session-not-ready gate — production handler's silent-ignore (E3)
# ──────────────────────────────────────────────────────────────────────


def test_request_quiz_ready_truth_table():
    """The gate predicate used by ``bot.py``'s ``request_quiz`` branch.

    Truth table:
      slug present  AND  scene_id present  ⇒  True   (handler proceeds)
      slug missing                          ⇒  False  (silent-ignore)
      scene_id missing                      ⇒  False  (silent-ignore)
      both missing                          ⇒  False  (silent-ignore)

    The production handler's body is ``if not request_quiz_ready(...):
    return``. With this predicate machine-verified, the handler's E3
    silent-ignore behavior is structurally guaranteed — no separate
    closure-extraction test of the handler body is needed.
    """

    def _ctx(slug, scene_id):
        c = MagicMock()
        c.get_slug.return_value = slug
        c.get_current_scene_id.return_value = scene_id
        return c

    # The only ready case: both populated.
    assert request_quiz_ready(_ctx("test-slug", "scene-1")) is True

    # Every not-ready combination — handler returns silently (no emit).
    assert request_quiz_ready(_ctx(None, "scene-1")) is False
    assert request_quiz_ready(_ctx("test-slug", None)) is False
    assert request_quiz_ready(_ctx(None, None)) is False
    # Empty strings count as not-ready (truthiness check inside the
    # predicate). Production never sees these — runner-args body
    # contains either a real string or None — but the predicate's
    # robustness here documents the contract.
    assert request_quiz_ready(_ctx("", "scene-1")) is False
    assert request_quiz_ready(_ctx("test-slug", "")) is False


def test_request_quiz_emits_session_error_when_slug_missing():
    """Calling the core with an empty SessionContext ⇒ single ``error``
    emit (the ``"no active live-room session"`` sentinel) AND the
    backend is NEVER called.

    Production handler note: ``bot.py``'s ``request_quiz`` branch
    pre-gates this case (the visitor sees no state event at all per
    E3's silent-ignore semantics). This test exercises the core's own
    defense-in-depth so a future refactor that drops the pre-gate
    doesn't silently regress to ``backend.generate_quiz`` being called
    with ``slug=None`` (which would hit the backend's own session
    validation and produce a less actionable error).
    """
    empty_ctx = MagicMock()
    empty_ctx.get_slug.return_value = None
    empty_ctx.get_current_scene_id.return_value = None

    backend_client = MagicMock()
    backend_client.generate_quiz = AsyncMock()

    on_state, emits = _record_state_emits()

    with patch(
        "tools.quiz_generation.dispatch_canvas_command",
        new=AsyncMock(),
    ) as mock_dispatch:
        result = _run(
            run_quiz_generation(
                backend_client=backend_client,
                session_context=empty_ctx,
                canvas_ctx=_canvas_ctx_wired(),
                count=3,
                language="en",
                on_state=on_state,
            )
        )

    assert result.ok is False
    assert result.error == "no active live-room session"
    # Backend NOT called — the session check short-circuited before it.
    backend_client.generate_quiz.assert_not_awaited()
    mock_dispatch.assert_not_awaited()
    # Exactly one emit: ("error", sentinel). No "generating" precedes
    # it because the session check fires BEFORE that emit.
    assert emits == [("error", "no active live-room session")]
