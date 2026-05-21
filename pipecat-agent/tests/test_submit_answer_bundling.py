"""Tests for ``handle_action`` around submit_answer.

Question-to-question advancement inside a quiz is owned by the Quiz Page
itself (public/canvas-pages/quiz/main.js auto-reveals the explanation
banner and auto-advances on a timer after submit_answer) so the visitor
can actually see the visual feedback — confetti, wrong-flag, explanation
banner — before the iframe moves on. The agent's submit_answer handler
is therefore a pass-through: it dispatches submit_answer, returns the
frontend's reply unchanged, and does NOT bundle next_question.

This file guards against regressions to that contract:
  - submit_answer success → exactly one dispatch (no follow-up
    next_question), result returned untouched
  - submit_answer error → exactly one dispatch, error surfaced cleanly
  - non-submit_answer action verbs → pass-through, no bundling
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from tools.canvas_protocol_tools import (
    CanvasCommandError,
    CanvasToolContext,
    PendingCommandRegistry,
    make_handlers,
)


def _run(coro):
    return asyncio.run(coro)


def _make_ctx(semantic_state: dict | None, send_responses: list):
    """Build a CanvasToolContext whose send_app_message resolves each
    dispatched future with the next entry in ``send_responses``.

    ``send_responses`` items can be:
      - dict                → resolves the future with that dict
      - CanvasCommandError  → rejects, preserving the original code/message
      - other BaseException → rejects with a synthetic TEST_ERROR
    """
    manifest_registry = MagicMock()
    manifest_registry.current = MagicMock(return_value={
        "pageType": "quiz",
        "capabilities": {
            "action": {"verbs": ["submit_answer", "request_hint", "show_explanation"]},
            "control": {"verbs": ["next_question", "previous_question", "restart", "clear"]},
        },
    })
    manifest_registry.state = MagicMock(return_value=semantic_state)

    pending = PendingCommandRegistry()
    responses_iter = iter(send_responses)

    async def fake_send(payload):
        cmd_id = payload["commandId"]
        try:
            resp = next(responses_iter)
        except StopIteration:
            raise AssertionError(
                f"unexpected dispatch — exhausted send_responses; payload={payload!r}"
            )
        if isinstance(resp, CanvasCommandError):
            pending.reject(cmd_id, {
                "code": resp.code,
                "message": resp.message,
                "details": resp.details,
            })
        elif isinstance(resp, BaseException):
            pending.reject(cmd_id, {"code": "TEST_ERROR", "message": str(resp)})
        else:
            pending.resolve(cmd_id, resp)

    ctx = CanvasToolContext(
        manifest_registry=manifest_registry,
        pending=pending,
        send_app_message=AsyncMock(side_effect=fake_send),
        element_alias_map={},
        command_timeout_s=2.0,
    )
    return ctx


def _capture_callback():
    """Return (callback_mock, results_list) where results_list captures
    the first positional arg of each call (the tool result the LLM sees)."""
    results: list = []

    async def cb(payload):
        results.append(payload)

    return AsyncMock(side_effect=cb), results


def test_submit_answer_is_passthrough_no_bundled_advance():
    """submit_answer dispatches ONCE. The frontend's reply (which already
    contains `completed`) is forwarded to the LLM unchanged. No
    next_question dispatch — that's the Quiz Page's job on a timer."""
    ctx = _make_ctx(
        semantic_state={"questionIndex": 0, "questionCount": 3},
        send_responses=[
            {"choice": "B", "correct": True, "completed": False},
        ],
    )
    handler = make_handlers(ctx)["canvas_action"]
    cb, results = _capture_callback()
    params = MagicMock()
    params.arguments = {"verb": "submit_answer", "args": {"choice": "B"}}
    params.result_callback = cb

    _run(handler(params))

    assert len(results) == 1
    assert results[0] == {"choice": "B", "correct": True, "completed": False}
    # Exactly one dispatch — no bundled next_question.
    assert ctx.send_app_message.await_count == 1


def test_submit_answer_last_question_passthrough_with_completed_true():
    """On the last question the frontend returns `completed: true`. The
    agent forwards it; no follow-up dispatch."""
    ctx = _make_ctx(
        semantic_state={"questionIndex": 2, "questionCount": 3},
        send_responses=[
            {"choice": "A", "correct": False, "completed": True},
        ],
    )
    handler = make_handlers(ctx)["canvas_action"]
    cb, results = _capture_callback()
    params = MagicMock()
    params.arguments = {"verb": "submit_answer", "args": {"choice": "A"}}
    params.result_callback = cb

    _run(handler(params))

    assert len(results) == 1
    assert results[0]["correct"] is False
    assert results[0]["completed"] is True
    assert ctx.send_app_message.await_count == 1


def test_submit_answer_error_surfaces_normally():
    """If the frontend rejects submit_answer (e.g. invalid choice id), the
    error reaches the LLM through the normal CANVAS_ACTION error path."""
    ctx = _make_ctx(
        semantic_state={"questionIndex": 0, "questionCount": 3},
        send_responses=[
            CanvasCommandError({"code": "INVALID_ARGS", "message": "bad choice"}),
        ],
    )
    handler = make_handlers(ctx)["canvas_action"]
    cb, results = _capture_callback()
    params = MagicMock()
    params.arguments = {"verb": "submit_answer", "args": {"choice": "Z"}}
    params.result_callback = cb

    _run(handler(params))

    assert len(results) == 1
    # _emit_error packages CanvasCommandError into a tool-result error shape.
    assert (
        results[0].get("error") == "INVALID_ARGS"
        or results[0].get("code") == "INVALID_ARGS"
    )
    assert ctx.send_app_message.await_count == 1


def test_non_submit_answer_action_verbs_are_passthrough():
    """show_explanation / request_hint dispatch once and return the
    frontend's reply unchanged."""
    ctx = _make_ctx(
        semantic_state={"questionIndex": 0, "questionCount": 3},
        send_responses=[
            {"ok": True},
        ],
    )
    handler = make_handlers(ctx)["canvas_action"]
    cb, results = _capture_callback()
    params = MagicMock()
    params.arguments = {"verb": "show_explanation", "args": {}}
    params.result_callback = cb

    _run(handler(params))

    assert len(results) == 1
    assert results[0] == {"ok": True}
    assert ctx.send_app_message.await_count == 1
