"""Tests for `make_handle_generate_quiz` — the S64e quiz tool handler.

The handler is factory-bound to a backend client (provides
``generate_quiz``), a session context (provides ``get_slug`` /
``get_current_scene_id``), and a canvas context (provides the
``send_app_message`` + ``PendingCommandRegistry`` substrate for the
bundled set_page dispatch — S64e Option D). Tests exercise:
  - happy path: backend returns a blob, the bundled set_page dispatch
    fires, the callback receives the blob
  - missing session: no slug or no scene_id → callback receives an
    ``ok: false`` error and the backend is NOT called
  - backend failure: backend raises → callback receives an ``ok: false``
    error stamped with the exception class name; set_page is NOT
    dispatched (no blob to dispatch with)
  - set_page failure: backend returns a blob but the dispatched set_page
    fails → callback receives an ``ok: false`` error so the LLM doesn't
    narrate questions to a stale iframe

Async patterns: the existing test suite doesn't use pytest-asyncio (not
installed), so each test wraps the awaitable in ``asyncio.run``. The
unittest.mock helpers are stdlib and require no new dependencies.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from tools.quiz_generation import make_handle_generate_quiz


def _run(coro):
    return asyncio.run(coro)


def _canvas_ctx_wired():
    """A canvas_ctx mock whose ``send_app_message`` is truthy — exercises the
    bundled set_page dispatch path."""
    ctx = MagicMock()
    ctx.send_app_message = AsyncMock()
    return ctx


def _canvas_ctx_unwired():
    """A canvas_ctx mock whose ``send_app_message`` is None — exercises the
    skip-dispatch fallback (tests, or canvas channel not wired)."""
    ctx = MagicMock()
    ctx.send_app_message = None
    return ctx


def test_quiz_generation_happy_path():
    backend_client = MagicMock()
    blob = {
        "questions": [
            {
                "id": "q1",
                "text": "?",
                "choices": [],
                "correct_choice_id": "A",
                "explanation": "",
            }
        ],
        "language": "en",
    }
    backend_client.generate_quiz = AsyncMock(return_value=blob)

    context = MagicMock()
    context.get_slug.return_value = "test-slug"
    context.get_current_scene_id.return_value = "scene-123"

    canvas_ctx = _canvas_ctx_wired()

    # Stub dispatch_canvas_command at the import site so we don't have to
    # construct a real PendingCommandRegistry + Future plumbing.
    with patch(
        "tools.quiz_generation.dispatch_canvas_command",
        new=AsyncMock(return_value={"pageType": "quiz", "version": "0.1"}),
    ) as mock_dispatch:
        handler = make_handle_generate_quiz(backend_client, context, canvas_ctx)

        params = MagicMock()
        params.arguments = {"count": 3, "language": "en"}
        params.result_callback = AsyncMock()

        _run(handler(params))

        # Positional args reach the backend in (slug, scene_id, count, language) order.
        backend_client.generate_quiz.assert_awaited_once_with(
            "test-slug", "scene-123", 3, "en"
        )
        # The bundled set_page dispatch fired with the blob as pageInit.
        mock_dispatch.assert_awaited_once()
        dispatch_args, _dispatch_kwargs = mock_dispatch.call_args
        assert dispatch_args[1] == "set_page"
        assert dispatch_args[2] == {"pageType": "quiz", "pageInit": blob}

        params.result_callback.assert_awaited_once()
        callback_args, _callback_kwargs = params.result_callback.call_args
        assert "questions" in callback_args[0]


def test_quiz_generation_no_session():
    """No slug or no scene_id → callback gets the 'no active live-room
    session' sentinel and the backend is NOT called."""
    backend_client = MagicMock()
    backend_client.generate_quiz = AsyncMock()

    context = MagicMock()
    context.get_slug.return_value = None
    context.get_current_scene_id.return_value = None

    canvas_ctx = _canvas_ctx_wired()

    with patch(
        "tools.quiz_generation.dispatch_canvas_command",
        new=AsyncMock(),
    ) as mock_dispatch:
        handler = make_handle_generate_quiz(backend_client, context, canvas_ctx)

        params = MagicMock()
        params.arguments = {}
        params.result_callback = AsyncMock()

        _run(handler(params))

        backend_client.generate_quiz.assert_not_called()
        mock_dispatch.assert_not_awaited()
        args, _kwargs = params.result_callback.call_args
        assert args[0]["ok"] is False
        assert "no active live-room session" in args[0]["error"]


def test_quiz_generation_backend_error():
    """Backend raises → callback gets an error result with the
    exception class name + message, not a raised exception (the LLM has
    to be able to read and recover). set_page is NOT dispatched."""
    backend_client = MagicMock()
    backend_client.generate_quiz = AsyncMock(side_effect=Exception("api down"))

    context = MagicMock()
    context.get_slug.return_value = "test-slug"
    context.get_current_scene_id.return_value = "scene-123"

    canvas_ctx = _canvas_ctx_wired()

    with patch(
        "tools.quiz_generation.dispatch_canvas_command",
        new=AsyncMock(),
    ) as mock_dispatch:
        handler = make_handle_generate_quiz(backend_client, context, canvas_ctx)

        params = MagicMock()
        params.arguments = {}
        params.result_callback = AsyncMock()

        _run(handler(params))

        # Backend raised before we had a blob — no point dispatching set_page.
        mock_dispatch.assert_not_awaited()

        args, _kwargs = params.result_callback.call_args
        assert args[0]["ok"] is False
        # Surface the underlying exception so the LLM can include it in its
        # apology — "Exception: api down" is greppable in production logs.
        assert "Exception" in args[0]["error"]
        assert "api down" in args[0]["error"]


def test_quiz_generation_set_page_failure_returns_error():
    """Backend returns a blob but the bundled set_page dispatch fails →
    callback receives an ``ok: false`` error rather than the blob. We do NOT
    want the LLM narrating new questions while the iframe is on the old
    quiz; surfacing the failure lets the model apologise instead."""
    from tools.canvas_protocol_tools import CanvasCommandError

    backend_client = MagicMock()
    blob = {
        "questions": [
            {
                "id": "q1",
                "text": "?",
                "choices": [],
                "correct_choice_id": "A",
                "explanation": "",
            }
        ],
        "language": "en",
    }
    backend_client.generate_quiz = AsyncMock(return_value=blob)

    context = MagicMock()
    context.get_slug.return_value = "test-slug"
    context.get_current_scene_id.return_value = "scene-123"

    canvas_ctx = _canvas_ctx_wired()

    swap_error = CanvasCommandError(
        {"code": "TIMEOUT", "message": "canvas set_page timed out after 6.0s"}
    )

    with patch(
        "tools.quiz_generation.dispatch_canvas_command",
        new=AsyncMock(side_effect=swap_error),
    ) as mock_dispatch:
        handler = make_handle_generate_quiz(backend_client, context, canvas_ctx)

        params = MagicMock()
        params.arguments = {}
        params.result_callback = AsyncMock()

        _run(handler(params))

        mock_dispatch.assert_awaited_once()
        args, _kwargs = params.result_callback.call_args
        assert args[0]["ok"] is False
        assert "page swap failed" in args[0]["error"]
        assert "TIMEOUT" in args[0]["error"]


def test_quiz_generation_skips_dispatch_when_unwired():
    """When canvas_ctx.send_app_message is None (e.g. canvas channel not
    wired in a degraded session), the bundled set_page dispatch is
    skipped and the blob is returned as a fallback. The LLM can still
    read questions; only the iframe-swap step is missing."""
    backend_client = MagicMock()
    blob = {
        "questions": [
            {
                "id": "q1",
                "text": "?",
                "choices": [],
                "correct_choice_id": "A",
                "explanation": "",
            }
        ],
        "language": "en",
    }
    backend_client.generate_quiz = AsyncMock(return_value=blob)

    context = MagicMock()
    context.get_slug.return_value = "test-slug"
    context.get_current_scene_id.return_value = "scene-123"

    canvas_ctx = _canvas_ctx_unwired()

    with patch(
        "tools.quiz_generation.dispatch_canvas_command",
        new=AsyncMock(),
    ) as mock_dispatch:
        handler = make_handle_generate_quiz(backend_client, context, canvas_ctx)

        params = MagicMock()
        params.arguments = {}
        params.result_callback = AsyncMock()

        _run(handler(params))

        mock_dispatch.assert_not_awaited()
        args, _kwargs = params.result_callback.call_args
        assert "questions" in args[0]
