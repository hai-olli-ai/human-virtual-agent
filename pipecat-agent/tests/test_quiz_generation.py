"""Tests for `make_handle_generate_quiz` — the S64e quiz tool handler.

The handler is factory-bound to a backend client (provides
``generate_quiz``) and a session context (provides ``get_slug`` /
``get_current_scene_id``). Tests exercise:
  - happy path: backend returns a blob, callback receives it
  - missing session: no slug or no scene_id → callback receives an
    ``ok: false`` error and the backend is NOT called
  - backend failure: backend raises → callback receives an ``ok: false``
    error stamped with the exception class name

Async patterns: the existing test suite doesn't use pytest-asyncio (not
installed), so each test wraps the awaitable in ``asyncio.run``. The
unittest.mock helpers are stdlib and require no new dependencies.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from tools.quiz_generation import make_handle_generate_quiz


def _run(coro):
    return asyncio.run(coro)


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

    handler = make_handle_generate_quiz(backend_client, context)

    params = MagicMock()
    params.arguments = {"count": 3, "language": "en"}
    params.result_callback = AsyncMock()

    _run(handler(params))

    # Positional args reach the backend in (slug, scene_id, count, language) order.
    backend_client.generate_quiz.assert_awaited_once_with(
        "test-slug", "scene-123", 3, "en"
    )
    params.result_callback.assert_awaited_once()
    args, _kwargs = params.result_callback.call_args
    assert "questions" in args[0]


def test_quiz_generation_no_session():
    """No slug or no scene_id → callback gets the 'no active live-room
    session' sentinel and the backend is NOT called."""
    backend_client = MagicMock()
    backend_client.generate_quiz = AsyncMock()

    context = MagicMock()
    context.get_slug.return_value = None
    context.get_current_scene_id.return_value = None

    handler = make_handle_generate_quiz(backend_client, context)

    params = MagicMock()
    params.arguments = {}
    params.result_callback = AsyncMock()

    _run(handler(params))

    backend_client.generate_quiz.assert_not_called()
    args, _kwargs = params.result_callback.call_args
    assert args[0]["ok"] is False
    assert "no active live-room session" in args[0]["error"]


def test_quiz_generation_backend_error():
    """Backend raises → callback gets an error result with the
    exception class name + message, not a raised exception (the LLM has
    to be able to read and recover)."""
    backend_client = MagicMock()
    backend_client.generate_quiz = AsyncMock(side_effect=Exception("api down"))

    context = MagicMock()
    context.get_slug.return_value = "test-slug"
    context.get_current_scene_id.return_value = "scene-123"

    handler = make_handle_generate_quiz(backend_client, context)

    params = MagicMock()
    params.arguments = {}
    params.result_callback = AsyncMock()

    _run(handler(params))

    args, _kwargs = params.result_callback.call_args
    assert args[0]["ok"] is False
    # Surface the underlying exception so the LLM can include it in its
    # apology — "Exception: api down" is greppable in production logs.
    assert "Exception" in args[0]["error"]
    assert "api down" in args[0]["error"]
