"""generate_quiz_from_knowledge — Pipecat tool surface for S64e.

Surfaces a single non-canvas tool that asks the backend to assemble the
active scene's knowledge_text and prompt Anthropic for a structured
multiple-choice quiz blob. The LLM is expected to follow this with a
canvas_set_page(pageType='quiz', pageInit=<blob>) call — the AGENT
PLAYBOOK section of the system prompt documents that two-step sequence.

Why a separate tool (not a canvas verb): canvas tools are 5 generic
protocol verbs (analyze/highlight/control/action/set_page) that travel
through the Daily app-message channel and bounce off the iframe. Quiz
generation is a backend call that happens before any Page is active,
so it skips the canvas protocol entirely and goes direct to the API.

The handler is factory-bound to:
- ``backend_client`` — the module (or object) that exposes
  ``generate_quiz(slug, scene_id, count, language)``. In the live agent
  this is the ``api_client`` module; tests pass a mock with the same
  callable shape.
- ``session_context`` — accessor that returns the current live-room slug
  and current scene id. ``bot.py`` constructs a ``SessionContext`` at
  session start (populating slug from the runner-args body) and updates
  ``current_scene_id`` on every ``canvas.sceneChanged`` event.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams


@dataclass
class SessionContext:
    """Session-scoped state for tool handlers that need live-room context.

    Kept minimal — only the fields any tool needs across the session.
    bot.py owns the lifecycle: ``set_slug`` once at session start (from
    runner-args body) and ``set_scene`` at session start + on every
    ``canvas.sceneChanged`` event. Tool handlers read via the getters.
    """

    slug: str | None = None
    current_scene_id: str | None = None

    def get_slug(self) -> str | None:
        return self.slug

    def get_current_scene_id(self) -> str | None:
        return self.current_scene_id

    def set_slug(self, slug: str | None) -> None:
        self.slug = slug

    def set_scene(self, scene_id: str | None) -> None:
        self.current_scene_id = scene_id


GENERATE_QUIZ_SCHEMA = FunctionSchema(
    name="generate_quiz_from_knowledge",
    description=(
        "Generate a multiple-choice quiz from the current scene's knowledge content. "
        "Returns a quiz blob containing the questions and correct answers. "
        "After calling this tool, you MUST call canvas_set_page with pageType='quiz' "
        "and pageInit set to the returned blob to display the quiz to the user. "
        "Narrate that you're putting together questions while this runs (it takes "
        "1-2 seconds). Use this when the user explicitly asks for a quiz, asks to "
        "be quizzed, or asks to test their knowledge."
    ),
    properties={
        "count": {
            "type": "integer",
            "description": "Number of questions to generate (1-10). Default 3.",
            "minimum": 1,
            "maximum": 10,
        },
        "language": {
            "type": "string",
            "description": "Language code for the quiz (en, es, fr, de, ja, ko, zh, pt, vi).",
        },
    },
    required=[],
)


def make_handle_generate_quiz(backend_client, session_context):
    """Factory that binds the handler to the active session's slug + scene id.

    ``backend_client`` must expose ``generate_quiz(slug, scene_id, count,
    language)`` returning the QuizBlob dict (or raising). The production
    binding passes the ``api_client`` module; tests pass a mock with the
    same shape.

    ``session_context`` must expose ``get_slug()`` and
    ``get_current_scene_id()``. See ``SessionContext`` above.

    Returned errors are handed back as a tool result (``{"ok": false,
    "error": "..."}``) rather than raised — same resilience pattern as
    the canvas tools, so a single failed quiz call doesn't break the
    conversation turn.
    """
    async def handle_generate_quiz(params: FunctionCallParams):
        args = params.arguments or {}
        count = args.get("count", 3)
        language = args.get("language", "en")
        slug = session_context.get_slug()
        scene_id = session_context.get_current_scene_id()
        logger.info(
            f"[GENERATE_QUIZ] called: count={count} language={language!r} "
            f"slug={slug!r} scene_id={scene_id!r}"
        )
        if not slug or not scene_id:
            await params.result_callback({
                "ok": False,
                "error": "no active live-room session",
            })
            return
        try:
            blob = await backend_client.generate_quiz(slug, scene_id, count, language)
            logger.info(
                "[GENERATE_QUIZ] generated quiz: questions={}",
                len((blob or {}).get("questions", [])),
            )
            await params.result_callback(blob)
        except Exception as exc:
            logger.exception("generate_quiz tool call failed")
            await params.result_callback({
                "ok": False,
                "error": f"quiz generation failed: {type(exc).__name__}: {str(exc)[:200]}",
            })

    return handle_generate_quiz
