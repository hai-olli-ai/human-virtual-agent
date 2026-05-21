"""generate_quiz_from_knowledge — Pipecat tool surface for S64e.

Surfaces a single non-canvas tool that asks the backend to assemble the
active scene's knowledge_text and prompt Anthropic for a structured
multiple-choice quiz blob. The tool also dispatches the canvas set_page
swap to the quiz Page in the same call (S64e Option D) — bundling the
two steps because LLMs unreliably copy large structured payloads like a
quiz blob between tool calls. Asking the LLM to follow up with
canvas_set_page(pageType='quiz', pageInit=<blob>) led to systematic
omission of pageInit, the frontend's same-pageType short-circuit firing,
and the iframe staying on the previous quiz while the agent narrated
the new questions.

Why a separate tool (not a canvas verb): canvas tools are 5 generic
protocol verbs (analyze/highlight/control/action/set_page) that travel
through the Daily app-message channel and bounce off the iframe. Quiz
generation is a backend call that happens before any Page is active;
the bundled set_page dispatch *does* go through the canvas protocol,
but it's an implementation detail the LLM doesn't see — the model just
gets back the quiz blob with the Page already showing it.

The handler is factory-bound to:
- ``backend_client`` — the module (or object) that exposes
  ``generate_quiz(slug, scene_id, count, language)``. In the live agent
  this is the ``api_client`` module; tests pass a mock with the same
  callable shape.
- ``session_context`` — accessor that returns the current live-room slug
  and current scene id. ``bot.py`` constructs a ``SessionContext`` at
  session start (populating slug from the runner-args body) and updates
  ``current_scene_id`` on every ``canvas.sceneChanged`` event.
- ``canvas_ctx`` — ``CanvasToolContext`` carrying the pending-command
  registry and Daily send_app_message hook. Used to dispatch the
  bundled set_page swap. Tests pass a mock with ``send_app_message``
  set to None to skip the dispatch step.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams

from tools.canvas_protocol_tools import (
    CanvasCommandError,
    dispatch_canvas_command,
)


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
        "Generate a multiple-choice quiz from the current scene's knowledge content "
        "AND activate the quiz Canvas Page in the same call. Returns a quiz blob "
        "containing the questions and correct answers; by the time the result is "
        "returned, the iframe is already showing the new quiz to the user. Do NOT "
        "call canvas_set_page after this — the page swap is bundled in. "
        "Narrate that you're putting together questions while this runs (it takes "
        "1-2 seconds). Use this when the user explicitly asks for a quiz, asks to "
        "be quizzed, or asks to test their knowledge — and also to auto-load a "
        "fresh quiz after the previous one finishes (see AGENT PLAYBOOK)."
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


def make_handle_generate_quiz(backend_client, session_context, canvas_ctx):
    """Factory that binds the handler to the active session's slug + scene id
    and the canvas dispatch substrate.

    ``backend_client`` must expose ``generate_quiz(slug, scene_id, count,
    language)`` returning the QuizBlob dict (or raising). The production
    binding passes the ``api_client`` module; tests pass a mock with the
    same shape.

    ``session_context`` must expose ``get_slug()`` and
    ``get_current_scene_id()``. See ``SessionContext`` above.

    ``canvas_ctx`` is the ``CanvasToolContext`` from bot.py — the same
    one threaded into the 5 canvas tool handlers. Used to dispatch the
    bundled set_page swap after the blob is generated. When
    ``canvas_ctx.send_app_message`` is None (tests, or the rare case
    where the canvas channel isn't wired), the dispatch step is skipped
    and the blob is returned unchanged.

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

        # Step 1 — generate the blob from the backend.
        try:
            blob = await backend_client.generate_quiz(slug, scene_id, count, language)
            logger.info(
                "[GENERATE_QUIZ] generated quiz: questions={}",
                len((blob or {}).get("questions", [])),
            )
        except Exception as exc:
            logger.exception("generate_quiz tool call failed during generation")
            await params.result_callback({
                "ok": False,
                "error": f"quiz generation failed: {type(exc).__name__}: {str(exc)[:200]}",
            })
            return

        # Step 2 — bundled set_page so the quiz Page activates with the
        # fresh blob in the same tool call. Without this, the LLM was
        # asked to follow up with canvas_set_page(pageType='quiz',
        # pageInit=<blob>) and routinely omitted pageInit — leading to
        # the iframe staying on the previous quiz while the agent
        # narrated the new questions. See S64e Option D.
        if canvas_ctx is not None and getattr(canvas_ctx, "send_app_message", None):
            try:
                swap_result = await dispatch_canvas_command(
                    canvas_ctx,
                    "set_page",
                    {"pageType": "quiz", "pageInit": blob},
                )
                logger.info("[GENERATE_QUIZ] page swap complete: {!r}", swap_result)
            except CanvasCommandError as exc:
                logger.warning(
                    "[GENERATE_QUIZ] page swap failed code={!r} message={!r}",
                    exc.code, exc.message,
                )
                await params.result_callback({
                    "ok": False,
                    "error": f"quiz page swap failed: {exc.code}: {exc.message[:200]}",
                })
                return
            except Exception as exc:
                logger.exception("generate_quiz page swap failed unexpectedly")
                await params.result_callback({
                    "ok": False,
                    "error": f"quiz page swap failed: {type(exc).__name__}: {str(exc)[:200]}",
                })
                return
        else:
            logger.info("[GENERATE_QUIZ] canvas dispatch not wired; skipping bundled set_page")

        # Step 3 — return the blob. The LLM uses it to read the first
        # question aloud; the iframe is already showing the same blob.
        await params.result_callback(blob)

    return handle_generate_quiz
