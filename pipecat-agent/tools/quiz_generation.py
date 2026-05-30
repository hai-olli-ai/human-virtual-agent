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
from typing import Awaitable, Callable, Optional

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams

from tools.canvas_protocol_tools import (
    CanvasCommandError,
    dispatch_canvas_command,
)


# S65c Block 3 — ``on_state`` callable contract for ``run_quiz_generation``.
# Two arguments so call sites pass an explicit ``None`` for error on the
# non-error transitions ("generating", "ready") — slightly more verbose at
# the call site, but the alternative (kwargs / variadic) would make Block 6's
# Daily emitter harder to type-check.
QuizStateFn = Callable[[str, Optional[str]], Awaitable[None]]


def request_quiz_ready(session_context) -> bool:
    """S65c Block 5+8 — production gate for the ``request_quiz`` handler.

    Returns ``True`` iff the agent's ``SessionContext`` is sufficiently
    populated to invoke :func:`run_quiz_generation` — slug AND
    current_scene_id must both be non-empty. ``bot.py``'s
    ``request_quiz`` branch uses this as an early-return guard per E3
    (silently ignore manual quiz requests that arrive before the agent
    has fully initialized — the wire sees zero events, the frontend
    button stays in whatever pre-click state it was in, the visitor
    can retry).

    Extracted as a standalone helper so the gate's predicate is
    machine-verified by ``test_request_quiz_ready_truth_table`` —
    without this, the silent-ignore behavior would rely on code review
    only. ``run_quiz_generation`` itself has a defense-in-depth check
    against the same condition (it returns
    ``QuizGenerationResult(ok=False, error="no active live-room
    session")``), but the handler's pre-gate is what keeps the wire
    silent in the silent-ignore case.
    """
    return bool(
        session_context.get_slug() and session_context.get_current_scene_id()
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


@dataclass
class QuizGenerationResult:
    """Outcome of one :func:`run_quiz_generation` invocation.

    Used by two callers with different LLM/UX consequences:
      * The LLM tool wrapper (``make_handle_generate_quiz``) returns
        ``blob`` to the LLM on success and the ``{"ok": false, "error":
        ...}`` sentinel on failure.
      * The S65c manual ``request_quiz`` button (Block 5) ignores the
        return shape and relies on ``on_state`` callbacks for UX state.

    ``blob`` may be populated even when ``ok=False`` — specifically when
    the backend produced a blob but the bundled set_page dispatch failed.
    The LLM tool wrapper still surfaces the failure (otherwise the LLM
    would narrate questions to a stale iframe), but the blob is retained
    on the result so future callers (e.g. retry logic) can read it.
    """

    ok: bool
    blob: Optional[dict] = None
    error: Optional[str] = None


async def run_quiz_generation(
    *,
    backend_client,
    session_context,
    canvas_ctx,
    count: int = 3,
    language: str = "en",
    on_state: Optional[QuizStateFn] = None,
) -> QuizGenerationResult:
    """Generate the quiz blob and bundle the canvas.set_page dispatch.

    Pure async core. Two callers wrap this:
      1. :func:`make_handle_generate_quiz` — the LLM tool handler.
      2. The S65c ``request_quiz`` inbound-message branch in ``bot.py``
         (manual visitor click; emits ``quiz_generation_state`` via the
         injected ``on_state`` hook).

    Contract preserved from the pre-Block-3 ``make_handle_generate_quiz``:
      * Backend ``generate_quiz`` invoked with POSITIONAL args
        ``(slug, scene_id, count, language)`` — keeps the existing mocks
        and any compatibility-leaning call-shape assertions intact.
      * Defaults ``count=3``, ``language="en"`` — same as today.
      * Missing slug or scene_id ⇒ ``ok=False`` with the literal
        ``"no active live-room session"`` error string (LLM tool path
        surfaces this verbatim).
      * Backend exception ⇒ error string
        ``"quiz generation failed: {ExcClass}: {str(exc)[:200]}"`` and
        ``canvas.set_page`` is NOT dispatched (no blob to dispatch with).
      * ``CanvasCommandError`` on set_page ⇒
        ``"quiz page swap failed: {exc.code}: {exc.message[:200]}"``;
        any other set_page exception ⇒
        ``"quiz page swap failed: {ExcClass}: {str(exc)[:200]}"``.
        Both paths keep ``blob`` on the result (creator-recoverable).
      * ``canvas_ctx.send_app_message is None`` (tests / unwired) ⇒
        skip dispatch silently and return ``ok=True`` with the blob.
        The LLM still reads questions; only the iframe-swap is missing.

    ``on_state`` (when supplied) fires:
      * ``("generating", None)`` immediately after the session check
        passes — so the button-driven UX can spinner before the backend
        roundtrip starts.
      * ``("ready", None)`` after the bundled set_page resolves (or after
        the skip-dispatch fallback). Lines up with frontend's
        "show the quiz Page is active" transition.
      * ``("error", message)`` on each failure branch. The shell uses
        this to flip the button back from spinner ⇒ error chip.

    The LLM tool path passes ``on_state=None`` so no Daily app-messages
    are emitted for tool-driven quizzes — they'd just race with the
    LLM's own narration and add noise.
    """
    slug = session_context.get_slug()
    scene_id = session_context.get_current_scene_id()
    logger.info(
        f"[GENERATE_QUIZ] called: count={count} language={language!r} "
        f"slug={slug!r} scene_id={scene_id!r}"
    )
    if not slug or not scene_id:
        msg = "no active live-room session"
        if on_state is not None:
            await on_state("error", msg)
        return QuizGenerationResult(ok=False, error=msg)

    if on_state is not None:
        await on_state("generating", None)

    # Step 1 — generate the blob from the backend.
    try:
        blob = await backend_client.generate_quiz(slug, scene_id, count, language)
        logger.info(
            "[GENERATE_QUIZ] generated quiz: questions={}",
            len((blob or {}).get("questions", [])),
        )
    except Exception as exc:
        logger.exception("generate_quiz tool call failed during generation")
        msg = f"quiz generation failed: {type(exc).__name__}: {str(exc)[:200]}"
        if on_state is not None:
            await on_state("error", msg)
        return QuizGenerationResult(ok=False, error=msg)

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
            msg = f"quiz page swap failed: {exc.code}: {exc.message[:200]}"
            if on_state is not None:
                await on_state("error", msg)
            return QuizGenerationResult(ok=False, blob=blob, error=msg)
        except Exception as exc:
            logger.exception("generate_quiz page swap failed unexpectedly")
            msg = f"quiz page swap failed: {type(exc).__name__}: {str(exc)[:200]}"
            if on_state is not None:
                await on_state("error", msg)
            return QuizGenerationResult(ok=False, blob=blob, error=msg)
    else:
        logger.info("[GENERATE_QUIZ] canvas dispatch not wired; skipping bundled set_page")

    if on_state is not None:
        await on_state("ready", None)
    return QuizGenerationResult(ok=True, blob=blob)


def make_handle_generate_quiz(backend_client, session_context, canvas_ctx):
    """Factory that binds the LLM tool handler to the session's session.

    Thin wrapper around :func:`run_quiz_generation` (S65c Block 3). The
    pre-refactor return-shape contract is preserved verbatim:

      * Success ⇒ ``params.result_callback(blob)`` — raw blob, not
        wrapped.
      * Failure ⇒ ``params.result_callback({"ok": False, "error": ...})``
        — the LLM uses the string to apologise to the visitor.

    Tests live in ``tests/test_quiz_generation.py``; they snapshot the
    pre-refactor behavior and act as the regression guard for this
    refactor. ``on_state`` is held at ``None`` here so the LLM tool path
    never emits Daily app-messages — those are S65c's manual-button
    path concern (Block 5 + Block 6).
    """
    async def handle_generate_quiz(params: FunctionCallParams):
        args = params.arguments or {}
        count = args.get("count", 3)
        language = args.get("language", "en")
        result = await run_quiz_generation(
            backend_client=backend_client,
            session_context=session_context,
            canvas_ctx=canvas_ctx,
            count=count,
            language=language,
            on_state=None,
        )
        if not result.ok:
            await params.result_callback({"ok": False, "error": result.error})
            return
        await params.result_callback(result.blob)

    return handle_generate_quiz
