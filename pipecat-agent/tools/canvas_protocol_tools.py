"""
Canvas Protocol generic tools (S64c).

Replaces the V2.13/S47 hardcoded tools (highlight_element, arrow_between,
add_annotation, navigate_scene, clear_overlays). The agent now calls 4 stable
generic verbs that work uniformly across Canvas Page types (composition,
youtube in S64d, quiz in S64e). The visitor-facing canvas tool set is 5 —
canvas_annotate (Block 8) is the 5th, registered separately in bot.py because it
draws on the shell overlay (an agent_annotate message) rather than dispatching a
canvas.command through these protocol verbs.

Each tool's handler:
  1. Validates the call against the active Page's manifest (canvas_manifest.py).
  2. Builds a wire-format CanvasCommand payload.
  3. Sends it as a Daily app-message via the transport.
  4. Awaits the matching CanvasCommandResult / CanvasCommandError reply.
  5. Returns the result (or raises) for the LLM service to incorporate.

Eager streaming dispatch (services/eager_dispatch/*) bypasses (1)-(5) for
arg-less verbs — when the LLM finishes streaming the verb token, the adapter
fires the Daily message immediately without waiting for the full tool call to
close. The eager path marks the call as `eager_dispatched=True` so this
handler skips re-dispatching when stop_reason finally arrives.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams

from context.canvas_manifest import CanvasManifestRegistry


# ----------------------------------------------------------------------------
# Verb classification
# ----------------------------------------------------------------------------

# Arg-less control/action verbs — eager dispatch fires the moment the verb
# token closes. The verb name alone is sufficient; no other args needed.
EAGER_DISPATCH_VERBS: frozenset[str] = frozenset(
    {
        "next_scene",
        "previous_scene",
        "clear",
        "next_question",
        "previous_question",
        "restart",
        "play",
        "pause",
    }
)

# Shell-level control verbs. These do NOT execute against the active Page's
# iframe — they're intercepted by the frontend's DailyRelay (see
# lib/canvas-protocol/daily-relay.ts SCENE_NAV_VERBS) and routed to the
# live-room shell's navigateToIndex. As a consequence:
#   1. They are always available regardless of which Canvas Page is active.
#   2. They are NOT (and should not be) listed in any iframe-side manifest's
#      capabilities.control.verbs — the iframe doesn't handle them.
#   3. The agent's per-verb manifest validation MUST exempt them or the
#      LLM hits a spurious UNSUPPORTED_VERB the moment the active Page (e.g.
#      YouTube, Quiz) doesn't declare them in its iframe manifest.
# Kept in sync with the relay-side constant by convention (two-line drift
# risk acceptable; these verbs are stable).
SCENE_NAV_VERBS: frozenset[str] = frozenset(
    {
        "next_scene",
        "previous_scene",
        "goto_scene",
    }
)


# ----------------------------------------------------------------------------
# Pending command tracking — bridges async tool-call handlers and Daily replies
# ----------------------------------------------------------------------------


@dataclass
class PendingCommand:
    command_id: str
    future: asyncio.Future
    eager_dispatched: bool = False


class PendingCommandRegistry:
    """Tracks in-flight canvas commands by commandId. The Daily message handler
    resolves them when reply messages arrive."""

    def __init__(self):
        self._pending: Dict[str, PendingCommand] = {}

    def open(self, command_id: str, eager: bool = False) -> asyncio.Future:
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._pending[command_id] = PendingCommand(
            command_id, fut, eager_dispatched=eager
        )
        return fut

    def is_eager(self, command_id: str) -> bool:
        p = self._pending.get(command_id)
        return bool(p and p.eager_dispatched)

    def resolve(self, command_id: str, result: dict) -> bool:
        p = self._pending.pop(command_id, None)
        if not p:
            return False
        if not p.future.done():
            p.future.set_result(result)
        return True

    def reject(self, command_id: str, error: dict) -> bool:
        p = self._pending.pop(command_id, None)
        if not p:
            return False
        if not p.future.done():
            p.future.set_exception(CanvasCommandError(error))
        return True

    def cancel_all(self, reason: str = "session_end"):
        for p in list(self._pending.values()):
            if not p.future.done():
                p.future.set_exception(
                    CanvasCommandError({"code": "CANCELLED", "message": reason})
                )
        self._pending.clear()


class CanvasCommandError(Exception):
    """Raised when a canvas command returns an error reply or times out."""

    def __init__(self, error: dict):
        self.code = error.get("code", "UNKNOWN")
        self.message = error.get("message", "")
        self.details = error.get("details")
        super().__init__(f"[{self.code}] {self.message}")


# ----------------------------------------------------------------------------
# Tool schemas — what the LLM sees
# ----------------------------------------------------------------------------


def make_tool_schemas(manifest: Optional[dict] = None) -> list[FunctionSchema]:
    """Build the 4 generic canvas-protocol tool schemas. The descriptions reference the active
    Page's manifest if available — this gives the LLM verb-level guidance
    without bloating the system prompt."""

    def verbs_str(section: str) -> str:
        if not manifest:
            return "(no page registered)"
        cap = manifest.get("capabilities", {}).get(section, {})
        verbs = cap.get("verbs", []) if section in ("control", "action") else []
        return " | ".join(verbs) if verbs else "(none)"

    return [
        FunctionSchema(
            name="canvas_analyze",
            description=(
                "Ask about anything visible on the visitor's screen. The visitor's "
                "LIVE screen — the actual rendered frame, any video, and any pen "
                "marks / highlights / text they've drawn — is NOT in your text "
                "context, so do NOT answer a 'what's on screen / what do you see' "
                "question from the scene description. Call this whenever the visitor "
                "asks what is on the screen, what you can see, to look at / read / "
                "check the screen, or about anything they've drawn or pointed at."
            ),
            properties={
                "question": {
                    "type": "string",
                    "description": "The question in natural language.",
                },
                "options": {
                    "type": "object",
                    "description": "Reserved for future provider hints. Pass {} for v0.1.",
                },
            },
            required=["question"],
        ),
        FunctionSchema(
            name="canvas_control",
            description=(
                f"Invoke a control verb on the active Canvas Page. Supported verbs: {verbs_str('control')}. "
                "Control verbs are state transitions (navigation, media playback, clearing). "
                "Verb-specific fields MUST be nested inside `args`, not placed at the top level alongside `verb`."
            ),
            properties={
                "verb": {
                    "type": "string",
                    "description": "The control verb to invoke.",
                },
                "args": {
                    "type": "object",
                    "description": (
                        "Verb-specific args object. "
                        "Argless verbs (next_scene, previous_scene, clear, play, pause, restart, "
                        "next_question, previous_question) take {}. "
                        'For seek: {"seconds": <non-negative number>} — absolute timestamp in seconds '
                        "(e.g. 120 for two minutes). "
                        'For set_speed: {"rate": <number>} — playback rate, 1.0 normal, 0.5 half, 2.0 double. '
                        'For goto_scene: {"index": <integer>} — zero-based scene index.'
                    ),
                },
            },
            required=["verb"],
        ),
        FunctionSchema(
            name="canvas_action",
            description=(
                f"Invoke an action verb on the active Canvas Page. Supported verbs: {verbs_str('action')}. "
                "Actions are content-producing operations (drawing arrows, adding annotations, submitting answers). "
                "Verb-specific fields MUST be nested inside `args`, not placed at the top level alongside `verb`."
            ),
            properties={
                "verb": {"type": "string", "description": "The action verb to invoke."},
                "args": {
                    "type": "object",
                    "description": (
                        "Verb-specific args object. "
                        'For draw_arrow: {"from": "<element_id>", "to": "<element_id>"} — '
                        "both ids must be element ids from CANVAS ELEMENTS (not overlay ids "
                        "like 'ovl_3' from prior tool results). "
                        'For add_annotation: {"text": "<string>", "x": <number>, '
                        '"y": <number>} — x and y in 1280x720 design-space coordinates.'
                    ),
                },
            },
            required=["verb"],
        ),
        FunctionSchema(
            name="canvas_set_page",
            description=(
                "Switch the active Canvas Page type. Use sparingly — typically only to return "
                "to 'composition' after a quiz. For quiz: do NOT call this tool — the quiz Page "
                "swap is bundled into generate_quiz_from_knowledge, which generates the blob "
                "and activates the Page in a single step. Calling canvas_set_page(pageType='quiz') "
                "directly will not display new questions because the LLM cannot reliably re-emit "
                "the full quiz blob as pageInit. pageType must be in the allowed set: "
                "composition, youtube, quiz."
            ),
            properties={
                "pageType": {
                    "type": "string",
                    "description": "One of: composition, youtube, quiz. Prefer 'composition' for returning to the scene view.",
                },
                "pageInit": {
                    "type": "object",
                    "description": "Page-specific seed data. Pass {} for composition (the snapshot drives the page) and youtube. Not used for quiz — see generate_quiz_from_knowledge.",
                },
            },
            required=["pageType"],
        ),
    ]


# ----------------------------------------------------------------------------
# Daily message helpers
# ----------------------------------------------------------------------------


def build_canvas_command(
    tool: str, args: dict, command_id: Optional[str] = None
) -> dict:
    """Build the Daily app-message payload for an outbound canvas command.
    Wire format mirrors the in-iframe postMessage protocol exactly so the
    frontend's Daily relay can forward it to the Canvas Service unchanged."""
    cmd = {
        "type": "canvas.command",
        "commandId": command_id or str(uuid.uuid4()),
        "tool": tool,
        "args": dict(args or {}),
    }
    # control/action carry the verb at the top level (mirrors the postMessage shape).
    if tool in ("control", "action"):
        verb = cmd["args"].pop("verb", None)
        if verb:
            cmd["verb"] = verb
    return cmd


# ----------------------------------------------------------------------------
# Tool handler factory — produces the async handlers Pipecat will register
# ----------------------------------------------------------------------------


@dataclass
class CanvasToolContext:
    """Shared state passed to tool handlers. Wired in agent.py during pipeline setup."""

    manifest_registry: CanvasManifestRegistry
    pending: PendingCommandRegistry
    send_app_message: Any  # callable(payload: dict) -> None — sends a Daily app-message
    # S64c — alias→element_id map for the active snapshot. The canvas_action
    # draw_arrow handler translates args.from / args.to through this map before
    # dispatching, so the LLM can reference elements by short, stable
    # aliases (text_1, avatar_1, …) instead of long UUID7 strings it
    # routinely fails to copy verbatim. bot.py refreshes this map on
    # session start and after every scene navigation; see persona.py's
    # aliases_out parameter.
    element_alias_map: dict[str, str] = field(default_factory=dict)
    command_timeout_s: float = 6.0
    # S66 Block 5a / S67b — lazy vision. handle_analyze calls this before
    # dispatching `analyze`; the closure runs the vision query and RETURNS the
    # vision answer text (or None). handle_analyze folds that text into the
    # canvas_analyze TOOL RESULT (in-band) — previously it was injected
    # out-of-band via context.add_message, which raced the function-call re-run
    # and intermittently left the spoken reply without the vision answer.
    # None ⇒ no vision this turn. bot.py constructs the closure (run_bot_*).
    ensure_vision: Optional[Callable[[str], Awaitable[Optional[str]]]] = None


async def dispatch_canvas_command(
    ctx: "CanvasToolContext",
    tool: str,
    args: dict,
    command_id: Optional[str] = None,
) -> dict:
    """Build, send, and await a canvas command via the protocol substrate.

    Shared between ``make_handlers`` (the 5 generic canvas tools) and
    ``tools.quiz_generation`` (generate_quiz_from_knowledge dispatches a
    set_page after the blob lands, so the LLM doesn't have to copy the
    blob into a follow-up canvas_set_page — see S64e Option D).
    """
    if not ctx.send_app_message:
        raise RuntimeError("CanvasToolContext.send_app_message not wired")

    cmd = build_canvas_command(tool, args, command_id=command_id)

    # If eager dispatch already opened the future, just attach the timeout.
    # Otherwise create a fresh one.
    if ctx.pending.is_eager(cmd["commandId"]):
        # Eager adapter has already sent the Daily message and registered
        # the future. We just need to await the existing future.
        existing = ctx.pending._pending.get(cmd["commandId"])
        future = existing.future if existing else ctx.pending.open(cmd["commandId"])
        logger.info(
            f"[CANVAS DISPATCH] tool={tool} commandId={cmd['commandId']} (eager already sent)"
        )
    else:
        future = ctx.pending.open(cmd["commandId"])
        await ctx.send_app_message(cmd)
        logger.info(
            f"[CANVAS DISPATCH] tool={tool} commandId={cmd['commandId']} payload={cmd!r}"
        )

    try:
        result = await asyncio.wait_for(future, timeout=ctx.command_timeout_s)
        logger.info(
            f"[CANVAS RESULT] tool={tool} commandId={cmd['commandId']} result={result!r}"
        )
        return result
    except asyncio.TimeoutError:
        ctx.pending._pending.pop(cmd["commandId"], None)
        logger.warning(
            f"[CANVAS TIMEOUT] tool={tool} commandId={cmd['commandId']} "
            f"after {ctx.command_timeout_s}s — frontend Canvas Service did not reply"
        )
        raise CanvasCommandError(
            {
                "code": "TIMEOUT",
                "message": f"canvas {tool} timed out after {ctx.command_timeout_s}s",
            }
        )


def _merge_analyze_result(vision_answer: Optional[str], page_state):
    """Fold the vision answer into the canvas_analyze tool result (in-band).

    S67b fix: the vision answer is the authoritative answer for visual questions
    ("what's on screen / what did I circle"), so it leads as ``answer``; the
    iframe page semantic state rides along as ``page_state`` for page-state
    questions (quiz / youtube). Either may be absent — with no vision answer this
    returns the page state unchanged (legacy behavior).
    """
    if vision_answer is not None and page_state is not None:
        return {"answer": vision_answer, "page_state": page_state}
    if vision_answer is not None:
        return {"answer": vision_answer}
    return page_state


def make_handlers(ctx: CanvasToolContext):
    """Build {tool_name: async_handler} for Pipecat's function-calling system.
    Each handler validates against the manifest, sends the Daily message, awaits
    the reply via Future."""

    async def _dispatch(
        tool: str, args: dict, command_id: Optional[str] = None
    ) -> dict:
        return await dispatch_canvas_command(ctx, tool, args, command_id=command_id)

    def _resolve_element_id(raw: Any) -> Any:
        """Translate an LLM-supplied element id from alias form to the real
        UUID. Pass-through when the value isn't a known alias (it's either
        already a UUID or some other shape — let the frontend reject it
        cleanly rather than masking the error here)."""
        if isinstance(raw, str) and raw in ctx.element_alias_map:
            return ctx.element_alias_map[raw]
        return raw

    async def _emit_error(
        params: FunctionCallParams, exc: CanvasCommandError, tool_label: str
    ) -> None:
        """Return the canvas error to the LLM as a tool result instead of
        letting it propagate as a pipeline ErrorFrame. Mirrors V2.13's
        always-result_callback pattern so a single failed call doesn't
        break the conversation turn — the LLM can read the error and
        recover (e.g. retry with a different argument shape, or apologize)."""
        logger.warning(f"[{tool_label}] error code={exc.code} message={exc.message!r}")
        await params.result_callback(
            {
                "error": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
        )

    async def handle_analyze(params: FunctionCallParams):
        args = params.arguments or {}
        question = args.get("question", "")
        logger.info(f"[CANVAS_ANALYZE] called: question={question!r}")
        # S67b fix — run vision and CAPTURE its answer to fold into the tool
        # result below (in-band). Previously the vision answer was injected
        # out-of-band via context.add_message (a separate developer message),
        # which raced the function-call re-run: the spoken reply was sometimes
        # generated from only the iframe page-state result and said "I can't
        # see" even though vision succeeded. Failures are logged, not fatal.
        vision_answer = None
        if ctx.ensure_vision is not None:
            try:
                # Pass the visitor's question so the vision path can derive its
                # mode (point / assess / describe) from the utterance.
                vision_answer = await ctx.ensure_vision(question)
            except Exception as exc:
                logger.warning(f"[CANVAS_ANALYZE] ensure_vision failed: {exc!r}")
        # Dispatch the iframe page-state analyze, but don't let its failure drop
        # the vision answer — for visual questions, vision IS the answer.
        page_state = None
        try:
            page_state = await _dispatch(
                "analyze", {"question": question, "options": args.get("options", {})}
            )
        except CanvasCommandError as exc:
            if vision_answer is None:
                await _emit_error(params, exc, "CANVAS_ANALYZE")
                return
            logger.warning(
                f"[CANVAS_ANALYZE] page-state analyze failed; returning vision answer only: {exc!r}"
            )
        await params.result_callback(_merge_analyze_result(vision_answer, page_state))

    async def handle_control(params: FunctionCallParams):
        args = params.arguments or {}
        verb = args.get("verb")
        verb_args = args.get("args") or {}
        logger.info(f"[CANVAS_CONTROL] called: verb={verb!r} args={verb_args!r}")
        try:
            # Manifest validation. Scene-nav verbs (SCENE_NAV_VERBS) bypass
            # the manifest check because they're handled by the live-room
            # shell, not by the active Page's iframe — they're always
            # available regardless of which Page is mounted. Without this
            # exemption, calling `next_scene` while on a YouTube or Quiz
            # scene hit UNSUPPORTED_VERB because those iframes' manifests
            # don't (and shouldn't) declare scene-nav verbs.
            if verb not in SCENE_NAV_VERBS:
                manifest = ctx.manifest_registry.current()
                if manifest:
                    cap = manifest.get("capabilities", {}).get("control", {})
                    if verb not in cap.get("verbs", []):
                        logger.warning(
                            f"[CANVAS_CONTROL] UNSUPPORTED_VERB verb={verb!r} "
                            f"available={cap.get('verbs', [])}"
                        )
                        raise CanvasCommandError(
                            {
                                "code": "UNSUPPORTED_VERB",
                                "message": f"Active page does not support control verb '{verb}'. Available: {cap.get('verbs', [])}",
                            }
                        )
            result = await _dispatch("control", {"verb": verb, **verb_args})
            # Scene-change refresh is no longer triggered from here. Both
            # voice and visitor-initiated nav flow through the frontend's
            # navigateToIndex, which emits a canvas.sceneChanged Daily
            # message; bot.py's on_app_message routes that to
            # refresh_agent_for_current_scene. Single source of truth.
            await params.result_callback(result)
        except CanvasCommandError as exc:
            await _emit_error(params, exc, "CANVAS_CONTROL")

    async def handle_action(params: FunctionCallParams):
        args = params.arguments or {}
        verb = args.get("verb")
        verb_args = args.get("args") or {}
        logger.info(f"[CANVAS_ACTION] called: verb={verb!r} args={verb_args!r}")
        # S64c — translate aliases → real UUIDs for verbs that take element
        # ids. draw_arrow's `from` and `to` are element ids; LLM uses
        # alias form. Other verbs (add_annotation) take text/x/y, not
        # element ids — pass-through.
        if verb == "draw_arrow" and isinstance(verb_args, dict):
            resolved_args = dict(verb_args)
            for key in ("from", "to"):
                if key in resolved_args:
                    resolved = _resolve_element_id(resolved_args[key])
                    if resolved != resolved_args[key]:
                        logger.info(
                            f"[CANVAS_ACTION] resolved alias {key}={resolved_args[key]!r} -> {resolved!r}"
                        )
                    resolved_args[key] = resolved
            verb_args = resolved_args
        try:
            manifest = ctx.manifest_registry.current()
            if manifest:
                cap = manifest.get("capabilities", {}).get("action", {})
                if verb not in cap.get("verbs", []):
                    logger.warning(
                        f"[CANVAS_ACTION] UNSUPPORTED_VERB verb={verb!r} "
                        f"available={cap.get('verbs', [])}"
                    )
                    raise CanvasCommandError(
                        {
                            "code": "UNSUPPORTED_VERB",
                            "message": f"Active page does not support action verb '{verb}'. Available: {cap.get('verbs', [])}",
                        }
                    )
            result = await _dispatch("action", {"verb": verb, **verb_args})
            # No bundled next_question here. Question-to-question advancement
            # is owned by the Quiz Page itself (it auto-reveals the explanation
            # and auto-advances on a timer after submit_answer) so the visitor
            # actually sees the visual feedback before the iframe moves on.
            # The submit_answer reply already includes `completed: bool` from
            # the frontend so the LLM knows when it just answered the last
            # question. See public/canvas-pages/quiz/main.js and the AGENT
            # PLAYBOOK quiz flow.
            await params.result_callback(result)
        except CanvasCommandError as exc:
            await _emit_error(params, exc, "CANVAS_ACTION")

    async def handle_set_page(params: FunctionCallParams):
        args = params.arguments or {}
        page_type = args.get("pageType")
        page_init = args.get("pageInit") or {}
        logger.info(
            f"[CANVAS_SET_PAGE] called: pageType={page_type!r} pageInit={page_init!r}"
        )
        try:
            # In v0.1 only composition is allowed. S64d/e expand the allowlist.
            if page_type not in {"composition", "youtube", "quiz"}:
                logger.warning(
                    f"[CANVAS_SET_PAGE] UNKNOWN_PAGE_TYPE pageType={page_type!r}"
                )
                raise CanvasCommandError(
                    {
                        "code": "UNKNOWN_PAGE_TYPE",
                        "message": f"pageType '{page_type}' is not in the allowlist",
                    }
                )
            result = await _dispatch(
                "set_page", {"pageType": page_type, "pageInit": page_init}
            )
            await params.result_callback(result)
        except CanvasCommandError as exc:
            await _emit_error(params, exc, "CANVAS_SET_PAGE")

    return {
        "canvas_analyze": handle_analyze,
        "canvas_control": handle_control,
        "canvas_action": handle_action,
        "canvas_set_page": handle_set_page,
    }
