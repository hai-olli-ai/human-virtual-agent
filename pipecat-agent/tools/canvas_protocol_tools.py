"""
Canvas Protocol generic tools (S64c).

Replaces the V2.13/S47 hardcoded tools (highlight_element, arrow_between,
add_annotation, navigate_scene, clear_overlays). The agent now calls 5 stable
generic verbs that work uniformly across Canvas Page types (composition,
youtube in S64d, quiz in S64e).

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
EAGER_DISPATCH_VERBS: frozenset[str] = frozenset({
    "next_scene", "previous_scene", "clear",
    "next_question", "previous_question", "restart",
    "play", "pause",
})

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
SCENE_NAV_VERBS: frozenset[str] = frozenset({
    "next_scene", "previous_scene", "goto_scene",
})


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
        self._pending[command_id] = PendingCommand(command_id, fut, eager_dispatched=eager)
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
    """Build the 5 generic tool schemas. The descriptions reference the active
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
                "question": {"type": "string", "description": "The question in natural language."},
                "options": {"type": "object", "description": "Reserved for future provider hints. Pass {} for v0.1."},
            },
            required=["question"],
        ),
        FunctionSchema(
            name="canvas_highlight",
            description=(
                "Draw a highlight overlay on the canvas. Each Canvas Page declares which "
                "target shape it accepts in the CANVAS PAGE section of the system prompt — "
                "composition pages accept element_id targets, the youtube page accepts box "
                "targets. Send the shape the active Page accepts; sending the wrong shape "
                "returns INVALID_ARGS and no highlight is drawn."
            ),
            properties={
                # Pipecat 0.0.108's FunctionSchema.to_default_dict() forwards
                # this property dict straight into OpenAI's tool spec. It does
                # NOT emit strict: true, so the LLM is allowed to violate
                # `required` and we observed gpt-5.4 emitting target=null on
                # box-only pages when it couldn't construct coordinates. The
                # oneOf branches below give the model a structurally
                # unambiguous schema (a second hint on top of the prompt
                # text); the agent-side _validate_highlight_target gate is
                # the contract that actually catches malformed calls before
                # dispatch.
                "target": {
                    "type": "object",
                    "description": (
                        "Highlight target. Send EXACTLY ONE of the two shapes documented "
                        "under oneOf — pick the shape the active Page accepts (see "
                        "CANVAS PAGE in the system prompt). Do not call this tool with "
                        "target=null or an empty target."
                    ),
                    "oneOf": [
                        {
                            "title": "element_id target",
                            "type": "object",
                            "properties": {
                                "element_id": {
                                    "type": "string",
                                    "description": (
                                        "Element id from CANVAS ELEMENTS — an alias like "
                                        "'text_1' or 'avatar_1', or a full UUID. Accepted "
                                        "by composition-style pages."
                                    ),
                                },
                            },
                            "required": ["element_id"],
                            "additionalProperties": False,
                        },
                        {
                            "title": "box target",
                            "type": "object",
                            "properties": {
                                "box": {
                                    "type": "array",
                                    "description": (
                                        "[x, y, width, height] in 1280x720 design-space "
                                        "coordinates. Accepted by pages that highlight "
                                        "arbitrary regions (e.g. youtube)."
                                    ),
                                    "items": {"type": "number"},
                                    "minItems": 4,
                                    "maxItems": 4,
                                },
                            },
                            "required": ["box"],
                            "additionalProperties": False,
                        },
                    ],
                },
                "options": {"type": "object", "description": "Reserved. Pass {} for v0.1."},
            },
            required=["target"],
        ),
        FunctionSchema(
            name="canvas_control",
            description=(
                f"Invoke a control verb on the active Canvas Page. Supported verbs: {verbs_str('control')}. "
                "Control verbs are state transitions (navigation, media playback, clearing). "
                "Verb-specific fields MUST be nested inside `args`, not placed at the top level alongside `verb`."
            ),
            properties={
                "verb": {"type": "string", "description": "The control verb to invoke."},
                "args": {
                    "type": "object",
                    "description": (
                        "Verb-specific args object. "
                        "Argless verbs (next_scene, previous_scene, clear, play, pause, restart, "
                        "next_question, previous_question) take {}. "
                        "For seek: {\"seconds\": <non-negative number>} — absolute timestamp in seconds "
                        "(e.g. 120 for two minutes). "
                        "For set_speed: {\"rate\": <number>} — playback rate, 1.0 normal, 0.5 half, 2.0 double. "
                        "For goto_scene: {\"index\": <integer>} — zero-based scene index."
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
                        "For draw_arrow: {\"from\": \"<element_id>\", \"to\": \"<element_id>\"} — "
                        "both ids must be element ids from CANVAS ELEMENTS (not overlay ids "
                        "like 'ovl_3' from prior tool results). "
                        "For add_annotation: {\"text\": \"<string>\", \"x\": <number>, "
                        "\"y\": <number>} — x and y in 1280x720 design-space coordinates."
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
                "pageType": {"type": "string", "description": "One of: composition, youtube, quiz. Prefer 'composition' for returning to the scene view."},
                "pageInit": {"type": "object", "description": "Page-specific seed data. Pass {} for composition (the snapshot drives the page) and youtube. Not used for quiz — see generate_quiz_from_knowledge."},
            },
            required=["pageType"],
        ),
    ]


# ----------------------------------------------------------------------------
# Daily message helpers
# ----------------------------------------------------------------------------

def build_canvas_command(tool: str, args: dict, command_id: Optional[str] = None) -> dict:
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
    # S64c — alias→element_id map for the active snapshot. Tool handlers
    # translate target.element_id (canvas_highlight) and args.from /
    # args.to (canvas_action verb=draw_arrow) through this map before
    # dispatching, so the LLM can reference elements by short, stable
    # aliases (text_1, avatar_1, …) instead of long UUID7 strings it
    # routinely fails to copy verbatim. bot.py refreshes this map on
    # session start and after every scene navigation; see persona.py's
    # aliases_out parameter.
    element_alias_map: dict[str, str] = field(default_factory=dict)
    command_timeout_s: float = 6.0
    # S66 Block 5a — lazy vision frame. When wired, handle_analyze calls
    # this before dispatching `analyze` so the LLM has a fresh scene
    # image in context for the active scene. None ⇒ no-op (legacy eager
    # behavior, where bot.py refreshes the vision frame on every scene
    # change). bot.py constructs the closure that bridges this to
    # vision_refresh.ensure_vision_frame_for_scene.
    ensure_vision: Optional[Callable[[str], Awaitable[None]]] = None


def _validate_highlight_target(target: Any) -> Optional["CanvasCommandError"]:
    """Shape-check the LLM-supplied highlight target before dispatch.

    Pipecat 0.0.108's `FunctionSchema.to_default_dict` does not emit
    `strict: true`, so OpenAI treats our `required: ["target"]` as a hint
    and the model can produce calls with `target` missing/null. We observed
    gpt-5.4 doing exactly that on YouTube scenes (box-only page) when it
    couldn't construct coordinates — see the 2026-05-13 production warning
    `INVALID_TOOL: highlight requires args.target`.

    Catching here (instead of letting the wire-format check in the
    frontend's daily-relay reject) lets us return a richer, instructive
    error to the LLM via the result_callback, which the model can act on
    the very next turn. Manifest-aware "page rejects this shape"
    validation deliberately stays in the frontend
    (lib/canvas-protocol/validate.ts) as the single source of truth — this
    gate only catches structural failures (no target, no shape).

    Returns None when the target is well-shaped, otherwise a
    CanvasCommandError ready to hand to `_emit_error`.
    """
    if not isinstance(target, dict):
        return CanvasCommandError({
            "code": "INVALID_ARGS",
            "message": (
                "canvas_highlight requires `target` as an object. "
                "Use {\"element_id\": \"<id>\"} on element-id pages, or "
                "{\"box\": [x, y, w, h]} on box pages. The active Page's accepted "
                "shape is listed in CANVAS PAGE — do not call canvas_highlight "
                "with a null or missing target. If you don't know what to "
                "highlight, call canvas_analyze first or respond verbally."
            ),
        })
    has_element_id = isinstance(target.get("element_id"), str) and bool(target["element_id"])
    box = target.get("box")
    has_box = (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in box)
    )
    if not has_element_id and not has_box:
        return CanvasCommandError({
            "code": "INVALID_ARGS",
            "message": (
                "canvas_highlight target must include either `element_id` "
                "(non-empty string) or `box` ([x, y, w, h] of four numbers). "
                "Check CANVAS PAGE for the shape the active Page accepts, "
                "then retry with a valid target."
            ),
        })
    return None


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
        logger.info(f"[CANVAS DISPATCH] tool={tool} commandId={cmd['commandId']} (eager already sent)")
    else:
        future = ctx.pending.open(cmd["commandId"])
        await ctx.send_app_message(cmd)
        logger.info(f"[CANVAS DISPATCH] tool={tool} commandId={cmd['commandId']} payload={cmd!r}")

    try:
        result = await asyncio.wait_for(future, timeout=ctx.command_timeout_s)
        logger.info(f"[CANVAS RESULT] tool={tool} commandId={cmd['commandId']} result={result!r}")
        return result
    except asyncio.TimeoutError:
        ctx.pending._pending.pop(cmd["commandId"], None)
        logger.warning(
            f"[CANVAS TIMEOUT] tool={tool} commandId={cmd['commandId']} "
            f"after {ctx.command_timeout_s}s — frontend Canvas Service did not reply"
        )
        raise CanvasCommandError(
            {"code": "TIMEOUT", "message": f"canvas {tool} timed out after {ctx.command_timeout_s}s"}
        )


def make_handlers(ctx: CanvasToolContext):
    """Build {tool_name: async_handler} for Pipecat's function-calling system.
    Each handler validates against the manifest, sends the Daily message, awaits
    the reply via Future."""

    async def _dispatch(tool: str, args: dict, command_id: Optional[str] = None) -> dict:
        return await dispatch_canvas_command(ctx, tool, args, command_id=command_id)

    def _resolve_element_id(raw: Any) -> Any:
        """Translate an LLM-supplied element id from alias form to the real
        UUID. Pass-through when the value isn't a known alias (it's either
        already a UUID or some other shape — let the frontend reject it
        cleanly rather than masking the error here)."""
        if isinstance(raw, str) and raw in ctx.element_alias_map:
            return ctx.element_alias_map[raw]
        return raw

    async def _emit_error(params: FunctionCallParams, exc: CanvasCommandError, tool_label: str) -> None:
        """Return the canvas error to the LLM as a tool result instead of
        letting it propagate as a pipeline ErrorFrame. Mirrors V2.13's
        always-result_callback pattern so a single failed call doesn't
        break the conversation turn — the LLM can read the error and
        recover (e.g. retry with a different argument shape, or apologize)."""
        logger.warning(f"[{tool_label}] error code={exc.code} message={exc.message!r}")
        await params.result_callback({
            "error": exc.code,
            "message": exc.message,
            "details": exc.details,
        })

    async def handle_analyze(params: FunctionCallParams):
        args = params.arguments or {}
        question = args.get("question", "")
        logger.info(f"[CANVAS_ANALYZE] called: question={question!r}")
        # S66 Block 5a — lazy vision frame. Fetch and add the current
        # scene's image to context if it isn't already loaded. Failures
        # are logged but don't fail the analyze call — the iframe's
        # semantic state alone still answers most questions.
        if ctx.ensure_vision is not None:
            try:
                # S67b — pass the visitor's question so the vision path can
                # derive its mode (point / assess / describe) from the utterance.
                await ctx.ensure_vision(question)
            except Exception as exc:
                logger.warning(f"[CANVAS_ANALYZE] ensure_vision failed: {exc!r}")
        try:
            result = await _dispatch("analyze", {"question": question, "options": args.get("options", {})})
            await params.result_callback(result)
        except CanvasCommandError as exc:
            await _emit_error(params, exc, "CANVAS_ANALYZE")

    async def handle_highlight(params: FunctionCallParams):
        args = params.arguments or {}
        target = args.get("target")
        logger.info(f"[CANVAS_HIGHLIGHT] called: target={target!r}")
        # Pre-dispatch shape check — see _validate_highlight_target for the
        # reasoning. Without this gate, malformed calls (target=None,
        # empty target object) round-trip through Daily to the frontend
        # relay, which returns a generic INVALID_TOOL the model has to
        # decode. Returning a tailored error here keeps the model in the
        # conversation with a clearer correction hint.
        shape_error = _validate_highlight_target(target)
        if shape_error is not None:
            await _emit_error(params, shape_error, "CANVAS_HIGHLIGHT")
            return
        # S64c — translate alias → real UUID before dispatch. The LLM sees
        # short aliases in the prompt's "Available canvas elements"
        # listing and reproduces them in target.element_id; the frontend
        # only knows real UUIDs. Safe to assume target is a dict here —
        # the shape check above rejected non-dict targets.
        if "element_id" in target:
            resolved = _resolve_element_id(target["element_id"])
            if resolved != target["element_id"]:
                logger.info(
                    f"[CANVAS_HIGHLIGHT] resolved alias {target['element_id']!r} -> {resolved!r}"
                )
            target = {**target, "element_id": resolved}
        try:
            result = await _dispatch("highlight", {"target": target, "options": args.get("options", {})})
            await params.result_callback(result)
        except CanvasCommandError as exc:
            await _emit_error(params, exc, "CANVAS_HIGHLIGHT")

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
                        raise CanvasCommandError({
                            "code": "UNSUPPORTED_VERB",
                            "message": f"Active page does not support control verb '{verb}'. Available: {cap.get('verbs', [])}",
                        })
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
                    raise CanvasCommandError({
                        "code": "UNSUPPORTED_VERB",
                        "message": f"Active page does not support action verb '{verb}'. Available: {cap.get('verbs', [])}",
                    })
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
        logger.info(f"[CANVAS_SET_PAGE] called: pageType={page_type!r} pageInit={page_init!r}")
        try:
            # In v0.1 only composition is allowed. S64d/e expand the allowlist.
            if page_type not in {"composition", "youtube", "quiz"}:
                logger.warning(f"[CANVAS_SET_PAGE] UNKNOWN_PAGE_TYPE pageType={page_type!r}")
                raise CanvasCommandError({
                    "code": "UNKNOWN_PAGE_TYPE",
                    "message": f"pageType '{page_type}' is not in the allowlist",
                })
            result = await _dispatch("set_page", {"pageType": page_type, "pageInit": page_init})
            await params.result_callback(result)
        except CanvasCommandError as exc:
            await _emit_error(params, exc, "CANVAS_SET_PAGE")

    return {
        "canvas_analyze": handle_analyze,
        "canvas_highlight": handle_highlight,
        "canvas_control": handle_control,
        "canvas_action": handle_action,
        "canvas_set_page": handle_set_page,
    }
