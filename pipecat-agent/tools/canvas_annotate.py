"""canvas_annotate — agent-drawn annotations on the live shell overlay (Block 8).

Unlike the 5 generic canvas protocol tools, ``canvas_annotate`` does NOT dispatch
a ``canvas.command`` through the iframe / DailyRelay. It emits a session-level
``agent_annotate`` Daily app-message (a sibling of ``request_canvas_capture``)
that the shell's general handler draws onto the S67a annotation overlay — pixels
above the iframe, not the iframe's own canvas protocol. (This is what the deleted
``canvas_highlight`` tool's job becomes once S67a moves annotation to a shell
overlay: the agent points things out by drawing on the overlay, not by asking the
Page to highlight an element.)

Wiring mirrors ``make_handle_generate_quiz`` (the standalone-tool template, A-AG-2)
rather than ``make_handlers``: a factory closes over one session's S67b machinery
and returns the Pipecat ``FunctionCallParams`` handler. The capture / vision /
snapshot deps are INJECTED (mirrors ``run_vision_query`` / ``run_quiz_generation``)
so the resolution logic is unit-testable without the full pipecat stack.

Wire op shape (the contract the frontend Block must match):

    {"type": "agent_annotate", "annotateId": <uuid>, "ops": [op, ...]}
    op = {"op": "circle"|"arrow"|"shape"|"highlight"|"text"|"erase",
          "box": {"x", "y", "w", "h"},   # normalized 0-1, canvas-relative, top-left origin
          "shape"?: <shape>,              # when op == "shape"
          "text"?: <string>,              # when op == "text"
          "from"?: {"x","y","w","h"}}     # when op == "arrow" (optional source box)
    The "erase" op carries no box.

All boxes are normalized 0-1 of the 1280x720 design canvas (resolution-independent).
Element geometry is stored as PERCENT 0-100 (A-AG-3 — nested ``position``/``size``),
so the element path divides by 100, NOT by 1280/720. The ``describe`` path's box
comes from Gemini ``locate`` and is relative to the captured frame; it is treated
as canvas-relative on the assumption that the capture surface is the canvas
(A-AG-5 §4 — the frontend capture must grab the canvas region for this to line up).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable, Optional

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams

# Shapes the overlay can stamp at a target box (op="shape").
AGENT_ANNOTATE_SHAPES = (
    "heart", "circle", "rectangle", "arrow", "star", "checkmark", "question_mark",
)
# Top-level operations. "erase" clears all annotations; everything else needs a target.
AGENT_ANNOTATE_OPS = ("circle", "arrow", "shape", "highlight", "text", "erase")


AGENT_ANNOTATE_SCHEMA = FunctionSchema(
    name="canvas_annotate",
    description=(
        "Draw a temporary annotation on top of what the visitor sees — circle, arrow, "
        "shape, highlight, or text — or erase all annotations. Use it to point things "
        "out visually while you talk (circle the answer, put a heart on an image, draw "
        "an arrow to a button). Specify WHAT to annotate via `target`, giving EXACTLY "
        "ONE of: `element` (an alias like 'text_1' or 'button_1' from CANVAS ELEMENTS — "
        "best on composition scenes), `region` ({x, y, w, h} as fractions 0-1 of the "
        "canvas, top-left origin), or `describe` (a natural-language description; you "
        "will look at the live screen to locate it — best for video or arbitrary "
        "content). `target` is required for every op except `erase`."
    ),
    properties={
        "op": {
            "type": "string",
            "enum": list(AGENT_ANNOTATE_OPS),
            "description": "The annotation to draw, or 'erase' to clear all annotations.",
        },
        "target": {
            "type": "object",
            "description": (
                "What to annotate (required unless op='erase'). Provide EXACTLY ONE of "
                "`element`, `region`, or `describe`."
            ),
            "properties": {
                "element": {
                    "type": "string",
                    "description": "Element alias from CANVAS ELEMENTS (e.g. 'text_1').",
                },
                "region": {
                    "type": "object",
                    "description": "Normalized box, each value 0-1 (top-left origin).",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "w": {"type": "number"},
                        "h": {"type": "number"},
                    },
                },
                "describe": {
                    "type": "string",
                    "description": "Natural-language target to locate visually.",
                },
            },
        },
        "shape": {
            "type": "string",
            "enum": list(AGENT_ANNOTATE_SHAPES),
            "description": "Shape to stamp when op='shape'.",
        },
        "text": {
            "type": "string",
            "description": "Text to write when op='text'.",
        },
        "from": {
            "type": "string",
            "description": (
                "Optional source element alias for op='arrow' — the arrow runs from "
                "this element to the target. Omit for an arrow that just points at the target."
            ),
        },
    },
    required=["op"],
)


def element_box_from_snapshot(
    alias: str,
    elements: list[dict],
    alias_map: dict[str, str],
) -> dict | None:
    """Map an element alias → normalized ``{x, y, w, h}`` (0-1) from snapshot geometry.

    A-AG-3: snapshot elements carry geometry as PERCENT (0-100) of the 1280x720
    reference canvas, nested under ``position`` (``{x, y}``) and ``size``
    (``{width, height}``). Normalize by /100 — NOT /1280 or /720. ``alias`` is
    resolved to a real element id via ``alias_map`` (pass-through when it's already
    a UUID or unknown, matching ``_resolve_element_id``). Returns None when the
    alias can't be resolved or the geometry is missing / degenerate.
    """
    if not alias:
        return None
    eid = alias_map.get(alias, alias)
    el = next((e for e in (elements or []) if e.get("id") == eid), None)
    if el is None:
        return None
    pos = el.get("position") or {}
    size = el.get("size") or {}
    try:
        x = float(pos["x"]) / 100.0
        y = float(pos["y"]) / 100.0
        w = float(size["width"]) / 100.0
        h = float(size["height"]) / 100.0
    except (KeyError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def make_handle_canvas_annotate(
    *,
    send_message: Callable[[dict], Awaitable[None]],
    pending: dict[str, asyncio.Future],
    vision_client: Any,
    request_capture: Callable[[str], Awaitable[tuple]],
    fetch_live_bytes: Callable[..., Awaitable[Optional[bytes]]],
    backend_client: Any,
    session_context: Any,
    element_alias_map: dict[str, str],
    room_id: Optional[str],
    api_url: Optional[str],
    timeout_s: float,
):
    """Bind ``canvas_annotate`` to one session's S67b machinery (Block 8).

    Returns the Pipecat ``FunctionCallParams`` handler. ``pending`` is the
    session's ``_pending_annotates`` registry (the ``on_app_message``
    ``agent_annotate_result`` branch resolves its futures). All other deps mirror
    the closures already built in ``run_bot_*`` (A-AG-4).
    """

    async def _emit(ops: list[dict]) -> bool:
        """Emit an agent_annotate message and await the shell ack.

        Best-effort (C8): a timeout is treated as rendered. Daily is best-effort
        and the overlay is cosmetic — telling the LLM "failed" on a slow ack would
        make it apologise for an annotation that very likely drew fine.
        """
        annotate_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        pending[annotate_id] = fut
        try:
            await send_message(
                {"type": "agent_annotate", "annotateId": annotate_id, "ops": ops}
            )
            res = await asyncio.wait_for(fut, timeout=timeout_s)
            return bool((res or {}).get("ok"))
        except asyncio.TimeoutError:
            logger.info(
                "[CANVAS_ANNOTATE] ack timeout {} — assuming rendered (best-effort)",
                annotate_id,
            )
            return True
        finally:
            pending.pop(annotate_id, None)

    async def _resolve_box(target: dict) -> dict | None:
        """C2 target resolution → normalized ``{x, y, w, h}`` (0-1) or None."""
        if not isinstance(target, dict):
            return None
        region = target.get("region")
        if isinstance(region, dict):
            try:
                return {k: float(region[k]) for k in ("x", "y", "w", "h")}
            except (KeyError, TypeError, ValueError):
                return None
        element = target.get("element")
        if element:
            # Geometry from a by-scene_id snapshot (cheap — backend Redis-cached, S66).
            scene_id = session_context.get_current_scene_id()
            snap = await backend_client.get_scene_snapshot(room_id, api_url, scene_id=scene_id)
            elements = ((snap or {}).get("current_scene") or {}).get("elements") or []
            return element_box_from_snapshot(element, elements, element_alias_map)
        describe = target.get("describe")
        if describe:
            # S67b: live capture → bytes → Gemini locate (A-AG-4 + Block 7). The
            # locate box is capture-relative; treated as canvas-relative (the capture
            # surface is expected to be the canvas — A-AG-5 §4).
            slug = session_context.get_slug()
            capture_id, capture_result = await request_capture("locate: " + str(describe))
            img = await fetch_live_bytes(capture_id, capture_result, backend_client, slug, api_url)
            if img is None:
                return None
            return await vision_client.locate(img, str(describe))
        return None

    async def handle_canvas_annotate(params: FunctionCallParams):
        args = params.arguments or {}
        op = args.get("op")
        logger.info("[CANVAS_ANNOTATE] called: op={!r} target={!r}", op, args.get("target"))

        if op not in AGENT_ANNOTATE_OPS:
            await params.result_callback({"ok": False, "message": f"Unknown annotate op '{op}'."})
            return

        if op == "erase":
            await _emit([{"op": "erase"}])
            await params.result_callback({"ok": True, "message": "Erased all annotations."})
            return

        box = await _resolve_box(args.get("target") or {})
        if box is None:
            await params.result_callback({
                "ok": False,
                "message": (
                    "I couldn't locate that on screen — ask the visitor to point at it "
                    "or describe it more specifically."
                ),
            })
            return

        op_obj: dict[str, Any] = {"op": op, "box": box}
        if op == "shape":
            op_obj["shape"] = args.get("shape", "circle")
        if op == "text":
            op_obj["text"] = args.get("text", "")
        if op == "arrow" and args.get("from"):
            # Resolve the arrow's source the same way as the target (element alias → box).
            from_box = await _resolve_box({"element": args["from"]})
            if from_box is not None:
                op_obj["from"] = from_box

        ok = await _emit([op_obj])
        await params.result_callback({"ok": ok, "message": f"Drew {op} on the canvas."})

    return handle_canvas_annotate
