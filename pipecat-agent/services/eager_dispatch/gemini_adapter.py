"""
Gemini streaming hook for eager canvas dispatch.

Gemini generateContentStream emits chunks as:
    chunk.candidates[*].content.parts[*].function_call

function_call.args may arrive whole (in one chunk) or partial (multiple chunks).
When whole, the chunk carries a fully-populated args dict — we detect the verb
from the dict directly, no JSON parsing needed.

Gemini gets the eager benefit on ~60% of calls (whole-args chunks); the other
40% (partial-args chunks) fall back to the normal stop_reason path. Acceptable
for v0.1.
"""

from __future__ import annotations

import json
import logging

from . import EagerToolCallTracker, maybe_fire_eager, is_arg_less_verb
from tools.canvas_protocol_tools import build_canvas_command, EAGER_DISPATCH_VERBS

logger = logging.getLogger(__name__)


class GeminiEagerHook:
    def __init__(self, pending, send_app_message):
        self.tracker = EagerToolCallTracker()
        self.pending = pending
        self.send_app_message = send_app_message

    async def on_chunk(self, chunk) -> None:
        """Called for each Gemini streaming chunk."""
        candidates = getattr(chunk, "candidates", None) or []
        for cand_idx, cand in enumerate(candidates):
            content = getattr(cand, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None) or []
            for part_idx, part in enumerate(parts):
                fc = getattr(part, "function_call", None)
                if not fc:
                    continue
                idx = (cand_idx << 8) | part_idx  # synthetic index per (candidate, part)
                name = getattr(fc, "name", None) or ""
                args = getattr(fc, "args", None)

                if not self.tracker.get_call(idx):
                    self.tracker.begin_tool_call(idx, name)

                # Gemini args is usually a dict (proto Struct) — try to grab verb directly.
                if args is None:
                    continue

                if isinstance(args, dict):
                    verb = args.get("verb")
                    if verb and name in ("canvas_control", "canvas_action") and is_arg_less_verb(verb):
                        # Fire eagerly using the dict directly.
                        call = self.tracker.get_call(idx)
                        if call and not call["eager_fired"]:
                            tool_short = name.split("_", 1)[1]
                            cmd = build_canvas_command(
                                tool_short, {"verb": verb}, command_id=call["command_id"],
                            )
                            self.pending.open(cmd["commandId"], eager=True)
                            try:
                                await self.send_app_message(cmd)
                                self.tracker.mark_fired(idx)
                                logger.info(
                                    "eager_dispatch[gemini]: fired %s verb=%s commandId=%s",
                                    name, verb, cmd["commandId"],
                                )
                            except Exception:
                                logger.exception("eager_dispatch[gemini]: send failed")
                else:
                    # Some Gemini SDK versions deliver partial JSON-as-string.
                    # Fall through to the shared partial-JSON path.
                    self.tracker.append_args(idx, json.dumps(args) if args else "")
                    await maybe_fire_eager(self.tracker, idx, self.pending, self.send_app_message)

    def reset(self):
        self.tracker.reset()
