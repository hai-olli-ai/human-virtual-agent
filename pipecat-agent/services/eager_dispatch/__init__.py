"""
Eager streaming dispatch — fires arg-less canvas verb commands the moment the
LLM finishes streaming the verb token, instead of waiting for stop_reason.

Saves 100-250ms per arg-less call (no_args overhead is purely the time
between "verb closed" and "tool_use stop_reason"). Especially valuable on
common navigation (next_scene, previous_scene), clears, and quiz transitions.

How it works:
  - Each provider adapter consumes the LLM's streaming events.
  - For each in-flight tool call, it accumulates the partial JSON args.
  - After every chunk, it tries to extract `verb` from the accumulated string.
  - If verb is set AND tool is canvas_control or canvas_action AND verb is in
    EAGER_DISPATCH_VERBS, the adapter fires the canvas command via the
    PendingCommandRegistry's eager path:
       1. Open the future with eager_dispatched=True.
       2. Send the Daily app-message immediately.
  - When stop_reason fires and Pipecat invokes the tool handler, the handler
    sees the future is already in flight and just awaits it (does NOT send
    the Daily message twice).

Verb detection uses regex on the accumulated args string, looking for:
    "verb"\\s*:\\s*"X"\\s*(,|})    -> verb=X complete

Once the closing punctuation arrives we know the verb value is final and the
LLM won't change it on the next chunk.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Optional

from tools.canvas_protocol_tools import (
    EAGER_DISPATCH_VERBS,
    PendingCommandRegistry,
    build_canvas_command,
)

logger = logging.getLogger(__name__)


# Match "verb"<ws>:<ws>"X"<ws>(,|}) — i.e., verb value followed by closing
# punctuation that proves the value is final.
_VERB_PATTERN = re.compile(r'"verb"\s*:\s*"([^"\\]+)"\s*(?:,|})')


def detect_completed_verb(accumulated_args: str) -> Optional[str]:
    """Return the verb if the accumulated partial JSON contains a completed
    `"verb": "X"` followed by , or }. Returns None otherwise.
    """
    if not accumulated_args:
        return None
    m = _VERB_PATTERN.search(accumulated_args)
    if not m:
        return None
    return m.group(1)


def is_arg_less_verb(verb: str) -> bool:
    return verb in EAGER_DISPATCH_VERBS


class EagerToolCallTracker:
    """Per-LLM-stream state for tracking in-flight tool calls and their
    accumulated partial JSON. Reset between LLM completions."""

    def __init__(self):
        # tool_call_index -> {tool: str, args: str, eager_fired: bool, command_id: str}
        self._calls: dict[int, dict] = {}

    def begin_tool_call(self, idx: int, tool_name: str) -> None:
        self._calls[idx] = {
            "tool": tool_name,
            "args": "",
            "eager_fired": False,
            "command_id": str(uuid.uuid4()),
        }

    def append_args(self, idx: int, partial: str) -> None:
        if idx in self._calls:
            self._calls[idx]["args"] += partial

    def get_call(self, idx: int) -> Optional[dict]:
        return self._calls.get(idx)

    def mark_fired(self, idx: int) -> None:
        if idx in self._calls:
            self._calls[idx]["eager_fired"] = True

    def reset(self):
        self._calls.clear()


async def maybe_fire_eager(
    tracker: EagerToolCallTracker,
    idx: int,
    pending: PendingCommandRegistry,
    send_app_message,
) -> None:
    """If the tool call at index `idx` qualifies for eager dispatch and hasn't
    fired yet, send the Daily app-message and register the eager future."""
    call = tracker.get_call(idx)
    if not call or call["eager_fired"]:
        return

    tool_full_name = call["tool"]  # e.g. "canvas_control"
    if tool_full_name not in ("canvas_control", "canvas_action"):
        return

    verb = detect_completed_verb(call["args"])
    if not verb or not is_arg_less_verb(verb):
        return

    # Fire it. Wire-format tool name is the part after "canvas_" — the
    # LLM-facing function uses underscore (OpenAI/Anthropic regex
    # `^[a-zA-Z0-9_-]+$` rejects dots), but the Daily message keeps the
    # short form for the frontend Canvas Service.
    tool_short = tool_full_name.split("_", 1)[1]  # "control" or "action"
    cmd = build_canvas_command(
        tool_short,
        {"verb": verb},
        command_id=call["command_id"],
    )

    pending.open(cmd["commandId"], eager=True)
    try:
        await send_app_message(cmd)
        tracker.mark_fired(idx)
        logger.info(
            "eager_dispatch: fired %s verb=%s commandId=%s",
            tool_full_name,
            verb,
            cmd["commandId"],
        )
    except Exception as exc:
        logger.exception(
            "eager_dispatch: send_app_message failed for %s verb=%s: %r",
            tool_full_name,
            verb,
            exc,
        )
        # Don't mark fired — let the regular handler retry.
