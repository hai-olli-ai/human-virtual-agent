"""
OpenAI streaming hook for eager canvas dispatch.

OpenAI Chat Completions streams tool calls as:
    chunk.choices[0].delta.tool_calls[
        { index, id, function: { name, arguments } }, ...
    ]

The `arguments` field is a STRING that accumulates over chunks. Same
verb-detection logic as Anthropic, just a different chunk shape.
"""

from __future__ import annotations

import logging

from . import EagerToolCallTracker, maybe_fire_eager

logger = logging.getLogger(__name__)


class OpenAIEagerHook:
    def __init__(self, pending, send_app_message):
        self.tracker = EagerToolCallTracker()
        self.pending = pending
        self.send_app_message = send_app_message

    async def on_chunk(self, chunk) -> None:
        """Called for each OpenAI streaming chunk (ChatCompletionChunk)."""
        if not chunk.choices:
            return
        delta = chunk.choices[0].delta
        if not getattr(delta, "tool_calls", None):
            return

        for tc in delta.tool_calls:
            idx = getattr(tc, "index", 0)
            fn = getattr(tc, "function", None)
            if not fn:
                continue
            name = getattr(fn, "name", None)
            args_chunk = getattr(fn, "arguments", None)

            # First chunk for this tool_call carries the name.
            if name and not self.tracker.get_call(idx):
                self.tracker.begin_tool_call(idx, name)

            if args_chunk:
                self.tracker.append_args(idx, args_chunk)
                await maybe_fire_eager(
                    self.tracker, idx, self.pending, self.send_app_message
                )

    def reset(self):
        self.tracker.reset()
