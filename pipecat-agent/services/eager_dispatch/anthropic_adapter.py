"""
Anthropic streaming hook for eager canvas dispatch.

Anthropic's Messages API streams tool-use as:
    content_block_start            { content_block: { type: 'tool_use', name, id, input: {} } }
    content_block_delta            { delta: { type: 'input_json_delta', partial_json: '...' } }
    content_block_delta            { delta: { type: 'input_json_delta', partial_json: '...' } }
    ...
    content_block_stop
    message_delta                  { stop_reason: 'tool_use' }
    message_stop

The hook reads partial_json deltas, accumulates per content_block index, and
tries to detect the verb after each chunk.
"""

from __future__ import annotations

import logging

from . import EagerToolCallTracker, maybe_fire_eager

logger = logging.getLogger(__name__)


class AnthropicEagerHook:
    def __init__(self, pending, send_app_message):
        self.tracker = EagerToolCallTracker()
        self.pending = pending
        self.send_app_message = send_app_message

    async def on_stream_event(self, event: dict) -> None:
        """Called for each Anthropic streaming event. event is the parsed JSON
        body (e.g. {"type": "content_block_start", "index": 0, "content_block": {...}}).
        """
        etype = event.get("type")

        if etype == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                idx = event.get("index", 0)
                self.tracker.begin_tool_call(idx, block.get("name", ""))

        elif etype == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "input_json_delta":
                idx = event.get("index", 0)
                partial = delta.get("partial_json", "")
                self.tracker.append_args(idx, partial)
                await maybe_fire_eager(self.tracker, idx, self.pending, self.send_app_message)

        elif etype == "message_stop":
            self.tracker.reset()

    def reset(self):
        self.tracker.reset()
