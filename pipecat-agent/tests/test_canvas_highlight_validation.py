"""Contract tests for `_validate_highlight_target`.

The validator is the agent-side gate that catches malformed
`canvas_highlight` tool calls before they're dispatched over Daily. It
exists because Pipecat 0.0.108 does not enable OpenAI's strict mode, so
the LLM is free to emit calls that violate `required: ["target"]` — and
we observed gpt-5.4 doing so in production (2026-05-13 INVALID_TOOL
warning, target=null on a YouTube scene).

Manifest-aware "page rejects this shape" validation is intentionally NOT
in scope here — that stays in the frontend's
lib/canvas-protocol/validate.ts so the manifest remains the single source
of truth for what each Page accepts.
"""

from __future__ import annotations

import pytest

from tools.canvas_protocol_tools import CanvasCommandError, _validate_highlight_target


class TestValidateHighlightTarget:
    def test_valid_element_id(self):
        assert _validate_highlight_target({"element_id": "text_1"}) is None

    def test_valid_element_id_uuid(self):
        assert _validate_highlight_target(
            {"element_id": "019e0e07-0215-7af4-9cc4-3f130fc91d9b"}
        ) is None

    def test_valid_box_ints(self):
        assert _validate_highlight_target({"box": [10, 20, 100, 50]}) is None

    def test_valid_box_floats(self):
        assert _validate_highlight_target({"box": [10.5, 20.25, 100.0, 50.0]}) is None

    def test_valid_box_tuple(self):
        # Pipecat may deserialize JSON arrays to lists, but accept tuples too —
        # the dispatch later JSON-encodes either, and the frontend doesn't care.
        assert _validate_highlight_target({"box": (10, 20, 100, 50)}) is None

    @pytest.mark.parametrize("bad_target", [
        None,
        "text_1",                    # bare string, LLM forgot to wrap in object
        ["text_1"],                  # list
        123,
    ])
    def test_non_dict_target_rejected(self, bad_target):
        err = _validate_highlight_target(bad_target)
        assert isinstance(err, CanvasCommandError)
        assert err.code == "INVALID_ARGS"
        # Error message must point the LLM at both legal shapes so it can
        # self-correct on the next turn.
        assert "element_id" in err.message
        assert "box" in err.message

    def test_empty_dict_rejected(self):
        err = _validate_highlight_target({})
        assert isinstance(err, CanvasCommandError)
        assert err.code == "INVALID_ARGS"
        assert "element_id" in err.message and "box" in err.message

    def test_empty_element_id_rejected(self):
        # Empty string isn't a usable element id — would never resolve to a
        # real UUID via the alias map.
        err = _validate_highlight_target({"element_id": ""})
        assert isinstance(err, CanvasCommandError)
        assert err.code == "INVALID_ARGS"

    def test_non_string_element_id_rejected(self):
        err = _validate_highlight_target({"element_id": 123})
        assert isinstance(err, CanvasCommandError)
        assert err.code == "INVALID_ARGS"

    def test_short_box_rejected(self):
        err = _validate_highlight_target({"box": [10, 20, 100]})
        assert isinstance(err, CanvasCommandError)

    def test_long_box_rejected(self):
        err = _validate_highlight_target({"box": [10, 20, 100, 50, 60]})
        assert isinstance(err, CanvasCommandError)

    def test_non_numeric_box_rejected(self):
        err = _validate_highlight_target({"box": [10, "twenty", 100, 50]})
        assert isinstance(err, CanvasCommandError)

    def test_boolean_box_rejected(self):
        # Python's bool is a subclass of int — explicitly reject so a
        # JSON-encoded `[true, false, 100, 50]` doesn't slip through.
        err = _validate_highlight_target({"box": [True, False, 100, 50]})
        assert isinstance(err, CanvasCommandError)

    def test_both_shapes_accepted(self):
        # If the LLM hedges by providing both, we accept — the frontend's
        # parseHighlightTarget picks element_id first, and the manifest
        # validator will reject if the page doesn't accept that shape.
        assert _validate_highlight_target(
            {"element_id": "text_1", "box": [10, 20, 100, 50]}
        ) is None

    def test_extra_keys_in_element_id_branch_accepted(self):
        # additionalProperties: false in the FunctionSchema is a hint to the
        # model, not enforced here — we shouldn't be stricter than the
        # frontend, which ignores unknown keys.
        assert _validate_highlight_target(
            {"element_id": "text_1", "color": "red"}
        ) is None
