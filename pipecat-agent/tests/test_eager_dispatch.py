"""Unit tests for partial-JSON verb detection. Provider adapters covered
indirectly — the detection logic is shared and is the only thing that can
have correctness bugs."""

import pytest
from services.eager_dispatch import detect_completed_verb, is_arg_less_verb


class TestDetectCompletedVerb:
    def test_empty(self):
        assert detect_completed_verb("") is None

    def test_partial_no_verb(self):
        assert detect_completed_verb('{"verb"') is None
        assert detect_completed_verb('{"verb":') is None
        assert detect_completed_verb('{"verb":"') is None
        assert detect_completed_verb('{"verb":"next_sce') is None

    def test_verb_value_unclosed(self):
        # Value not yet followed by , or } — not safe to fire.
        assert detect_completed_verb('{"verb":"next_scene"') is None
        assert detect_completed_verb('{"verb":"next_scene"  ') is None

    def test_verb_complete_with_closing_brace(self):
        assert detect_completed_verb('{"verb":"next_scene"}') == "next_scene"

    def test_verb_complete_with_comma(self):
        assert detect_completed_verb('{"verb":"clear",') == "clear"
        assert detect_completed_verb('{"verb":"clear", "args":') == "clear"

    def test_verb_with_whitespace(self):
        assert detect_completed_verb('{ "verb"  :  "pause" }') == "pause"

    def test_verb_after_other_field(self):
        # If args came first (rare but valid), the verb still parses out.
        assert detect_completed_verb('{"args":{},"verb":"play"}') == "play"


class TestIsArgLessVerb:
    @pytest.mark.parametrize("verb", [
        "next_scene", "previous_scene", "clear",
        "next_question", "previous_question", "restart",
        "play", "pause",
    ])
    def test_known_arg_less(self, verb):
        assert is_arg_less_verb(verb) is True

    @pytest.mark.parametrize("verb", [
        "draw_arrow", "add_annotation", "submit_answer",
        "goto_scene", "seek", "set_speed",
    ])
    def test_known_arg_having(self, verb):
        assert is_arg_less_verb(verb) is False

    def test_unknown_verb(self):
        assert is_arg_less_verb("nonexistent") is False
