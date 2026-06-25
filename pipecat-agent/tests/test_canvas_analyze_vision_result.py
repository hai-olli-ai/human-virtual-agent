"""canvas_analyze folds the vision answer into its TOOL RESULT (S67b fix).

Bug: the vision answer was injected out-of-band as a separate ``developer``
message (``context.add_message``), which raced the function-call re-run — the
spoken reply was sometimes generated from only the iframe page-state result
("video paused at 36:25") and said "I can't see the circle" even though vision
succeeded. Fix: ``ensure_vision`` RETURNS the vision text and ``handle_analyze``
folds it into the canvas_analyze tool result via ``_merge_analyze_result``, so
it's in-band and guaranteed-present in the re-run.

These tests pin the merge logic (the pure, importable core of the fix).
"""

from tools.canvas_protocol_tools import _merge_analyze_result

VISION = "[vision: point] The circled area highlights cooked yellow peas."
PAGE = {"answer": "Video '8D China City' is paused at 36:25 of 1:39:08."}


def test_vision_leads_when_both_present():
    # Visual question: vision is the authoritative answer; page state rides along.
    assert _merge_analyze_result(VISION, PAGE) == {"answer": VISION, "page_state": PAGE}


def test_vision_only_when_page_state_missing():
    # Page-state dispatch failed but vision succeeded → still return the vision answer.
    assert _merge_analyze_result(VISION, None) == {"answer": VISION}


def test_page_state_passthrough_when_no_vision():
    # No vision answer → legacy behavior: return the page state unchanged.
    assert _merge_analyze_result(None, PAGE) == PAGE


def test_none_when_neither():
    assert _merge_analyze_result(None, None) is None


def test_vision_answer_is_the_top_level_answer_key():
    # The LLM reads `answer` as the response to its canvas_analyze call — the
    # circle description must live there, not buried under page_state (the bug
    # was the LLM seeing only the page state and saying "I can't see").
    merged = _merge_analyze_result(VISION, PAGE)
    assert merged["answer"] == VISION and "circled area" in merged["answer"]
    assert merged["page_state"] == PAGE  # contradictory page state demoted, not primary
