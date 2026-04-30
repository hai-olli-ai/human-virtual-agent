"""Tests for the S63 Block 7 LINK NARRATION directive.

The directive lives in scene_context.build_link_narration_directive and
slots between KNOWLEDGE and SCENE INSTRUCTION in the system-prompt
sandwich. It tells the LLM HOW to use linked content already injected
into the KNOWLEDGE section — not whether to use it.
"""
from scene_context import build_link_narration_directive


def test_no_link_returns_empty():
    assert build_link_narration_directive(None) == ""
    assert build_link_narration_directive({}) == ""


def test_walk_through_includes_keyword():
    out = build_link_narration_directive({
        "url": "https://youtu.be/X",
        "source": "youtube",
        "narration_mode": "walk_through",
    })
    assert "step-by-step" in out
    assert "youtube" in out
    assert "https://youtu.be/X" in out


def test_summarize_directive():
    out = build_link_narration_directive({
        "url": "https://en.wikipedia.org/wiki/X",
        "source": "wikipedia",
        "narration_mode": "summarize",
    })
    assert "Summarize" in out


def test_answer_questions_directive():
    out = build_link_narration_directive({
        "url": "https://www.canva.com/design/X",
        "source": "canva",
        "narration_mode": "answer_questions",
    })
    assert "reactively" in out


def test_reference_as_needed_directive():
    out = build_link_narration_directive({
        "url": "https://www.zillow.com/x",
        "source": "zillow",
        "narration_mode": "reference_as_needed",
    })
    assert "background" in out


def test_unknown_mode_returns_empty():
    out = build_link_narration_directive({
        "url": "https://x.com/y",
        "source": "youtube",
        "narration_mode": "freestyle_jazz",
    })
    assert out == ""
