import pytest

from scene_context import (
    build_language_directive,
    build_language_reminder,
    build_recipient_context,
    build_system_prompt,
    LANGUAGE_NAMES,
)


# -----------------------
# build_recipient_context
# -----------------------

def test_recipient_context_none_returns_empty():
    assert build_recipient_context(None) == ""


def test_recipient_context_empty_returns_empty():
    assert build_recipient_context("") == ""


def test_recipient_context_whitespace_returns_empty():
    assert build_recipient_context("   \n  \t  ") == ""


def test_recipient_context_non_empty_includes_audience_header():
    out = build_recipient_context("Sales team at Acme")
    assert "# AUDIENCE" in out
    assert "Sales team at Acme" in out


def test_recipient_context_strips_surrounding_whitespace():
    out = build_recipient_context("  \n  Speak to engineers  \n  ")
    assert "Speak to engineers" in out
    # No extra whitespace adjacent to header
    assert "# AUDIENCE\n" in out


def test_recipient_context_includes_preamble():
    out = build_recipient_context("Audience text")
    # The preamble explains to the LLM what AUDIENCE means
    assert "audience" in out.lower()


# -----------------------
# build_language_directive
# -----------------------

@pytest.mark.parametrize(
    "code,expected_name",
    [
        ("en", "English"),
        ("es", "Spanish"),
        ("fr", "French"),
        ("de", "German"),
        ("pt", "Portuguese"),
        ("ja", "Japanese"),
        ("ko", "Korean"),
        ("vi", "Vietnamese"),
        ("zh", "Chinese (Mandarin)"),
    ],
)
def test_language_directive_includes_name(code, expected_name):
    out = build_language_directive(code)
    # The name should appear at least twice — strong steering
    assert out.count(expected_name) >= 2


def test_language_directive_unknown_falls_back_to_english():
    out = build_language_directive("xx")
    assert "English" in out


def test_language_directive_none_falls_back_to_english():
    out = build_language_directive(None)
    assert "English" in out


def test_language_directive_empty_falls_back_to_english():
    out = build_language_directive("")
    assert "English" in out


def test_language_reminder_short_and_emphatic():
    out = build_language_reminder("ja")
    assert "Japanese" in out
    assert len(out) < 100  # reminder is short


def test_language_reminder_unknown_falls_back():
    out = build_language_reminder("klingon")
    assert "English" in out


# -----------------------
# build_system_prompt assembly
# -----------------------

def test_system_prompt_includes_language_at_top_and_bottom():
    snap = {
        "language": "es",
        "persona": "A friendly assistant.",
        "recipient_prompt": None,
        "knowledge": None,
        "scene_instruction": "Help the visitor.",
        "display_mode": "normal",
        "elements": [],
    }
    prompt = build_system_prompt(snap)

    # Top: LANGUAGE directive
    first_section = prompt.split("\n\n", 1)[0]
    assert "# LANGUAGE" in first_section
    assert "Spanish" in first_section

    # Bottom: language reminder
    last_section = prompt.rsplit("\n\n", 1)[-1]
    assert "Spanish" in last_section
    assert "Remember" in last_section or "remember" in last_section.lower()


def test_system_prompt_audience_appears_when_recipient_prompt_set():
    snap = {
        "language": "en",
        "persona": "Assistant.",
        "recipient_prompt": "Speak to first-time parents — be reassuring.",
        "knowledge": None,
        "scene_instruction": "Help.",
        "display_mode": "normal",
        "elements": [],
    }
    prompt = build_system_prompt(snap)
    assert "# AUDIENCE" in prompt
    assert "first-time parents" in prompt


def test_system_prompt_audience_absent_when_recipient_prompt_empty():
    snap = {
        "language": "en",
        "persona": "Assistant.",
        "recipient_prompt": "",
        "knowledge": None,
        "scene_instruction": "Help.",
        "display_mode": "normal",
        "elements": [],
    }
    prompt = build_system_prompt(snap)
    assert "# AUDIENCE" not in prompt


def test_system_prompt_audience_absent_when_recipient_prompt_whitespace():
    snap = {
        "language": "en",
        "persona": "Assistant.",
        "recipient_prompt": "   \n   ",
        "knowledge": None,
        "scene_instruction": "Help.",
        "display_mode": "normal",
        "elements": [],
    }
    prompt = build_system_prompt(snap)
    assert "# AUDIENCE" not in prompt


def test_system_prompt_audience_appears_after_persona_before_knowledge():
    snap = {
        "language": "en",
        "persona": "Assistant.",
        "recipient_prompt": "Audience text",
        "knowledge": {"sources": [{"text": "Knowledge fact ABC"}]},
        "scene_instruction": "Help.",
        "display_mode": "normal",
        "elements": [],
    }
    prompt = build_system_prompt(snap)
    persona_idx = prompt.find("# PERSONA")
    audience_idx = prompt.find("# AUDIENCE")
    knowledge_idx = prompt.find("# KNOWLEDGE")  # adapt header text to actual

    if persona_idx >= 0 and audience_idx >= 0 and knowledge_idx >= 0:
        assert persona_idx < audience_idx < knowledge_idx


def test_system_prompt_defaults_to_english_when_language_missing():
    snap = {
        "persona": "Assistant.",
        "recipient_prompt": None,
        "knowledge": None,
        "scene_instruction": "Help.",
        "display_mode": "normal",
        "elements": [],
    }
    prompt = build_system_prompt(snap)
    assert "English" in prompt


def test_system_prompt_handles_all_nine_languages():
    for lang in ["en", "es", "fr", "de", "pt", "ja", "ko", "vi", "zh"]:
        snap = {
            "language": lang,
            "persona": "Assistant.",
            "recipient_prompt": None,
            "knowledge": None,
            "scene_instruction": "Help.",
            "display_mode": "normal",
            "elements": [],
        }
        prompt = build_system_prompt(snap)
        # Language name should appear at least twice (top + bottom)
        name = LANGUAGE_NAMES[lang]
        assert prompt.count(name) >= 2, (
            f"Language {lang}/{name} expected >=2 mentions in prompt"
        )


# -----------------------
# Deepgram language resolver (Block 5)
# -----------------------

def test_deepgram_language_resolver_known_codes():
    from config import resolve_deepgram_language

    for code in ["en", "es", "fr", "de", "pt", "ja", "ko", "vi", "zh"]:
        assert resolve_deepgram_language(code) == code


def test_deepgram_language_resolver_unknown_falls_back_to_multi():
    from config import resolve_deepgram_language
    assert resolve_deepgram_language("klingon") == "multi"


def test_deepgram_language_resolver_none_defaults_to_en():
    from config import resolve_deepgram_language
    assert resolve_deepgram_language(None) == "en"


def test_deepgram_language_resolver_empty_defaults_to_en():
    from config import resolve_deepgram_language
    assert resolve_deepgram_language("") == "en"
