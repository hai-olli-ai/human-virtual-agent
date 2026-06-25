"""VOICE OUTPUT STYLE directive (#1) — keeps gpt-oss output plain-spoken.

The conversational reply is read aloud by the TTS engine and shown as a live
caption, so Markdown (`**`), stray symbols, and trailing ellipses ("…"/"...")
must not appear. gpt-oss-120b is markdown-happy and will otherwise leak them
(observed in the wild as `**… **… Oops! Looks …` captions). This directive is
the SOURCE-level fix: it also cleans the caption, which is forwarded UPSTREAM
of the TTS-side MarkdownTextFilter (the audio-only safety net, #2).

These tests pin that the directive exists and forbids the specific artifacts.
"""

from context.prompt_builder import render_voice_output_style_section


def test_directive_present_and_titled():
    s = render_voice_output_style_section()
    assert "VOICE OUTPUT" in s
    assert s.strip()  # non-empty


def test_forbids_the_observed_artifacts():
    s = render_voice_output_style_section()
    low = s.lower()
    # Markdown / asterisks (the `**` leak)
    assert "markdown" in low
    assert "asterisk" in low or "**" in s
    # ellipses / trailing-off (the `…` filler)
    assert "ellips" in low
    assert "…" in s or "..." in s
    # emojis explicitly disallowed
    assert "emoji" in low
