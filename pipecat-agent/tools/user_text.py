"""S83 PR-13 — the typed-question inbound (`user_text`).

The shell's transcript panel gains a typed input (P-11's deferred half,
lifted by Hai); the agent treats a typed question EXACTLY like a spoken
one: the router queues the VAD trio (UserStartedSpeaking →
Transcription → UserStoppedSpeaking), so it rides the identical
aggregator → LLM → TTS turn, interrupts ongoing narration like real
barge-in, and echoes back to the shell through the existing
TranscriptForwarder (no separate echo path, no double bubbles).

The validation lives here (the ``request_quiz_ready`` extraction
precedent) so contract-mirror tests machine-verify it.
"""

USER_TEXT_MAX_CHARS = 500


def clean_user_text(message: object) -> str | None:
    """The wire's ``text`` field, cleaned: stripped, length-capped, or
    None when absent/blank/non-string (the router then ignores the
    message silently — never an error turn)."""
    if not isinstance(message, dict):
        return None
    text = message.get("text")
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    return text[:USER_TEXT_MAX_CHARS]
