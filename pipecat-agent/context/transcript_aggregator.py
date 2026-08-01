"""Word-boundary aggregator for the avatar caption transcript.

The conversational LLM streams sub-token text deltas — for Vietnamese a single
syllable arrives in several pieces (gpt-oss: "Chào" → ["Ch", "ào"], "phân" →
["ph", "ân"]). ``TranscriptForwarder`` forwards each avatar delta to the shell
as its own ``{type:'transcript'}`` message, and the shell's caption renders the
messages separated (a separator between every message). Forwarding raw deltas
therefore rendered as "ph ân đo ạn" — every multi-token word split apart.
Single-token words ("chuyển") were unaffected, which is exactly why the
artifact was Vietnamese-specific (English deltas are ~word-sized).

This aggregator buffers deltas and releases only WHOLE words (text terminated by
whitespace), trimmed, so the shell's per-message separator falls between words
and the caption reads naturally regardless of how finely the model tokenizes.

It is deliberately word-level (not sentence-level) to keep the live, incremental
caption feel. It does NOT touch the pipeline frame stream — TTS/audio still sees
the raw deltas (the TTS has its own sentence aggregator), so synthesis is
unchanged; this only reshapes the transcript side-channel.
"""

from __future__ import annotations


class WordBoundaryAggregator:
    """Release streamed text deltas only at whitespace (word) boundaries."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str) -> str:
        """Append a delta; return any newly-completed words (or ``""``).

        Everything up to the last whitespace in the buffer is "complete" and
        returned (trimmed; internal spacing preserved). The trailing partial
        word stays buffered for the next ``feed`` / ``flush``.
        """
        if not text:
            return ""
        self._buf += text
        last_ws = max(
            self._buf.rfind(" "),
            self._buf.rfind("\n"),
            self._buf.rfind("\t"),
        )
        if last_ws < 0:
            return ""
        ready = self._buf[:last_ws].strip()
        self._buf = self._buf[last_ws + 1 :]
        return ready

    def flush(self) -> str:
        """Return the buffered remainder (final partial word) and clear."""
        ready = self._buf.strip()
        self._buf = ""
        return ready
