"""WordBoundaryAggregator (#1) — caption fix for Vietnamese sub-token deltas.

The conversational LLM streams sub-token deltas; for Vietnamese a single
syllable arrives in pieces (gpt-oss-120b: "Chào" → ["Ch","ào"], "phân" →
["ph","ân"]). The shell's caption renders transcript messages separated, so
forwarding raw deltas rendered as "ph ân đo ạn" — every multi-token word split.
The aggregator releases only whole words so the caption reads naturally. The
gpt-oss output itself is clean (verified) — this is purely a caption-side fix,
so prompting/model-swap don't address it; this does.
"""

from context.transcript_aggregator import WordBoundaryAggregator


def _run(deltas):
    agg = WordBoundaryAggregator()
    out = []
    for d in deltas:
        r = agg.feed(d)
        if r:
            out.append(r)
    tail = agg.flush()
    if tail:
        out.append(tail)
    return out


def test_vietnamese_subtoken_deltas_become_whole_words():
    # Real gpt-oss-120b streaming deltas for Vietnamese (observed via probe).
    deltas = ["Ch", "ào", " bạn", "!", " Trong", " b", "ức", " tranh"]
    out = _run(deltas)
    assert out == ["Chào", "bạn!", "Trong", "bức", "tranh"]
    # The shell joins messages with a single space → clean caption (the bug
    # was raw per-delta forwarding rendering as "Ch ào b ức").
    assert " ".join(out) == "Chào bạn! Trong bức tranh"
    # No sub-syllable fragment is ever emitted on its own.
    for frag in ("Ch", "ào", "b", "ức"):
        assert frag not in out


def test_english_word_deltas_round_trip():
    out = _run(["Hello", " there", ",", " friend", "."])
    assert out == ["Hello", "there,", "friend."]


def test_flush_releases_trailing_partial_word():
    agg = WordBoundaryAggregator()
    assert agg.feed("Xin") == ""        # no whitespace yet — buffered
    assert agg.feed(" chào") == "Xin"   # "Xin" complete, "chào" still buffered
    assert agg.flush() == "chào"        # released at the turn boundary
    assert agg.flush() == ""            # idempotent / empty afterwards


def test_no_whitespace_run_buffers_until_flush():
    agg = WordBoundaryAggregator()
    for piece in ["mộtt", "ừ", "dài"]:
        assert agg.feed(piece) == ""
    assert agg.flush() == "mộttừdài"


def test_multiword_delta_preserves_internal_single_spaces():
    out = _run(["Chào bạn thân ", "mến"])
    assert out == ["Chào bạn thân", "mến"]


def test_empty_delta_is_ignored():
    agg = WordBoundaryAggregator()
    assert agg.feed("") == ""
    assert agg.feed("hi") == ""
    assert agg.flush() == "hi"
