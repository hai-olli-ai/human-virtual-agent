"""Tests for ``flow_knowledge_cache.py`` (S66 Block 5b).

Covers:

  * :class:`FlowKnowledgeCache` — miss-then-hit, change detection,
    invalidation, no-flow-knowledge edge case.
  * :func:`scene_context.build_flow_knowledge_section` /
    :func:`scene_context.build_scene_knowledge_section` — split helpers
    behave identically to the legacy concatenated output.
  * Integration: ``persona._build_knowledge_block`` reuses the cache
    across calls when supplied; rebuilds when the cache is None.

Follows the existing tests/ convention: no pytest-asyncio (not in the
dependency closure).
"""

from __future__ import annotations

from flow_knowledge_cache import FlowKnowledgeCache, _hash_flow
from scene_context import (
    build_flow_knowledge_section,
    build_knowledge_context,
    build_scene_knowledge_section,
)


# ──────────────────────────────────────────────────────────────────────
# Split helpers
# ──────────────────────────────────────────────────────────────────────


def _kn(flow_text: str = "", scene_text: str = ""):
    """Build a knowledge dict with optional FLOW + SCENE documents."""
    out: dict = {}
    if flow_text:
        out["flow"] = {
            "sources": [{"file_name": "flow.md", "extracted_text": flow_text}],
        }
    if scene_text:
        out["scene"] = {
            "sources": [{"file_name": "scene.md", "extracted_text": scene_text}],
        }
    return out


def test_split_matches_combined_output():
    knowledge = _kn("FLOW DOC", "SCENE DOC")
    flow = build_flow_knowledge_section(knowledge)
    scene = build_scene_knowledge_section(knowledge)
    combined = build_knowledge_context(knowledge)
    assert flow
    assert scene
    assert combined == "\n\n".join([flow, scene])


def test_split_empty_when_missing_scopes():
    assert build_flow_knowledge_section(None) == ""
    assert build_scene_knowledge_section(None) == ""
    assert build_flow_knowledge_section({}) == ""
    assert build_scene_knowledge_section({}) == ""
    only_scene = _kn(scene_text="SCENE")
    assert build_flow_knowledge_section(only_scene) == ""
    assert build_scene_knowledge_section(only_scene)


# ──────────────────────────────────────────────────────────────────────
# FlowKnowledgeCache
# ──────────────────────────────────────────────────────────────────────


def test_cache_starts_empty():
    c = FlowKnowledgeCache()
    assert c.hash is None
    assert c.hits == 0
    assert c.misses == 0


def test_cache_miss_then_hit_returns_same_section():
    c = FlowKnowledgeCache()
    knowledge = _kn(flow_text="The capital of France is Paris.")
    s1 = c.get_or_build(knowledge)
    assert s1
    assert c.misses == 1
    assert c.hits == 0

    s2 = c.get_or_build(knowledge)
    assert s2 == s1
    assert c.misses == 1
    assert c.hits == 1


def test_cache_misses_on_content_change():
    c = FlowKnowledgeCache()
    s1 = c.get_or_build(_kn(flow_text="A"))
    s2 = c.get_or_build(_kn(flow_text="B"))
    assert s1 != s2
    assert c.misses == 2
    assert c.hits == 0


def test_cache_handles_no_flow():
    c = FlowKnowledgeCache()
    assert c.get_or_build(None) == ""
    assert c.get_or_build({}) == ""
    assert c.get_or_build(_kn(scene_text="SCENE")) == ""
    assert c.misses == 0
    assert c.hits == 0


def test_cache_invalidate_forces_rebuild():
    c = FlowKnowledgeCache()
    knowledge = _kn(flow_text="X")
    c.get_or_build(knowledge)
    c.invalidate()
    assert c.hash is None
    c.get_or_build(knowledge)
    assert c.misses == 2  # invalidate forced a second build


def test_hash_deterministic_across_dict_orderings():
    """Snapshot dict ordering must not affect the cache key — sort_keys=True
    canonicalises the JSON so two equivalent dicts hash equal."""
    a = {"flow": {"sources": [{"file_name": "a.md", "extracted_text": "x"}]}}
    b = {"flow": {"sources": [{"extracted_text": "x", "file_name": "a.md"}]}}
    assert _hash_flow(a["flow"]) == _hash_flow(b["flow"])


# ──────────────────────────────────────────────────────────────────────
# Integration: persona._build_knowledge_block uses the cache
# ──────────────────────────────────────────────────────────────────────


def test_build_knowledge_block_uses_cache_on_repeated_calls():
    """Two consecutive _build_knowledge_block calls on the same
    knowledge content should hit the cache the second time."""
    from persona import _build_knowledge_block

    cache = FlowKnowledgeCache()
    snap = {"knowledge": _kn(flow_text="FLOW", scene_text="SCENE-1")}
    out1 = _build_knowledge_block(snap, flow_cache=cache)
    assert "FLOW" in out1
    assert "SCENE-1" in out1
    assert cache.misses == 1

    # New scene-scope content, same flow content → cache hit on flow,
    # scene re-stitched.
    snap2 = {"knowledge": _kn(flow_text="FLOW", scene_text="SCENE-2")}
    out2 = _build_knowledge_block(snap2, flow_cache=cache)
    assert "FLOW" in out2
    assert "SCENE-2" in out2
    assert "SCENE-1" not in out2
    assert cache.misses == 1
    assert cache.hits == 1


def test_build_knowledge_block_no_cache_still_works():
    """flow_cache=None reproduces the pre-Block-5b path."""
    from persona import _build_knowledge_block

    snap = {"knowledge": _kn(flow_text="FLOW", scene_text="SCENE")}
    out = _build_knowledge_block(snap, flow_cache=None)
    assert "FLOW" in out
    assert "SCENE" in out


def test_build_knowledge_block_returns_empty_with_no_knowledge():
    from persona import _build_knowledge_block

    assert _build_knowledge_block({}, flow_cache=FlowKnowledgeCache()) == ""
    assert _build_knowledge_block({"knowledge": None}, flow_cache=None) == ""
