"""In-session cache for the FLOW-scope knowledge prompt block (S66 Block 5b).

The agent fetches scene snapshots with ``?include_all_scene_knowledge=true``
so the snapshot's ``knowledge.flow`` payload aggregates every sibling
scene's knowledge. Within a single live-room session that aggregated
block is stable across scene navigations (it only changes when a
creator edits the underlying knowledge in another tab) — but pre-5b,
the agent re-formatted it on every scene-change prompt rebuild.

This cache memoises the rendered FLOW section keyed on a sha256 of the
canonical JSON form of ``knowledge["flow"]``. Cache hits skip the
``_format_scope`` iteration over sources/urls/faqs; misses rebuild +
remember. The SCENE-scope block is intentionally NOT cached — it
varies per navigation and is the part the LLM uses for scene-specific
context.

Used by ``persona.build_system_prompt(flow_cache=…)`` and constructed
once per session in ``bot.py``'s ``run_bot_classic`` / ``run_bot_relay``.
Re-running ``build_system_prompt`` after a scene change reuses the same
cache instance, so the FLOW section is built once and reused for the
remainder of the session (unless ``knowledge.flow`` content changes).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from loguru import logger

from scene_context import build_flow_knowledge_section


class FlowKnowledgeCache:
    """Per-session cache of the rendered FLOW-scope knowledge block."""

    def __init__(self) -> None:
        self._hash: Optional[str] = None
        self._section: str = ""
        self.hits: int = 0
        self.misses: int = 0

    @property
    def hash(self) -> Optional[str]:
        return self._hash

    def get_or_build(self, knowledge: Optional[dict[str, Any]]) -> str:
        """Return the FLOW-scope section, reusing the cached render on hit.

        Returns "" when there's no flow knowledge to render — callers
        should treat that identically to a cache miss with no content.
        """
        if not knowledge:
            return ""
        flow = knowledge.get("flow")
        h = _hash_flow(flow)
        if h is None:
            return ""
        if h == self._hash:
            self.hits += 1
            return self._section
        section = build_flow_knowledge_section(knowledge)
        self._hash = h
        self._section = section
        self.misses += 1
        logger.info(
            "[FLOW_KNOWLEDGE_CACHE] miss — rebuilt section ({} chars, hash={})",
            len(section),
            h[:8],
        )
        return section

    def invalidate(self) -> None:
        """Drop the cached section. Forces the next get_or_build to rebuild."""
        self._hash = None
        self._section = ""


def _hash_flow(flow: Optional[dict[str, Any]]) -> Optional[str]:
    """Stable sha256 of the canonical JSON form of the flow scope.

    sort_keys=True belt-and-suspenders against snapshot dict order
    changes from future backend versions; Python 3.7+ dicts preserve
    insertion order so the canonical form is already deterministic on
    the current backend.
    """
    if not flow:
        return None
    canonical = json.dumps(flow, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
