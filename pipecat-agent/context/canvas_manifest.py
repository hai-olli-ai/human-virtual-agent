"""
Canvas Manifest registry — in-memory holder for the active Page's manifest.

Fed by Daily app-messages from the frontend's Canvas Service:
  - canvas.register      -> set_manifest()
  - canvas.stateChange   -> update_state()

Read by tools/canvas_protocol_tools.py (verb validation) and
context/prompt_builder.py (CANVAS PAGE section assembly).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CanvasManifestRegistry:
    _manifest: Optional[dict] = None
    _semantic_state: Optional[dict] = None

    def set_manifest(self, manifest: dict) -> None:
        """Called when canvas.register arrives from the frontend."""
        if not isinstance(manifest, dict):
            logger.warning("canvas_manifest: ignoring non-dict manifest %r", manifest)
            return
        self._manifest = manifest
        self._semantic_state = manifest.get("semanticState")
        logger.info(
            "canvas_manifest: registered pageType=%s version=%s capabilities=%s",
            manifest.get("pageType"), manifest.get("version"),
            list((manifest.get("capabilities") or {}).keys()),
        )

    def update_state(self, semantic_state: dict) -> None:
        """Called when canvas.stateChange arrives. Replaces the cached state."""
        if not isinstance(semantic_state, dict):
            return
        self._semantic_state = semantic_state

    def clear(self) -> None:
        self._manifest = None
        self._semantic_state = None

    def current(self) -> Optional[dict]:
        return self._manifest

    def state(self) -> Optional[dict]:
        return self._semantic_state

    def page_type(self) -> Optional[str]:
        return (self._manifest or {}).get("pageType")

    def supported_verbs(self, section: str) -> list[str]:
        """Returns control or action verb list, or [] if unset."""
        if not self._manifest:
            return []
        cap = (self._manifest.get("capabilities") or {}).get(section, {})
        return list(cap.get("verbs", []) or [])

    def supports_tool(self, tool: str) -> bool:
        """Returns whether the active Page's manifest supports a given top-level tool."""
        if not self._manifest:
            return False
        cap = (self._manifest.get("capabilities") or {}).get(tool, {})
        if tool in ("control", "action"):
            return bool(cap.get("verbs"))
        return bool(cap.get("supported", False))
