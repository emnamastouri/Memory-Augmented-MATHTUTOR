"""Graph exercise support validation."""

from __future__ import annotations

import re
from typing import Any


GRAPH_KEYWORDS = ("graphe", "sommets", "aretes", "arêtes", "chaine", "chaîne", "euler", "hamilton", "coloriage", "adjacence")


def validate_graph_support(exercise: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    text = " ".join(str(exercise.get(field, "")) for field in ("prompt", "context", "subtopic", "topic")).lower()
    applicable = any(keyword in text for keyword in GRAPH_KEYWORDS)
    if not applicable:
        return True, [], {"applicable": False}
    graph_data = exercise.get("graph_data")
    if _valid_graph_data(graph_data):
        return True, [], {"applicable": True, "support": "graph_data"}
    if _has_explicit_edge_list(text):
        return True, [], {"applicable": True, "support": "edge_list_text"}
    return False, ["Graph support validator failed: graph_data ou liste explicite d'aretes absente."], {"applicable": True}


def _valid_graph_data(graph_data: Any) -> bool:
    if not isinstance(graph_data, dict):
        return False
    vertices = graph_data.get("vertices") or graph_data.get("nodes")
    edges = graph_data.get("edges")
    return isinstance(vertices, list) and len(vertices) >= 2 and isinstance(edges, list) and len(edges) >= 1


def _has_explicit_edge_list(text: str) -> bool:
    return bool(re.search(r"\b[ev]\s*=\s*\{[^}]+\}", text, flags=re.IGNORECASE)) or "aretes" in text and re.search(r"\([a-z0-9]\s*,\s*[a-z0-9]\)", text)
