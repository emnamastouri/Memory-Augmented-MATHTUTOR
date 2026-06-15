"""Strict domain routing for deterministic validators."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_label(text: str) -> str:
    """Lowercase, remove accents and normalize spaces/punctuation."""
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^a-zA-Z0-9]+", " ", ascii_text)
    return re.sub(r"\s+", " ", ascii_text).strip().lower()


def get_domain_validator_key(
    topic: str,
    subtopic: str,
    exercise_metadata: dict[str, Any] | None = None,
) -> str | None:
    """Return exactly one deterministic validator key from topic/subtopic labels."""
    metadata = exercise_metadata or {}
    label = normalize_label(" ".join([topic, subtopic, str(metadata.get("target_subtopic", ""))]))

    if "nombres complexes" in label or "nombre complexe" in label:
        return "complex_numbers"
    if any(token in label for token in ("matrices", "determinants", "systemes lineaires", "systeme lineaire")):
        return "linear_systems"
    if any(token in label for token in ("series a deux caracteres", "regression", "correlation")):
        return "regression"
    if any(token in label for token in ("conditionnement", "probabilites totales", "bayes")):
        return "bayes"
    if any(token in label for token in ("variables aleatoires", "variable aleatoire", "esperance", "variance")):
        return "finite_probability"
    if "loi exponentielle" in label:
        return "exponential_law"
    if "suites numeriques" in label or "suite numerique" in label:
        return "sequences"
    if "equations differentielles" in label or "equation differentielle" in label:
        return "ode"
    if "graphes" in label or "graphe" in label:
        return "graphs"
    return None


def explain_domain_route(topic: str, subtopic: str, metadata: dict[str, Any] | None = None) -> str:
    key = get_domain_validator_key(topic, subtopic, metadata)
    return f"Routeur domaine: {key or 'aucun'} depuis topic='{topic}' et subtopic='{subtopic}'."
