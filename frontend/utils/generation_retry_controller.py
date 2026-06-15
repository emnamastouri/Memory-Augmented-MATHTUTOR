from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationRetryController:
    max_attempts: int = 8
    failures: list[dict[str, Any]] = field(default_factory=list)

    def register_failure(
        self,
        attempt: int,
        issues: list[str],
        raw_output: str | None = None,
        parsed_exercise: dict[str, Any] | None = None,
    ) -> None:
        categories = classify_failure_categories(issues)
        self.failures.append(
            {
                "attempt": attempt,
                "issues": list(issues),
                "categories": categories,
                "raw_output": raw_output or "",
                "parsed_exercise": parsed_exercise or {},
            }
        )

    def next_strategy(self) -> str:
        if not self.failures:
            return "normal_memory_adapted_generation"

        counts = Counter(category for item in self.failures for category in item.get("categories", []))
        if counts["context_only"] >= 2:
            return "topic_template_guided_generation"
        if counts["context_only"] >= 1:
            return "strict_schema_generation"
        if counts["invalid_json"] >= 2:
            return "simple_exercise_generation"
        if counts["probability_inconsistent"] >= 1:
            return "deterministic_arithmetic_repair"
        if counts["expected_answer_mismatch"] >= 1:
            return "strict_schema_generation"

        if len(self.failures) >= 3:
            last_three = [tuple(item.get("categories", [])) for item in self.failures[-3:]]
            if len(set(last_three)) == 1:
                return "topic_template_guided_generation"
        if len(self.failures) >= self.max_attempts - 1:
            return "final_fail_no_fallback"
        return "normal_memory_adapted_generation"

    def previous_errors(self) -> list[str]:
        messages: list[str] = []
        for failure in self.failures[-5:]:
            messages.extend(str(issue) for issue in failure.get("issues", []) if str(issue).strip())
        deduped: list[str] = []
        seen: set[str] = set()
        for message in messages:
            lowered = message.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(message)
        return deduped

    def failure_categories(self) -> list[str]:
        categories = [category for failure in self.failures for category in failure.get("categories", [])]
        return list(dict.fromkeys(categories))


def classify_failure_categories(issues: list[str]) -> list[str]:
    categories: list[str] = []
    normalized = " ".join(str(issue or "").lower() for issue in issues)

    if any(token in normalized for token in ("connection_error", "connection error", "connexion", "network", "connect error")):
        categories.append("connection_error")
    if "timeout" in normalized or "timed out" in normalized:
        categories.append("timeout")
    if "unsupported_response_format" in normalized or "response_format" in normalized or "json_schema" in normalized:
        categories.append("unsupported_response_format")
    if "empty_response" in normalized or "contenu textuel" in normalized or "choices vide" in normalized:
        categories.append("empty_response")
    if "auth_error" in normalized or "api key" in normalized or "401" in normalized or "403" in normalized:
        categories.append("auth_error")
    if "rate_limit" in normalized or "429" in normalized:
        categories.append("rate_limit")
    if "http_error" in normalized:
        categories.append("http_error")
    if (
        any(token in normalized for token in ("invalid_json", "objet json", "json inexploitable", "json invalide", "json est invalide", "valid json", "exploitable"))
        and "connection_error" not in categories
        and "unsupported_response_format" not in categories
    ):
        categories.append("invalid_json")
    if "sans question" in normalized or "contexte sans question" in normalized or "consigne explicite" in normalized:
        categories.append("context_only")
    if "probabil" in normalized:
        categories.append("probability_inconsistent")
    if "schema" in normalized or "champ requis" in normalized:
        categories.append("schema_invalid")
    if "latex" in normalized or "format" in normalized or "frace" in normalized:
        categories.append("format_invalid")
    if "align" in normalized or "referentiel" in normalized:
        categories.append("alignment_failed")
    if "symbol" in normalized or "sympy" in normalized:
        categories.append("symbolic_failed")
    if "reponse attendue" in normalized or "expected_answer" in normalized or "contrad" in normalized:
        categories.append("expected_answer_mismatch")
    if "similar" in normalized or "copie" in normalized:
        categories.append("too_similar_to_source_case")

    return categories or ["unknown_failure"]
