"""Deterministic validator for simple exponential-law probability exercises."""

from __future__ import annotations

import math
import re
from typing import Any


def parse_exponential_context(text: str) -> dict[str, Any] | None:
    source = str(text or "")
    normalized = source.lower()
    if "exponentielle" not in normalized:
        return None
    lam_match = re.search(r"(?:lambda|λ|param[eè]tre)\s*(?:=|:|vaut)?\s*([0-9]+(?:[.,][0-9]+)?)", source, flags=re.IGNORECASE)
    if not lam_match:
        return {"missing_lambda": True}
    lam = float(lam_match.group(1).replace(",", "."))
    query_match = re.search(r"P\(\s*([0-9]+(?:[.,][0-9]+)?)\s*<\s*X\s*<\s*([0-9]+(?:[.,][0-9]+)?)\s*\)", source, flags=re.IGNORECASE)
    if query_match:
        return {"lambda": lam, "query": ("between", float(query_match.group(1).replace(",", ".")), float(query_match.group(2).replace(",", ".")))}
    query_match = re.search(r"P\(\s*X\s*>\s*([0-9]+(?:[.,][0-9]+)?)\s*\)", source, flags=re.IGNORECASE)
    if query_match:
        return {"lambda": lam, "query": ("gt", float(query_match.group(1).replace(",", ".")))}
    query_match = re.search(r"P\(\s*X\s*(?:<=|≤|<)\s*([0-9]+(?:[.,][0-9]+)?)\s*\)", source, flags=re.IGNORECASE)
    if query_match:
        return {"lambda": lam, "query": ("le", float(query_match.group(1).replace(",", ".")))}
    return {"lambda": lam}


def compute_exponential_probability(lam: float, query: tuple[Any, ...]) -> float:
    if query[0] == "between":
        return math.exp(-lam * float(query[1])) - math.exp(-lam * float(query[2]))
    if query[0] == "gt":
        return math.exp(-lam * float(query[1]))
    if query[0] == "le":
        return 1 - math.exp(-lam * float(query[1]))
    raise ValueError("Unsupported exponential query")


def validate_exponential_law_exercise(exercise: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    text = "\n".join(str(exercise.get(field, "")) for field in ("context", "prompt", "hidden_solution", "display_answer", "subtopic"))
    parsed = parse_exponential_context(text)
    if not parsed:
        return True, [], {"applicable": False}
    if parsed.get("missing_lambda"):
        return False, ["Exponential-law validator failed: le parametre lambda est manquant."], {"applicable": True}
    if not parsed.get("query"):
        return False, ["Exponential-law validator failed: la probabilite demandee n'est pas interpretable."], {"applicable": True, "parsed": parsed}
    expected = compute_exponential_probability(parsed["lambda"], parsed["query"])
    numbers = [float(item.replace(",", ".")) for item in re.findall(r"(?<![A-Za-z])0[.,]\d+", text)]
    if numbers and all(abs(number - expected) > 1e-3 for number in numbers):
        return False, [f"Exponential-law validator failed: probabilite attendue {expected:.6f}."], {"applicable": True, "expected": expected, "parsed": parsed}
    return True, [], {"applicable": True, "expected": expected, "parsed": parsed}


def repair_exponential_law_solution(exercise: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_exponential_context("\n".join(str(exercise.get(field, "")) for field in ("context", "prompt")))
    if not parsed or parsed.get("missing_lambda") or not parsed.get("query"):
        return exercise
    expected = compute_exponential_probability(parsed["lambda"], parsed["query"])
    repaired = dict(exercise)
    repaired["hidden_solution"] = f"Pour une loi exponentielle de parametre \\(\\lambda={parsed['lambda']}\\), la probabilite demandee vaut \\({expected:.6f}\\)."
    repaired["display_answer"] = f"\\({expected:.6f}\\)"
    repaired["accepted_answers"] = [repaired["display_answer"]]
    repaired["corrected_fields_applied"] = True
    repaired["deterministic_repair_applied"] = True
    repaired["values_recomputed"] = {"exponential_law": expected}
    repaired["domain_validator_name"] = "exponential_law_solver"
    return repaired
