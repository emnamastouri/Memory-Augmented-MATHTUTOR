"""Deterministic Bayes/conditional probability checker."""

from __future__ import annotations

from fractions import Fraction
import re
from typing import Any


def parse_bayes_context(text: str) -> dict[str, Fraction] | None:
    source = str(text or "")
    normalized = _norm(source)
    values = [_parse_probability(item) for item in re.findall(r"[-+]?\d+(?:[.,]\d+)?\s*%?|[-+]?\d+\s*/\s*\d+", source)]
    values = [value for value in values if value is not None]
    params: dict[str, Fraction] = {}

    direct_patterns = {
        "p_d": r"P\(\s*D\s*\)\s*=\s*([^,;.\n]+)",
        "p_h_given_d": r"P\(\s*H\s*\|\s*D\s*\)\s*=\s*([^,;.\n]+)",
        "p_h_given_not_d": r"P\(\s*H\s*\|\s*(?:non\s*D|Dbar|\\bar\{D\}|D̄)\s*\)\s*=\s*([^,;.\n]+)",
    }
    for key, pattern in direct_patterns.items():
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if match:
            parsed = _parse_probability(match.group(1))
            if parsed is not None:
                params[key] = parsed

    if "p_d" not in params:
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*%\s+des\s+\w+\s+sont\s+defect", normalized)
        if match:
            params["p_d"] = _parse_probability(match.group(1) + "%") or Fraction(0)
    if "p_h_given_d" not in params:
        match = re.search(r"parmi\s+ceux[- ]?ci[^.]{0,80}?(\d+(?:[.,]\d+)?)\s*%", normalized)
        if match:
            params["p_h_given_d"] = _parse_probability(match.group(1) + "%") or Fraction(0)
    if "p_h_given_not_d" not in params:
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*%\s+des\s+\w+\s+non\s+defect[^.]{0,80}?detect", normalized)
        if match:
            params["p_h_given_not_d"] = _parse_probability(match.group(1) + "%") or Fraction(0)
    if len(params) < 3 and len(values) >= 3 and any(token in normalized for token in ("bayes", "condition", "defect", "detect")):
        params.setdefault("p_d", values[0])
        params.setdefault("p_h_given_d", values[1])
        params.setdefault("p_h_given_not_d", values[2])
    if {"p_d", "p_h_given_d", "p_h_given_not_d"} <= set(params):
        return params
    return None


def compute_bayes_values(params: dict[str, Fraction]) -> dict[str, Fraction]:
    p_d = params["p_d"]
    p_not_d = Fraction(1) - p_d
    p_h_given_d = params["p_h_given_d"]
    p_h_given_not_d = params["p_h_given_not_d"]
    p_d_and_h = p_d * p_h_given_d
    p_not_d_and_h = p_not_d * p_h_given_not_d
    p_h = p_d_and_h + p_not_d_and_h
    return {
        "p_not_d": p_not_d,
        "p_d_and_h": p_d_and_h,
        "p_not_d_and_h": p_not_d_and_h,
        "p_h": p_h,
        "p_d_given_h": p_d_and_h / p_h if p_h else Fraction(0),
    }


def extract_probability_claims(text: str) -> dict[str, Fraction]:
    claims: dict[str, Fraction] = {}
    patterns = {
        "p_h": r"P\(\s*H\s*\)\s*(?:≈|\\approx|=)\s*([^,;.\n]+)",
        "p_d_given_h": r"P\(\s*D\s*\|\s*H\s*\)\s*(?:≈|\\approx|=)\s*([^,;.\n]+)",
        "p_not_d_and_h": r"P\(\s*(?:non\s*D|\\bar\{D\}|Dbar)\s*(?:∩|\\cap|inter)\s*H\s*\)\s*(?:≈|\\approx|=)\s*([^,;.\n]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            parsed = _parse_probability(match.group(1))
            if parsed is not None:
                claims[key] = parsed
    return claims


def validate_bayes_exercise(exercise: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    text = "\n".join(str(exercise.get(field, "")) for field in ("context", "prompt", "hidden_solution", "display_answer", "subtopic", "topic"))
    if not _is_bayes_topic(text):
        return True, [], {"applicable": False}
    params = parse_bayes_context(text)
    if not params:
        return False, ["Bayes validator failed: donnees P(D), P(H|D), P(H|non D) incompletes."], {"applicable": True}
    if any(value < 0 or value > 1 for value in params.values()):
        return False, ["Bayes validator failed: une probabilite du contexte n'appartient pas a [0,1]."], {"applicable": True, "params": params}
    values = compute_bayes_values(params)
    claims = extract_probability_claims(text)
    issues: list[str] = []
    for key, expected in values.items():
        if key in claims and abs(float(claims[key] - expected)) > 1e-3:
            issues.append(f"Bayes validator failed: {key} vaut {float(expected):.6g}, pas {float(claims[key]):.6g}.")
    metadata = {"applicable": True, "params": params, "values": values, "claims": claims}
    return not issues, issues, metadata


def repair_bayes_solution(exercise: dict[str, Any]) -> dict[str, Any]:
    params = parse_bayes_context("\n".join(str(exercise.get(field, "")) for field in ("context", "prompt")))
    if not params:
        return exercise
    values = compute_bayes_values(params)
    repaired = dict(exercise)
    solution = (
        f"On a \\(P(D)={_fmt(params['p_d'])}\\), \\(P(H|D)={_fmt(params['p_h_given_d'])}\\) et "
        f"\\(P(H|\\bar{{D}})={_fmt(params['p_h_given_not_d'])}\\). "
        f"Donc \\(P(H)=P(D)P(H|D)+P(\\bar{{D}})P(H|\\bar{{D}})={_fmt(values['p_h'])}\\). "
        f"Par Bayes, \\(P(D|H)=\\frac{{P(D\\cap H)}}{{P(H)}}={_fmt(values['p_d_given_h'])}\\). "
        f"Enfin \\(P(\\bar{{D}}\\cap H)={_fmt(values['p_not_d_and_h'])}\\)."
    )
    repaired["hidden_solution"] = solution
    repaired["display_answer"] = (
        f"\\(P(H)={_fmt(values['p_h'])}\\), \\(P(D|H)\\approx {float(values['p_d_given_h']):.3f}\\), "
        f"\\(P(\\bar{{D}}\\cap H)={_fmt(values['p_not_d_and_h'])}\\)."
    )
    repaired["accepted_answers"] = [repaired["display_answer"]]
    repaired["corrected_fields_applied"] = True
    repaired["deterministic_repair_applied"] = True
    repaired["values_recomputed"] = {"bayes": {key: float(value) for key, value in values.items()}}
    repaired["domain_validator_name"] = "bayes_solver"
    return repaired


def _parse_probability(value: str) -> Fraction | None:
    text = str(value or "").strip().replace(",", ".")
    try:
        if text.endswith("%"):
            return Fraction(text[:-1]) / 100
        if "/" in text:
            left, right = text.split("/", 1)
            return Fraction(left.strip()) / Fraction(right.strip())
        return Fraction(text)
    except Exception:
        return None


def _fmt(value: Fraction) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _norm(value: str) -> str:
    return str(value or "").lower().replace("dé", "de").replace("é", "e").replace("è", "e").replace("fectueux", "fectueux")


def _is_bayes_topic(text: str) -> bool:
    normalized = _norm(text)
    return any(token in normalized for token in ("bayes", "condition", "probabilites totales", "probabilités totales", "defect", "detect"))
