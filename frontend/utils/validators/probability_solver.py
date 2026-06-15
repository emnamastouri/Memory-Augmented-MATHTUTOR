from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import math
import re
from typing import Any
import unicodedata

from frontend.utils.exercise_schema import has_explicit_questions


FRENCH_NUMBER_WORDS = {
    "zero": 0,
    "un": 1,
    "une": 1,
    "deux": 2,
    "trois": 3,
    "quatre": 4,
    "cinq": 5,
    "six": 6,
    "sept": 7,
    "huit": 8,
    "neuf": 9,
    "dix": 10,
}


def parse_urn_counts(text: str) -> dict[int, int] | None:
    """Parse simple French urn descriptions with signed integer labels."""
    normalized = _normalize_lookup(text)
    if "boule" not in normalized:
        return None

    counts: dict[int, int] = {}
    pattern = re.compile(
        r"\b(?P<count>\d+|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\s+boules?\b[^.\n;,:]*?\b(?:portent|portees|portant|marquees|marque|de valeur|valant)\b[^.\n;,:]*?(?P<value>[-+]?\d+)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(normalized):
        count = _parse_count_token(match.group("count"))
        value = int(match.group("value"))
        if count is None:
            continue
        counts[value] = counts.get(value, 0) + count

    return counts or None


def compute_two_draw_sum_distribution(counts: dict[int, int]) -> dict[int, Fraction]:
    """Compute the exact law of the sum for two simultaneous draws without replacement."""
    total_balls = sum(int(count) for count in counts.values())
    if total_balls < 2:
        return {}

    denominator = math.comb(total_balls, 2)
    distribution: defaultdict[int, Fraction] = defaultdict(Fraction)
    items = sorted((int(value), int(count)) for value, count in counts.items() if int(count) > 0)
    for index, (left_value, left_count) in enumerate(items):
        if left_count >= 2:
            distribution[left_value + left_value] += Fraction(math.comb(left_count, 2), denominator)
        for right_value, right_count in items[index + 1 :]:
            ways = left_count * right_count
            distribution[left_value + right_value] += Fraction(ways, denominator)
    return dict(sorted(distribution.items()))


def compute_expectation(distribution: dict[int, Fraction]) -> Fraction:
    return sum(Fraction(value) * probability for value, probability in distribution.items())


def compute_variance(distribution: dict[int, Fraction]) -> Fraction:
    expectation = compute_expectation(distribution)
    second_moment = sum(Fraction(value * value) * probability for value, probability in distribution.items())
    return second_moment - expectation * expectation


def validate_probability_distribution(distribution: dict[int, Fraction]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    total = Fraction(0, 1)
    for value, probability in distribution.items():
        if probability < 0 or probability > 1:
            issues.append(f"La probabilite P(X={value})={_format_fraction(probability)} n'appartient pas a [0,1].")
        total += probability
    if total != 1:
        issues.append(f"La somme des probabilites vaut {_format_fraction(total)} au lieu de 1.")
    return not issues, issues


def extract_probability_values_from_text(text: str) -> dict[str, Fraction]:
    """Extract simple probability/expectation/variance assignments from free text."""
    values: dict[str, Fraction] = {}
    compact = str(text or "").replace("−", "-").replace(",", ".")
    value_pattern = r"(?:-?\\frac\{-?\d+\}\{\d+\}|[-+]?\d+(?:/\d+)?|\d+\.\d+)"

    for match in re.finditer(
        rf"P\s*\(\s*X\s*=\s*([-+]?\d+)\s*\)\s*=\s*({value_pattern})",
        compact,
        flags=re.IGNORECASE,
    ):
        fraction = _to_fraction(match.group(2))
        if fraction is not None:
            values[f"X={int(match.group(1))}"] = fraction

    for label in ("E", "V"):
        match = re.search(
            rf"{label}\s*\(\s*X\s*\)\s*=\s*({value_pattern})",
            compact,
            flags=re.IGNORECASE,
        )
        if match:
            fraction = _to_fraction(match.group(1))
            if fraction is not None:
                values[f"{label}(X)"] = fraction
    return values


def validate_probability_exercise(exercise: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate deterministic urn/sum probability exercises when the pattern is recognized."""
    statement_text = _build_statement_text(exercise)
    text_parts = [
        str(exercise.get("title", "")),
        statement_text,
        str(exercise.get("hidden_solution", "")),
        str(exercise.get("display_answer", "")),
        str(exercise.get("learning_objective", "")),
    ]
    combined = " ".join(part for part in text_parts if part)
    counts = parse_urn_counts(statement_text)
    if not counts or not _looks_like_two_draw_sum_problem(statement_text):
        return True, [], {"applicable": False}

    distribution = compute_two_draw_sum_distribution(counts)
    distribution_ok, distribution_issues = validate_probability_distribution(distribution)
    issues = list(distribution_issues)
    metadata: dict[str, Any] = {
        "applicable": True,
        "counts": counts,
        "distribution": {str(value): _format_fraction(probability) for value, probability in distribution.items()},
        "expectation": _format_fraction(compute_expectation(distribution)),
        "variance": _format_fraction(compute_variance(distribution)),
    }

    solution_values = extract_probability_values_from_text(str(exercise.get("hidden_solution", "")))
    answer_values = extract_probability_values_from_text(str(exercise.get("display_answer", "")))
    target_values = {
        **{f"X={value}": probability for value, probability in distribution.items()},
        "E(X)": compute_expectation(distribution),
        "V(X)": compute_variance(distribution),
    }

    instruction_text = statement_text
    objective_text = _normalize_lookup(exercise.get("learning_objective", ""))
    needs_expectation = any(marker in _normalize_lookup(instruction_text) for marker in ("e(x)", "esperance")) or "esperance" in objective_text
    needs_variance = any(marker in _normalize_lookup(instruction_text) for marker in ("v(x)", "variance", "sigma")) or "variance" in objective_text

    for key, expected in target_values.items():
        if key.startswith("X="):
            for label, extracted in (("solution", solution_values), ("expected_answer", answer_values)):
                if key not in extracted:
                    issues.append(f"La {label} ne fournit pas {key} dans la loi de probabilite.")
                    continue
                if extracted[key] != expected:
                    issues.append(
                        f"La {label} donne {key}={_format_fraction(extracted[key])} au lieu de {_format_fraction(expected)}."
                    )
        elif key == "E(X)" and needs_expectation:
            if solution_values.get(key) != expected or answer_values.get(key) != expected:
                issues.append(f"L'esperance doit valoir {_format_fraction(expected)}.")
        elif key == "V(X)" and needs_variance:
            if solution_values.get(key) != expected or answer_values.get(key) != expected:
                issues.append(f"La variance doit valoir {_format_fraction(expected)}.")

    if needs_expectation and "E(X)" not in solution_values:
        issues.append("La solution doit calculer explicitement E(X).")
    if needs_variance and "V(X)" not in solution_values:
        issues.append("La solution doit calculer explicitement V(X).")
    if ("esperance" in objective_text or "variance" in objective_text) and not has_explicit_questions(
        instruction_text,
        exercise.get("questions"),
    ):
        issues.append("L'objectif mentionne espérance/variance mais l'enonce ne formule pas correctement les questions.")

    return distribution_ok and not issues, issues, metadata


def repair_probability_exercise_with_deterministic_solution(exercise: dict[str, Any]) -> dict[str, Any]:
    """Rewrite solution/expected answer using deterministic arithmetic for supported urn problems."""
    repaired = dict(exercise)
    text = _build_statement_text(exercise)
    counts = parse_urn_counts(text)
    if not counts or not _looks_like_two_draw_sum_problem(text):
        return repaired
    if not has_explicit_questions(text, exercise.get("questions")):
        return repaired

    distribution = compute_two_draw_sum_distribution(counts)
    expectation = compute_expectation(distribution)
    variance = compute_variance(distribution)
    distribution_lines = [f"\\(P(X={value})={_format_fraction(probability)}\\)" for value, probability in distribution.items()]
    solution_parts = [
        "On tire simultanément deux boules sans remise, donc le nombre total d'issues équiprobables vaut "
        f"\\(\\binom{{{sum(counts.values())}}}{{2}}={math.comb(sum(counts.values()), 2)}\\).",
        "La loi de probabilité de \\(X\\) est : " + ", ".join(distribution_lines) + ".",
        f"On en déduit \\(E(X)={_format_fraction(expectation)}\\).",
        f"Puis \\(V(X)={_format_fraction(variance)}\\).",
    ]
    repaired["hidden_solution"] = " ".join(solution_parts)
    repaired["display_answer"] = (
        "Loi : "
        + ", ".join(distribution_lines)
        + f" ; \\(E(X)={_format_fraction(expectation)}\\) ; \\(V(X)={_format_fraction(variance)}\\)."
    )
    repaired["accepted_answers"] = [repaired["display_answer"]]
    repaired["solution_steps"] = [
        "Compter toutes les paires possibles de boules.",
        "Regrouper les issues selon la somme obtenue pour X.",
        "Calculer l'espérance puis la variance à partir de la loi de X.",
    ]
    repaired["corrected_fields_applied"] = True
    repaired["judge_corrections_applied"] = True
    repaired["probability_repair_note"] = "Probability arithmetic corrected by deterministic checker."
    return repaired


def _parse_count_token(token: str) -> int | None:
    cleaned = _normalize_lookup(token)
    if cleaned.isdigit():
        return int(cleaned)
    return FRENCH_NUMBER_WORDS.get(cleaned)


def _to_fraction(value: str) -> Fraction | None:
    text = str(value or "").strip().replace(",", ".")
    if not text:
        return None
    latex_fraction = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", text)
    if latex_fraction:
        sign = -1 if latex_fraction.group(1) == "-" else 1
        numerator = int(latex_fraction.group(2)) * sign
        denominator = int(latex_fraction.group(3))
        return Fraction(numerator, denominator)
    try:
        return Fraction(text)
    except Exception:
        return None


def _looks_like_two_draw_sum_problem(text: str) -> bool:
    normalized = _normalize_lookup(text)
    return (
        "simultan" in normalized
        and "deux boule" in normalized
        and ("somme" in normalized or "x" in normalized)
    )


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"\\frac{{{value.numerator}}}{{{value.denominator}}}"


def _build_statement_text(exercise: dict[str, Any]) -> str:
    context = str(exercise.get("context", "") or "").strip()
    prompt = str(exercise.get("prompt", "") or exercise.get("instruction", "") or "").strip()
    questions = [str(item or "").strip() for item in (exercise.get("questions") or []) if str(item or "").strip()]

    parts: list[str] = []
    if context and context not in prompt:
        parts.append(context)
    if prompt:
        parts.append(prompt)
    for question in questions:
        if question not in prompt:
            parts.append(question)
    return " ".join(part for part in parts if part).strip()


def _normalize_lookup(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_value).strip().lower()
