"""Deterministic regression/statistics validation helpers."""

from __future__ import annotations

import math
import re
from typing import Any


def parse_numeric_series_from_text(text: str) -> dict[str, Any] | None:
    """Extract simple t/x/y data and transformations from French statements."""
    source = str(text or "")
    data: dict[str, Any] = {}
    for name in ("t", "x", "y"):
        match = re.search(rf"\b{name}\s*=\s*([0-9,.;\s\-]+)", source, flags=re.IGNORECASE)
        if match:
            values = _parse_number_list(match.group(1))
            if len(values) >= 2:
                data[name] = values
    if re.search(r"y\s*=\s*\\?ln\s*\(?\s*x\s*\)?", source, flags=re.IGNORECASE):
        data["y_formula"] = "ln(x)"
    else:
        formula = re.search(r"y\s*=\s*([0-9.,]+\s*\*?\s*x\s*[+-]\s*[0-9.,]+\s*\*?\s*t)", source, flags=re.IGNORECASE)
        data["y_formula"] = formula.group(1).replace(",", ".") if formula else None
    return data if data else None


def parse_numeric_series_from_table(table_data: Any) -> dict[str, Any] | None:
    if not isinstance(table_data, dict):
        return None
    rows = table_data.get("rows")
    headers = table_data.get("headers") or []
    if not isinstance(rows, list) or len(rows) < 3:
        return None
    numeric_rows = []
    for row in rows:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            try:
                numeric_rows.append([_to_float(row[0]), _to_float(row[1])])
            except Exception:
                continue
    if len(numeric_rows) < 3:
        return None
    first_label = str(headers[0] if headers else "x").lower()
    second_label = str(headers[1] if len(headers) > 1 else "y").lower()
    return {
        "x_axis_name": first_label,
        "y_axis_name": second_label,
        "x": [row[0] for row in numeric_rows],
        "y": [row[1] for row in numeric_rows],
        "y_formula": None,
    }


def compute_transformed_y(t_values: list[float] | None, x_values: list[float], y_formula: str | None) -> list[float] | None:
    if not y_formula:
        return None
    normalized = y_formula.replace(" ", "").lower()
    if normalized in {"ln(x)", "\\ln(x)"}:
        if any(value <= 0 for value in x_values):
            return None
        return [math.log(value) for value in x_values]
    if t_values and re.fullmatch(r"[0-9.]+[*]?x[+-][0-9.]+[*]?t", normalized):
        signs = re.split(r"([+-])", normalized)
        values: list[float] = []
        for t_value, x_value in zip(t_values, x_values):
            total = 0.0
            sign = 1.0
            for part in signs:
                if part == "+":
                    sign = 1.0
                    continue
                if part == "-":
                    sign = -1.0
                    continue
                if not part:
                    continue
                if "x" in part:
                    coeff = part.replace("*", "").replace("x", "")
                    total += sign * (float(coeff) if coeff else 1.0) * x_value
                elif "t" in part:
                    coeff = part.replace("*", "").replace("t", "")
                    total += sign * (float(coeff) if coeff else 1.0) * t_value
            values.append(total)
        return values
    return None


def compute_linear_regression(x_values: list[float], y_values: list[float]) -> dict[str, float] | None:
    if len(x_values) != len(y_values) or len(x_values) < 3:
        return None
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    sxx = sum((x - mean_x) ** 2 for x in x_values)
    syy = sum((y - mean_y) ** 2 for y in y_values)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    return {"slope": slope, "intercept": intercept, "r": sxy / math.sqrt(sxx * syy)}


def extract_numeric_claims_from_solution(text: str) -> dict[str, Any]:
    source = str(text or "")
    claims: dict[str, Any] = {}
    ln_values = re.findall(r"\\?ln\s*\(\s*[-+]?\d+(?:[.,]\d+)?\s*\)\s*(?:≈|\\approx|=)\s*([-+]?\d+(?:[.,]\d+)?)", source)
    if ln_values:
        claims["y_values"] = [_to_float(value) for value in ln_values]
    r_match = re.search(r"\br\s*(?:≈|\\approx|=)\s*([-+]?\d+(?:[.,]\d+)?)", source, flags=re.IGNORECASE)
    if r_match:
        claims["r"] = _to_float(r_match.group(1))
    line_match = re.search(
        r"(?:y|x|p)\s*(?:≈|\\approx|=)\s*([-+]?\d+(?:[.,]\d+)?)\s*(?:t|x|i)\s*([+-]\s*\d+(?:[.,]\d+)?)",
        source,
        flags=re.IGNORECASE,
    )
    if line_match:
        claims["slope"] = _to_float(line_match.group(1))
        claims["intercept"] = _to_float(line_match.group(2).replace(" ", ""))
    return claims


def validate_regression_exercise(exercise: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    text = "\n".join(
        str(exercise.get(field, ""))
        for field in ("context", "prompt", "instruction", "hidden_solution", "display_answer", "learning_objective", "subtopic")
    )
    if not _is_regression_context(text):
        return True, [], {"applicable": False}
    parsed = parse_numeric_series_from_table(exercise.get("table_data")) or parse_numeric_series_from_text(text)
    if not parsed:
        return False, ["Regression validator failed: les donnees numeriques de regression ne sont pas interpretables."], {"applicable": True}
    x_values = parsed.get("t") or parsed.get("x")
    raw_x = parsed.get("x")
    y_values = parsed.get("y")
    if y_values is None and raw_x:
        y_values = compute_transformed_y(parsed.get("t"), raw_x, parsed.get("y_formula"))
    if not x_values or not y_values or len(x_values) != len(y_values):
        return False, ["Regression validator failed: les valeurs x/y ou la transformation y ne sont pas coherentes."], {"applicable": True, "parsed": parsed}
    stats = compute_linear_regression(x_values, y_values)
    if not stats:
        return False, ["Regression validator failed: impossible de recalculer la droite de regression."], {"applicable": True, "parsed": parsed}
    claims = extract_numeric_claims_from_solution(text)
    issues: list[str] = []
    if claims.get("y_values") and parsed.get("y_formula"):
        for index, (claimed, expected) in enumerate(zip(claims["y_values"], y_values), start=1):
            if abs(claimed - expected) > 0.02:
                issues.append(f"Regression validator failed: y_{index} vaut {expected:.3f}, pas {claimed:.3f}.")
                break
    if "slope" in claims and abs(claims["slope"] - stats["slope"]) > 0.03:
        issues.append(f"Regression validator failed: pente attendue {stats['slope']:.3f}, pas {claims['slope']:.3f}.")
    if "intercept" in claims and abs(claims["intercept"] - stats["intercept"]) > 0.03:
        issues.append(f"Regression validator failed: ordonnee attendue {stats['intercept']:.3f}, pas {claims['intercept']:.3f}.")
    if "r" in claims and abs(claims["r"] - stats["r"]) > 0.01:
        issues.append(f"Regression validator failed: coefficient r attendu {stats['r']:.3f}, pas {claims['r']:.3f}.")
    metadata = {"applicable": True, "parsed": parsed, "computed_y": y_values, "regression": stats, "claims": claims}
    return not issues, issues, metadata


def repair_regression_solution(exercise: dict[str, Any]) -> dict[str, Any]:
    ok, _issues, metadata = validate_regression_exercise({**exercise, "hidden_solution": "", "display_answer": ""})
    if not metadata.get("applicable") or not ok:
        return exercise
    y_values = metadata["computed_y"]
    stats = metadata["regression"]
    repaired = dict(exercise)
    y_table = "; ".join(f"{value:.3f}" for value in y_values)
    solution = (
        f"On calcule les valeurs transformees : {y_table}. "
        f"Le coefficient de correlation vaut \\(r \\approx {stats['r']:.3f}\\). "
        f"La droite de regression est \\(y \\approx {stats['slope']:.3f}x"
        f"{stats['intercept']:+.3f}\\)."
    )
    repaired["hidden_solution"] = solution
    repaired["display_answer"] = f"\\(r \\approx {stats['r']:.3f}\\), droite : \\(y \\approx {stats['slope']:.3f}x{stats['intercept']:+.3f}\\)."
    repaired["accepted_answers"] = [repaired["display_answer"]]
    repaired["corrected_fields_applied"] = True
    repaired["deterministic_repair_applied"] = True
    repaired["values_recomputed"] = {"regression": stats, "y": y_values}
    repaired["domain_validator_name"] = "regression_solver"
    return repaired


def _parse_number_list(value: str) -> list[float]:
    return [_to_float(item) for item in re.findall(r"[-+]?\d+(?:[.,]\d+)?", value)]


def _to_float(value: Any) -> float:
    return float(str(value).replace(",", "."))


def _is_regression_context(text: str) -> bool:
    normalized = str(text or "").lower()
    return any(token in normalized for token in ("regression", "corrélation", "correlation", "ajustement", "serie statistique", "série statistique"))
