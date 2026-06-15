"""Deterministic validators for matrices, determinants and linear systems."""

from __future__ import annotations

import re
from typing import Any

try:
    import sympy as sp
except ImportError:  # pragma: no cover
    sp = None


def repair_latex_cases_environment(text: str) -> str:
    repaired = str(text or "")
    repaired = re.sub(r"(?<!\\)\b(?:egin|begin)\{(cases|pmatrix|bmatrix)\}", r"\\begin{\1}", repaired)
    repaired = re.sub(r"(?<!\\)\bend\{(cases|pmatrix|bmatrix)\}", r"\\end{\1}", repaired)
    repaired = re.sub(r"\\{2,}(begin|end)\{(cases|pmatrix|bmatrix)\}", r"\\\1{\2}", repaired)
    return repaired


def parse_linear_system(text: str) -> tuple[list[Any], list[Any]] | None:
    if sp is None:
        return None
    source = repair_latex_cases_environment(text).replace("\\\\", ";")
    case_match = re.search(r"\\begin\{cases\}(.+?)\\end\{cases\}", source, flags=re.DOTALL)
    if case_match:
        source = case_match.group(1)
    candidates = [part.strip() for part in re.split(r"[;\n]", source) if "=" in part]
    variables = sorted(set(re.findall(r"\b[x-z]\b", source)))
    if not variables or len(candidates) < len(variables):
        return None
    symbols = sp.symbols(" ".join(variables))
    if not isinstance(symbols, tuple):
        symbols = (symbols,)
    equations = []
    for candidate in candidates:
        left, right = candidate.split("=", 1)
        try:
            equations.append(sp.Eq(sp.sympify(_normalize_expr(left)), sp.sympify(_normalize_expr(right))))
        except Exception:
            continue
    if len(equations) < len(symbols):
        return None
    return equations, list(symbols)


def solve_linear_system(equations: list[Any], variables: list[Any]) -> dict[str, Any] | None:
    if sp is None:
        return None
    solution = sp.solve(equations, variables, dict=True)
    if not solution:
        return None
    return {str(key): sp.simplify(value) for key, value in solution[0].items()}


def extract_claimed_solution(text: str) -> dict[str, Any]:
    if sp is None:
        return {}
    claims: dict[str, Any] = {}
    for variable, value in re.findall(r"\b([xyz])\s*=\s*([-+]?\d+(?:[.,]\d+)?(?:/\d+)?)", str(text), flags=re.IGNORECASE):
        try:
            claims[variable.lower()] = sp.sympify(value.replace(",", "."))
        except Exception:
            pass
    return claims


def validate_linear_system_exercise(exercise: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    text = "\n".join(str(exercise.get(field, "")) for field in ("prompt", "hidden_solution", "display_answer", "context", "subtopic"))
    parsed = parse_linear_system(text)
    if parsed is None:
        return False, ["Linear-system validator failed: systeme lineaire non interpretable."], {"applicable": True}
    equations, variables = parsed
    solution = solve_linear_system(equations, variables)
    if not solution:
        return False, ["Linear-system validator failed: impossible de resoudre le systeme."], {"applicable": True}
    claims = extract_claimed_solution(text)
    issues: list[str] = []
    for variable, expected in solution.items():
        if variable in claims and sp.simplify(claims[variable] - expected) != 0:
            issues.append(f"Linear-system validator failed: {variable}={expected}, pas {claims[variable]}.")
    if not claims:
        issues.append("Linear-system validator failed: aucune solution x/y/z explicite n'est donnee.")
    return not issues, issues, {"applicable": True, "solution": solution}


def repair_linear_system_solution(exercise: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_linear_system(str(exercise.get("prompt", "")))
    if parsed is None:
        return exercise
    solution = solve_linear_system(*parsed)
    if not solution:
        return exercise
    repaired = dict(exercise)
    answer = ", ".join(f"\\({key}={value}\\)" for key, value in solution.items())
    repaired["hidden_solution"] = f"En resolvant le systeme lineaire, on obtient {answer}."
    repaired["display_answer"] = answer
    repaired["accepted_answers"] = [answer]
    repaired["corrected_fields_applied"] = True
    repaired["deterministic_repair_applied"] = True
    repaired["values_recomputed"] = {"linear_system": {key: str(value) for key, value in solution.items()}}
    repaired["domain_validator_name"] = "linear_system_solver"
    return repaired


def _normalize_expr(value: str) -> str:
    text = str(value or "").replace(",", ".").replace("^", "**")
    text = re.sub(r"(?<=\d)(?=[xyz])", "*", text)
    return text
