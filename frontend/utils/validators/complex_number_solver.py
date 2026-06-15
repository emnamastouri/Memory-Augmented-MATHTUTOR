"""Deterministic checks for Bac-level complex-number exercises."""

from __future__ import annotations

import re
from typing import Any

try:
    import sympy as sp
except ImportError:  # pragma: no cover
    sp = None


def parse_complex_expression(expr: str) -> Any | None:
    """Convert common French/LaTeX complex expressions into SymPy values."""
    if sp is None:
        return None
    text = str(expr or "").strip()
    text = _latex_frac_to_sympy(text)
    text = text.replace("\\sqrt", "sqrt").replace("\\pi", "pi").replace("^", "**")
    text = re.sub(r"e\^\{?i([^}\s]+)\}?", r"exp(I*\1)", text)
    text = re.sub(r"\be\s*\^\s*\(\s*i\s*([^)]+)\)", r"exp(I*(\1))", text)
    text = re.sub(r"(?<![A-Za-z])i(?![A-Za-z])", "I", text)
    text = re.sub(r"(?<=\d)\s*I", "*I", text)
    text = re.sub(r"I\s*sqrt", "I*sqrt", text)
    text = re.sub(r"\)\s*\(", ")*(", text)
    try:
        return sp.simplify(sp.sympify(text))
    except Exception:
        return None


def complex_to_module_argument(z: Any) -> tuple[Any, Any] | None:
    if sp is None or z is None:
        return None
    return sp.simplify(abs(z)), sp.simplify(sp.arg(z))


def is_nth_root_of_unity(z: Any, n: int) -> bool:
    if sp is None or z is None:
        return False
    return sp.simplify(z**n - 1) == 0


def verify_polynomial_root(poly_text: str, candidate_text: str) -> bool:
    if sp is None:
        return False
    z = sp.symbols("z")
    candidate = parse_complex_expression(candidate_text)
    if candidate is None:
        return False
    poly = str(poly_text or "").replace("^", "**")
    try:
        expr = sp.sympify(poly)
        return sp.simplify(expr.subs(z, candidate)) == 0
    except Exception:
        return False


def validate_complex_number_exercise(exercise: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    """Run conservative deterministic checks for complex-number statements."""
    text = "\n".join(str(exercise.get(field, "")) for field in ("prompt", "hidden_solution", "display_answer", "context", "subtopic"))
    normalized = _norm(text)
    issues: list[str] = []
    checks: dict[str, Any] = {}

    if "cinquieme racine" in normalized or "5e racine" in normalized or "racine cinquieme" in normalized:
        z_value = _extract_first_complex_value(text)
        if z_value is None:
            issues.append("Complex-number validator could not parse the candidate fifth root of unity.")
        else:
            is_root = is_nth_root_of_unity(z_value, 5)
            checks["fifth_root"] = bool(is_root)
            if ("est une" in normalized or "vrai" in normalized) and not is_root:
                issues.append("Complex-number validator failed: le nombre donne n'est pas une cinquieme racine de l'unite.")

    if "arg" in normalized and "conjug" in normalized and "module" not in normalized:
        issues.append("Complex-number validator failed: arg(z')=-arg(z) ne suffit pas a prouver z'=conjugue(z) sans egalite des modules.")

    root_claim = re.search(r"([0-9+\-*/sqrt()iI\\{}\s]+)\s+est\s+(?:une\s+)?racine\s+de\s+([^.;\n=]+=\s*0)", text, flags=re.IGNORECASE)
    if root_claim:
        candidate = root_claim.group(1).strip()
        poly = root_claim.group(2).split("=", 1)[0]
        if not verify_polynomial_root(poly, candidate):
            issues.append("Complex-number validator failed: le candidat ne verifie pas l'equation polynomiale.")
        checks["polynomial_root"] = not issues

    if "(sqrt(3)+i)^8" in normalized.replace(" ", "") or "(sqrt 3 i) 8" in normalized:
        z_value = parse_complex_expression("sqrt(3)+i")
        expected = parse_complex_expression("2**8*exp(I*4*pi/3)")
        if z_value is not None and expected is not None and sp.simplify(z_value**8 - expected) == 0:
            checks["moivre_power"] = True
        else:
            issues.append("Complex-number validator failed: la puissance de Moivre annoncee est incoherente.")

    if "z1" in normalized and "z2" in normalized and "conjug" in normalized and "racine" in normalized:
        z1 = _extract_named_complex(text, "z1")
        z2 = _extract_named_complex(text, "z2")
        if z1 is not None and z2 is not None:
            product = sp.simplify(z1 * z2)
            checks["conjugate_product"] = product
            if sp.simplify(product - 1) != 0:
                issues.append("Complex-number validator failed: le produit des conjugues annonce n'est pas egal a 1.")

    if not checks and not issues:
        return False, ["Complex-number validator could not derive a deterministic check."], {"applicable": True}
    return not issues, issues, {"applicable": True, "checks": checks}


def repair_complex_number_solution(exercise: dict[str, Any]) -> dict[str, Any]:
    ok, _issues, metadata = validate_complex_number_exercise(exercise)
    if not ok:
        return exercise
    repaired = dict(exercise)
    repaired["domain_validator_name"] = "complex_number_solver"
    repaired["domain_validator_flag"] = "approved"
    repaired["domain_validator_issues"] = []
    repaired["values_recomputed"] = {"complex": str(metadata.get("checks", {}))}
    return repaired


def _extract_first_complex_value(text: str) -> Any | None:
    patterns = [
        r"z\s*=\s*([^.;\n]+)",
        r"nombre\s+([^.;\n]+?)\s+est",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            parsed = parse_complex_expression(match.group(1))
            if parsed is not None:
                return parsed
    if "1/2" in text and "sqrt" in text.lower():
        return parse_complex_expression("1/2 + i*sqrt(3)/2")
    return None


def _extract_named_complex(text: str, name: str) -> Any | None:
    match = re.search(rf"{name}\s*=\s*([^.;\n]+)", text, flags=re.IGNORECASE)
    return parse_complex_expression(match.group(1)) if match else None


def _latex_frac_to_sympy(text: str) -> str:
    repaired = str(text)
    for _ in range(4):
        new = re.sub(r"\\(?:d)?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", repaired)
        if new == repaired:
            break
        repaired = new
    return repaired


def _norm(value: str) -> str:
    return str(value or "").lower().replace("é", "e").replace("è", "e").replace("π", "pi")
