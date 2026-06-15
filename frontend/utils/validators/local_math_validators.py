from __future__ import annotations

import ast
from fractions import Fraction
import json
import math
import re
from typing import Any
import unicodedata

try:
    import sympy as sp
except ImportError:  # pragma: no cover
    sp = None

from frontend.utils.math_format_guard import repair_math_text_locally
from frontend.utils.validators.probability_solver import (
    repair_probability_exercise_with_deterministic_solution,
    validate_probability_exercise,
)
from frontend.utils.validators.bayes_solver import validate_bayes_exercise
from frontend.utils.validators.complex_number_solver import validate_complex_number_exercise
from frontend.utils.validators.domain_router import get_domain_validator_key, explain_domain_route
from frontend.utils.validators.exponential_law_solver import validate_exponential_law_exercise
from frontend.utils.validators.graph_support_validator import validate_graph_support
from frontend.utils.validators.linear_system_solver import validate_linear_system_exercise
from frontend.utils.validators.question_coverage import validate_question_answer_coverage
from frontend.utils.validators.regression_solver import validate_regression_exercise


VISUAL_CHART_PATTERNS = (
    r"\bgraphique\b",
    r"\bgraphe ci[- ]dessous\b",
    r"\bcourbe donnee\b",
    r"\bcourbe fournie\b",
    r"\bcourbe representee\b",
    r"\bci[- ]dessous\b",
    r"\bpar lecture graphique\b",
    r"\btrace\b",
    r"\btracé\b",
    r"\bfigure\b",
    r"\bfeuille annexe\b",
)
VISUAL_TABLE_PATTERNS = (
    r"\btableau de variation donne\b",
    r"\btableau de variation donné\b",
    r"\ble tableau suivant\b",
    r"\ble tableau ci dessous\b",
    r"\ble tableau ci-dessous\b",
)
PROBLEMATIC_CORRECTION_PHRASES = (
    "l'enonce est problematique",
    "l'énoncé est problématique",
    "il y a une erreur dans l'enonce",
    "il y a une erreur dans l'énoncé",
    "la solution initiale est fausse",
    "il semble y avoir une incomprehension",
    "il semble y avoir une incompréhension",
)
LATEX_CORRUPTION_PATTERNS = (
    (r"\bfrace\b", "Le token 'frace' indique une fraction LaTeX corrompue."),
    (r"\bsqrtt\b", "Le token 'sqrtt' indique une racine LaTeX corrompue."),
    (r"\bfracpi\b", "Le token 'fracpi' indique une fraction de pi corrompue."),
    (r"\bfrac[0-9]", "Un token de fraction compacte du type 'frac1' n'a pas ete converti proprement."),
    (r"\bmathbb\s*[a-z]\b", "Le token 'mathbbR/mathbbN/...' n'a pas ete converti proprement."),
    (r"\bvec[ijk]\b", "Les vecteurs unitaires 'veci/vecj/veck' n'ont pas ete convertis proprement."),
    (r"(?:\+|-|->|vers|tend(?:re)?\s+vers|lim(?:ite)?)[^.\n]{0,30}\bin\s+fty\b", "La borne infinie contient le token corrompu 'in fty'."),
    (r"(?:\+|-|->|vers|tend(?:re)?\s+vers|lim(?:ite)?)[^.\n]{0,30}\bin\s+t\b", "La borne infinie contient le token corrompu 'in t'."),
    (r"\bin\s+t_", "Le token 'in t_' indique une integrale corrompue."),
)


def validate_exercise_locally(record: dict[str, Any]) -> dict[str, Any]:
    """Run deterministic local math and structure checks before presentation."""
    prompt = str(record.get("prompt", "")).strip()
    display_answer = str(record.get("display_answer", "")).strip()
    extracted_answer = (
        str(record.get("solution_validation_llm_answer", "")).strip()
        or _extract_final_answer_from_solution_text(str(record.get("hidden_solution", "")))
    )
    answer_kind = str(record.get("answer_kind", "text")).strip().lower() or "text"
    domain_key = get_domain_validator_key(record.get("topic", ""), record.get("subtopic", ""), record.get("generation_metadata") or {})

    checks = {
        "formatting": _validate_formatting(record),
        "correction_guard": _check_problematic_correction_language(record),
        "derivative": _validate_derivative(record, display_answer, extracted_answer, answer_kind),
        "function_membership": _validate_function_membership(record),
        "ode": _validate_differential_equation(record),
        "inverse_function": _validate_inverse_function(record),
        "probability": _validate_probability(record, display_answer, extracted_answer, answer_kind),
        "bayes": _validate_bayes(record),
        "exponential_law": _validate_exponential_law(record),
        "complex": _validate_complex(record),
        "complex_numbers": _validate_complex_numbers(record),
        "linear_systems": _validate_linear_systems(record),
        "conics": _validate_conics(record),
        "regression": _validate_regression_line(record),
        "regression_numeric": _validate_regression_numeric(record),
        "regression_deterministic": _validate_regression_deterministic(record),
        "pedagogical_completeness": _validate_pedagogical_completeness(record),
        "integral_area": _validate_integral_area(record, display_answer, extracted_answer, answer_kind),
        "visual_support": _validate_visual_support(record),
        "graph_support": _validate_graph_support(record),
        "question_coverage": _validate_question_coverage(record),
        "graph_theory": _validate_graph_theory(record),
        "recurrence_relation": _validate_recurrence_relations(record),
        "sequence_numeric": _validate_numeric_sequence_patterns(record),
        "adjacent_sequences": _validate_adjacent_sequences(record),
        "domain": _validate_domain(record),
    }
    checks = _route_domain_checks(domain_key, checks)

    issues: list[str] = []
    symbolic_checks_required = False
    symbolic_checks_ran = False
    symbolic_failures: list[str] = []
    any_applied = False
    for name, outcome in checks.items():
        if outcome["applicable"]:
            any_applied = True
        if outcome["requires_symbolic"]:
            symbolic_checks_required = True
        if outcome["symbolic_ran"]:
            symbolic_checks_ran = True
        if outcome["status"] == "failed":
            issues.extend(outcome["issues"])
            if outcome["requires_symbolic"]:
                symbolic_failures.append(name)

    issues = _deduplicate_preserving_order(issues)
    if _exercise_forces_symbolic(record) and not symbolic_checks_required:
        symbolic_checks_required = True
    if symbolic_checks_required and not symbolic_checks_ran and not issues:
        issues.append("Aucune validation symbolique derivee de l'enonce n'a pu etre executee localement.")

    if symbolic_checks_required:
        symbolic_checks_passed = bool(symbolic_checks_ran and not symbolic_failures)
    elif symbolic_checks_ran:
        symbolic_checks_passed = not symbolic_failures
    else:
        symbolic_checks_passed = None

    local_validation_flag = "wrong" if issues else "approved"
    if not any_applied and local_validation_flag == "approved":
        local_validation_summary = "Validation symbolique non applicable a cet exercice."
    elif issues:
        local_validation_summary = "La validation locale a detecte des incoherences mathematiques ou structurelles."
    elif symbolic_checks_ran:
        local_validation_summary = "Les controles locaux applicables ont valide la coherence mathematique."
    else:
        local_validation_summary = "Les controles structurels et arithmetiques applicables ont passe."

    pedagogical_outcome = checks["pedagogical_completeness"]
    pedagogical_flag = (
        "not_applicable"
        if not pedagogical_outcome["applicable"]
        else ("approved" if pedagogical_outcome["status"] == "passed" else "incomplete")
    )

    return {
        "local_validation_flag": local_validation_flag,
        "local_validation_summary": local_validation_summary,
        "local_validation_issues": issues,
        "checks": checks,
        "pedagogical_completeness_flag": pedagogical_flag,
        "pedagogical_completeness_summary": pedagogical_outcome.get("summary", ""),
        "pedagogical_completeness_issues": list(pedagogical_outcome.get("issues", []) or []),
        "symbolic_checks_ran": symbolic_checks_ran,
        "symbolic_checks_passed": symbolic_checks_passed,
        "symbolic_checks_required": symbolic_checks_required,
        "domain_router_key": domain_key,
        "domain_router_reason": explain_domain_route(record.get("topic", ""), record.get("subtopic", ""), record.get("generation_metadata") or {}),
    }


def instruction_requires_visual_support(record: dict[str, Any]) -> bool:
    """Return whether the statement explicitly relies on an attached support."""
    prompt = _normalize_lookup(record.get("prompt", ""))
    return any(re.search(pattern, prompt) for pattern in [*VISUAL_CHART_PATTERNS, *VISUAL_TABLE_PATTERNS])


def _build_check_result(
    *,
    name: str,
    applicable: bool,
    passed: bool | None,
    issues: list[str] | None = None,
    summary: str = "",
    requires_symbolic: bool = False,
    symbolic_ran: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "applicable": applicable,
        "status": "not_applicable" if not applicable else ("passed" if passed else "failed"),
        "issues": list(issues or []),
        "summary": summary,
        "requires_symbolic": requires_symbolic,
        "symbolic_ran": symbolic_ran,
        "metadata": metadata or {},
    }


def _check_problematic_correction_language(record: dict[str, Any]) -> dict[str, Any]:
    text = _normalize_lookup(
        " ".join(
            [
                str(record.get("judge_summary", "")),
                str(record.get("hidden_solution", "")),
                str(record.get("display_answer", "")),
            ]
        )
    )
    for phrase in PROBLEMATIC_CORRECTION_PHRASES:
        if _normalize_lookup(phrase) in text:
            return _build_check_result(
                name="correction_guard",
                applicable=True,
                passed=False,
                issues=["La correction signale elle-meme un probleme d'enonce ou de solution."],
                summary="Le contenu corrige reste problematique.",
            )
    return _build_check_result(name="correction_guard", applicable=bool(text), passed=True)


def _validate_formatting(record: dict[str, Any]) -> dict[str, Any]:
    """Block obvious LaTeX corruption that should never reach the student."""
    raw_segments = [
        str(record.get("prompt", "")),
        str(record.get("display_answer", "")),
        str(record.get("hidden_solution", "")),
        *[str(option) for option in (record.get("options", []) or [])],
    ]
    combined_text = "\n".join(segment for segment in raw_segments if segment).strip()
    if not combined_text:
        return _build_check_result(name="formatting", applicable=False, passed=None)

    issues: list[str] = []
    lowered_text = combined_text.lower()
    for pattern, message in LATEX_CORRUPTION_PATTERNS:
        if re.search(pattern, lowered_text, flags=re.IGNORECASE):
            issues.append(f"Formatting validator failed: {message}")

    return _build_check_result(
        name="formatting",
        applicable=True,
        passed=not issues,
        issues=issues,
        summary="Controle de corruption LaTeX et de format student-facing.",
        requires_symbolic=False,
        symbolic_ran=False,
    )


def _validate_derivative(
    record: dict[str, Any],
    display_answer: str,
    extracted_answer: str,
    answer_kind: str,
) -> dict[str, Any]:
    if sp is None:
        return _build_check_result(
            name="derivative",
            applicable=_looks_like_derivative_context(record),
            passed=False,
            issues=["Validation symbolique requise pour la derivee, mais SymPy est indisponible."],
            summary="SymPy indisponible pour le controle de derivee.",
            requires_symbolic=_looks_like_derivative_context(record),
            symbolic_ran=False,
        )

    prompt = str(record.get("prompt", "")).strip()
    function_expression = _extract_function_expression(prompt)
    derivative_context = _looks_like_derivative_context(record)
    if not derivative_context or not function_expression:
        return _build_check_result(name="derivative", applicable=False, passed=None)

    try:
        x = sp.symbols("x")
        computed = sp.simplify(sp.diff(sp.sympify(_normalize_for_sympy(function_expression)), x))
    except Exception:
        return _build_check_result(
            name="derivative",
            applicable=True,
            passed=False,
            issues=["La fonction de depart n'est pas assez lisible pour recalculer sa derivee localement."],
            summary="Echec du recalcul local de derivee.",
            requires_symbolic=True,
            symbolic_ran=False,
        )

    issues: list[str] = []
    claims = _extract_derivative_claims(record)
    computed_text = sp.sstr(computed)
    for claim in claims:
        try:
            claimed_expr = sp.sympify(_normalize_for_sympy(claim))
        except Exception:
            issues.append(f"Une derivee annoncee n'est pas interpretable localement : {claim}.")
            continue
        if sp.simplify(claimed_expr - computed) != 0:
            issues.append(
                f"Derivative validator failed: claimed f'(x)={claim}, expected {computed_text}."
            )

    if answer_kind in {"numeric", "expression"} and display_answer:
        if not _answers_equivalent(display_answer, computed_text, answer_kind):
            issues.append(
                f"Derivative validator failed: expected answer {display_answer} does not match {computed_text}."
            )
    if extracted_answer and answer_kind in {"numeric", "expression"}:
        if not _answers_equivalent(extracted_answer, computed_text, answer_kind):
            issues.append(
                f"Derivative validator failed: solution final answer {extracted_answer} does not match {computed_text}."
            )

    return _build_check_result(
        name="derivative",
        applicable=True,
        passed=not issues,
        issues=issues,
        summary=f"Derivee recalculee localement : {computed_text}.",
        requires_symbolic=True,
        symbolic_ran=True,
    )


def _validate_function_membership(record: dict[str, Any]) -> dict[str, Any]:
    """Substitute a candidate function into the defining condition of an ensemble."""
    prompt = str(record.get("prompt", "")).strip()
    normalized_prompt = _normalize_lookup(prompt)
    if sp is None:
        applicable = "appart" in normalized_prompt and "ensemble" in normalized_prompt and "=" in prompt
        return _build_check_result(
            name="function_membership",
            applicable=applicable,
            passed=False if applicable else None,
            issues=["Validation symbolique requise pour verifier l'appartenance fonctionnelle, mais SymPy est indisponible."] if applicable else [],
            summary="SymPy indisponible pour le controle d'appartenance." if applicable else "",
            requires_symbolic=applicable,
            symbolic_ran=False,
        )

    function_info = _extract_named_function_expression(prompt) or _extract_named_function_expression(str(record.get("hidden_solution", "")))
    condition = _extract_membership_condition(prompt)
    applicable = bool(function_info and condition and "appart" in normalized_prompt)
    if not applicable:
        return _build_check_result(name="function_membership", applicable=False, passed=None)

    function_name, function_expr = function_info
    try:
        x = sp.symbols("x")
        candidate_expr = sp.sympify(_normalize_for_sympy(function_expr))
        left_condition, right_condition = [part.strip() for part in condition.split("=", 1)]
        substituted_left = _substitute_function_condition(left_condition, function_name, candidate_expr, x)
        substituted_right = _substitute_function_condition(right_condition, function_name, candidate_expr, x)
        difference = sp.simplify(substituted_left - substituted_right)
    except Exception:
        return _build_check_result(
            name="function_membership",
            applicable=True,
            passed=False,
            issues=["Function-membership validator failed: la condition definissant l'ensemble n'est pas interpretable localement."],
            summary="Echec du controle d'appartenance fonctionnelle.",
            requires_symbolic=True,
            symbolic_ran=False,
        )

    if difference != 0:
        return _build_check_result(
            name="function_membership",
            applicable=True,
            passed=False,
            issues=[
                f"Function-membership validator failed: en substituant {function_name}(x)={function_expr}, on obtient {sp.sstr(substituted_left)} =/= {sp.sstr(substituted_right)}."
            ],
            summary="La fonction proposee ne verifie pas la condition de l'ensemble.",
            requires_symbolic=True,
            symbolic_ran=True,
        )

    return _build_check_result(
        name="function_membership",
        applicable=True,
        passed=True,
        summary="La fonction candidate verifie la condition definissant l'ensemble.",
        requires_symbolic=True,
        symbolic_ran=True,
    )


def _validate_differential_equation(record: dict[str, Any]) -> dict[str, Any]:
    """Validate common Bac differential-equation patterns deterministically.

    The checker is deliberately conservative: it only applies to patterns that it
    can verify without interpreting a complete free-form proof.  It currently
    covers the frequent Bac pattern ``y'' + y = 0`` and the associated functional
    set condition ``f'(x) + f(pi/2 - x) = 0``.
    """
    combined = "\n".join(
        str(record.get(field, ""))
        for field in (
            "title",
            "topic",
            "subtopic",
            "prompt",
            "hidden_solution",
            "display_answer",
        )
    )
    normalized = _normalize_lookup(combined)
    prompt_normalized = _normalize_lookup(record.get("prompt", ""))
    applicable = any(
        token in normalized
        for token in (
            "equation differentielle",
            "equations differentielles",
            "y''+y=0",
            "y'' + y = 0",
            "y''+ a^2y=0",
            "y'' + a^2y = 0",
        )
    )
    if not applicable:
        return _build_check_result(name="ode", applicable=False, passed=None)

    if sp is None:
        return _build_check_result(
            name="ode",
            applicable=True,
            passed=False,
            issues=["Validation symbolique requise pour les equations differentielles, mais SymPy est indisponible."],
            summary="SymPy indisponible pour le controle d'equation differentielle.",
            requires_symbolic=True,
            symbolic_ran=False,
        )

    issues: list[str] = []
    report_parts: list[str] = []
    x = sp.symbols("x")
    A, B = sp.symbols("A B")

    # Pattern 1: y'' + y = 0 has the general solution A sin(x) + B cos(x).
    if _contains_ode_y_second_plus_y_zero(combined):
        candidate = A * sp.sin(x) + B * sp.cos(x)
        residual = sp.simplify(sp.diff(candidate, x, 2) + candidate)
        if residual != 0:
            issues.append("ODE validator failed: A sin(x)+B cos(x) ne verifie pas y''+y=0.")
        else:
            report_parts.append("ODE y''+y=0 verifiee par substitution de A sin(x)+B cos(x).")

        if not _solution_mentions_sine_cosine_family(combined):
            issues.append("ODE validator failed: la solution generale A sin(x)+B cos(x) n'est pas clairement annoncee.")

    # Pattern 2: E = {f | f'(x)+f(pi/2-x)=0}.  Verify the final family B cos(x).
    if _contains_functional_ode_condition(prompt_normalized):
        candidate_e = B * sp.cos(x)
        residual_e = sp.simplify(sp.diff(candidate_e, x) + candidate_e.subs(x, sp.pi / 2 - x))
        if residual_e != 0:
            issues.append("ODE validator failed: B cos(x) ne verifie pas f'(x)+f(pi/2-x)=0.")
        else:
            report_parts.append("Condition de l'ensemble E verifiee pour B cos(x).")

        general = A * sp.sin(x) + B * sp.cos(x)
        membership_residual = sp.simplify(
            sp.diff(general, x) + general.subs(x, sp.pi / 2 - x)
        )
        # The residual must be proportional to A*cos(x); hence forcing it to be
        # zero for every x gives A=0 and the family B cos(x).
        if sp.simplify(membership_residual - 2 * A * sp.cos(x)) != 0:
            issues.append("ODE validator failed: la reduction de la condition E vers A=0 n'a pas ete confirmee.")
        if "cos" not in normalized:
            issues.append("ODE validator failed: la famille finale de E doit contenir des fonctions en cosinus.")

    return _build_check_result(
        name="ode",
        applicable=True,
        passed=not issues,
        issues=_deduplicate_preserving_order(issues),
        summary=" | ".join(report_parts) or "Controle symbolique d'equation differentielle execute.",
        requires_symbolic=True,
        symbolic_ran=True,
    )


def _contains_ode_y_second_plus_y_zero(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or "").lower())
    compact = compact.replace("\\", "")
    return "y''+y=0" in compact or "y''+1*y=0" in compact


def _contains_functional_ode_condition(normalized_prompt: str) -> bool:
    compact = normalized_prompt.replace(" ", "")
    return (
        ("f'(x)+f" in compact or "f(x)+f" in compact or "fprime(x)+f" in compact)
        and ("pi/2-x" in compact or "dfracpi2-x" in compact or "fracpi2-x" in compact)
        and "=0" in compact
    )


def _solution_mentions_sine_cosine_family(text: str) -> bool:
    normalized = _normalize_lookup(text)
    return (
        "sin" in normalized
        and "cos" in normalized
        and ("a" in normalized or "alpha" in normalized)
        and ("b" in normalized or "beta" in normalized)
    )


def _validate_inverse_function(record: dict[str, Any]) -> dict[str, Any]:
    """Validate claimed inverse functions and basic domain consistency."""
    prompt = str(record.get("prompt", "")).strip()
    solution = str(record.get("hidden_solution", "")).strip()
    combined = f"{prompt}\n{solution}\n{record.get('display_answer', '')}"
    normalized = _normalize_lookup(combined)
    applicable = any(token in normalized for token in ("reciproque", "inverse", "^-1"))
    if not applicable:
        return _build_check_result(name="inverse_function", applicable=False, passed=None)
    if sp is None:
        return _build_check_result(
            name="inverse_function",
            applicable=True,
            passed=False,
            issues=["Validation symbolique requise pour la fonction reciproque, mais SymPy est indisponible."],
            summary="SymPy indisponible pour le controle de reciproque.",
            requires_symbolic=True,
            symbolic_ran=False,
        )

    function_info = _extract_named_function_expression(prompt)
    inverse_claim = _extract_inverse_claim(combined)
    if not function_info or not inverse_claim:
        return _build_check_result(
            name="inverse_function",
            applicable=True,
            passed=False,
            issues=["Inverse-function validator failed: la fonction ou la reciproque annoncee n'est pas assez lisible."],
            summary="Controle de reciproque impossible.",
            requires_symbolic=True,
            symbolic_ran=False,
        )

    function_name, function_expr = function_info
    issues: list[str] = []
    try:
        x = sp.symbols("x")
        y = sp.symbols("y")
        original_expr = sp.sympify(_normalize_for_sympy(function_expr))
        domain_issue = _check_declared_domain_consistency(prompt, original_expr, x, function_name)
        if domain_issue:
            issues.append(domain_issue)

        claimed_inverse_expr = sp.sympify(_normalize_for_sympy(inverse_claim), locals={"x": x, "y": y}).subs({x: y})
        candidate_solutions = sp.solve(sp.Eq(y, original_expr), x)
        normalized_solutions = [
            sp.simplify(solution_expr).subs({x: y})
            for solution_expr in candidate_solutions
        ]
        if not any(sp.simplify(claimed_inverse_expr - solution_expr) == 0 for solution_expr in normalized_solutions):
            issues.append(
                f"Inverse-function validator failed: la reciproque annoncee {inverse_claim} ne correspond pas a la resolution symbolique de y={function_expr}."
            )
    except Exception:
        issues.append("Inverse-function validator failed: la resolution symbolique de la reciproque a echoue.")

    return _build_check_result(
        name="inverse_function",
        applicable=True,
        passed=not issues,
        issues=issues,
        summary="Controle local de la fonction reciproque et de son domaine.",
        requires_symbolic=True,
        symbolic_ran=True,
    )


def _validate_probability(
    record: dict[str, Any],
    display_answer: str,
    extracted_answer: str,
    answer_kind: str,
) -> dict[str, Any]:
    if not _looks_like_probability_context(record):
        return _build_check_result(name="probability", applicable=False, passed=None)

    issues: list[str] = []
    report_parts: list[str] = []
    prompt = str(record.get("prompt", ""))
    text = "\n".join(
        [
            prompt,
            str(record.get("hidden_solution", "")),
            str(record.get("solution_validation_llm_answer", "")),
        ]
    )

    parsed_assignments, unsupported_assignment_count = _extract_probability_assignments(prompt)
    if unsupported_assignment_count:
        issues.append(
            "Probability validator failed: certaines donnees de probabilite du sujet n'ont pas pu etre verifiees localement."
        )
    issues.extend(_validate_probability_assignment_consistency(parsed_assignments, prompt))

    binomial_problem = _extract_binomial_probability_problem(prompt)
    if binomial_problem:
        binomial_result = _validate_binomial_probability_problem(
            binomial_problem,
            display_answer=display_answer,
            extracted_answer=extracted_answer,
        )
        issues.extend(binomial_result["issues"])
        if binomial_result["summary"]:
            report_parts.append(binomial_result["summary"])

    for label, candidate in (("reponse attendue", display_answer), ("reponse finale", extracted_answer)):
        if not candidate:
            continue
        value = _evaluate_fraction_expression(candidate)
        if value is not None and not (Fraction(0, 1) <= value <= Fraction(1, 1)):
            issues.append(f"La {label} n'est pas une probabilite valide dans [0, 1].")

    for expression, claimed in _extract_probability_equations(text):
        expected = _evaluate_fraction_expression(expression)
        obtained = _evaluate_fraction_expression(claimed)
        if expected is None or obtained is None:
            continue
        report_parts.append(f"{expression} = {float(expected):.6g}")
        if expected != obtained:
            issues.append(
                f"Probability validator failed: {expression} = {float(expected):.6g}, pas {float(obtained):.6g}."
            )

    deterministic_ok, deterministic_issues, deterministic_metadata = validate_probability_exercise(record)
    if deterministic_metadata.get("applicable"):
        report_parts.append("Controle d'urne/local: " + json.dumps(deterministic_metadata, ensure_ascii=False))
    issues.extend(deterministic_issues)

    return _build_check_result(
        name="probability",
        applicable=True,
        passed=not issues and deterministic_ok,
        issues=issues,
        summary=" | ".join(report_parts) if report_parts else "Probabilites controlees localement.",
        requires_symbolic=True,
        symbolic_ran=True,
        metadata=deterministic_metadata,
    )


def _validate_bayes(record: dict[str, Any]) -> dict[str, Any]:
    ok, issues, metadata = validate_bayes_exercise(record)
    applicable = bool(metadata.get("applicable"))
    return _build_check_result(
        name="bayes",
        applicable=applicable,
        passed=ok if applicable else None,
        issues=issues,
        summary="Controle deterministe Bayes/probabilites conditionnelles.",
        requires_symbolic=applicable,
        symbolic_ran=applicable,
        metadata=metadata,
    )


def _validate_exponential_law(record: dict[str, Any]) -> dict[str, Any]:
    ok, issues, metadata = validate_exponential_law_exercise(record)
    applicable = bool(metadata.get("applicable"))
    return _build_check_result(
        name="exponential_law",
        applicable=applicable,
        passed=ok if applicable else None,
        issues=issues,
        summary="Controle deterministe de loi exponentielle.",
        requires_symbolic=applicable,
        symbolic_ran=applicable,
        metadata=metadata,
    )


def _validate_regression_deterministic(record: dict[str, Any]) -> dict[str, Any]:
    ok, issues, metadata = validate_regression_exercise(record)
    applicable = bool(metadata.get("applicable"))
    return _build_check_result(
        name="regression_deterministic",
        applicable=applicable,
        passed=ok if applicable else None,
        issues=issues,
        summary="Controle deterministe des donnees et calculs de regression.",
        requires_symbolic=applicable,
        symbolic_ran=applicable,
        metadata=metadata,
    )


def _validate_complex_numbers(record: dict[str, Any]) -> dict[str, Any]:
    ok, issues, metadata = validate_complex_number_exercise(record)
    return _build_check_result(
        name="complex_numbers",
        applicable=bool(metadata.get("applicable", True)),
        passed=ok,
        issues=issues,
        summary="Controle deterministe des nombres complexes.",
        requires_symbolic=True,
        symbolic_ran=True,
        metadata=metadata,
    )


def _validate_linear_systems(record: dict[str, Any]) -> dict[str, Any]:
    ok, issues, metadata = validate_linear_system_exercise(record)
    return _build_check_result(
        name="linear_systems",
        applicable=True,
        passed=ok,
        issues=issues,
        summary="Controle deterministe des matrices/systemes lineaires.",
        requires_symbolic=True,
        symbolic_ran=True,
        metadata=metadata,
    )


def _validate_graph_support(record: dict[str, Any]) -> dict[str, Any]:
    ok, issues, metadata = validate_graph_support(record)
    return _build_check_result(
        name="graph_support",
        applicable=bool(metadata.get("applicable")),
        passed=ok if metadata.get("applicable") else None,
        issues=issues,
        summary="Controle du support visuel ou de la liste d'aretes pour les graphes.",
        metadata=metadata,
    )


def _validate_question_coverage(record: dict[str, Any]) -> dict[str, Any]:
    ok, issues, metadata = validate_question_answer_coverage(record)
    return _build_check_result(
        name="question_coverage",
        applicable=True,
        passed=ok,
        issues=issues,
        summary="Controle question par question de la solution.",
        metadata=metadata,
    )


def _route_domain_checks(domain_key: str | None, checks: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep non-domain checks, but suppress unrelated domain validator outcomes."""
    domain_groups = {
        "complex_numbers": {"complex", "complex_numbers"},
        "linear_systems": {"linear_systems"},
        "regression": {"regression", "regression_numeric", "regression_deterministic", "pedagogical_completeness"},
        "bayes": {"bayes"},
        "finite_probability": {"probability"},
        "exponential_law": {"exponential_law"},
        "sequences": {"recurrence_relation", "sequence_numeric", "adjacent_sequences"},
        "ode": {"ode"},
        "graphs": {"graph_support", "graph_theory", "visual_support"},
    }
    all_domain_checks = set().union(*domain_groups.values())
    allowed = domain_groups.get(domain_key or "", set())
    routed = dict(checks)
    for name in all_domain_checks:
        if name in routed and name not in allowed:
            routed[name] = _build_check_result(name=name, applicable=False, passed=None)
    return routed


def _validate_complex(record: dict[str, Any]) -> dict[str, Any]:
    if sp is None:
        applicable = _looks_like_complex_context(record)
        return _build_check_result(
            name="complex",
            applicable=applicable,
            passed=not applicable,
            issues=["Validation symbolique requise pour les racines complexes, mais SymPy est indisponible."] if applicable else [],
            summary="SymPy indisponible pour le controle complexe." if applicable else "",
            requires_symbolic=applicable,
            symbolic_ran=False,
        )

    if not _looks_like_complex_context(record):
        return _build_check_result(name="complex", applicable=False, passed=None)

    prompt = str(record.get("prompt", ""))
    solution = str(record.get("hidden_solution", ""))
    combined = f"{prompt}\n{solution}"
    issues: list[str] = []
    z = sp.symbols("z")
    summaries: list[str] = []

    rotation_check = _validate_complex_rotation(prompt, solution)
    if rotation_check["applicable"]:
        issues.extend(rotation_check["issues"])
        if rotation_check["summary"]:
            summaries.append(rotation_check["summary"])

    roots = _extract_complex_roots(solution)
    if roots:
        polynomial = _extract_quadratic_expression(combined)
        if polynomial is not None:
            for label, root_expr in roots.items():
                try:
                    root_value = sp.sympify(_normalize_for_sympy(root_expr))
                except Exception:
                    issues.append(f"La racine {label} n'est pas interpretable : {root_expr}.")
                    continue
                if sp.simplify(polynomial.subs({z: root_value})) != 0:
                    issues.append(f"Complex validator failed: {label}={root_expr} ne verifie pas l'equation quadratique.")

        sum_target, product_target = _extract_complex_sum_product(combined)
        if sum_target is not None and product_target is not None and {"z1", "z2"} <= set(roots):
            try:
                z1_expr = sp.sympify(_normalize_for_sympy(roots["z1"]))
                z2_expr = sp.sympify(_normalize_for_sympy(roots["z2"]))
                expected_sum = sp.sympify(_normalize_for_sympy(sum_target))
                expected_product = sp.sympify(_normalize_for_sympy(product_target))
                if sp.simplify(z1_expr + z2_expr - expected_sum) != 0:
                    issues.append("Complex validator failed: z1 + z2 ne correspond pas a la somme attendue.")
                if sp.simplify(z1_expr * z2_expr - expected_product) != 0:
                    issues.append("Complex validator failed: z1 * z2 ne correspond pas au produit attendu.")
            except Exception:
                issues.append("Complex validator failed: le controle local des racines a echoue.")
        summaries.append("Controle des racines complexes effectue localement.")
    elif not rotation_check["applicable"]:
        return _build_check_result(name="complex", applicable=False, passed=None)

    return _build_check_result(
        name="complex",
        applicable=True,
        passed=not issues,
        issues=issues,
        summary=" | ".join(summaries) if summaries else "Controle complexe effectue localement.",
        requires_symbolic=True,
        symbolic_ran=True,
    )


def _validate_conics(record: dict[str, Any]) -> dict[str, Any]:
    """Validate simple geometric claims for the parabola y^2 = 4px."""
    prompt = str(record.get("prompt", "")).strip()
    solution = str(record.get("hidden_solution", "")).strip()
    combined = f"{prompt}\n{solution}"
    conic_match = re.search(r"y\s*\^?\s*2\s*=\s*4\s*([A-Za-z0-9_./+\-]+)\s*x", combined, flags=re.IGNORECASE)
    if not conic_match:
        return _build_check_result(name="conics", applicable=False, passed=None)

    issues: list[str] = []
    p_token = conic_match.group(1).strip()
    directrix_match = re.search(r"directrice[^.\n:=]*[:=]?\s*x\s*=\s*([^\n;,.]+)", combined, flags=re.IGNORECASE)
    if directrix_match:
        claimed_directrix = _normalize_for_sympy(directrix_match.group(1))
        expected_directrix = _normalize_for_sympy(f"-({p_token})")
        if claimed_directrix != expected_directrix:
            if not _answers_equivalent(claimed_directrix, expected_directrix, "expression"):
                issues.append(
                    f"Conics validator failed: la directrice devrait etre x = -{p_token}, pas x = {directrix_match.group(1).strip()}."
                )

    vertex_match = re.search(
        r"(?:sommet|vertex)[^(\n]*\(\s*([-+]?\d+(?:[.,]\d+)?)\s*[;,]\s*([-+]?\d+(?:[.,]\d+)?)\s*\)",
        combined,
        flags=re.IGNORECASE,
    )
    if vertex_match and (abs(_to_float(vertex_match.group(1))) > 1e-9 or abs(_to_float(vertex_match.group(2))) > 1e-9):
        issues.append("Conics validator failed: le sommet de y^2 = 4px doit etre O(0,0).")

    tangent_match = re.search(
        r"tangente[^.\n:=]*[:=]?\s*([^\n]+)",
        combined,
        flags=re.IGNORECASE,
    )
    if tangent_match:
        tangent_text = tangent_match.group(1).strip().rstrip(".")
        tangent_normalized = _normalize_lookup(tangent_text)
        if "x" not in tangent_normalized:
            issues.append(
                f"Conics validator failed: l'equation de tangente '{tangent_text}' ne contient aucune variable x."
            )

    return _build_check_result(
        name="conics",
        applicable=True,
        passed=not issues,
        issues=issues,
        summary="Controle local des invariants de la parabole y^2 = 4px.",
        requires_symbolic=True,
        symbolic_ran=True,
    )


def _validate_regression_line(record: dict[str, Any]) -> dict[str, Any]:
    """Validate line equations built from reported points such as G and G1."""
    prompt = str(record.get("prompt", ""))
    solution = str(record.get("hidden_solution", ""))
    combined = f"{prompt}\n{solution}"
    normalized = _normalize_lookup(combined)
    applicable = "regression" in normalized or "(gg1)" in normalized or "droite gg1" in normalized
    if not applicable:
        return _build_check_result(name="regression", applicable=False, passed=None)

    point_g = _extract_named_point(combined, "G")
    point_g1 = _extract_named_point(combined, "G1")
    line_equation = _extract_line_equation(solution)
    issues: list[str] = []
    if point_g and point_g1 and line_equation:
        if not _line_contains_point(line_equation, *point_g) or not _line_contains_point(line_equation, *point_g1):
            issues.append(
                "Regression validator failed: l'equation annoncee pour la droite (GG1) ne passe pas par G et G1."
            )
    elif "gg1" in normalized:
        issues.append("Regression validator failed: impossible de verifier localement l'equation de la droite (GG1).")

    return _build_check_result(
        name="regression",
        applicable=True,
        passed=not issues,
        issues=issues,
        summary="Controle local de la droite passant par G et G1.",
        requires_symbolic=True,
        symbolic_ran=True,
    )


def _validate_pedagogical_completeness(record: dict[str, Any]) -> dict[str, Any]:
    """Ensure regression exercises are complete without confusing classical regression with Mayer.

    Two pedagogical families are accepted:
    - classical least-squares/correlation: r, regression line, optional estimation;
    - Mayer adjustment: point moyen G, grouped means G1/G2, and adjustment line.
    """
    prompt = str(record.get("prompt", "")).strip()
    title = str(record.get("title", "")).strip()
    objective = str(record.get("learning_objective", "")).strip()
    context = _normalize_lookup(" ".join([title, objective, prompt]))
    prompt_norm = _normalize_lookup(prompt)
    regression_markers = (
        "droite de mayer",
        "ajuster cette serie double",
        "ajuster cette serie",
        "droite de regression",
        "regression",
        "correlation",
        "coefficient de correlation",
    )
    applicable = any(marker in context for marker in regression_markers)
    if not applicable:
        return _build_check_result(name="pedagogical_completeness", applicable=False, passed=None)

    issues: list[str] = []
    has_total_mean_task = _has_total_mean_task(prompt)
    has_grouped_means_task = _has_grouped_means_task(prompt)
    has_line_task = _has_adjustment_line_task(prompt)
    has_correlation_task = "correlation" in prompt_norm or "coefficient de correlation" in prompt_norm
    has_estimation_task = any(token in prompt_norm for token in ("estimer", "prevoir", "prediction", "determiner a partir de", "depassera"))
    asks_mayer = _asks_mayer_method(" ".join([title, objective, prompt]))

    if asks_mayer:
        if not has_total_mean_task:
            issues.append(
                "Pedagogical-completeness validator failed: l'enonce de Mayer ne demande pas le calcul du point moyen G."
            )
        if not has_grouped_means_task:
            issues.append(
                "Pedagogical-completeness validator failed: l'enonce de Mayer ne demande pas les points moyens groupes G1/G2 ou les moyennes partielles."
            )
        if not has_line_task:
            issues.append(
                "Pedagogical-completeness validator failed: l'enonce de Mayer ne demande pas l'equation de la droite d'ajustement."
            )
    else:
        # Classical regression exercises do not have to ask for G, G1 or G2.
        if not (has_correlation_task or has_line_task):
            issues.append(
                "Pedagogical-completeness validator failed: l'enonce de regression classique doit demander un coefficient de correlation ou une droite de regression."
            )
        if "regression" in context and not has_line_task:
            issues.append(
                "Pedagogical-completeness validator failed: l'enonce annonce une regression mais ne demande pas l'equation de la droite de regression."
            )
        if has_estimation_task and not has_line_task:
            issues.append(
                "Pedagogical-completeness validator failed: l'enonce demande une estimation sans demander ou fournir une droite d'ajustement."
            )

    table_issue = _check_two_variable_table_caption(record.get("table_data"))
    if table_issue:
        issues.append(table_issue)

    return _build_check_result(
        name="pedagogical_completeness",
        applicable=True,
        passed=not issues,
        issues=issues,
        summary="Controle de completude pedagogique pour regression classique ou methode de Mayer.",
        requires_symbolic=False,
        symbolic_ran=False,
    )


def _validate_regression_numeric(record: dict[str, Any]) -> dict[str, Any]:
    """Recompute r, the regression line and common estimates from attached table_data."""
    prompt = str(record.get("prompt", ""))
    solution = str(record.get("hidden_solution", ""))
    display_answer = str(record.get("display_answer", ""))
    combined = f"{prompt}\n{solution}\n{display_answer}"
    normalized = _normalize_lookup(combined)
    applicable = any(token in normalized for token in ("regression", "correlation", "droite d ajustement", "droite de mayer"))
    if not applicable:
        return _build_check_result(name="regression_numeric", applicable=False, passed=None)

    table_info = _extract_regression_table_xy(record.get("table_data"), prompt)
    if not table_info:
        # If no table is attached, another support validator will handle missing supports.
        return _build_check_result(name="regression_numeric", applicable=False, passed=None)

    x_values, y_values, year_offset = table_info
    if len(x_values) < 3 or len(x_values) != len(y_values):
        return _build_check_result(name="regression_numeric", applicable=False, passed=None)

    expected = _compute_regression_values(x_values, y_values)
    if expected is None:
        return _build_check_result(name="regression_numeric", applicable=False, passed=None)

    issues: list[str] = []
    claimed_r = _extract_claimed_correlation(combined)
    if claimed_r is not None and abs(abs(claimed_r) - abs(expected["r"])) > 0.025:
        issues.append(
            "Regression numeric validator failed: le coefficient de correlation annonce "
            f"({claimed_r:.4g}) ne correspond pas au tableau ({expected['r']:.4g})."
        )

    line = _extract_regression_equation(combined)
    if line is not None:
        claimed_slope, claimed_intercept = line
        if abs(claimed_slope - expected["slope"]) > max(0.25, 0.015 * abs(expected["slope"])):
            issues.append(
                "Regression numeric validator failed: la pente de la droite de regression annoncee "
                f"({claimed_slope:.4g}) ne correspond pas au tableau ({expected['slope']:.4g})."
            )
        if abs(claimed_intercept - expected["intercept"]) > max(2.0, 0.01 * abs(expected["intercept"])):
            issues.append(
                "Regression numeric validator failed: l'ordonnee a l'origine annoncee "
                f"({claimed_intercept:.4g}) ne correspond pas au tableau ({expected['intercept']:.4g})."
            )

    predicted_anchors = _extract_years_asked_for_estimation(prompt)
    for year in predicted_anchors[:3]:
        if year_offset is None:
            continue
        rank = year - year_offset
        predicted = expected["slope"] * rank + expected["intercept"]
        if _has_number_near_anchor(solution + " " + display_answer, str(year), predicted, tolerance=max(3.0, abs(predicted) * 0.005)):
            continue
        issues.append(
            "Regression numeric validator failed: l'estimation pour "
            f"{year} devrait etre proche de {predicted:.2f} d'apres le tableau et la droite recalculee."
        )

    threshold = _extract_threshold_value(prompt)
    if threshold is not None and expected["slope"] > 0 and year_offset is not None:
        raw_rank = (threshold - expected["intercept"]) / expected["slope"]
        threshold_rank = math.floor(raw_rank) + 1
        threshold_year = year_offset + threshold_rank
        # Only enforce the year when the solution explicitly discusses the threshold.
        if "depass" in normalized or "seuil" in normalized:
            years_in_solution = {int(value) for value in re.findall(r"(20\d{2}|19\d{2})", solution + " " + display_answer)}
            if years_in_solution and threshold_year not in years_in_solution:
                issues.append(
                    "Regression numeric validator failed: l'annee de depassement du seuil "
                    f"{threshold:g} devrait etre {threshold_year} avec les donnees du tableau."
                )

    return _build_check_result(
        name="regression_numeric",
        applicable=True,
        passed=not issues,
        issues=issues,
        summary="Controle numerique local de la regression a partir du tableau fourni.",
        requires_symbolic=True,
        symbolic_ran=True,
    )


def _validate_integral_area(
    record: dict[str, Any],
    display_answer: str,
    extracted_answer: str,
    answer_kind: str,
) -> dict[str, Any]:
    prompt = str(record.get("prompt", ""))
    normalized_prompt = _normalize_lookup(prompt)
    applicable = any(token in normalized_prompt for token in ("aire", "surface", "partie hachuree", "partie hachurée"))
    if not applicable:
        return _build_check_result(name="integral_area", applicable=False, passed=None)

    issues: list[str] = []
    for label, candidate in (("reponse attendue", display_answer), ("reponse finale", extracted_answer)):
        if not candidate or answer_kind not in {"numeric", "expression"}:
            continue
        value = _sympy_value_if_possible(candidate)
        if value is None:
            continue
        try:
            is_negative = bool(value.is_negative)
            if value.is_negative is None:
                is_negative = float(value.evalf()) < 0
        except Exception:
            is_negative = False
        if is_negative:
            issues.append(f"Area validator failed: la {label} est negative alors qu'une aire doit etre positive.")

    return _build_check_result(
        name="integral_area",
        applicable=True,
        passed=not issues,
        issues=issues,
        summary="Controle du signe de l'aire effectue localement.",
        requires_symbolic=sp is not None,
        symbolic_ran=sp is not None,
    )


def _validate_domain(record: dict[str, Any]) -> dict[str, Any]:
    prompt = str(record.get("prompt", "")).strip()
    normalized_prompt = _normalize_lookup(prompt)
    positive_domain_patterns = [r"sur\s*0\s*,\s*\+?\s*(?:inf|oo)", r"sur\s*0\s*;\s*\+?\s*(?:inf|oo)"]
    uses_negated_argument = bool(re.search(r"\b[a-z]\s*\(\s*-\s*x\s*\)", normalized_prompt))
    positive_only_domain = any(re.search(pattern, normalized_prompt) for pattern in positive_domain_patterns)
    if not (uses_negated_argument and positive_only_domain):
        return _build_check_result(name="domain", applicable=False, passed=None)
    return _build_check_result(
        name="domain",
        applicable=True,
        passed=False,
        issues=["Domain validator failed: l'enonce utilise f(-x) alors que la fonction est definie sur un domaine positif."],
        summary="Controle de domaine negatif pour f(-x).",
        requires_symbolic=False,
        symbolic_ran=False,
    )


def _validate_visual_support(record: dict[str, Any]) -> dict[str, Any]:
    prompt = str(record.get("prompt", ""))
    normalized_prompt = _normalize_lookup(prompt)
    needs_chart = any(re.search(pattern, normalized_prompt) for pattern in VISUAL_CHART_PATTERNS)
    needs_table = any(re.search(pattern, normalized_prompt) for pattern in VISUAL_TABLE_PATTERNS)
    needs_graph = "graphe" in normalized_prompt
    if not needs_chart and not needs_table and not needs_graph:
        return _build_check_result(name="visual_support", applicable=False, passed=None)

    chart_data = record.get("chart_data")
    table_data = record.get("table_data")
    graph_data = record.get("graph_data")
    issues: list[str] = []
    if needs_chart and chart_data is None and table_data is None and graph_data is None:
        issues.append("Visual-support validator failed: l'enonce demande un support graphique, mais aucun chart_data/table_data/graph_data n'est fourni.")
    if needs_table and table_data is None:
        issues.append("Visual-support validator failed: l'enonce annonce un tableau fourni, mais table_data est absent.")
    if needs_graph and chart_data is None and table_data is None and graph_data is None:
        issues.append("Visual-support validator failed: l'enonce exige un graphe ou une figure, mais aucun support exploitable n'est fourni.")

    semantic_issues = _check_visual_semantic_consistency(prompt, chart_data, graph_data)
    issues.extend(semantic_issues)
    support_ready = bool(record.get("support_ready", False))
    if (needs_chart or needs_table or needs_graph) and not support_ready:
        issues.append("Visual-support validator failed: support_ready est faux alors que le support est requis.")

    return _build_check_result(
        name="visual_support",
        applicable=True,
        passed=not issues,
        issues=issues,
        summary="Presence et coherence du support visuel controlees localement.",
        requires_symbolic=False,
        symbolic_ran=False,
    )


def _validate_graph_theory(record: dict[str, Any]) -> dict[str, Any]:
    """Check simple Eulerian-chain reasoning from degree parity."""
    prompt = str(record.get("prompt", ""))
    solution = str(record.get("hidden_solution", ""))
    combined = f"{prompt}\n{solution}"
    normalized = _normalize_lookup(combined)
    if "euler" not in normalized:
        return _build_check_result(name="graph_theory", applicable=False, passed=None)

    degrees = _extract_degree_sequence(combined)
    if not degrees:
        return _build_check_result(name="graph_theory", applicable=False, passed=None)

    odd_count = sum(1 for value in degrees if value % 2 == 1)
    issues: list[str] = []
    if "tous les degres sont pairs" in normalized or "all degrees are even" in normalized:
        if odd_count != 0:
            issues.append(
                f"Graph validator failed: la solution affirme que tous les degres sont pairs alors que {odd_count} sommets sont de degre impair."
            )
    if odd_count == 2 and any(
        token in normalized
        for token in (
            "pas de chaine eulerienne",
            "n existe pas de chaine eulerienne",
            "aucune chaine eulerienne",
        )
    ):
        issues.append(
            "Graph validator failed: avec exactement deux sommets impairs, une chaine eulerienne existe."
        )
    if odd_count not in {0, 2} and any(
        token in normalized
        for token in (
            "une chaine eulerienne existe",
            "admet une chaine eulerienne",
            "il existe une chaine eulerienne",
        )
    ):
        issues.append(
            f"Graph validator failed: le graphe a {odd_count} sommets impairs, donc il n'admet pas de chaine eulerienne."
        )

    return _build_check_result(
        name="graph_theory",
        applicable=True,
        passed=not issues,
        issues=issues,
        summary=f"Parite des degres controlee localement (sequence {degrees}).",
        requires_symbolic=False,
        symbolic_ran=False,
    )


def _validate_recurrence_relations(record: dict[str, Any]) -> dict[str, Any]:
    """Validate claimed recurrence equalities and difference implications."""
    prompt = str(record.get("prompt", ""))
    solution = str(record.get("hidden_solution", ""))
    normalized = _normalize_lookup(f"{prompt}\n{solution}")
    raw_lower = f"{prompt}\n{solution}".lower()
    if not _looks_like_recurrence_defined_sequences(prompt, solution):
        return _build_check_result(name="recurrence_relation", applicable=False, passed=None)
    if sp is None:
        return _build_check_result(
            name="recurrence_relation",
            applicable=True,
            passed=False,
            issues=["Validation symbolique requise pour la recurrence, mais SymPy est indisponible."],
            summary="SymPy indisponible pour le controle de recurrence.",
            requires_symbolic=True,
            symbolic_ran=False,
        )

    definitions = _extract_recurrence_definitions(prompt)
    if not definitions:
        return _build_check_result(name="recurrence_relation", applicable=False, passed=None)

    sequence_names = [name for name in ("a", "b", "u", "v") if name in definitions]
    if not sequence_names:
        return _build_check_result(name="recurrence_relation", applicable=False, passed=None)

    locals_map = {f"{name}_n": sp.symbols(f"{name}_n") for name in sequence_names}
    issues: list[str] = []
    try:
        next_terms = {
            name: sp.sympify(_normalize_recurrence_expression(definitions.get(name, "")), locals=locals_map)
            for name in sequence_names
        }
    except Exception:
        return _build_check_result(
            name="recurrence_relation",
            applicable=True,
            passed=False,
            issues=["Recurrence validator failed: les relations de recurrence du prompt ne sont pas interpretables."],
            summary="Echec du calcul local de recurrence.",
            requires_symbolic=True,
            symbolic_ran=False,
        )

    primary_name = sequence_names[0]
    secondary_name = sequence_names[1] if len(sequence_names) >= 2 else ""
    expected_gap = None
    if secondary_name:
        primary_symbol = locals_map[f"{primary_name}_n"]
        secondary_symbol = locals_map[f"{secondary_name}_n"]
        expected_gap = sp.simplify(next_terms[secondary_name] - next_terms[primary_name])
        if _solution_mentions_recurrence_inequality(f"{prompt}\n{solution}", primary_name, secondary_name):
            gap_dependence = _analyze_recurrence_gap_dependence(
                expected_gap,
                primary_symbol,
                secondary_symbol,
                primary_name,
                secondary_name,
            )
            if not gap_dependence["passed"]:
                issues.append(gap_dependence["issue"])

        claimed_gap = _extract_claimed_gap_expression(solution, primary_name, secondary_name)
        if claimed_gap:
            try:
                claimed_gap_expr = sp.sympify(_normalize_recurrence_expression(claimed_gap), locals=locals_map)
                if sp.simplify(claimed_gap_expr - expected_gap) != 0:
                    issues.append(
                        f"Recurrence validator failed: {secondary_name}_(n+1)-{primary_name}_(n+1) devrait valoir {sp.sstr(expected_gap)}, pas {claimed_gap}."
                    )
            except Exception:
                issues.append(
                    f"Recurrence validator failed: l'expression annoncee pour {secondary_name}_(n+1)-{primary_name}_(n+1) n'est pas interpretable."
                )

        false_comparison_issue = _detect_false_comparison_to_next_term(
            solution,
            sequence_name=secondary_name,
            expected_expr=next_terms[secondary_name],
            locals_map=locals_map,
        )
        if false_comparison_issue:
            issues.append(false_comparison_issue)

    for sequence_name, expected_expr in next_terms.items():
        claimed_expr_text = _extract_claimed_recurrence_expression(solution, sequence_name)
        if not claimed_expr_text:
            continue
        try:
            claimed_expr = sp.sympify(_normalize_recurrence_expression(claimed_expr_text), locals=locals_map)
            if sp.simplify(claimed_expr - expected_expr) != 0:
                issues.append(
                    f"Recurrence validator failed: {sequence_name}_(n+1) devrait valoir {sp.sstr(expected_expr)}, pas {claimed_expr_text}."
                )
        except Exception:
            issues.append(
                f"Recurrence validator failed: l'expression annoncee pour {sequence_name}_(n+1) n'est pas interpretable."
            )

    return _build_check_result(
        name="recurrence_relation",
        applicable=True,
        passed=not issues,
        issues=issues,
        summary=(
            f"Controle local des relations de recurrence, ecart attendu {sp.sstr(expected_gap)}."
            if expected_gap is not None
            else "Controle local des relations de recurrence."
        ),
        requires_symbolic=True,
        symbolic_ran=True,
    )


def _validate_adjacent_sequences(record: dict[str, Any]) -> dict[str, Any]:
    """Require the classical distance-to-zero argument for adjacent sequences."""
    prompt = str(record.get("prompt", ""))
    solution = str(record.get("hidden_solution", ""))
    combined = f"{prompt}\n{solution}"
    normalized = _normalize_lookup(combined)
    if "adjacente" not in normalized:
        return _build_check_result(name="adjacent_sequences", applicable=False, passed=None)

    solution_normalized = _normalize_lookup(solution)
    raw_solution_lower = solution.lower()
    has_gap_expression = bool(
        re.search(
            r"(v_n\s*-\s*u_n|u_n\s*-\s*v_n|b_n\s*-\s*a_n|a_n\s*-\s*b_n|\bdifference\b|\bdistance\b|\becart\b)",
            raw_solution_lower,
        )
    )
    has_zero_limit = bool(
        re.search(r"(?:->|tend(?:re)?\s+vers|converge\s+vers|lim(?:ite)?)[^.\n]{0,25}0", solution_normalized)
    )
    if has_gap_expression and has_zero_limit:
        return _build_check_result(
            name="adjacent_sequences",
            applicable=True,
            passed=True,
            summary="La preuve de suites adjacentes contient bien un ecart qui tend vers 0.",
        )
    return _build_check_result(
        name="adjacent_sequences",
        applicable=True,
        passed=False,
        issues=[
            "Adjacent-sequences validator failed: la solution ne montre pas explicitement que la distance b_n-a_n (ou v_n-u_n) tend vers 0."
        ],
        summary="Preuve des suites adjacentes incomplete.",
        requires_symbolic=False,
        symbolic_ran=False,
    )


def _validate_numeric_sequence_patterns(record: dict[str, Any]) -> dict[str, Any]:
    """Validate common Bac suite-numerique patterns deterministically when recognized."""
    prompt = repair_math_text_locally(str(record.get("prompt", "")))
    solution = repair_math_text_locally(str(record.get("hidden_solution", "")))
    combined = f"{prompt}\n{solution}"
    normalized = _normalize_lookup(combined)
    suite_context = any(
        token in _normalize_lookup(
            " ".join(
                [
                    str(record.get("title", "")),
                    str(record.get("topic", "")),
                    str(record.get("subtopic", "")),
                    prompt,
                ]
            )
        )
        for token in ("suite", "suites numeriques")
    )
    if not suite_context and "u_(n+1)" not in combined.lower() and "u_{n+1}" not in combined.lower():
        return _build_check_result(name="sequence_numeric", applicable=False, passed=None)
    if sp is None:
        return _build_check_result(
            name="sequence_numeric",
            applicable=True,
            passed=False,
            issues=["Validation symbolique requise pour les suites numeriques, mais SymPy est indisponible."],
            summary="SymPy indisponible pour le controle de suite numerique.",
            requires_symbolic=True,
            symbolic_ran=False,
        )

    initial_match = re.search(r"[uU]_0\s*=\s*1\b", prompt)
    recurrence_match = re.search(r"[uU]_\{?(?:n|k)\+1\}?\s*=\s*e\^\{-?n\}\s*[uU]_\{?(?:n|k)\}?", prompt)
    if suite_context and not recurrence_match:
        return _build_check_result(
            name="sequence_numeric",
            applicable=True,
            passed=False,
            issues=["Suite-numerique validator failed: la recurrence de la suite est absente ou non parsable dans l'enonce."],
            summary="L'enonce annonce une suite numerique sans recurrence exploitable.",
            requires_symbolic=True,
            symbolic_ran=False,
        )
    if not initial_match or not recurrence_match:
        return _build_check_result(name="sequence_numeric", applicable=False, passed=None)

    issues: list[str] = []
    n = sp.symbols("n", integer=True, nonnegative=True)
    expected_u2 = sp.exp(-1)
    expected_v_diff = -n
    expected_vn = -n * (n - 1) / 2
    expected_un = sp.exp(expected_vn)

    if "u_2" in combined.lower():
        claimed_u2 = _extract_sequence_value(solution, "u", 2)
        if claimed_u2:
            try:
                claimed_u2_expr = sp.sympify(_normalize_recurrence_expression(claimed_u2))
                if sp.simplify(claimed_u2_expr - expected_u2) != 0:
                    issues.append("Suite-numerique validator failed: U_2 devrait valoir e^{-1}.")
            except Exception:
                issues.append("Suite-numerique validator failed: la valeur annoncee pour U_2 n'est pas interpretable.")

    if "v_(n+1)-v_n" in combined.lower() or "v_{n+1}-v_n" in combined.lower():
        claimed_v_diff = _extract_sequence_difference(solution, "v")
        if claimed_v_diff:
            try:
                claimed_v_diff_expr = sp.sympify(_normalize_recurrence_expression(claimed_v_diff), locals={"n": n})
                if sp.simplify(claimed_v_diff_expr - expected_v_diff) != 0:
                    issues.append("Suite-numerique validator failed: V_(n+1)-V_n devrait valoir -n.")
            except Exception:
                issues.append("Suite-numerique validator failed: la relation sur V_(n+1)-V_n n'est pas interpretable.")

    claimed_vn = _extract_general_sequence_expression(solution, "v")
    if claimed_vn:
        try:
            claimed_vn_expr = sp.sympify(_normalize_recurrence_expression(claimed_vn), locals={"n": n})
            if sp.simplify(claimed_vn_expr - expected_vn) != 0:
                issues.append("Suite-numerique validator failed: la forme fermee de V_n est incorrecte.")
        except Exception:
            issues.append("Suite-numerique validator failed: la forme fermee annoncee pour V_n n'est pas interpretable.")

    claimed_un = _extract_general_sequence_expression(solution, "u")
    if claimed_un:
        try:
            claimed_un_expr = sp.sympify(_normalize_recurrence_expression(claimed_un), locals={"n": n})
            if sp.simplify(claimed_un_expr - expected_un) != 0:
                issues.append("Suite-numerique validator failed: la forme fermee de U_n est incorrecte.")
        except Exception:
            issues.append("Suite-numerique validator failed: la forme fermee annoncee pour U_n n'est pas interpretable.")

    if "lim" in normalized and "u_n" in combined.lower():
        if "0" not in _normalize_lookup(solution):
            issues.append("Suite-numerique validator failed: la limite attendue de U_n est 0.")

    return _build_check_result(
        name="sequence_numeric",
        applicable=True,
        passed=not issues,
        issues=issues,
        summary="Controle deterministe de la suite U_(n+1)=e^{-n}U_n avec U_0=1.",
        requires_symbolic=True,
        symbolic_ran=True,
    )


def _check_visual_semantic_consistency(prompt: str, chart_data: Any, graph_data: Any) -> list[str]:
    if not isinstance(chart_data, dict) and not isinstance(graph_data, dict):
        return []

    issues: list[str] = []
    chart_like = chart_data if isinstance(chart_data, dict) else graph_data
    point = _extract_named_point(prompt, "A")
    tangent = _extract_tangent_equation(prompt)
    if point:
        x0, y0 = point
        if not _chart_contains_point(chart_like, x0, y0):
            issues.append(
                f"Visual-support validator failed: le graphique fourni ne contient pas le point A({x0},{y0}) annonce."
            )
        if tangent and chart_like.get("series"):
            tangent_series = [
                series for series in chart_like.get("series", [])
                if "tangent" in _normalize_lookup(series.get("name", ""))
            ]
            for series in tangent_series:
                if not _series_matches_line(series, tangent["slope"], tangent["intercept"]):
                    issues.append("Visual-support validator failed: la serie de tangente ne correspond pas a l'equation annoncee.")
                    break
    if "branche parabolique" in _normalize_lookup(prompt) and _chart_is_almost_linear(chart_like):
        issues.append("Visual-support validator failed: la branche annoncee comme parabolique est representee par un support quasi lineaire.")
    return issues


def _extract_named_point(prompt: str, point_name: str) -> tuple[float, float] | None:
    match = re.search(
        rf"\b{re.escape(point_name)}\s*\(\s*([-+]?\d+(?:[.,]\d+)?)\s*[;,]\s*([-+]?\d+(?:[.,]\d+)?)\s*\)",
        prompt,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return (_to_float(match.group(1)), _to_float(match.group(2)))


def _extract_tangent_equation(prompt: str) -> dict[str, float] | None:
    match = re.search(
        r"y\s*=\s*([-+]?\d+(?:[.,]\d+)?)\s*x\s*([+-]\s*\d+(?:[.,]\d+)?)",
        prompt.replace("−", "-"),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return {
        "slope": _to_float(match.group(1)),
        "intercept": _to_float(match.group(2).replace(" ", "")),
    }


def _chart_contains_point(chart_data: dict[str, Any], x0: float, y0: float, tolerance: float = 0.35) -> bool:
    for series in chart_data.get("series", []):
        for x_value, y_value in zip(series.get("x", []), series.get("y", [])):
            try:
                if abs(float(x_value) - x0) <= tolerance and abs(float(y_value) - y0) <= tolerance:
                    return True
            except Exception:
                continue
    return False


def _series_matches_line(series: dict[str, Any], slope: float, intercept: float, tolerance: float = 0.35) -> bool:
    pairs = list(zip(series.get("x", []), series.get("y", [])))
    if len(pairs) < 2:
        return False
    for x_value, y_value in pairs[:3]:
        try:
            expected = slope * float(x_value) + intercept
            if abs(float(y_value) - expected) > tolerance:
                return False
        except Exception:
            return False
    return True


def _chart_is_almost_linear(chart_data: dict[str, Any], tolerance: float = 0.2) -> bool:
    series = chart_data.get("series", [])
    if not series:
        return False
    numeric_y: list[float] = []
    for value in series[0].get("y", []):
        try:
            numeric_y.append(float(value))
        except Exception:
            return False
    if len(numeric_y) < 4:
        return False
    first_diffs = [numeric_y[index + 1] - numeric_y[index] for index in range(len(numeric_y) - 1)]
    second_diffs = [first_diffs[index + 1] - first_diffs[index] for index in range(len(first_diffs) - 1)]
    return all(abs(value) <= tolerance for value in second_diffs)


def _looks_like_derivative_context(record: dict[str, Any]) -> bool:
    return any(
        token in _normalize_lookup(" ".join([record.get("title", ""), record.get("topic", ""), record.get("subtopic", ""), record.get("prompt", "")]))
        for token in ("derivee", "derivation", "deriver", "f'(x)")
    )


def _looks_like_probability_context(record: dict[str, Any]) -> bool:
    context = _normalize_lookup(" ".join([record.get("title", ""), record.get("topic", ""), record.get("subtopic", ""), record.get("prompt", ""), record.get("hidden_solution", "")]))
    return (
        "probabilite" in context
        or "p(" in context
        or "binomiale" in context
        or "bernoulli" in context
        or _extract_binomial_probability_problem(str(record.get("prompt", ""))) is not None
    )


def _looks_like_complex_context(record: dict[str, Any]) -> bool:
    context = _normalize_lookup(" ".join([record.get("title", ""), record.get("topic", ""), record.get("subtopic", ""), record.get("prompt", ""), record.get("hidden_solution", "")]))
    return (
        "complex" in context
        or "affixe" in context
        or "rotation" in context
        or " z1 " in f" {context} "
        or " z2 " in f" {context} "
    )


def _exercise_forces_symbolic(record: dict[str, Any]) -> bool:
    context = _normalize_lookup(
        " ".join(
            [
                str(record.get("title", "")),
                str(record.get("topic", "")),
                str(record.get("subtopic", "")),
                str(record.get("prompt", "")),
            ]
        )
    )
    if _looks_like_recurrence_defined_sequences(
        str(record.get("prompt", "")),
        str(record.get("hidden_solution", "")),
    ):
        return True
    return any(
        token in context
        for token in (
            "appart",
            "reciproque",
            "inverse",
            "recurrence",
            "equation differentielle",
            "differentielles",
            "regression",
            "probabilite",
            "complex",
            "conique",
        )
    )


def _extract_named_function_expression(text: str) -> tuple[str, str] | None:
    match = re.search(r"\b([a-z])\s*\(\s*x\s*\)\s*=\s*([^\n]+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    function_name = match.group(1)
    expression = re.split(
        r"(?:[.;]|(?:\b(calculer|determiner|donner|montrer|etudier|trouver|verifier)\b))",
        match.group(2).strip(),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    return function_name, expression


def _extract_membership_condition(prompt: str) -> str:
    patterns = [
        r"(?:condition|verifie|v[ée]rifie|defini par|telle que)\s*[:=]?\s*([a-z]'\(x\)[^\n=]*=[^\n.]+)",
        r"\{[^{}]*\|\s*([a-z]'\(x\)[^\n=]*=[^\n}]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _substitute_function_condition(condition_side: str, function_name: str, function_expr: Any, x_symbol: Any) -> Any:
    side = str(condition_side or "")
    derivative_pattern = rf"{re.escape(function_name)}'\(x\)"
    side = re.sub(derivative_pattern, f"({sp.sstr(sp.diff(function_expr, x_symbol))})", side)

    def _replace_function_call(match: re.Match[str]) -> str:
        argument = match.group(1).strip()
        argument_expr = sp.sympify(_normalize_for_sympy(argument))
        substituted = sp.simplify(function_expr.subs({x_symbol: argument_expr}))
        return f"({sp.sstr(substituted)})"

    call_pattern = rf"{re.escape(function_name)}\(([^()]+)\)"
    side = re.sub(call_pattern, _replace_function_call, side)
    return sp.sympify(_normalize_for_sympy(side))


def _extract_inverse_claim(text: str) -> str:
    patterns = [
        r"[a-z]\s*\^\s*-?1\s*\(\s*[xy]\s*\)\s*=\s*([^\n]+)",
        r"[a-z]\s*\^\{\s*-?1\s*\}\s*\(\s*[xy]\s*\)\s*=\s*([^\n]+)",
        r"reciproque[^:=]*[:=]\s*([^\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return re.split(r"[.;]", match.group(1).strip(), maxsplit=1)[0].strip()
    return ""


def _check_declared_domain_consistency(prompt: str, original_expr: Any, x_symbol: Any, function_name: str) -> str:
    match = re.search(rf"{re.escape(function_name)}\s+.*?sur\s*([\[\]])\s*([-+]?\d+(?:[.,]\d+)?)\s*[,;]\s*([-+]?\d+(?:[.,]\d+)?|\+?inf)\s*([\[\]])", prompt, flags=re.IGNORECASE)
    if not match:
        return ""
    lower_closed = match.group(1) == "["
    lower_value = _to_float(match.group(2))
    if not lower_closed:
        return ""
    try:
        evaluated = sp.simplify(original_expr.subs({x_symbol: lower_value}))
    except Exception:
        return ""
    if evaluated.has(sp.zoo, sp.oo, -sp.oo) or evaluated.is_real is False:
        return (
            f"Inverse-function validator failed: la fonction {function_name} n'est pas definie en x={lower_value} alors que ce point appartient au domaine annonce."
        )
    if evaluated.is_finite is False:
        return (
            f"Inverse-function validator failed: la fonction {function_name} n'est pas finie en x={lower_value} alors que ce point appartient au domaine annonce."
        )
    return ""


def _extract_degree_sequence(text: str) -> list[int]:
    for match in re.finditer(r"\b\d+(?:\s*[,;]\s*\d+){3,}\b", text):
        values = [int(chunk.strip()) for chunk in re.split(r"[,;]", match.group(0)) if chunk.strip()]
        if len(values) >= 4:
            return values
    return []


def _extract_line_equation(text: str) -> str:
    patterns = [
        r"(?:droite|equation)[^:=\n]*[:=]\s*([^\n]+)",
        r"\(GG1\)[^:=\n]*[:=]\s*([^\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            equation = re.split(r"[.;]", match.group(1).strip(), maxsplit=1)[0].strip()
            if "x" in equation and "y" in equation:
                return equation
    return ""


def _asks_mayer_method(text: str) -> bool:
    normalized = _normalize_lookup(text)
    return any(
        token in normalized
        for token in (
            "mayer",
            "g1",
            "g2",
            "g_1",
            "g_2",
            "points moyens",
            "moyennes partielles",
            "groupes de meme effectif",
        )
    )


def _extract_regression_table_xy(table_data: Any, prompt: str) -> tuple[list[float], list[float], int | None] | None:
    if not isinstance(table_data, dict):
        return None
    rows = table_data.get("rows") or []
    headers = [_normalize_lookup(header) for header in (table_data.get("headers") or [])]
    if len(headers) < 2 or len(rows) < 3:
        return None

    numeric_rows: list[tuple[float, float]] = []
    first_column_years = True
    for row in rows:
        if not isinstance(row, list) or len(row) < 2:
            continue
        try:
            first = _to_float(str(row[0]))
            second = _to_float(str(row[1]))
        except Exception:
            continue
        numeric_rows.append((first, second))
        if not (1900 <= first <= 2100 and float(first).is_integer()):
            first_column_years = False

    if len(numeric_rows) < 3:
        return None

    prompt_norm = _normalize_lookup(prompt)
    header0 = headers[0] if headers else ""
    uses_rank = (
        "rang" in prompt_norm
        or "i est le rang" in prompt_norm
        or "rang de l annee" in prompt_norm
        or "annee" in header0
    ) and first_column_years

    if uses_rank:
        years = [int(pair[0]) for pair in numeric_rows]
        x_values = [float(index) for index in range(1, len(numeric_rows) + 1)]
        year_offset = years[0] - 1
    else:
        x_values = [pair[0] for pair in numeric_rows]
        year_offset = None
    y_values = [pair[1] for pair in numeric_rows]
    return x_values, y_values, year_offset


def _compute_regression_values(x_values: list[float], y_values: list[float]) -> dict[str, float] | None:
    n = len(x_values)
    if n < 3 or n != len(y_values):
        return None
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n
    sxx = sum((x - mean_x) ** 2 for x in x_values)
    syy = sum((y - mean_y) ** 2 for y in y_values)
    if abs(sxx) < 1e-12 or abs(syy) < 1e-12:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    r_value = sxy / math.sqrt(sxx * syy)
    return {"slope": slope, "intercept": intercept, "r": r_value}


def _extract_claimed_correlation(text: str) -> float | None:
    normalized = _latex_decimal_to_float_text(text)
    patterns = [
        r"\br\s*(?:=|≈|\s+vaut|\s+est)\s*([-+]?\d+(?:\.\d+)?)",
        r"coefficient[^.\n]{0,80}?(?:=|vaut|est)\s*([-+]?\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
            except ValueError:
                continue
            if -1.05 <= value <= 1.05:
                return value
    return None


def _extract_regression_equation(text: str) -> tuple[float, float] | None:
    normalized = _latex_decimal_to_float_text(text)
    normalized = normalized.replace("\\times", "*").replace("×", "*").replace("·", "*")
    normalized = re.sub(r"\\[a-zA-Z]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    patterns = [
        r"\b[PpYy]\s*=\s*([-+]?\d+(?:\.\d+)?)\s*\*?\s*[IiXx]\s*([+-])\s*(\d+(?:\.\d+)?)",
        r"\b[PpYy]\s*=\s*([-+]?\d+(?:\.\d+)?)\s*([IiXx])\s*([+-])\s*(\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        try:
            slope = float(match.group(1))
            sign = match.group(2) if len(match.groups()) == 3 else match.group(3)
            intercept_token = match.group(3) if len(match.groups()) == 3 else match.group(4)
            intercept = float(intercept_token)
        except (ValueError, IndexError):
            continue
        if sign == "-":
            intercept = -intercept
        return slope, intercept
    return None


def _extract_years_asked_for_estimation(prompt: str) -> list[int]:
    # Ignore introductory years used only to define the table period. Keep years appearing in the numbered tasks.
    task_start_candidates = [
        index for index in (prompt.find("1)"), prompt.find("1."), prompt.find("a)")) if index >= 0
    ]
    task_text = prompt[min(task_start_candidates) :] if task_start_candidates else prompt
    normalized_task = _normalize_lookup(task_text)
    years: list[int] = []
    for match in re.finditer(r"\b(20\d{2}|19\d{2})\b", task_text):
        year = int(match.group(1))
        window_start = max(0, match.start() - 90)
        window_end = min(len(normalized_task), match.end() + 160)
        window = normalized_task[window_start:window_end]
        if any(token in window for token in ("estimer", "prevoir", "prediction", "besoin", "repondra", "population", "consomme")):
            years.append(year)
    return _deduplicate_preserving_order_int(years)


def _extract_threshold_value(prompt: str) -> float | None:
    normalized = _latex_decimal_to_float_text(prompt)
    match = re.search(r"depass(?:era|er|e|ent)?[^0-9]{0,40}(\d+(?:\.\d+)?)", normalized, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _has_number_near_anchor(text: str, anchor: str, expected: float, tolerance: float) -> bool:
    normalized = _latex_decimal_to_float_text(text)
    found_anchor = False
    for match in re.finditer(re.escape(anchor), normalized):
        found_anchor = True
        window = normalized[match.start() : match.end() + 220]
        if _window_contains_number_close_to(window, expected, tolerance):
            return True
    if not found_anchor:
        # Some concise expected answers omit the year label; accept any close predicted value.
        return _window_contains_number_close_to(normalized, expected, tolerance)
    return False


def _window_contains_number_close_to(window: str, expected: float, tolerance: float) -> bool:
    for token in re.findall(r"[-+]?\d+(?:\.\d+)?", window):
        try:
            value = float(token)
        except ValueError:
            continue
        # Ignore years and small ranks when checking large production estimates.
        if expected > 100 and 1900 <= value <= 2100:
            continue
        if abs(value - expected) <= tolerance:
            return True
    return False


def _latex_decimal_to_float_text(text: str) -> str:
    normalized = str(text or "")
    normalized = normalized.replace("{,}", ".").replace(",", ".")
    normalized = normalized.replace("\\(", " ").replace("\\)", " ")
    normalized = normalized.replace("\\[", " ").replace("\\]", " ")
    normalized = re.sub(r"[{}]", "", normalized)
    return normalized


def _deduplicate_preserving_order_int(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _has_total_mean_task(prompt: str) -> bool:
    normalized = _normalize_lookup(prompt)
    return any(
        token in normalized
        for token in (
            "point moyen",
            "coordonnees du point moyen",
            "coordonnees du point moyen",
            "calculer g",
            "determiner g",
            "point g",
            "moyenne generale",
        )
    )


def _has_grouped_means_task(prompt: str) -> bool:
    normalized = _normalize_lookup(prompt)
    return (
        ("g1" in normalized and "g2" in normalized)
        or "points moyens partiels" in normalized
        or "moyennes partielles" in normalized
        or "deux groupes" in normalized
        or "groupes de meme effectif" in normalized
    )


def _has_adjustment_line_task(prompt: str) -> bool:
    normalized = _normalize_lookup(prompt)
    return any(
        token in normalized
        for token in (
            "equation de la droite",
            "droite de mayer",
            "droite de regression",
            "droite d ajustement",
            "ajustement affine",
            "(gg1)",
            "(g1g2)",
        )
    )


def _check_two_variable_table_caption(table_data: Any) -> str:
    if not isinstance(table_data, dict):
        return ""
    caption = _normalize_lookup(table_data.get("caption", ""))
    headers = [_normalize_lookup(header) for header in (table_data.get("headers", []) or [])]
    if " et " not in caption or len(headers) < 2:
        return ""

    left_phrase, right_phrase = caption.rsplit(" et ", 1)
    left_tokens = _significant_caption_tokens(left_phrase)
    right_tokens = _significant_caption_tokens(right_phrase)
    if not left_tokens or not right_tokens:
        return ""

    left_match = any(any(token in header for token in left_tokens) for header in headers)
    right_match = any(any(token in header for token in right_tokens) for header in headers)
    if left_match and right_match:
        return ""

    return (
        "Pedagogical-completeness validator failed: la legende du tableau annonce deux variables, "
        "mais les colonnes ne reprennent pas clairement ces deux grandeurs."
    )


def _significant_caption_tokens(text: str) -> list[str]:
    stopwords = {
        "tableau",
        "donnees",
        "donnee",
        "serie",
        "double",
        "valeurs",
        "mesures",
        "production",
        "productions",
        "energie",
        "energies",
        "de",
        "des",
        "du",
        "la",
        "le",
        "les",
        "d",
        "l",
    }
    tokens = [
        token
        for token in re.findall(r"[a-z]{4,}", _normalize_lookup(text))
        if token not in stopwords
    ]
    if not tokens:
        return []
    return tokens[-2:]


def _line_contains_point(equation: str, x_value: float, y_value: float) -> bool:
    if sp is None:
        return False
    x_symbol, y_symbol = sp.symbols("x y")
    equation_text = equation.replace("^", "**")
    if "=" in equation_text:
        left, right = equation_text.split("=", 1)
        expression = f"({left})-({right})"
    else:
        expression = equation_text
    try:
        value = sp.simplify(sp.sympify(_normalize_for_sympy(expression)).subs({x_symbol: x_value, y_symbol: y_value}))
        return value == 0
    except Exception:
        return False


def _extract_recurrence_definitions(prompt: str) -> dict[str, str]:
    definitions: dict[str, str] = {}
    cleaned_prompt = repair_math_text_locally(prompt).replace("\\left", "").replace("\\right", "")
    pattern = re.compile(
        r"([abuvABUV])_\{?(?:n|k)\+1\}?\s*=\s*(.+?)(?=\s+(?:et|,)\s*[abuvABUV]_\{?(?:n|k)\+1\}?\s*=|[.;\n]|$)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(cleaned_prompt):
        expression = match.group(2).strip().replace("\\)", "").replace("\\]", "").strip()
        definitions[match.group(1).lower()] = expression
    return definitions


def _normalize_recurrence_expression(expression: str) -> str:
    text = repair_math_text_locally(expression)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\mathrm{e}", "e").replace("\\mathrm e", "e")
    text = text.replace("\\ln", "ln")
    text = _normalize_for_sympy(text)
    text = re.sub(r"(?<=n)\(", "*(", text)
    text = re.sub(r"(?<=\))(?=[A-Za-z])", "*", text)
    for key in ("a", "b", "u", "v"):
        text = re.sub(rf"[{key.upper()}{key}]_\{{?(?:n|k)\+1\}}?", f"{key}_next", text, flags=re.IGNORECASE)
        text = re.sub(rf"[{key.upper()}{key}]_\{{?(?:n|k)\}}?", f"{key}_n", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=[0-9A-Za-z_)])(?=(?:a_n|b_n|u_n|v_n)\b)", "*", text)
    return text


def _extract_claimed_gap_expression(solution: str, primary_name: str, secondary_name: str) -> str:
    match = re.search(
        rf"{secondary_name}_\{{?(?:n|k)\+1\}}?\s*-\s*{primary_name}_\{{?(?:n|k)\+1\}}?\s*=\s*(.+?)(?=>=|<=|[.;,\n]|$)",
        solution,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    raw_claim = match.group(1).strip()
    segments = [segment.strip() for segment in raw_claim.split("=") if segment.strip()]
    return segments[-1] if segments else raw_claim


def _extract_claimed_recurrence_expression(solution: str, sequence_name: str) -> str:
    match = re.search(
        rf"(?<![-A-Za-z0-9_]){sequence_name}_\{{?(?:n|k)\+1\}}?\s*=\s*(.+?)(?=[.;,\n]|$)",
        solution,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    raw_claim = match.group(1).strip()
    return re.split(r"<=|>=", raw_claim, maxsplit=1)[0].strip()


def _solution_mentions_recurrence_inequality(solution: str, primary_name: str, secondary_name: str) -> bool:
    return bool(
        re.search(
            rf"{primary_name}_\{{?(?:n|k)\}}?\s*<=\s*{secondary_name}_\{{?(?:n|k)\}}?",
            solution,
            flags=re.IGNORECASE,
        )
    )


def _analyze_recurrence_gap_dependence(
    expected_gap: Any,
    primary_symbol: Any,
    secondary_symbol: Any,
    primary_name: str,
    secondary_name: str,
) -> dict[str, Any]:
    gap_symbol = secondary_symbol - primary_symbol
    try:
        coefficient = sp.simplify(expected_gap / gap_symbol)
    except Exception:
        coefficient = None
    if coefficient is None or sp.simplify(expected_gap - coefficient * gap_symbol) != 0:
        return {
            "passed": False,
            "issue": (
                f"Recurrence validator failed: l'ecart {secondary_name}_(n+1)-{primary_name}_(n+1) ne se reecrit pas "
                f"comme multiple de {secondary_name}_n-{primary_name}_n."
            ),
        }
    if not _sympy_expression_is_nonnegative(coefficient):
        return {
            "passed": False,
            "issue": (
                f"Recurrence validator failed: l'ecart suivant vaut {sp.sstr(expected_gap)}, ce qui n'est pas un "
                f"multiple positif de {secondary_name}_n-{primary_name}_n."
            ),
        }
    return {"passed": True, "issue": ""}


def _detect_false_comparison_to_next_term(
    solution: str,
    *,
    sequence_name: str,
    expected_expr: Any,
    locals_map: dict[str, Any],
) -> str:
    pattern = rf"(?:<=|>=|=)\s*([^=\n<>]+?)\s*=\s*{sequence_name}_\{{?(?:n|k)\+1\}}?"
    for match in re.finditer(pattern, solution, flags=re.IGNORECASE):
        candidate_text = match.group(1).strip()
        if not candidate_text:
            continue
        try:
            candidate_expr = sp.sympify(_normalize_recurrence_expression(candidate_text), locals=locals_map)
        except Exception:
            continue
        if sp.simplify(candidate_expr - expected_expr) != 0:
            return (
                f"Recurrence validator failed: la comparaison finale identifie a tort {candidate_text} avec "
                f"{sequence_name}_(n+1), alors que {sequence_name}_(n+1)={sp.sstr(expected_expr)}."
            )
    return ""


def _sympy_expression_is_nonnegative(expression: Any) -> bool:
    if expression is None:
        return False
    if getattr(expression, "is_nonnegative", None) is True:
        return True
    if getattr(expression, "is_positive", None) is True:
        return True
    if getattr(expression, "is_number", False):
        try:
            return float(expression) >= 0.0
        except Exception:
            return False
    return False


def _looks_like_recurrence_defined_sequences(prompt: str, solution: str = "") -> bool:
    text = repair_math_text_locally(f"{prompt}\n{solution}")
    normalized = _normalize_lookup(text)
    raw_lower = text.lower()
    has_step_definition = bool(re.search(r"[abuv]_\{?(?:n|k)\+1\}?\s*=", raw_lower))
    has_sequence_reference = bool(re.search(r"[abuv]_\{?(?:n|k)\}?", raw_lower))
    return has_step_definition and (has_sequence_reference or "suite" in normalized or "recurrence" in normalized)


def _extract_sequence_value(solution: str, sequence_name: str, index_value: int) -> str:
    match = re.search(
        rf"[{sequence_name}{sequence_name.upper()}]_\{{?{index_value}\}}?\s*=\s*([^\n.;]+)",
        repair_math_text_locally(solution),
        flags=re.IGNORECASE,
    )
    return match.group(1).strip().replace("\\)", "").replace("\\]", "").strip() if match else ""


def _extract_sequence_difference(solution: str, sequence_name: str) -> str:
    match = re.search(
        rf"[{sequence_name}{sequence_name.upper()}]_\{{?(?:n|k)\+1\}}?\s*-\s*[{sequence_name}{sequence_name.upper()}]_\{{?(?:n|k)\}}?\s*=\s*([^\n.;]+)",
        repair_math_text_locally(solution),
        flags=re.IGNORECASE,
    )
    return match.group(1).strip().replace("\\)", "").replace("\\]", "").strip() if match else ""


def _extract_general_sequence_expression(solution: str, sequence_name: str) -> str:
    match = re.search(
        rf"(?<![-A-Za-z0-9_])[{sequence_name}{sequence_name.upper()}]_\{{?(?:n|k)\}}?\s*=\s*([^\n.;]+)",
        repair_math_text_locally(solution),
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    candidate = match.group(1).strip().replace("\\)", "").replace("\\]", "").strip()
    if re.search(rf"[{sequence_name}{sequence_name.upper()}]_\{{?(?:n|k)\+1\}}?", candidate, flags=re.IGNORECASE):
        return ""
    return candidate


def _extract_derivative_claims(record: dict[str, Any]) -> list[str]:
    text = "\n".join([str(record.get("prompt", "")), str(record.get("hidden_solution", "")), str(record.get("display_answer", ""))])
    claims = re.findall(r"f'\(x\)\s*=\s*([^\n;,.]+(?:/[^\n;,.]+)?)", text, flags=re.IGNORECASE)
    return _deduplicate_preserving_order([claim.strip() for claim in claims if claim.strip()])


def _extract_function_expression(prompt: str) -> str:
    patterns = [
        r"f\(x\)\s*=\s*([^\n]+)",
        r"fonction\s+f.*?f\(x\)\s*=\s*([^\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if not match:
            continue
        expression = match.group(1).strip()
        expression = re.split(
            r"(?:[.;]|(?:\b(calculer|determiner|donner|montrer|etudier|trouver)\b))",
            expression,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        if expression:
            return expression
    return ""


def _extract_probability_equations(text: str) -> list[tuple[str, str]]:
    equations: list[tuple[str, str]] = []
    for raw_line in re.split(r"[\n]|(?<=\.)\s+", text):
        line = raw_line.strip()
        if "P(" not in line and "p(" not in line.lower():
            continue
        segments = [segment.strip().rstrip(".") for segment in line.split("=") if segment.strip()]
        if len(segments) < 3:
            continue
        expression = segments[-2]
        claimed = segments[-1]
        if any(operator in expression for operator in "+-*/"):
            equations.append((expression, claimed))
    return equations


def _evaluate_fraction_expression(expression: str) -> Fraction | None:
    text = str(expression or "").strip()
    if not text:
        return None
    text = text.replace(",", ".").replace(" ", "")
    try:
        node = ast.parse(text, mode="eval")
    except SyntaxError:
        return None
    return _eval_fraction_node(node.body)


def _eval_fraction_node(node: ast.AST) -> Fraction | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _eval_fraction_node(node.operand)
        return -value if value is not None else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _eval_fraction_node(node.operand)
    if isinstance(node, ast.BinOp):
        left = _eval_fraction_node(node.left)
        right = _eval_fraction_node(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                return None
            return left / right
        if isinstance(node.op, ast.Pow) and right.denominator == 1 and right >= 0 and right <= 6:
            return left ** right.numerator
    return None


def _extract_complex_roots(text: str) -> dict[str, str]:
    roots: dict[str, str] = {}
    for label in ("z1", "z2"):
        match = re.search(
            rf"\b{label}\s*=\s*(.+?)(?=\s*(?:,|;|\bet\b|\n|$))",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            roots[label] = match.group(1).strip()
    return roots


def _validate_complex_rotation(prompt: str, solution: str) -> dict[str, Any]:
    combined = f"{prompt}\n{solution}"
    normalized = _normalize_lookup(combined)
    applicable = "rotation" in normalized
    if not applicable:
        return {"applicable": False, "issues": [], "summary": ""}
    if sp is None:
        return {
            "applicable": True,
            "issues": ["Complex validator failed: validation symbolique requise pour la rotation, mais SymPy est indisponible."],
            "summary": "SymPy indisponible pour le controle de rotation.",
        }

    issues: list[str] = []
    points = _extract_affix_points(combined)
    mapping_pair = _extract_rotation_point_pair(combined)
    prompt_center = _extract_rotation_center(prompt, points)
    prompt_angle = _extract_rotation_angle(prompt)
    prompt_transform = _extract_rotation_transform(prompt)
    center_expr = _extract_rotation_center(combined, points)
    angle_expr = _extract_rotation_angle(combined)
    transform_expr = _extract_rotation_transform(solution)

    if _claims_unique_rotation(prompt) and mapping_pair and prompt_center is None and prompt_angle is None and prompt_transform is None:
        issues.append(
            "Complex validator failed: une rotation unique ne peut pas etre determinee a partir d'un seul couple point-image sans centre, angle ou contrainte supplementaire."
        )

    z = sp.symbols("z")
    omega = center_expr
    rotation_factor = None
    transform_rotation_factor = None
    transform_omega = None
    if angle_expr is not None:
        rotation_factor = sp.simplify(sp.exp(sp.I * angle_expr))
    if transform_expr is not None:
        transform_rotation_factor, transform_omega = _recover_rotation_parameters_from_transform(transform_expr, z)
        if rotation_factor is None:
            rotation_factor = transform_rotation_factor
        if omega is None:
            omega = transform_omega
    if (
        transform_rotation_factor is not None
        and rotation_factor is not None
        and sp.simplify(transform_rotation_factor - rotation_factor) != 0
    ):
        issues.append("Complex validator failed: la formule proposee pour la rotation n'est pas coherente avec l'angle annonce.")
    if transform_omega is not None and omega is not None and sp.simplify(transform_omega - omega) != 0:
        issues.append("Complex validator failed: la formule proposee pour la rotation n'est pas coherente avec le centre annonce.")

    if mapping_pair and omega is not None and rotation_factor is not None:
        source_name, target_name = mapping_pair
        source_value = points.get(source_name)
        target_value = points.get(target_name)
        if source_value is not None and target_value is not None:
            mapped_value = sp.simplify(omega + rotation_factor * (source_value - omega))
            if sp.simplify(mapped_value - target_value) != 0:
                issues.append(
                    f"Complex validator failed: la rotation proposee n'envoie pas {source_name} sur {target_name}."
                )

    image_claims = _extract_rotation_image_claims(solution)
    if image_claims and (omega is None or rotation_factor is None):
        issues.append(
            "Complex validator failed: la solution annonce des images par rotation sans fournir un centre et un angle (ou une ecriture equivalente) verifiables."
        )
    for source_name, target_name in image_claims:
        source_value = points.get(source_name)
        target_value = points.get(target_name)
        if source_value is None or target_value is None or omega is None or rotation_factor is None:
            continue
        mapped_value = sp.simplify(omega + rotation_factor * (source_value - omega))
        if sp.simplify(mapped_value - target_value) != 0:
            issues.append(
                f"Complex validator failed: l'image annoncee {source_name}->{target_name} n'est pas coherente avec la rotation proposee."
            )

    if "exp(i*theta)" in _normalize_for_sympy(solution).lower() and "exp(-i*theta)" in _normalize_for_sympy(solution).lower():
        issues.append("Complex validator failed: la solution alterne exp(i*theta) et exp(-i*theta) sans justification coherente.")

    return {
        "applicable": True,
        "issues": issues,
        "summary": "Controle local des rotations complexes et des images d'affixes.",
    }


def _extract_affix_points(text: str) -> dict[str, Any]:
    points: dict[str, Any] = {}
    patterns = [
        r"\b([A-Z])\s*\(\s*([^)]+?)\s*\)",
        r"\b([A-Z])\s+d['’]affixe\s*([^\n;,.]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            name = match.group(1).upper()
            if name in points:
                continue
            candidate = match.group(2).strip()
            parsed = _sympy_value_if_possible(candidate)
            if parsed is not None:
                points[name] = parsed
    return points


def _extract_rotation_point_pair(text: str) -> tuple[str, str] | None:
    patterns = [
        r"envoie\s+([A-Z])(?:\s*\([^)]*\))?\s+sur\s+([A-Z])(?:\s*\([^)]*\))?",
        r"transforme\s+([A-Z])(?:\s*\([^)]*\))?\s+en\s+([A-Z])(?:\s*\([^)]*\))?",
        r"image\s+de\s+([A-Z])(?:\s*\([^)]*\))?\s+est\s+([A-Z])(?:\s*\([^)]*\))?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper(), match.group(2).upper()
    return None


def _extract_rotation_image_claims(text: str) -> list[tuple[str, str]]:
    claims: list[tuple[str, str]] = []
    for match in re.finditer(r"r\(\s*([A-Z])\s*\)\s*=\s*([A-Z])", text, flags=re.IGNORECASE):
        claims.append((match.group(1).upper(), match.group(2).upper()))
    pair = _extract_rotation_point_pair(text)
    if pair and pair not in claims:
        claims.append(pair)
    return claims


def _extract_rotation_center(text: str, points: dict[str, Any]) -> Any | None:
    for pattern in (
        r"omega\s*=\s*(.+?)(?=\s+\b(?:et|qui|puis)\b|[.;,\n]|$)",
        r"centre\s+omega[^=:\n]*[:=]?\s*(.+?)(?=\s+\b(?:et|qui|puis)\b|[.;,\n]|$)",
        r"centre\s+d['’]affixe\s*(.+?)(?=\s+\b(?:et|qui|puis)\b|[.;,\n]|$)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _sympy_value_if_possible(match.group(1).strip().rstrip("."))
    point_match = re.search(r"centre\s+([A-Z])\b", text, flags=re.IGNORECASE)
    if point_match:
        return points.get(point_match.group(1).upper())
    return None


def _extract_rotation_angle(text: str) -> Any | None:
    for pattern in (
        r"theta\s*=\s*(.+?)(?=\s+\b(?:et|qui|puis)\b|[.;,\n]|$)",
        r"angle\s+(?:de\s+)?(?:rotation\s+)?[:=]?\s*(.+?)(?=\s+\b(?:et|qui|puis)\b|[.;,\n]|$)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _sympy_value_if_possible(match.group(1).strip().rstrip("."))
    return None


def _extract_rotation_transform(text: str) -> Any | None:
    for pattern in (
        r"r\(z\)\s*=\s*(.+?)(?=\s+\b(?:et|qui|puis)\b|[.;,\n]|$)",
        r"z'\s*=\s*(.+?)(?=\s+\b(?:et|qui|puis)\b|[.;,\n]|$)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _sympy_value_if_possible(match.group(1).strip().rstrip("."))
    return None


def _recover_rotation_parameters_from_transform(transform_expr: Any, z_symbol: Any) -> tuple[Any | None, Any | None]:
    try:
        coefficient = sp.simplify(transform_expr.expand().coeff(z_symbol, 1))
        constant = sp.simplify(transform_expr.subs({z_symbol: 0}))
    except Exception:
        return None, None
    if coefficient is None:
        return None, None
    if sp.simplify(1 - coefficient) == 0:
        return coefficient, 0
    try:
        omega = sp.simplify(constant / (1 - coefficient))
    except Exception:
        omega = None
    return coefficient, omega


def _claims_unique_rotation(text: str) -> bool:
    normalized = _normalize_lookup(text)
    return "rotation unique" in normalized or "l'unique rotation" in normalized or "unique rotation" in normalized


def _extract_complex_sum_product(text: str) -> tuple[str | None, str | None]:
    sum_match = re.search(r"z1\s*\+\s*z2\s*=\s*([^\n;,]+)", text, flags=re.IGNORECASE)
    product_match = re.search(r"z1\s*z2\s*=\s*([^\n;,]+)", text, flags=re.IGNORECASE)
    return (
        sum_match.group(1).strip() if sum_match else None,
        product_match.group(1).strip() if product_match else None,
    )


def _extract_quadratic_expression(text: str) -> Any | None:
    if sp is None:
        return None
    match = re.search(r"(z\^2[^=\n]+)=\s*0", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return sp.sympify(_normalize_for_sympy(match.group(1)))
    except Exception:
        return None


def _extract_probability_assignments(text: str) -> tuple[dict[tuple[str, ...], Fraction], int]:
    assignments: dict[tuple[str, ...], Fraction] = {}
    parsed_spans: list[tuple[int, int]] = []
    cleaned = str(text or "").replace("∩", " n ").replace("⋂", " n ")
    value_pattern = r"([-+]?\d+(?:[.,]\d+)?(?:\s*/\s*[-+]?\d+(?:[.,]\d+)?)?)"

    conditional_pattern = re.compile(
        rf"P\(\s*([A-Za-z])\s*\|\s*([A-Za-z])\s*\)\s*=\s*{value_pattern}",
        flags=re.IGNORECASE,
    )
    intersection_pattern = re.compile(
        rf"P\(\s*([A-Za-z])\s*(?:n|inter)\s*([A-Za-z])\s*\)\s*=\s*{value_pattern}",
        flags=re.IGNORECASE,
    )
    simple_pattern = re.compile(
        rf"P\(\s*([A-Za-z])\s*\)\s*=\s*{value_pattern}",
        flags=re.IGNORECASE,
    )

    for match in conditional_pattern.finditer(cleaned):
        value = _evaluate_fraction_expression(match.group(3))
        if value is None:
            continue
        assignments[("cond", match.group(1).upper(), match.group(2).upper())] = value
        parsed_spans.append(match.span())

    for match in intersection_pattern.finditer(cleaned):
        value = _evaluate_fraction_expression(match.group(3))
        if value is None:
            continue
        left = match.group(1).upper()
        right = match.group(2).upper()
        assignments[("inter", *sorted((left, right)))] = value
        parsed_spans.append(match.span())

    for match in simple_pattern.finditer(cleaned):
        if any(start <= match.start() and match.end() <= end for start, end in parsed_spans):
            continue
        value = _evaluate_fraction_expression(match.group(2))
        if value is None:
            continue
        assignments[("simple", match.group(1).upper())] = value
        parsed_spans.append(match.span())

    unsupported_count = 0
    broad_pattern = re.compile(r"P\([^)]*\)\s*=\s*[-+]?\d+(?:[.,]\d+)?(?:\s*/\s*[-+]?\d+(?:[.,]\d+)?)?", flags=re.IGNORECASE)
    for match in broad_pattern.finditer(cleaned):
        if not any(start <= match.start() and match.end() <= end for start, end in parsed_spans):
            unsupported_count += 1
    return assignments, unsupported_count


def _validate_probability_assignment_consistency(
    assignments: dict[tuple[str, ...], Fraction],
    source_text: str,
) -> list[str]:
    issues: list[str] = []
    for key, value in assignments.items():
        if not (Fraction(0, 1) <= value <= Fraction(1, 1)):
            issues.append(
                f"Probability validator failed: la probabilite donnee {_format_probability_key(key)}={float(value):.6g} n'appartient pas a [0,1]."
            )

    simple_values = {key[1]: value for key, value in assignments.items() if key[0] == "simple"}
    intersection_values = {(key[1], key[2]): value for key, value in assignments.items() if key[0] == "inter"}
    conditional_values = {(key[1], key[2]): value for key, value in assignments.items() if key[0] == "cond"}

    for (event_a, event_b), value in intersection_values.items():
        if event_a in simple_values and value > simple_values[event_a]:
            issues.append(
                f"Probability validator failed: P({event_a}∩{event_b})={float(value):.6g} ne peut pas depasser P({event_a})={float(simple_values[event_a]):.6g}."
            )
        if event_b in simple_values and value > simple_values[event_b]:
            issues.append(
                f"Probability validator failed: P({event_a}∩{event_b})={float(value):.6g} ne peut pas depasser P({event_b})={float(simple_values[event_b]):.6g}."
            )

        if (event_b, event_a) in conditional_values and event_a in simple_values and simple_values[event_a] != 0:
            expected = value / simple_values[event_a]
            claimed = conditional_values[(event_b, event_a)]
            if expected != claimed:
                issues.append(
                    f"Probability validator failed: P({event_b}|{event_a}) devrait valoir {float(expected):.6g}, pas {float(claimed):.6g}."
                )

        if (event_a, event_b) in conditional_values and event_b in simple_values and simple_values[event_b] != 0:
            expected = value / simple_values[event_b]
            claimed = conditional_values[(event_a, event_b)]
            if expected != claimed:
                issues.append(
                    f"Probability validator failed: P({event_a}|{event_b}) devrait valoir {float(expected):.6g}, pas {float(claimed):.6g}."
                )

        if "indep" in _normalize_lookup(source_text) and event_a in simple_values and event_b in simple_values:
            expected = simple_values[event_a] * simple_values[event_b]
            if expected != value:
                issues.append(
                    f"Probability validator failed: l'independance exigerait P({event_a}∩{event_b})={float(expected):.6g}, pas {float(value):.6g}."
                )

    return issues


def _format_probability_key(key: tuple[str, ...]) -> str:
    if not key:
        return "P(?)"
    if key[0] == "simple":
        return f"P({key[1]})"
    if key[0] == "inter":
        return f"P({key[1]}∩{key[2]})"
    if key[0] == "cond":
        return f"P({key[1]}|{key[2]})"
    return "P(?)"


def _extract_binomial_probability_problem(prompt: str) -> dict[str, Any] | None:
    normalized = _normalize_lookup(prompt)

    n_value: int | None = None
    p_value: Fraction | None = None

    b_match = re.search(
        r"\bB\s*\(\s*(\d+)\s*[,;]\s*([-+]?\d+(?:[.,]\d+)?(?:\s*/\s*[-+]?\d+(?:[.,]\d+)?)?)\s*\)",
        prompt,
        flags=re.IGNORECASE,
    )
    if b_match:
        n_value = int(b_match.group(1))
        p_value = _evaluate_fraction_expression(b_match.group(2))

    if n_value is None:
        n_match = re.search(r"\bn\s*=\s*(\d+)", prompt, flags=re.IGNORECASE)
        if n_match:
            n_value = int(n_match.group(1))
    if p_value is None:
        p_match = re.search(
            r"\bp\s*=\s*([-+]?\d+(?:[.,]\d+)?(?:\s*/\s*[-+]?\d+(?:[.,]\d+)?)?)",
            prompt,
            flags=re.IGNORECASE,
        )
        if p_match:
            p_value = _evaluate_fraction_expression(p_match.group(1))

    threshold_patterns = [
        ("at_least", r"au moins\s+(\d+)"),
        ("more_than", r"plus de\s+(\d+)"),
        ("at_most", r"au plus\s+(\d+)"),
        ("less_than", r"moins de\s+(\d+)"),
        ("exactly", r"exactement\s+(\d+)"),
    ]
    comparison: str | None = None
    threshold_value: int | None = None
    for comparison_name, pattern in threshold_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            comparison = comparison_name
            threshold_value = int(match.group(1))
            break

    explicit_binomial_context = any(marker in normalized for marker in ("binomiale", "bernoulli", "schema de bernoulli"))
    inferred_binomial_context = n_value is not None and p_value is not None and comparison is not None
    if not explicit_binomial_context and not inferred_binomial_context:
        return None
    if n_value is None or p_value is None or comparison is None or threshold_value is None:
        return None
    return {
        "n": n_value,
        "p": p_value,
        "comparison": comparison,
        "k": threshold_value,
    }


def _validate_binomial_probability_problem(
    problem: dict[str, Any],
    *,
    display_answer: str,
    extracted_answer: str,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    n_value = int(problem["n"])
    p_fraction = problem["p"]
    comparison = str(problem["comparison"])
    threshold = int(problem["k"])
    issues: list[str] = []

    computed_probability = _compute_binomial_probability(
        n_value,
        float(p_fraction),
        comparison,
        threshold,
    )
    human_label = _describe_binomial_event(comparison, threshold)
    summary = f"Loi binomiale locale : n={n_value}, p={float(p_fraction):.6g}, {human_label} => {computed_probability:.10f}"

    candidates_checked = 0
    for label, candidate in (("reponse attendue", display_answer), ("reponse finale", extracted_answer)):
        numeric_candidate = _evaluate_fraction_expression(candidate)
        if numeric_candidate is None:
            continue
        candidates_checked += 1
        if abs(float(numeric_candidate) - computed_probability) > tolerance:
            issues.append(
                f"Probability validator failed: pour n={n_value}, p={float(p_fraction):.6g} et '{human_label}', la probabilite vaut {computed_probability:.10f}, pas {float(numeric_candidate):.6g} dans la {label}."
            )

    if candidates_checked == 0:
        issues.append(
            "Probability validator failed: aucun resultat numerique exploitable n'a ete fourni pour la question binomiale."
        )

    return {
        "issues": issues,
        "summary": summary,
    }


def _compute_binomial_probability(n_value: int, p_value: float, comparison: str, threshold: int) -> float:
    if comparison == "at_least":
        indices = range(max(0, threshold), n_value + 1)
    elif comparison == "more_than":
        indices = range(max(0, threshold + 1), n_value + 1)
    elif comparison == "at_most":
        indices = range(0, min(n_value, threshold) + 1)
    elif comparison == "less_than":
        indices = range(0, min(n_value, threshold - 1) + 1)
    else:
        indices = [threshold]

    total = 0.0
    for k_value in indices:
        if 0 <= k_value <= n_value:
            total += math.comb(n_value, k_value) * (p_value ** k_value) * ((1 - p_value) ** (n_value - k_value))
    return total


def _describe_binomial_event(comparison: str, threshold: int) -> str:
    return {
        "at_least": f"au moins {threshold}",
        "more_than": f"plus de {threshold}",
        "at_most": f"au plus {threshold}",
        "less_than": f"moins de {threshold}",
        "exactly": f"exactement {threshold}",
    }.get(comparison, f"exactement {threshold}")


def _extract_final_answer_from_solution_text(solution_text: str) -> str:
    text = str(solution_text or "").strip()
    if not text:
        return ""
    patterns = [
        r"reponse finale\s*[:\-]\s*(.+?)(?:(?<!\d)\.(?!\d)|\n|$)",
        r"donc\s+(.+?)(?:(?<!\d)\.(?!\d)|\n|$)",
        r"ainsi\s+(.+?)(?:(?<!\d)\.(?!\d)|\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _sympy_value_if_possible(value: str) -> Any | None:
    if not value or sp is None:
        return None
    try:
        return sp.sympify(_normalize_for_sympy(value))
    except Exception:
        return None


def _answers_equivalent(left: str, right: str, answer_kind: str) -> bool:
    if not left or not right:
        return False
    if _normalize_text(left) == _normalize_text(right):
        return True
    if answer_kind == "set":
        return _parse_set_answer(left) == _parse_set_answer(right)
    if sp is None:
        return False
    try:
        left_expr = sp.sympify(_normalize_for_sympy(left))
        right_expr = sp.sympify(_normalize_for_sympy(right))
        return sp.simplify(left_expr - right_expr) == 0
    except Exception:
        return False


def _parse_set_answer(value: str) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\b(or|ou|et)\b", ",", text, flags=re.IGNORECASE)
    raw_parts = [part.strip() for part in re.split(r"[;,]", text) if part.strip()]
    normalized_parts: set[str] = set()
    for part in raw_parts:
        cleaned = re.sub(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*", "", part).strip()
        normalized_parts.add(_normalize_for_sympy(cleaned))
    return normalized_parts


def _normalize_for_sympy(value: str) -> str:
    text = str(value or "").strip()
    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("π", "pi").replace("^", "**").replace("−", "-")
    text = text.replace("×", "*").replace("÷", "/")
    text = re.sub(r"\bln\s*\(", "log(", text)
    text = re.sub(r"\be\*\*\(([^)]+)\)", r"exp(\1)", text)
    text = re.sub(r"\be\*\*([A-Za-z0-9_]+)", r"exp(\1)", text)
    text = re.sub(r"(?<![A-Za-z])i(?![A-Za-z])", "I", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", text)
    text = re.sub(r"(?<=[A-Za-z)])(?=\d)", "*", text)
    if re.fullmatch(r"-?\d+,\d+", text):
        text = text.replace(",", ".")
    return text


def _normalize_lookup(value: str) -> str:
    raw_text = str(value or "").replace("∞", "inf").replace("−", "-")
    ascii_text = unicodedata.normalize("NFKD", raw_text).encode("ascii", "ignore").decode("ascii")
    compact = re.sub(r"[^a-zA-Z0-9()=+*/.,;\-\s]+", " ", ascii_text)
    return re.sub(r"\s+", " ", compact).strip().lower()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _deduplicate_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean_value = str(value or "").strip()
        if not clean_value:
            continue
        key = _normalize_text(clean_value)
        if key in seen:
            continue
        seen.add(key)
        result.append(clean_value)
    return result


def _to_float(value: str) -> float:
    return float(str(value).replace(",", "."))
