from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from frontend.utils.exercise_schema import has_explicit_questions
from frontend.utils.math_format_guard import find_math_format_issues, repair_corrupted_latex_commands, repair_exercise_math_locally
from frontend.utils.validators.domain_router import get_domain_validator_key, explain_domain_route
from frontend.utils.validators.graph_support_validator import validate_graph_support
from frontend.utils.validators.question_coverage import validate_question_answer_coverage
from frontend.utils.validators.local_math_validators import instruction_requires_visual_support

BLOCKING_JUDGE_FLAGS = {
    "judge_error",
    "wrong",
    "misaligned",
    "unknown",
    "blocked_after_retries",
    "blocked_missing_alignment",
}

STUDENT_FACING_BAD_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bfrace\b", "Le token 'frace' apparait encore dans le texte eleve."),
    (r"\bextbf\{", "La commande \\textbf est corrompue dans le texte eleve."),
    (r"\bextit\{", "La commande \\textit est corrompue dans le texte eleve."),
    (r"\bext\{", "La commande \\text est corrompue dans le texte eleve."),
    (r"\brac\{", "La commande \\frac est corrompue dans le texte eleve."),
    (r"\bhickapprox\b", "La commande \\approx est corrompue dans le texte eleve."),
    (r"\bextasciitilde\b", "La commande \\sim est corrompue dans le texte eleve."),
    (r"\\\\{2,}(?:ln|frac|dfrac|text|textit)\b", "Une commande LaTeX contient trop d'antislashs."),
    (r"\begin\{", "Un environnement LaTeX begin est corrompu dans le texte eleve."),
    (r"\bend\{", "Un environnement LaTeX end est corrompu dans le texte eleve."),
    (r"\\\\{2,}[\(\)\[\]]", "Un delimiteur LaTeX inline/display est sur-echappe."),
    (r"\bfracpi\b", "Le token 'fracpi' apparait encore dans le texte eleve."),
    (r"\bmathbb\s+R\b", "La notation 'mathbb R' n'a pas ete convertie en \\mathbb{R}."),
    (r"\bmathbb\s+N\b", "La notation 'mathbb N' n'a pas ete convertie en \\mathbb{N}."),
    (r"\bin\s+fty\b", "Le token corrompu 'in fty' apparait encore."),
    (r"\+\s*in\s+fty\b", "Le token corrompu '+ in fty' apparait encore."),
    (r"e\^\{0U\}", "La notation corrompue 'e^{0U}' apparait encore."),
    (r"e\^\{-nU\}", "La notation corrompue 'e^{-nU}' apparait encore."),
    (r"\{\{\{", "Des accolades triples non resolues apparaissent encore."),
    (r"\}\}\}", "Des accolades triples non resolues apparaissent encore."),
    (r"\\\\int_\\alpha\^\{\{\{0 f\}\}\}", "Une integrale corrompue apparait encore."),
    (r"\\int_\\alpha\^\{\{\{0 f\}\}\}", "Une integrale corrompue apparait encore."),
    (r"\\\\mathbb", "Une commande \\mathbb contient des antislashs repetes anormaux."),
)

PLACEHOLDER_PATTERNS = (
    r"\bvoir annexe\b",
    r"\bcompleter ici\b",
    r"\btodo\b",
    r"\breponse en attente\b",
    r"\bsolution detaillee indisponible\b",
)


def validate_student_facing_math_text(exercise: dict[str, Any]) -> tuple[bool, list[str]]:
    """Block malformed or placeholder-ridden math text before student display."""
    fields = [
        str(exercise.get("title", "")),
        str(exercise.get("prompt", "")),
        str(exercise.get("display_answer", "")),
        str(exercise.get("hidden_solution", "")),
        *[str(step) for step in (exercise.get("solution_steps") or [])],
    ]
    combined = "\n".join(part for part in fields if part).strip()
    combined = repair_corrupted_latex_commands(combined)
    if not combined:
        return False, ["Le texte eleve est vide apres generation."]

    issues = list(find_math_format_issues(exercise))
    lowered = combined.lower()
    for pattern, message in STUDENT_FACING_BAD_PATTERNS:
        if re.search(pattern, combined, flags=re.IGNORECASE):
            issues.append(f"Format student-facing invalide: {message}")
    for pattern in PLACEHOLDER_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            issues.append("Le texte eleve contient encore un placeholder non resolu.")

    deduped = _deduplicate(issues)
    return not deduped, deduped


def can_present_exercise(record: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return whether an exercise may be shown to the student, with blocking reasons."""
    if not record:
        return False, ["Aucun exercice n'a ete genere."]

    reasons: list[str] = []
    judge_flag = _normalized(record.get("judge_validation_flag"))
    judge_status = _normalized(record.get("judge_status"))
    alignment_status = _normalized(record.get("judge_alignment_status"))
    solution_flag = _normalized(record.get("solution_validation_flag"))
    local_flag = _normalized(record.get("local_validation_flag"))
    generation_backend = _normalized(record.get("generation_backend"))
    display_source_category = _normalized(record.get("display_source_category"))
    symbolic_required = bool(record.get("symbolic_checks_required", False))
    symbolic_ran = bool(record.get("symbolic_checks_ran", False))
    symbolic_passed = record.get("symbolic_checks_passed")
    pedagogical_flag = _normalized(record.get("pedagogical_completeness_flag"))
    corrected_fields_applied = bool(
        record.get("corrected_fields_applied", record.get("judge_corrections_applied", False))
    )
    questions = [str(item).strip() for item in (record.get("questions") or []) if str(item).strip()]
    domain_key = get_domain_validator_key(record.get("topic", ""), record.get("subtopic", ""), record.get("generation_metadata") or {})

    student_format_ok, student_format_issues = validate_student_facing_math_text(record)
    if not student_format_ok:
        reasons.extend(student_format_issues)

    if judge_status == "juge indisponible":
        reasons.append("Le juge est indisponible.")
    if judge_flag not in {"approved", "corrected"}:
        reasons.append(f"Le flag du juge est bloqueur : {judge_flag or 'absent'}.")
    if judge_flag in BLOCKING_JUDGE_FLAGS:
        reasons.append(f"Le flag du juge interdit l'affichage : {judge_flag}.")
    if alignment_status != "aligned":
        reasons.append("L'exercice n'est pas aligne avec le referentiel officiel.")
    if solution_flag != "approved":
        reasons.append("La validation de solution n'a pas approuve l'exercice.")
    if local_flag != "approved":
        reasons.append("La validation locale n'a pas approuve l'exercice.")
    if bool(record.get("blocked_before_display")) or bool(record.get("judge_blocked")):
        reasons.append("L'exercice a deja ete bloque avant affichage.")
    if not str(record.get("display_answer", "")).strip():
        reasons.append("La reponse attendue est vide.")
    if not str(record.get("hidden_solution", "")).strip():
        reasons.append("La solution complete est vide.")
    if len(questions) < 2:
        reasons.append("Le schema eleve ne contient pas au moins deux questions explicites.")
    if not has_explicit_questions(str(record.get("prompt", "")), questions):
        reasons.append("L'enonce eleve ne contient pas de questions explicites detectables.")
    coverage_ok, coverage_issues, _coverage_meta = validate_question_answer_coverage(record)
    if not coverage_ok:
        reasons.extend(coverage_issues)
    if not bool(record.get("is_true_llm_generation")) and not bool(record.get("demo_mode_used", False)):
        reasons.append("L'exercice n'est pas une vraie generation LLM et le mode demonstration n'est pas actif.")
    if display_source_category not in {"llm_generated", "demo_dataset", "blocked"}:
        reasons.append("La categorie de source d'affichage est invalide.")
    if display_source_category == "demo_dataset" and not bool(record.get("demo_mode_used", False)):
        reasons.append("Le mode demonstration dataset n'a pas ete active explicitement.")
    if bool(record.get("too_similar_to_source_case")):
        reasons.append("L'enonce genere reste trop similaire a un cas source recupere.")

    if symbolic_required:
        if not symbolic_ran:
            reasons.append("Une validation symbolique etait requise mais n'a pas ete executee.")
        if symbolic_passed is not True:
            reasons.append("Une validation symbolique etait requise mais n'a pas ete confirmee.")

    if pedagogical_flag == "incomplete":
        reasons.append("L'exercice reste pedagogiquement incomplet pour la notion annoncee.")

    if generation_backend in {"dataset-fallback", "local-fallback", "dataset-fallback-blocked"}:
        reasons.append(f"La sortie {generation_backend} ne peut pas etre affichee comme generation finale.")
    if generation_backend == "trusted-dataset-demo" and not bool(record.get("demo_mode_used", False)):
        reasons.append("Le mode demonstration dataset n'a pas ete active explicitement.")

    if judge_flag == "corrected":
        if not corrected_fields_applied:
            reasons.append("Le juge a corrige l'exercice, mais les champs corriges n'ont pas ete appliques.")
        reasons.extend(_extract_unresolved_corrected_issues(record.get("judge_issues")))

    if instruction_requires_visual_support(record):
        if not bool(record.get("support_ready", False)):
            reasons.append("Le support visuel requis n'est pas pret.")
        if not record.get("chart_data") and not record.get("table_data") and not record.get("graph_data"):
            reasons.append("Le support visuel requis est absent.")

    probability_check = ((record.get("local_validation_checks") or {}).get("probability") or {})
    regression_check = ((record.get("local_validation_checks") or {}).get("regression_numeric") or {})
    regression_domain_check = ((record.get("local_validation_checks") or {}).get("regression_deterministic") or {})
    bayes_check = ((record.get("local_validation_checks") or {}).get("bayes") or {})
    exponential_check = ((record.get("local_validation_checks") or {}).get("exponential_law") or {})
    sequence_check = ((record.get("local_validation_checks") or {}).get("sequence_numeric") or {})
    ode_check = ((record.get("local_validation_checks") or {}).get("ode") or {})
    topic_text = _normalized(" ".join([str(record.get("topic", "")), str(record.get("subtopic", ""))]))
    if domain_key == "finite_probability" and probability_check.get("status") == "failed":
        reasons.append("Le controle probabiliste local a echoue.")
    if domain_key == "bayes":
        if bayes_check.get("applicable") and bayes_check.get("status") != "passed":
            reasons.append("Le controle deterministe de Bayes a echoue.")
        elif not bayes_check.get("applicable"):
            reasons.append("Le controle deterministe de Bayes est requis mais non applicable.")
    if domain_key == "regression" and regression_check.get("status") == "failed":
        reasons.append("Le controle numerique de regression a echoue.")
    if domain_key == "regression":
        if regression_domain_check.get("applicable") and regression_domain_check.get("status") != "passed":
            reasons.append("Les donnees numeriques et le corrige de regression ne correspondent pas.")
    if domain_key == "exponential_law":
        if exponential_check.get("applicable") and exponential_check.get("status") != "passed":
            reasons.append("Le controle deterministe de loi exponentielle a echoue.")
        elif not exponential_check.get("applicable"):
            reasons.append("Le controle de loi exponentielle est requis mais non applicable.")
    if domain_key == "complex_numbers":
        complex_check = ((record.get("local_validation_checks") or {}).get("complex_numbers") or {})
        if complex_check.get("applicable") and complex_check.get("status") != "passed":
            reasons.append("Le controle deterministe des nombres complexes a echoue.")
        elif not complex_check.get("applicable"):
            reasons.append("Le controle deterministe des nombres complexes est requis mais non applicable.")
    if domain_key == "linear_systems":
        linear_check = ((record.get("local_validation_checks") or {}).get("linear_systems") or {})
        if linear_check.get("applicable") and linear_check.get("status") != "passed":
            reasons.append("Le controle deterministe du systeme lineaire a echoue.")
        elif not linear_check.get("applicable"):
            reasons.append("Le controle deterministe du systeme lineaire est requis mais non applicable.")
    if domain_key == "graphs":
        graph_ok, graph_issues, _graph_meta = validate_graph_support(record)
        if not graph_ok:
            reasons.extend(graph_issues)
    if "suite" in topic_text and sequence_check.get("status") == "failed":
        reasons.append("Le controle local de suites numeriques a echoue.")
    if "equation differentielle" in topic_text and ode_check.get("status") == "failed":
        reasons.append("Le controle local d'equation differentielle a echoue.")

    for issue in _ensure_list(record.get("local_validation_issues")):
        if issue not in reasons:
            reasons.append(str(issue))
    for issue in _ensure_list(record.get("pedagogical_completeness_issues")):
        if issue not in reasons:
            reasons.append(str(issue))
    for issue in _ensure_list(record.get("judge_issues")):
        lowered = _normalized(issue)
        if any(token in lowered for token in ("juge indisponible", "couple non couvert", "erreur", "incorrect", "incoher")):
            reasons.append(str(issue))

    reasons = _deduplicate(reasons)
    return not reasons, reasons


def apply_final_display_decision(record: dict[str, Any]) -> dict[str, Any]:
    """Persist the central display gate result back onto the exercise record."""
    exercise, latex_changed = repair_exercise_math_locally(deepcopy(record))
    exercise["latex_repair_applied"] = bool(latex_changed or exercise.get("math_format_repair_applied"))
    student_format_ok, student_format_issues = validate_student_facing_math_text(exercise)
    coverage_ok, coverage_issues, coverage_meta = validate_question_answer_coverage(exercise)
    domain_key = get_domain_validator_key(exercise.get("topic", ""), exercise.get("subtopic", ""), exercise.get("generation_metadata") or {})
    allowed, reasons = can_present_exercise(exercise)
    exercise["student_facing_format_flag"] = "approved" if student_format_ok else "wrong"
    exercise["student_facing_format_issues"] = student_format_issues
    exercise["student_facing_format_after_repair"] = "approved" if student_format_ok else "wrong"
    exercise["latex_repair_issues_remaining"] = student_format_issues
    exercise["question_coverage_flag"] = "approved" if coverage_ok else "wrong"
    exercise["question_coverage_issues"] = coverage_issues
    exercise["unanswered_question_indices"] = coverage_meta.get("unanswered_question_indices", [])
    exercise["domain_router_key"] = domain_key or ""
    exercise["domain_router_reason"] = explain_domain_route(exercise.get("topic", ""), exercise.get("subtopic", ""), exercise.get("generation_metadata") or {})
    exercise["final_gate_steps"] = ["latex_repair", "student_format", "schema_questions", "question_coverage", "domain_router", "domain_validator"]
    exercise["final_gate_failed_step"] = "" if allowed else (reasons[0] if reasons else "unknown")
    exercise["final_display_decision"] = "presented" if allowed else "blocked"
    exercise["final_display_blocking_reasons"] = reasons
    exercise["blocked_before_display"] = not allowed
    return exercise


def _normalized(value: Any) -> str:
    return str(value or "").strip().lower()


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean_value = str(value or "").strip()
        if not clean_value:
            continue
        key = clean_value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(clean_value)
    return result


def _extract_unresolved_corrected_issues(issues: Any) -> list[str]:
    unresolved: list[str] = []
    for issue in _ensure_list(issues):
        text = str(issue or "").strip()
        lowered = _normalized(text)
        if not lowered:
            continue
        if any(
            marker in lowered
            for marker in (
                "erreur",
                "incorrect",
                "incoher",
                "contradic",
                "problemat",
                "faux",
                "fausse",
                "mauvais",
                "non defini",
                "derivee",
                "probabil",
                "racine",
                "suite",
                "recurrence",
                "conique",
            )
        ):
            unresolved.append(f"Issue du juge non resolue apres correction: {text}")
    return unresolved
