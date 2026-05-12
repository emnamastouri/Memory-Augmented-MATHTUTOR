from __future__ import annotations

from copy import deepcopy
from typing import Any

from frontend.utils.validators.local_math_validators import instruction_requires_visual_support


BLOCKING_JUDGE_FLAGS = {
    "judge_error",
    "wrong",
    "misaligned",
    "unknown",
    "blocked_after_retries",
    "blocked_missing_alignment",
}


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
    symbolic_required = bool(record.get("symbolic_checks_required", False))
    symbolic_passed = record.get("symbolic_checks_passed")
    formatting_status = _extract_check_status(record, "formatting")
    pedagogical_flag = _normalized(record.get("pedagogical_completeness_flag"))

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
    if generation_backend in {"dataset-fallback", "local-fallback"} and not bool(record.get("fallback_revalidated", False)):
        reasons.append(f"La sortie {generation_backend} n'a pas ete revalidee completement.")
    if generation_backend in {"dataset-fallback", "local-fallback"}:
        if not bool(record.get("symbolic_checks_ran", False)) or symbolic_passed is not True:
            reasons.append(f"La sortie {generation_backend} n'a pas passe une validation symbolique deterministe.")
        if formatting_status != "passed":
            reasons.append(f"La sortie {generation_backend} n'a pas passe le controle de format student-facing.")
    if judge_flag == "corrected":
        corrected_fields_applied = bool(
            record.get("corrected_fields_applied", record.get("judge_corrections_applied", False))
        )
        if not corrected_fields_applied:
            reasons.append("Le juge a corrige l'exercice, mais les champs corriges n'ont pas ete appliques.")
        unresolved_correction_issues = _extract_unresolved_corrected_issues(record.get("judge_issues"))
        reasons.extend(unresolved_correction_issues)
    if symbolic_required and symbolic_passed is not True:
        reasons.append("Une validation symbolique etait requise, mais elle n'a pas ete confirmee.")
    if pedagogical_flag == "incomplete":
        reasons.append("L'exercice reste pedagogiquement incomplet pour la notion annoncee.")

    if instruction_requires_visual_support(record):
        if not bool(record.get("support_ready", False)):
            reasons.append("Le support visuel requis n'est pas pret.")
        if not record.get("chart_data") and not record.get("table_data") and not record.get("graph_data"):
            reasons.append("Le support visuel requis est absent.")

    for issue in _ensure_list(record.get("local_validation_issues")):
        if issue not in reasons:
            reasons.append(str(issue))
    for issue in _ensure_list(record.get("pedagogical_completeness_issues")):
        if issue not in reasons:
            reasons.append(str(issue))
    for issue in _ensure_list(record.get("judge_issues")):
        lowered = _normalized(issue)
        if "couple non couvert" in lowered or "juge indisponible" in lowered:
            reasons.append(str(issue))

    reasons = _deduplicate(reasons)
    return not reasons, reasons


def apply_final_display_decision(record: dict[str, Any]) -> dict[str, Any]:
    """Persist the central display gate result back onto the exercise record."""
    exercise = deepcopy(record)
    allowed, reasons = can_present_exercise(exercise)
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


def _extract_check_status(record: dict[str, Any], check_name: str) -> str:
    checks = record.get("local_validation_checks")
    if not isinstance(checks, dict):
        return ""
    check = checks.get(check_name)
    if not isinstance(check, dict):
        return ""
    return _normalized(check.get("status"))


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
