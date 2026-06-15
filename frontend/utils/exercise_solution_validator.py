"""Secondary validation of generated exercises using Qwen plus local SymPy checks."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any
import unicodedata

try:
    import sympy as sp
except ImportError:  # pragma: no cover
    sp = None

from frontend.utils.openrouter_client import (
    call_openrouter_chat,
    extract_json_object,
    extract_openrouter_text,
    get_openrouter_client,
    get_openrouter_settings,
    has_openrouter_config,
    parse_json_object_detailed,
    summarize_openrouter_response_issue,
)
from frontend.utils.validators.local_math_validators import validate_exercise_locally


@dataclass(frozen=True)
class ExerciseSolutionValidationDecision:
    """Structured result returned by the secondary validator."""

    decision: str
    summary: str
    issues: list[str]
    confidence: float
    model_name: str
    sympy_report: str
    normalized_fields: dict[str, Any]
    validation_status_label: str
    local_validation_flag: str
    local_validation_summary: str
    local_validation_issues: list[str]
    pedagogical_completeness_flag: str
    pedagogical_completeness_summary: str
    pedagogical_completeness_issues: list[str]
    symbolic_checks_ran: bool
    symbolic_checks_passed: bool | None
    symbolic_checks_required: bool
    error_message: str = ""


def validate_exercise_solution(
    exercise: dict[str, Any],
    *,
    level: str,
    section: str,
    topic: str,
    subtopic: str,
    exercise_type: str,
) -> ExerciseSolutionValidationDecision:
    """Validate that the generated solution is coherent with the statement and SymPy-checkable when possible."""
    settings = get_openrouter_settings()
    model_name = settings.exercise_model if settings is not None else "qwen/qwen-2.5-7b-instruct"

    local_issues = _collect_local_issues(exercise)
    if local_issues:
        return ExerciseSolutionValidationDecision(
            decision="rejected",
            summary="La solution interne est trop incomplete pour etre validee.",
            issues=local_issues,
            confidence=1.0,
            model_name="local-sympy-guard",
            sympy_report="Controle local interrompu avant appel LLM.",
            normalized_fields={},
            validation_status_label="Rejetée par validation locale",
            local_validation_flag="wrong",
            local_validation_summary="La solution interne est trop incomplete pour etre validee.",
            local_validation_issues=local_issues,
            pedagogical_completeness_flag="not_applicable",
            pedagogical_completeness_summary="",
            pedagogical_completeness_issues=[],
            symbolic_checks_ran=False,
            symbolic_checks_passed=False,
            symbolic_checks_required=False,
            error_message="Validation locale incomplete.",
        )

    if not has_openrouter_config() or settings is None:
        return ExerciseSolutionValidationDecision(
            decision="error",
            summary="Le validateur LLM de solution n'est pas configure.",
            issues=["Ajoutez la section [openrouter] avec api_key dans .streamlit/secrets.toml."],
            confidence=0.0,
            model_name=model_name,
            sympy_report="Aucun controle LLM disponible.",
            normalized_fields={},
            validation_status_label="Rejetée par validation locale",
            local_validation_flag="wrong",
            local_validation_summary="Le validateur LLM de solution n'est pas configure.",
            local_validation_issues=["Ajoutez la section [openrouter] avec api_key dans .streamlit/secrets.toml."],
            pedagogical_completeness_flag="not_applicable",
            pedagogical_completeness_summary="",
            pedagogical_completeness_issues=[],
            symbolic_checks_ran=False,
            symbolic_checks_passed=False,
            symbolic_checks_required=False,
            error_message="Configuration OpenRouter absente.",
        )

    prompt = _build_solution_validator_prompt(
        exercise=exercise,
        level=level,
        section=section,
        topic=topic,
        subtopic=subtopic,
        exercise_type=exercise_type,
    )

    try:
        payload = _call_openrouter_solution_validator(prompt, model_name)
    except RuntimeError as exc:
        return ExerciseSolutionValidationDecision(
            decision="error",
            summary="Le validateur LLM de solution est temporairement indisponible.",
            issues=[str(exc)],
            confidence=0.0,
            model_name=model_name,
            sympy_report="Le controle symbolique n'a pas pu etre confirme par l'agent LLM.",
            normalized_fields={},
            validation_status_label="Rejetée par validation locale",
            local_validation_flag="wrong",
            local_validation_summary="Le validateur LLM de solution est temporairement indisponible.",
            local_validation_issues=[str(exc)],
            pedagogical_completeness_flag="not_applicable",
            pedagogical_completeness_summary="",
            pedagogical_completeness_issues=[],
            symbolic_checks_ran=False,
            symbolic_checks_passed=False,
            symbolic_checks_required=False,
            error_message=str(exc),
        )

    decision = str(payload.get("decision", "")).strip().lower()
    if decision not in {"approved", "rejected"}:
        decision = "error"

    issues = payload.get("issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)]
    issues = [str(item).strip() for item in issues if str(item).strip()]
    solution_final_answer = str(payload.get("solution_final_answer", "")).strip()
    correct_option_text = str(payload.get("correct_option_text", "")).strip()

    sympy_outcome = _run_sympy_consistency_checks(
        exercise=exercise,
        solution_final_answer=solution_final_answer,
        correct_option_text=correct_option_text,
    )

    combined_issues = [*issues, *sympy_outcome["issues"]]
    combined_issues = [item for item in combined_issues if item]

    if decision == "rejected" or sympy_outcome["decision"] == "rejected":
        rejection_summary = str(payload.get("summary", "")).strip()
        if sympy_outcome["decision"] == "rejected" and decision != "rejected":
            rejection_summary = sympy_outcome["summary"]
        return ExerciseSolutionValidationDecision(
            decision="rejected",
            summary=rejection_summary or sympy_outcome["summary"] or "La solution interne ne correspond pas assez clairement a l'enonce.",
            issues=combined_issues,
            confidence=_coerce_confidence(payload.get("confidence")),
            model_name=model_name,
            sympy_report=sympy_outcome["report"],
            normalized_fields={},
            validation_status_label=sympy_outcome["status_label"],
            local_validation_flag=sympy_outcome["local_validation_flag"],
            local_validation_summary=sympy_outcome["local_validation_summary"],
            local_validation_issues=sympy_outcome["local_validation_issues"],
            pedagogical_completeness_flag=sympy_outcome["pedagogical_completeness_flag"],
            pedagogical_completeness_summary=sympy_outcome["pedagogical_completeness_summary"],
            pedagogical_completeness_issues=sympy_outcome["pedagogical_completeness_issues"],
            symbolic_checks_ran=sympy_outcome["symbolic_checks_ran"],
            symbolic_checks_passed=sympy_outcome["symbolic_checks_passed"],
            symbolic_checks_required=sympy_outcome["symbolic_checks_required"],
            error_message="",
        )

    if decision == "error":
        return ExerciseSolutionValidationDecision(
            decision="error",
            summary="Le format de sortie du validateur de solution est invalide.",
            issues=combined_issues or ["Le validateur n'a pas renvoye un JSON conforme."],
            confidence=0.0,
            model_name=model_name,
            sympy_report=sympy_outcome["report"],
            normalized_fields={},
            validation_status_label=sympy_outcome["status_label"],
            local_validation_flag=sympy_outcome["local_validation_flag"],
            local_validation_summary=sympy_outcome["local_validation_summary"],
            local_validation_issues=sympy_outcome["local_validation_issues"],
            pedagogical_completeness_flag=sympy_outcome["pedagogical_completeness_flag"],
            pedagogical_completeness_summary=sympy_outcome["pedagogical_completeness_summary"],
            pedagogical_completeness_issues=sympy_outcome["pedagogical_completeness_issues"],
            symbolic_checks_ran=sympy_outcome["symbolic_checks_ran"],
            symbolic_checks_passed=sympy_outcome["symbolic_checks_passed"],
            symbolic_checks_required=sympy_outcome["symbolic_checks_required"],
            error_message="Format JSON invalide.",
        )

    normalized_fields = dict(sympy_outcome["normalized_fields"])
    normalized_fields["solution_validation_llm_answer"] = solution_final_answer
    normalized_fields["solution_validation_correct_option"] = correct_option_text

    return ExerciseSolutionValidationDecision(
        decision="approved",
        summary=str(payload.get("summary", "")).strip() or sympy_outcome["summary"] or "Solution validee.",
        issues=combined_issues,
        confidence=_coerce_confidence(payload.get("confidence")),
        model_name=model_name,
        sympy_report=sympy_outcome["report"],
        normalized_fields=normalized_fields,
        validation_status_label=sympy_outcome["status_label"],
        local_validation_flag=sympy_outcome["local_validation_flag"],
        local_validation_summary=sympy_outcome["local_validation_summary"],
        local_validation_issues=sympy_outcome["local_validation_issues"],
        pedagogical_completeness_flag=sympy_outcome["pedagogical_completeness_flag"],
        pedagogical_completeness_summary=sympy_outcome["pedagogical_completeness_summary"],
        pedagogical_completeness_issues=sympy_outcome["pedagogical_completeness_issues"],
        symbolic_checks_ran=sympy_outcome["symbolic_checks_ran"],
        symbolic_checks_passed=sympy_outcome["symbolic_checks_passed"],
        symbolic_checks_required=sympy_outcome["symbolic_checks_required"],
        error_message="",
    )


def _build_solution_validator_prompt(
    *,
    exercise: dict[str, Any],
    level: str,
    section: str,
    topic: str,
    subtopic: str,
    exercise_type: str,
) -> str:
    """Create the prompt used to validate statement/solution coherence."""
    options = exercise.get("options", []) or []
    option_block = "\n".join([f"- {option}" for option in options]) if options else "- Aucun choix propose"
    accepted_answers = exercise.get("accepted_answers", []) or []
    accepted_block = "\n".join([f"- {answer}" for answer in accepted_answers]) if accepted_answers else "- Aucune"
    solution_steps = "\n".join([f"- {step}" for step in exercise.get("solution_steps", [])]) or "- Aucune etape fournie"

    return (
        "Tu es l'agent de validation mathematique interne de MathTutorAI. "
        "Tu passes apres le juge d'alignement. Ta mission est de verifier que l'enonce, "
        "la reponse attendue et la solution complete sont coherents entre eux et mathematiquement defendables. "
        "Quand c'est possible, extrais la reponse finale de la solution dans une forme courte compatible avec SymPy.\n\n"
        "Regles:\n"
        "- approved : la solution interne est compatible avec l'enonce et la reponse attendue.\n"
        "- rejected : l'enonce, la reponse attendue ou la solution sont incoherents, faux, ambigus ou non verifiables.\n"
        "- Si tu as un doute mathematique, choisis rejected.\n"
        "- Pour un QCM, indique le texte exact de la bonne option dans correct_option_text si tu peux l'identifier.\n"
        "- Pour une reponse mathematique, indique la reponse finale courte dans solution_final_answer.\n\n"
        f"Demande cible\n- Niveau : {level}\n- Section : {section}\n- Theme : {topic}\n- Sous-theme : {subtopic}\n- Type : {exercise_type}\n\n"
        "Exercice a valider\n"
        f"- Titre : {exercise.get('title', '')}\n"
        f"- Enonce : {exercise.get('prompt', '')}\n"
        f"- Reponse attendue : {exercise.get('display_answer', '')}\n"
        f"- Nature de reponse : {exercise.get('answer_kind', '')}\n"
        "Reponses acceptees\n"
        f"{accepted_block}\n"
        "Choix proposes\n"
        f"{option_block}\n"
        "Etapes de solution\n"
        f"{solution_steps}\n"
        "Solution complete interne\n"
        f"{exercise.get('hidden_solution', '')}\n\n"
        "Reponds avec un seul objet JSON valide de la forme : "
        "{decision, summary, issues, confidence, solution_final_answer, correct_option_text}. "
        "N'ajoute aucun texte avant ou apres le JSON."
    )


def _call_openrouter_solution_validator(prompt: str, model_name: str) -> dict[str, Any]:
    """Call the Qwen validator model and parse its JSON output."""
    messages = [
        {
            "role": "system",
            "content": (
                "Tu es l'agent de validation mathematique de MathTutorAI. "
                "Retourne uniquement un objet JSON valide, sans markdown et sans texte additionnel."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    result = call_openrouter_chat(
        model=model_name,
        messages=messages,
        temperature=0,
        top_p=0.1,
        max_tokens=3000,
        purpose="validator",
        json_schema=_solution_validator_schema(),
    )
    if not result.ok:
        raise RuntimeError(result.error_message or result.error_type or "OpenRouter validator call failed.")
    parse_result = parse_json_object_detailed(result.content)
    payload = parse_result.data or {}
    if not payload:
        raise RuntimeError("Le validateur de solution n'a pas renvoye un JSON exploitable.")
    return payload


def _solution_validator_schema() -> dict[str, Any]:
    return {
        "name": "mathtutorai_solution_validator",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": True,
            "required": ["decision", "summary", "issues", "confidence", "solution_final_answer", "correct_option_text"],
            "properties": {
                "decision": {"type": "string", "enum": ["approved", "rejected"]},
                "summary": {"type": "string"},
                "issues": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
                "solution_final_answer": {"type": "string"},
                "correct_option_text": {"type": "string"},
                "requires_symbolic_check": {"type": "boolean"},
                "symbolic_check_recommendation": {"type": "string"},
            },
        },
    }


def _collect_local_issues(exercise: dict[str, Any]) -> list[str]:
    """Reject obviously incomplete exercises before the expensive model call."""
    issues: list[str] = []
    if not str(exercise.get("prompt", "")).strip():
        issues.append("L'enonce est vide.")
    if not str(exercise.get("display_answer", "")).strip():
        issues.append("La reponse attendue est absente.")
    if not str(exercise.get("hidden_solution", "")).strip():
        issues.append("La solution complete interne est absente.")
    if exercise.get("exercise_type") == "QCM" and len(exercise.get("options", []) or []) < 2:
        issues.append("Le QCM ne contient pas assez de choix pour une verification fiable.")
    return issues


def _domain_validator_summary(checks: dict[str, Any]) -> tuple[str, str, list[str]]:
    for name in ("bayes", "regression_deterministic", "exponential_law", "probability"):
        outcome = checks.get(name) or {}
        if outcome.get("applicable"):
            return name, "approved" if outcome.get("status") == "passed" else "wrong", list(outcome.get("issues") or [])
    return "", "", []


def _run_sympy_consistency_checks(
    *,
    exercise: dict[str, Any],
    solution_final_answer: str,
    correct_option_text: str,
) -> dict[str, Any]:
    """Check the mathematical consistency of the expected answer and solution."""
    display_answer = str(exercise.get("display_answer", "")).strip()
    accepted_answers = [str(answer).strip() for answer in (exercise.get("accepted_answers", []) or []) if str(answer).strip()]
    answer_kind = str(exercise.get("answer_kind", "text")).strip().lower() or "text"
    exercise_type = str(exercise.get("exercise_type", "")).strip()
    options = [str(option).strip() for option in (exercise.get("options", []) or []) if str(option).strip()]
    extracted_from_solution = solution_final_answer or _extract_final_answer_from_solution_text(
        str(exercise.get("hidden_solution", ""))
    )

    issues: list[str] = []
    report_parts: list[str] = []
    local_record = dict(exercise)
    local_record["solution_validation_llm_answer"] = solution_final_answer
    local_record["solution_validation_correct_option"] = correct_option_text
    local_outcome = validate_exercise_locally(local_record)
    normalized_fields: dict[str, Any] = {
        "local_validation_flag": local_outcome["local_validation_flag"],
        "local_validation_summary": local_outcome["local_validation_summary"],
        "local_validation_issues": local_outcome["local_validation_issues"],
        "local_validation_checks": local_outcome["checks"],
        "pedagogical_completeness_flag": local_outcome["pedagogical_completeness_flag"],
        "pedagogical_completeness_summary": local_outcome["pedagogical_completeness_summary"],
        "pedagogical_completeness_issues": local_outcome["pedagogical_completeness_issues"],
        "symbolic_checks_ran": local_outcome["symbolic_checks_ran"],
        "symbolic_checks_passed": local_outcome["symbolic_checks_passed"],
        "symbolic_checks_required": local_outcome["symbolic_checks_required"],
        "domain_router_key": local_outcome.get("domain_router_key", ""),
        "domain_router_reason": local_outcome.get("domain_router_reason", ""),
    }
    domain_name, domain_flag, domain_issues = _domain_validator_summary(local_outcome.get("checks", {}))
    if domain_name:
        normalized_fields.update(
            {
                "domain_validator_name": domain_name,
                "domain_validator_flag": domain_flag,
                "domain_validator_issues": domain_issues,
            }
        )
    symbolic_checks_ran = bool(local_outcome["symbolic_checks_ran"])
    symbolic_checks_passed = local_outcome["symbolic_checks_passed"]
    symbolic_checks_required = bool(local_outcome["symbolic_checks_required"])
    issues.extend(local_outcome["local_validation_issues"])
    if local_outcome["local_validation_summary"]:
        report_parts.append(local_outcome["local_validation_summary"])
    if local_outcome["local_validation_flag"] == "wrong":
        return {
            "decision": "rejected",
            "summary": local_outcome["local_validation_summary"] or "La validation locale a rejete cet exercice.",
            "issues": _deduplicate_preserving_order(issues),
            "report": " | ".join(part for part in report_parts if part),
            "normalized_fields": normalized_fields,
            "status_label": "Rejetée par validation locale",
            "local_validation_flag": local_outcome["local_validation_flag"],
            "local_validation_summary": local_outcome["local_validation_summary"],
            "local_validation_issues": local_outcome["local_validation_issues"],
            "pedagogical_completeness_flag": local_outcome["pedagogical_completeness_flag"],
            "pedagogical_completeness_summary": local_outcome["pedagogical_completeness_summary"],
            "pedagogical_completeness_issues": local_outcome["pedagogical_completeness_issues"],
            "symbolic_checks_ran": symbolic_checks_ran,
            "symbolic_checks_passed": symbolic_checks_passed,
            "symbolic_checks_required": symbolic_checks_required,
        }

    domain_checks = local_outcome.get("checks", {})
    approved_domain = next(
        (
            name
            for name in ("bayes", "regression_deterministic", "exponential_law", "probability")
            if (domain_checks.get(name) or {}).get("applicable")
            and (domain_checks.get(name) or {}).get("status") == "passed"
        ),
        "",
    )
    if approved_domain:
        normalized_fields.update(
            {
                "domain_validator_name": approved_domain,
                "domain_validator_flag": "approved",
                "domain_validator_issues": [],
                "verification_ready": True,
                "verification_message": "La solution a ete controlee par un validateur deterministe de domaine.",
            }
        )
        return {
            "decision": "approved",
            "summary": f"Validation deterministe de domaine approuvee : {approved_domain}.",
            "issues": [],
            "report": " | ".join(part for part in report_parts if part),
            "normalized_fields": normalized_fields,
            "status_label": _validation_status_label(
                symbolic_checks_ran=symbolic_checks_ran,
                symbolic_checks_passed=symbolic_checks_passed,
                symbolic_checks_required=symbolic_checks_required,
            ),
            "local_validation_flag": local_outcome["local_validation_flag"],
            "local_validation_summary": local_outcome["local_validation_summary"],
            "local_validation_issues": local_outcome["local_validation_issues"],
            "pedagogical_completeness_flag": local_outcome["pedagogical_completeness_flag"],
            "pedagogical_completeness_summary": local_outcome["pedagogical_completeness_summary"],
            "pedagogical_completeness_issues": local_outcome["pedagogical_completeness_issues"],
            "symbolic_checks_ran": symbolic_checks_ran,
            "symbolic_checks_passed": symbolic_checks_passed,
            "symbolic_checks_required": symbolic_checks_required,
        }

    if exercise_type == "QCM":
        match = _resolve_correct_qcm_option(
            options=options,
            display_answer=display_answer,
            extracted_answer=extracted_from_solution,
            declared_option=correct_option_text,
            answer_kind=answer_kind,
        )
        issues.extend(match["issues"])
        report_parts.append(match["report"])
        if match["decision"] == "rejected" or issues:
            return {
                "decision": "rejected",
                "summary": "Le QCM n'expose pas clairement une seule bonne option mathematiquement defendable.",
                "issues": _deduplicate_preserving_order(issues),
                "report": " | ".join(part for part in report_parts if part),
                "normalized_fields": {},
                "status_label": "Rejetée par validation locale",
                "local_validation_flag": local_outcome["local_validation_flag"],
                "local_validation_summary": local_outcome["local_validation_summary"],
                "local_validation_issues": local_outcome["local_validation_issues"],
                "pedagogical_completeness_flag": local_outcome["pedagogical_completeness_flag"],
                "pedagogical_completeness_summary": local_outcome["pedagogical_completeness_summary"],
                "pedagogical_completeness_issues": local_outcome["pedagogical_completeness_issues"],
                "symbolic_checks_ran": symbolic_checks_ran,
                "symbolic_checks_passed": symbolic_checks_passed,
                "symbolic_checks_required": symbolic_checks_required,
            }
        accepted = _deduplicate_preserving_order(
            [
                match["correct_option"],
                display_answer,
                extracted_from_solution,
                *accepted_answers,
            ]
        )
        normalized_fields.update(
            {
                "accepted_answers": accepted,
                "answer_kind": "text",
                "verification_ready": True,
                "verification_message": "La bonne option a ete revalidee par Qwen et controlee localement.",
            }
        )
        status_label = _validation_status_label(
            symbolic_checks_ran=symbolic_checks_ran,
            symbolic_checks_passed=symbolic_checks_passed,
            symbolic_checks_required=symbolic_checks_required,
        )
        return {
            "decision": "approved",
            "summary": "La bonne option du QCM est coherente avec la solution interne.",
            "issues": issues,
            "report": " | ".join(part for part in report_parts if part),
            "normalized_fields": normalized_fields,
            "status_label": status_label,
            "local_validation_flag": local_outcome["local_validation_flag"],
            "local_validation_summary": local_outcome["local_validation_summary"],
            "local_validation_issues": local_outcome["local_validation_issues"],
            "pedagogical_completeness_flag": local_outcome["pedagogical_completeness_flag"],
            "pedagogical_completeness_summary": local_outcome["pedagogical_completeness_summary"],
            "pedagogical_completeness_issues": local_outcome["pedagogical_completeness_issues"],
            "symbolic_checks_ran": symbolic_checks_ran,
            "symbolic_checks_passed": symbolic_checks_passed,
            "symbolic_checks_required": symbolic_checks_required,
        }

    if answer_kind in {"numeric", "expression"}:
        symbolic_checks_required = True
        if sp is None:
            return {
                "decision": "rejected",
                "summary": "Validation symbolique requise mais SymPy est indisponible localement.",
                "issues": ["SymPy est requis pour verifier cet exercice numerique ou symbolique."],
                "report": "Validation symbolique requise mais indisponible.",
                "normalized_fields": normalized_fields,
                "status_label": "Rejetée par validation locale",
                "local_validation_flag": "wrong",
                "local_validation_summary": "Validation symbolique requise mais SymPy est indisponible localement.",
                "local_validation_issues": ["SymPy est requis pour verifier cet exercice numerique ou symbolique."],
                "pedagogical_completeness_flag": local_outcome["pedagogical_completeness_flag"],
                "pedagogical_completeness_summary": local_outcome["pedagogical_completeness_summary"],
                "pedagogical_completeness_issues": local_outcome["pedagogical_completeness_issues"],
                "symbolic_checks_ran": False,
                "symbolic_checks_passed": False,
                "symbolic_checks_required": True,
            }

        symbolic_checks_ran = True
        if not _is_sympy_parseable(display_answer):
            issues.append("La reponse attendue n'est pas interpretable par SymPy.")
        inconsistent_answers = [
            answer
            for answer in accepted_answers
            if not _answers_equivalent(answer, display_answer, answer_kind)
        ]
        if inconsistent_answers:
            issues.append("Au moins une reponse acceptee diverge de la reponse attendue.")
            report_parts.append("Reponses acceptees incoherentes : " + " | ".join(inconsistent_answers[:3]))

        if extracted_from_solution:
            if _answers_equivalent(extracted_from_solution, display_answer, answer_kind):
                report_parts.append("La reponse finale extraite de la solution est equivalente a la cible attendue.")
            else:
                issues.append("La solution complete conduit a une reponse finale differente de la cible attendue.")
                report_parts.append(
                    f"Solution finale extraite = {extracted_from_solution} ; cible attendue = {display_answer}."
                )
        else:
            issues.append("Impossible d'extraire une reponse finale exploitable depuis la solution complete.")
        symbolic_checks_passed = not issues

        if issues:
            return {
                "decision": "rejected",
                "summary": "La reponse attendue et la solution interne ne passent pas le controle symbolique.",
                "issues": _deduplicate_preserving_order(issues),
                "report": " | ".join(part for part in report_parts if part),
                "normalized_fields": normalized_fields,
                "status_label": "Rejetée par validation locale",
                "local_validation_flag": "wrong",
                "local_validation_summary": "La reponse attendue et la solution interne ne passent pas le controle symbolique.",
                "local_validation_issues": _deduplicate_preserving_order(issues),
                "pedagogical_completeness_flag": local_outcome["pedagogical_completeness_flag"],
                "pedagogical_completeness_summary": local_outcome["pedagogical_completeness_summary"],
                "pedagogical_completeness_issues": local_outcome["pedagogical_completeness_issues"],
                "symbolic_checks_ran": symbolic_checks_ran,
                "symbolic_checks_passed": False,
                "symbolic_checks_required": symbolic_checks_required,
            }

        normalized_fields.update(
            {
                "accepted_answers": _deduplicate_preserving_order(
                    [display_answer, extracted_from_solution, *accepted_answers]
                ),
                "verification_ready": True,
                "verification_message": "La solution interne a ete revalidee par Qwen puis controlee avec SymPy.",
            }
        )
        return {
            "decision": "approved",
            "summary": "La reponse attendue est coherente avec la solution et verifiable symboliquement.",
            "issues": [],
            "report": " | ".join(part for part in report_parts if part),
            "normalized_fields": normalized_fields,
            "status_label": _validation_status_label(
                symbolic_checks_ran=True,
                symbolic_checks_passed=True,
                symbolic_checks_required=True,
            ),
            "local_validation_flag": local_outcome["local_validation_flag"],
            "local_validation_summary": local_outcome["local_validation_summary"],
            "local_validation_issues": local_outcome["local_validation_issues"],
            "pedagogical_completeness_flag": local_outcome["pedagogical_completeness_flag"],
            "pedagogical_completeness_summary": local_outcome["pedagogical_completeness_summary"],
            "pedagogical_completeness_issues": local_outcome["pedagogical_completeness_issues"],
            "symbolic_checks_ran": True,
            "symbolic_checks_passed": True,
            "symbolic_checks_required": True,
        }

    if answer_kind == "set":
        expected_set = _parse_set_answer(display_answer)
        if not expected_set:
            issues.append("La reponse attendue sous forme d'ensemble est trop ambigue pour un controle local.")
        for answer in accepted_answers:
            if _parse_set_answer(answer) != expected_set:
                issues.append("Une forme acceptee de l'ensemble de solutions n'est pas coherente.")
                break
        if extracted_from_solution and _parse_set_answer(extracted_from_solution) != expected_set:
            issues.append("La solution complete mene a un ensemble de solutions different.")

        if issues:
            return {
                "decision": "rejected",
                "summary": "L'ensemble de solutions n'est pas coherent entre cible et solution interne.",
                "issues": _deduplicate_preserving_order(issues),
                "report": "Controle local des ensembles non concluant.",
                "normalized_fields": normalized_fields,
                "status_label": "Rejetée par validation locale",
                "local_validation_flag": "wrong",
                "local_validation_summary": "L'ensemble de solutions n'est pas coherent entre cible et solution interne.",
                "local_validation_issues": _deduplicate_preserving_order(issues),
                "pedagogical_completeness_flag": local_outcome["pedagogical_completeness_flag"],
                "pedagogical_completeness_summary": local_outcome["pedagogical_completeness_summary"],
                "pedagogical_completeness_issues": local_outcome["pedagogical_completeness_issues"],
                "symbolic_checks_ran": symbolic_checks_ran,
                "symbolic_checks_passed": False if symbolic_checks_required else symbolic_checks_passed,
                "symbolic_checks_required": symbolic_checks_required,
            }

        normalized_fields.update(
            {
                "accepted_answers": _deduplicate_preserving_order([display_answer, extracted_from_solution, *accepted_answers]),
                "verification_ready": True,
                "verification_message": "L'ensemble de solutions a ete revalide et normalise localement.",
            }
        )
        return {
            "decision": "approved",
            "summary": "L'ensemble de solutions reste coherent avec la solution interne.",
            "issues": [],
            "report": "Controle local des ensembles reussi.",
            "normalized_fields": normalized_fields,
            "status_label": _validation_status_label(
                symbolic_checks_ran=symbolic_checks_ran,
                symbolic_checks_passed=symbolic_checks_passed,
                symbolic_checks_required=symbolic_checks_required,
            ),
            "local_validation_flag": local_outcome["local_validation_flag"],
            "local_validation_summary": local_outcome["local_validation_summary"],
            "local_validation_issues": local_outcome["local_validation_issues"],
            "pedagogical_completeness_flag": local_outcome["pedagogical_completeness_flag"],
            "pedagogical_completeness_summary": local_outcome["pedagogical_completeness_summary"],
            "pedagogical_completeness_issues": local_outcome["pedagogical_completeness_issues"],
            "symbolic_checks_ran": symbolic_checks_ran,
            "symbolic_checks_passed": symbolic_checks_passed,
            "symbolic_checks_required": symbolic_checks_required,
        }

    if issues:
        return {
            "decision": "rejected",
            "summary": "Le format textuel n'a pas passe les controles locaux de coherence.",
            "issues": _deduplicate_preserving_order(issues),
            "report": " | ".join(part for part in report_parts if part),
            "normalized_fields": normalized_fields,
            "status_label": "Rejetée par validation locale",
            "local_validation_flag": "wrong",
            "local_validation_summary": "Le format textuel n'a pas passe les controles locaux de coherence.",
            "local_validation_issues": _deduplicate_preserving_order(issues),
            "pedagogical_completeness_flag": local_outcome["pedagogical_completeness_flag"],
            "pedagogical_completeness_summary": local_outcome["pedagogical_completeness_summary"],
            "pedagogical_completeness_issues": local_outcome["pedagogical_completeness_issues"],
            "symbolic_checks_ran": symbolic_checks_ran,
            "symbolic_checks_passed": False if symbolic_checks_required else symbolic_checks_passed,
            "symbolic_checks_required": symbolic_checks_required,
        }

    status_label = _validation_status_label(
        symbolic_checks_ran=symbolic_checks_ran,
        symbolic_checks_passed=symbolic_checks_passed,
        symbolic_checks_required=symbolic_checks_required,
    )
    report = " | ".join(
        part
        for part in [
            *report_parts,
            "Controle symbolique local execute." if symbolic_checks_ran else "Format non symbolique : controle LLM seulement.",
        ]
        if part
    )
    normalized_fields.update(
        {
            "verification_ready": bool(symbolic_checks_ran),
            "verification_message": (
                "Validation LLM terminee et controle local deterministe confirme."
                if symbolic_checks_ran
                else "Validation LLM terminee, mais ce format textuel n'appelle pas de controle symbolique complet."
            ),
        }
    )
    return {
        "decision": "approved",
        "summary": (
            "Le format textuel a ete approuve par l'agent LLM et par les controles locaux applicables."
            if symbolic_checks_ran
            else "Le format textuel a ete approuve par l'agent LLM, sans controle SymPy complet."
        ),
        "issues": [],
        "report": report,
        "normalized_fields": normalized_fields,
        "status_label": status_label,
        "local_validation_flag": local_outcome["local_validation_flag"],
        "local_validation_summary": local_outcome["local_validation_summary"],
        "local_validation_issues": local_outcome["local_validation_issues"],
        "pedagogical_completeness_flag": local_outcome["pedagogical_completeness_flag"],
        "pedagogical_completeness_summary": local_outcome["pedagogical_completeness_summary"],
        "pedagogical_completeness_issues": local_outcome["pedagogical_completeness_issues"],
        "symbolic_checks_ran": symbolic_checks_ran,
        "symbolic_checks_passed": symbolic_checks_passed,
        "symbolic_checks_required": symbolic_checks_required,
    }


def _run_domain_specific_checks(
    *,
    exercise: dict[str, Any],
    display_answer: str,
    extracted_answer: str,
    answer_kind: str,
) -> dict[str, Any]:
    """Run stricter local math checks for common high-signal Bac topics."""
    if sp is None:
        return {"issues": [], "report": ""}

    prompt = str(exercise.get("prompt", "")).strip()
    context = _normalize_lookup(
        " ".join(
            [
                str(exercise.get("title", "")),
                str(exercise.get("topic", "")),
                str(exercise.get("subtopic", "")),
                prompt,
            ]
        )
    )
    issues: list[str] = []
    report_parts: list[str] = []

    if _looks_like_derivative_context(context):
        derivative_check = _check_derivative_consistency(prompt, display_answer, extracted_answer, answer_kind)
        issues.extend(derivative_check["issues"])
        if derivative_check["report"]:
            report_parts.append(derivative_check["report"])

    if _looks_like_limit_context(context):
        limit_check = _check_simple_limit_consistency(prompt, display_answer, extracted_answer, answer_kind)
        issues.extend(limit_check["issues"])
        if limit_check["report"]:
            report_parts.append(limit_check["report"])

    if _looks_like_probability_context(context):
        probability_check = _check_probability_consistency(prompt, display_answer, extracted_answer, answer_kind)
        issues.extend(probability_check["issues"])
        if probability_check["report"]:
            report_parts.append(probability_check["report"])

    if _looks_like_algebraic_equality_context(context, answer_kind):
        equality_check = _check_algebraic_equality_consistency(display_answer, extracted_answer, answer_kind)
        issues.extend(equality_check["issues"])
        if equality_check["report"]:
            report_parts.append(equality_check["report"])

    deduped_issues = _deduplicate_preserving_order(issues)
    return {"issues": deduped_issues, "report": " | ".join(part for part in report_parts if part)}


def _run_global_exercise_guards(
    *,
    exercise: dict[str, Any],
    display_answer: str,
    extracted_answer: str,
    answer_kind: str,
) -> dict[str, Any]:
    """Run cross-cutting guards that should block any invalid exercise shape."""
    prompt = str(exercise.get("prompt", "")).strip()
    issues: list[str] = []
    report_parts: list[str] = []

    support_outcome = _check_visual_support_presence(exercise, prompt)
    issues.extend(support_outcome["issues"])
    if support_outcome["report"]:
        report_parts.append(support_outcome["report"])

    domain_outcome = _check_positive_domain_negated_argument(prompt)
    issues.extend(domain_outcome["issues"])
    if domain_outcome["report"]:
        report_parts.append(domain_outcome["report"])

    membership_outcome = _check_membership_condition_consistency(prompt)
    issues.extend(membership_outcome["issues"])
    if membership_outcome["report"]:
        report_parts.append(membership_outcome["report"])

    area_outcome = _check_area_sign_consistency(prompt, display_answer, extracted_answer, answer_kind)
    issues.extend(area_outcome["issues"])
    if area_outcome["report"]:
        report_parts.append(area_outcome["report"])

    return {
        "issues": _deduplicate_preserving_order(issues),
        "report": " | ".join(part for part in report_parts if part),
    }


def _looks_like_derivative_context(context: str) -> bool:
    return any(token in context for token in ("derivee", "derivation", "deriver"))


def _looks_like_limit_context(context: str) -> bool:
    return "limite" in context or "lim(" in context or "quand x tend vers" in context


def _looks_like_probability_context(context: str) -> bool:
    return "probabilite" in context and "esperance" not in context and "variance" not in context


def _looks_like_algebraic_equality_context(context: str, answer_kind: str) -> bool:
    return answer_kind in {"numeric", "expression"} and any(
        token in context for token in ("egalite", "simplifier", "developper", "factoriser", "resoudre", "verifier")
    )


def _check_derivative_consistency(
    prompt: str,
    display_answer: str,
    extracted_answer: str,
    answer_kind: str,
) -> dict[str, Any]:
    expression = _extract_function_expression(prompt)
    if not expression:
        return {"issues": [], "report": ""}
    try:
        x = sp.symbols("x")
        derivative = sp.diff(sp.sympify(_normalize_for_sympy(expression)), x)
        derivative_text = sp.sstr(derivative)
    except Exception:
        return {"issues": ["La fonction de depart n'est pas assez lisible pour recalculer sa derivee localement."], "report": ""}

    issues: list[str] = []
    if not _answers_equivalent(display_answer, derivative_text, answer_kind):
        issues.append("Le calcul local de la derivee ne correspond pas a la reponse attendue.")
    if extracted_answer and not _answers_equivalent(extracted_answer, derivative_text, answer_kind):
        issues.append("La solution complete n'aboutit pas a la derivee recalculee localement.")
    return {
        "issues": issues,
        "report": f"Derivee recalculee localement : {derivative_text}.",
    }


def _check_simple_limit_consistency(
    prompt: str,
    display_answer: str,
    extracted_answer: str,
    answer_kind: str,
) -> dict[str, Any]:
    limit_data = _extract_limit_expression(prompt)
    if not limit_data:
        return {"issues": [], "report": ""}
    point, expression = limit_data
    try:
        x = sp.symbols("x")
        limit_value = sp.limit(
            sp.sympify(_normalize_for_sympy(expression)),
            x,
            sp.sympify(_normalize_for_sympy(point)),
        )
        limit_text = sp.sstr(limit_value)
    except Exception:
        return {"issues": ["La limite simple n'a pas pu etre recontrolee localement."], "report": ""}

    issues: list[str] = []
    if not _answers_equivalent(display_answer, limit_text, answer_kind):
        issues.append("Le calcul local de la limite ne correspond pas a la reponse attendue.")
    if extracted_answer and not _answers_equivalent(extracted_answer, limit_text, answer_kind):
        issues.append("La solution complete n'aboutit pas a la limite recalculee localement.")
    return {
        "issues": issues,
        "report": f"Limite recalculee localement : {limit_text}.",
    }


def _check_probability_consistency(
    prompt: str,
    display_answer: str,
    extracted_answer: str,
    answer_kind: str,
) -> dict[str, Any]:
    issues: list[str] = []
    report_parts: list[str] = []

    probability_value = _sympy_value_if_possible(display_answer)
    if probability_value is not None:
        try:
            probability_float = float(probability_value)
            if probability_float < 0 or probability_float > 1:
                issues.append("Une probabilite attendue doit rester comprise entre 0 et 1.")
        except Exception:
            pass

    if extracted_answer:
        extracted_probability = _sympy_value_if_possible(extracted_answer)
        if extracted_probability is not None:
            try:
                extracted_float = float(extracted_probability)
                if extracted_float < 0 or extracted_float > 1:
                    issues.append("La probabilite finale extraite de la solution sort de l'intervalle [0, 1].")
            except Exception:
                pass

    die_probability = _extract_simple_die_probability(prompt)
    if die_probability is not None:
        die_text = sp.sstr(die_probability)
        report_parts.append(f"Probabilite simple recalculee localement : {die_text}.")
        if not _answers_equivalent(display_answer, die_text, answer_kind):
            issues.append("Le calcul local de probabilite ne correspond pas a la reponse attendue.")
        if extracted_answer and not _answers_equivalent(extracted_answer, die_text, answer_kind):
            issues.append("La solution complete n'aboutit pas a la probabilite recalculee localement.")

    return {"issues": issues, "report": " | ".join(report_parts)}


def _check_algebraic_equality_consistency(
    display_answer: str,
    extracted_answer: str,
    answer_kind: str,
) -> dict[str, Any]:
    if not display_answer:
        return {"issues": [], "report": ""}
    if not _is_sympy_parseable(display_answer):
        return {"issues": ["La cible algebrique attendue n'est pas interpretable localement."], "report": ""}
    if extracted_answer and not _answers_equivalent(extracted_answer, display_answer, answer_kind):
        return {
            "issues": ["La solution complete n'aboutit pas a une egalite algebrique equivalente a la cible."],
            "report": "Equivalence algebrique locale non confirmee.",
        }
    return {"issues": [], "report": "Equivalence algebrique confirmee localement."}


def _check_visual_support_presence(exercise: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Reject statements that explicitly mention a provided visual support but ship none."""
    normalized_prompt = _normalize_lookup(prompt)
    chart_required = any(
        token in normalized_prompt
        for token in ("courbe", "trace", "trac", "figure", "graphe", "graphique")
    )
    table_required = "tableau de variation donne" in normalized_prompt or "tableau de variation donné" in prompt.lower()

    issues: list[str] = []
    report_parts: list[str] = []
    has_chart = exercise.get("chart_data") is not None
    has_table = exercise.get("table_data") is not None

    if chart_required and not (has_chart or has_table):
        issues.append("L'enonce annonce un support visuel fourni, mais aucun chart_data ou table_data n'est attache.")
    if table_required and not has_table:
        issues.append("L'enonce mentionne un tableau de variation donne, mais table_data est absent.")

    if chart_required or table_required:
        report_parts.append(
            "Support visuel detecte ; chart_data present = "
            f"{'oui' if has_chart else 'non'} ; table_data present = {'oui' if has_table else 'non'}."
        )
    return {"issues": issues, "report": " | ".join(report_parts)}


def _check_positive_domain_negated_argument(prompt: str) -> dict[str, Any]:
    """Catch impossible f(-x) uses when the function domain is only positive."""
    normalized_prompt = _normalize_lookup(prompt)
    positive_domain_patterns = [
        r"sur\s*[\[\]]\s*0\s*,\s*\+\s*(?:inf|oo)",
        r"sur\s*[\]\[]\s*0\s*,\s*\+\s*(?:inf|oo)",
        r"sur\s*\]0\s*,\s*\+\s*(?:inf|oo)\[",
        r"sur\s*\[0\s*,\s*\+\s*(?:inf|oo)\[",
    ]
    uses_negated_argument = bool(re.search(r"\b[a-z]\s*\(\s*-\s*x\s*\)", normalized_prompt))
    positive_only_domain = any(re.search(pattern, normalized_prompt) for pattern in positive_domain_patterns)
    if uses_negated_argument and positive_only_domain:
        return {
            "issues": ["L'enonce utilise f(-x) alors que la fonction est definie seulement sur un domaine positif."],
            "report": "Controle de domaine : argument negatif detecte pour une fonction definie sur [0,+inf[ ou ]0,+inf[.",
        }
    return {"issues": [], "report": ""}


def _check_membership_condition_consistency(prompt: str) -> dict[str, Any]:
    """Substitute g(x) or y into the announced condition and verify the equality locally."""
    if sp is None:
        return {"issues": [], "report": ""}

    normalized_prompt = _normalize_lookup(prompt)
    if "appart" not in normalized_prompt and "verifie" not in normalized_prompt and "satisf" not in normalized_prompt:
        return {"issues": [], "report": ""}

    function_match = re.search(r"g\(x\)\s*=\s*([^\n;.,]+)", prompt, flags=re.IGNORECASE)
    if not function_match:
        return {"issues": [], "report": ""}

    g_expression = function_match.group(1).strip()
    condition_match = re.search(
        r"(?:condition|relation|equation|egalite|courbe d[' ]equation|droite d[' ]equation)\s*(?:\([A-Za-z]\))?\s*[:=]\s*([^\n]+=[^\n]+)",
        prompt,
        flags=re.IGNORECASE,
    )
    if not condition_match:
        return {"issues": [], "report": ""}

    condition = re.split(r"[.;]", condition_match.group(1).strip(), maxsplit=1)[0].strip()
    if "=" not in condition:
        return {"issues": [], "report": ""}
    left_text, right_text = [part.strip() for part in condition.split("=", 1)]

    try:
        x, y = sp.symbols("x y")
        g_expr = sp.sympify(_normalize_for_sympy(g_expression))
        left_prepared = _normalize_for_sympy(left_text).replace("g(x)", f"({sp.sstr(g_expr)})")
        right_prepared = _normalize_for_sympy(right_text).replace("g(x)", f"({sp.sstr(g_expr)})")
        left_expr = sp.sympify(left_prepared).subs({y: g_expr})
        right_expr = sp.sympify(right_prepared).subs({y: g_expr})
        if sp.simplify(left_expr - right_expr) != 0:
            return {
                "issues": ["La substitution de g(x) dans la condition d'appartenance ne verifie pas l'egalite annoncee."],
                "report": f"Controle d'appartenance : {sp.sstr(left_expr)} != {sp.sstr(right_expr)}.",
            }
        return {
            "issues": [],
            "report": f"Controle d'appartenance : {sp.sstr(left_expr)} = {sp.sstr(right_expr)} apres substitution.",
        }
    except Exception:
        return {
            "issues": ["La condition d'appartenance n'a pas pu etre verifiee localement par substitution."],
            "report": "",
        }


def _check_area_sign_consistency(
    prompt: str,
    display_answer: str,
    extracted_answer: str,
    answer_kind: str,
) -> dict[str, Any]:
    """Reject negative answers when the statement explicitly asks for an area."""
    normalized_prompt = _normalize_lookup(prompt)
    if not any(token in normalized_prompt for token in ("aire", "surface", "partie hachuree", "partie hachuree")):
        return {"issues": [], "report": ""}

    issues: list[str] = []
    report_parts: list[str] = []
    for label, candidate in (("cible attendue", display_answer), ("solution finale", extracted_answer)):
        if not candidate or answer_kind not in {"numeric", "expression"}:
            continue
        value = _sympy_value_if_possible(candidate)
        if value is None:
            continue
        try:
            is_negative = bool(getattr(value, "is_negative", False))
            if not is_negative:
                is_negative = float(value) < 0
        except Exception:
            is_negative = False
        if is_negative:
            article = "La" if label.startswith("cible") or label.startswith("solution") else "L'"
            issues.append(f"{article} {label} est negative alors que la question demande une aire.")
            report_parts.append(f"{label.capitalize()} = {candidate}.")

    if report_parts:
        report_parts.insert(0, "Controle d'aire : une aire doit etre positive ou nulle.")
    return {"issues": issues, "report": " | ".join(report_parts)}


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


def _extract_limit_expression(prompt: str) -> tuple[str, str] | None:
    compact_prompt = re.sub(r"\s+", " ", prompt)
    patterns = [
        r"lim\s*\(\s*x\s*->\s*([^)]+)\)\s*([^\n]+)",
        r"limite\s+de\s+(.+?)\s+quand\s+x\s+tend\s+vers\s+([^\s\.,;]+)",
    ]
    first_match = re.search(patterns[0], compact_prompt, flags=re.IGNORECASE)
    if first_match:
        point = first_match.group(1).strip()
        expression = re.split(r"[.;]", first_match.group(2), maxsplit=1)[0].strip()
        return (point, expression) if point and expression else None

    second_match = re.search(patterns[1], compact_prompt, flags=re.IGNORECASE)
    if second_match:
        expression = second_match.group(1).strip()
        point = second_match.group(2).strip()
        return (point, expression) if point and expression else None
    return None


def _extract_simple_die_probability(prompt: str) -> Any | None:
    normalized_prompt = _normalize_lookup(prompt)
    if "de equilibre" not in normalized_prompt and "de " not in normalized_prompt:
        return None
    match = re.search(r"inferieur ou egal a\s*(\d+)", normalized_prompt)
    if not match:
        return None
    target = max(0, min(6, int(match.group(1))))
    return sp.Rational(target, 6)


def _sympy_value_if_possible(value: str) -> Any | None:
    if not value or sp is None:
        return None
    try:
        return sp.sympify(_normalize_for_sympy(value))
    except Exception:
        return None


def _resolve_correct_qcm_option(
    *,
    options: list[str],
    display_answer: str,
    extracted_answer: str,
    declared_option: str,
    answer_kind: str,
) -> dict[str, Any]:
    """Locate the unique correct QCM option."""
    if len(options) < 2:
        return {
            "decision": "rejected",
            "issues": ["Le QCM ne contient pas assez d'options."],
            "report": "Moins de deux options disponibles.",
            "correct_option": "",
        }

    if declared_option and declared_option in options:
        return {
            "decision": "approved",
            "issues": [],
            "report": f"Bonne option annoncee explicitement : {declared_option}.",
            "correct_option": declared_option,
        }

    candidates = [value for value in [display_answer, extracted_answer] if value]
    matching_options: list[str] = []
    for option in options:
        if any(_answers_equivalent(option, candidate, answer_kind) or _normalize_text(option) == _normalize_text(candidate) for candidate in candidates):
            matching_options.append(option)

    matching_options = _deduplicate_preserving_order(matching_options)
    if len(matching_options) == 1:
        return {
            "decision": "approved",
            "issues": [],
            "report": f"Une seule option est compatible avec la cible attendue : {matching_options[0]}.",
            "correct_option": matching_options[0],
        }
    if not matching_options:
        return {
            "decision": "rejected",
            "issues": ["Aucune option ne correspond a la reponse attendue ou a la solution extraite."],
            "report": "Aucune correspondance QCM n'a ete detectee.",
            "correct_option": "",
        }
    return {
        "decision": "rejected",
        "issues": ["Plusieurs options semblent correctes, ce qui rend le QCM ambigu."],
        "report": "Options compatibles multiples : " + " | ".join(matching_options[:4]),
        "correct_option": "",
    }


def _extract_final_answer_from_solution_text(solution_text: str) -> str:
    """Extract a concise final answer from the internal solution text when possible."""
    text = str(solution_text or "").strip()
    if not text:
        return ""

    patterns = [
        r"r[eé]ponse finale\s*[:\-]\s*(.+?)(?:[\n\.]|$)",
        r"donc\s+([^\n\.]+?)(?:[\n\.]|$)",
        r"ainsi\s+([^\n\.]+?)(?:[\n\.]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _answers_equivalent(left: str, right: str, answer_kind: str) -> bool:
    """Compare two answers using exact, set, or SymPy-aware equivalence."""
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


def _is_sympy_parseable(value: str) -> bool:
    """Return whether one value can be parsed by SymPy."""
    if sp is None:
        return False
    try:
        sp.sympify(_normalize_for_sympy(value))
        return True
    except Exception:
        return False


def _parse_set_answer(value: str) -> set[str]:
    """Parse a textual set of solutions into normalized items."""
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
    """Normalize common math display variants for SymPy parsing."""
    text = str(value or "").strip()
    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("π", "pi").replace("^", "**").replace("−", "-")
    text = text.replace("×", "*").replace("÷", "/")
    text = re.sub(r"\bln\s*\(", "log(", text)
    text = re.sub(r"\be\*\*\(([^)]+)\)", r"exp(\1)", text)
    text = re.sub(r"\be\*\*([A-Za-z0-9_]+)", r"exp(\1)", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", text)
    text = re.sub(r"(?<=[A-Za-z)])(?=\d)", "*", text)
    if re.fullmatch(r"-?\d+,\d+", text):
        text = text.replace(",", ".")
    return text


def _normalize_lookup(value: str) -> str:
    """Normalize free text into a lowercase ASCII lookup key."""
    raw_text = str(value or "").replace("∞", "inf").replace("−", "-")
    ascii_text = unicodedata.normalize("NFKD", raw_text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text).strip().lower()


def _normalize_text(value: str) -> str:
    """Normalize text for low-cost exact matching."""
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _deduplicate_preserving_order(values: list[str]) -> list[str]:
    """Deduplicate strings while preserving order and ignoring empty values."""
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


def _extract_json_payload(raw_content: str) -> dict[str, Any]:
    """Parse the first valid JSON object returned by the model."""
    parsed = extract_json_object(raw_content)
    return parsed or {}


def _load_json_candidate(candidate: str) -> dict[str, Any]:
    """Load one candidate JSON object."""
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _coerce_confidence(value: Any) -> float:
    """Clamp the model confidence between 0 and 1."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _validation_status_label(
    *,
    symbolic_checks_ran: bool,
    symbolic_checks_passed: bool | None,
    symbolic_checks_required: bool,
) -> str:
    """Return a truthful validation label for the student-facing metadata."""
    if symbolic_checks_required and symbolic_checks_passed is False:
        return "Rejetee par validation locale"
    if symbolic_checks_ran and symbolic_checks_passed is True:
        return "Validee par LLM + SymPy"
    if symbolic_checks_required and not symbolic_checks_ran:
        return "Validation symbolique non applicable"
    return "Validee par LLM seulement"
