"""Services frontend avec données réalistes et placeholders backend."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import random
import re
from typing import Any
import unicodedata
from uuid import uuid4

try:
    import sympy as sp
except ImportError:  # pragma: no cover
    sp = None

from frontend.utils.alignment_catalog import get_alignment_record
from frontend.utils.constants import DEFAULT_STUDENT_PROFILE
from frontend.utils.generation_retry_controller import GenerationRetryController, classify_failure_categories
from frontend.utils.exercise_agent import assess_exercise_completeness, generate_exercise_with_memory_adaptation
from frontend.utils.exercise_audit_log import persist_rejected_attempt_record
from frontend.utils.exercise_judge import judge_generated_exercise
from frontend.utils.exercise_memory import register_generation_outcome, retrieve_dataset_cases
from frontend.utils.exercise_schema import has_explicit_questions, split_instruction_into_context_and_questions
from frontend.utils.math_format_guard import (
    find_math_format_issues,
    repair_exercise_math_locally,
    repair_exercise_math_with_openrouter,
)
from frontend.utils.exercise_presentation_gate import apply_final_display_decision
from frontend.utils.exercise_solution_validator import validate_exercise_solution
from frontend.utils.mongo_learning import get_user_dashboard_payload, get_user_progress_analytics
from frontend.utils.mongo_teacher import (
    add_student_to_group,
    assign_exercise_to_group as assign_teacher_exercise_to_group,
    create_teacher_group,
    get_teacher_panel_snapshot,
    get_teacher_supervision_view,
)
from frontend.utils.exercise_supports import enrich_exercise_supports
from frontend.utils.openrouter_client import has_openrouter_config
from frontend.utils.tutoring_agent import generate_tutor_reply
from frontend.utils.validators.probability_solver import repair_probability_exercise_with_deterministic_solution
from frontend.utils.validators.bayes_solver import repair_bayes_solution
from frontend.utils.validators.complex_number_solver import repair_complex_number_solution
from frontend.utils.validators.domain_router import get_domain_validator_key
from frontend.utils.validators.exponential_law_solver import repair_exponential_law_solution
from frontend.utils.validators.linear_system_solver import repair_linear_system_solution
from frontend.utils.validators.regression_solver import repair_regression_solution


@dataclass
class BackendConfig:
    """Configuration placeholder pour une future API FastAPI."""

    base_url: str | None = None
    use_mock: bool = True


class MathTutorApiClient:
    """Façade légère prête à être remplacée par des appels HTTP réels."""

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig()

    def get_student_dashboard(
        self,
        student_id: str,
        *,
        user_email: str = "",
        profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retourner les données du tableau de bord étudiant depuis Mongo."""
        return get_user_dashboard_payload(
            student_id=student_id,
            user_email=user_email,
            profile=profile,
        )

    def generate_exercise(
        self,
        level: str,
        section: str,
        topic: str,
        subtopic: str,
        difficulty: str,
        exercise_type: str,
        audit_context: dict[str, str] | None = None,
        demo_mode_requested: bool = False,
        generation_strategy_override: str = "",
    ) -> dict[str, Any]:
        """Créer un exercice, le faire juger, puis le préparer pour l'élève."""
        rejected_attempts: list[dict[str, Any]] = []
        quality_feedback = ""
        generation_trace_id = uuid4().hex
        attempt_number = 1
        technical_retry_count = 0
        max_technical_retry_budget = 6
        retry_controller = GenerationRetryController(max_attempts=8)
        previous_bad_outputs: list[str] = []
        alignment_record = get_alignment_record(section, topic, subtopic)
        if alignment_record is None:
            blocked = self._build_blocked_generation_result(
                exercise=self._build_generation_shell(
                    level=level,
                    section=section,
                    topic=topic,
                    subtopic=subtopic,
                    difficulty=difficulty,
                    exercise_type=exercise_type,
                    generation_trace_id=generation_trace_id,
                ),
                status="Bloque par l'alignement officiel",
                summary="Couple absent du referentiel officiel.",
                issues=[
                    "Couple absent du referentiel officiel.",
                    "Aucun appel LLM n'a ete lance pour ce couple non reference.",
                ],
                model_name="local-alignment-precheck",
                attempt_number=0,
                rejected_attempts=[],
                flag="misaligned",
                alignment_status="misaligned",
                alignment_reason="Aucune entree d'alignement officielle n'a ete trouvee pour ce couple.",
                warning="Couple absent du referentiel officiel.",
            )
            return self._finalize_generation_result(blocked)

        last_exercise = self._build_generation_shell(
            level=level,
            section=section,
            topic=topic,
            subtopic=subtopic,
            difficulty=difficulty,
            exercise_type=exercise_type,
            generation_trace_id=generation_trace_id,
        )
        last_summary = "Aucun exercice valide n'a encore ete obtenu."
        last_issues: list[str] = []
        last_model_name = "qwen/qwen-2.5-7b-instruct"
        last_alignment_status = "aligned"
        last_alignment_reason = alignment_record.official_program_scope

        if not has_openrouter_config() or demo_mode_requested:
            demo_exercise = self._generate_trusted_dataset_demo_exercise(
                level=level,
                section=section,
                topic=topic,
                subtopic=subtopic,
                difficulty=difficulty,
                exercise_type=exercise_type,
                generation_trace_id=generation_trace_id,
                demo_mode_requested=demo_mode_requested,
            )
            if demo_exercise is not None:
                return self._finalize_generation_result(demo_exercise)

        while attempt_number <= retry_controller.max_attempts:
            retry_strategy = generation_strategy_override if attempt_number == 1 and generation_strategy_override else retry_controller.next_strategy()
            exercise = self._generate_candidate_exercise(
                level=level,
                section=section,
                topic=topic,
                subtopic=subtopic,
                difficulty=difficulty,
                exercise_type=exercise_type,
                quality_feedback=quality_feedback,
                allow_dataset_demo=demo_mode_requested,
                previous_errors=retry_controller.previous_errors(),
                previous_bad_outputs=previous_bad_outputs,
                generation_strategy=retry_strategy,
            )
            exercise["generation_trace_id"] = generation_trace_id
            exercise["generation_attempt_number"] = attempt_number
            exercise["llm_generation_attempts_count"] = attempt_number
            exercise["retry_strategy"] = retry_strategy
            exercise["previous_errors_injected"] = retry_controller.previous_errors()
            exercise["failure_categories"] = retry_controller.failure_categories()
            exercise.setdefault("generated_at", datetime.now().isoformat(timespec="seconds"))
            exercise["estimated_time"] = self._estimate_time_label(difficulty)
            exercise["fallback_revalidated"] = False

            if exercise.get("generation_backend") == "dataset-fallback-blocked":
                technical_retry_count = 0
                openrouter_error_type = str(exercise.get("openrouter_error_type") or exercise.get("llm_json_parse_status") or "").strip()
                openrouter_raw_preview = str(exercise.get("llm_raw_response_preview") or exercise.get("generation_warning") or "")
                generation_issues = [
                    item
                    for item in [
                        openrouter_error_type,
                        str(exercise.get("openrouter_error_message") or exercise.get("generation_warning") or "").strip(),
                        str(exercise.get("llm_json_parse_error") or "").strip(),
                    ]
                    if item
                ]
                if not generation_issues:
                    generation_issues = ["Le modele n'a pas renvoye un JSON exploitable pour l'exercice."]
                rejected_attempt = self._snapshot_rejected_attempt(
                    exercise,
                    "La generation LLM a echoue avant obtention d'un exercice exploitable.",
                    generation_issues,
                    str(exercise.get("openrouter_model_used") or "openrouter-llm"),
                    attempt_number,
                    alignment_status="aligned",
                    alignment_reason=alignment_record.official_program_scope,
                    flag="unknown",
                    review_stage="llm_json_generation",
                )
                rejected_attempts = self._append_rejected_attempt(rejected_attempts, rejected_attempt)
                self._persist_rejected_attempt_if_needed(rejected_attempt, exercise, audit_context)
                retry_controller.register_failure(
                    attempt_number,
                    generation_issues + [rejected_attempt["summary"]],
                    raw_output=openrouter_raw_preview,
                    parsed_exercise=exercise,
                )
                if openrouter_raw_preview:
                    previous_bad_outputs.append(openrouter_raw_preview)
                last_summary = rejected_attempt["summary"]
                last_issues = generation_issues
                last_model_name = str(exercise.get("openrouter_model_used") or "openrouter-llm")
                quality_feedback = self._build_judge_feedback(
                    rejected_attempt["summary"],
                    [
                        *generation_issues,
                        "Previous attempt failed before a displayable exercise was produced. Return ONLY valid JSON and do not repeat these mistakes.",
                    ],
                )
                attempt_number += 1
                continue

            exercise = enrich_exercise_supports(exercise)
            exercise, local_math_repaired = repair_exercise_math_locally(exercise)
            format_issues = find_math_format_issues(exercise)
            if format_issues and has_openrouter_config():
                exercise, remote_math_repaired, repair_issues = repair_exercise_math_with_openrouter(
                    exercise,
                    level=level,
                    section=section,
                    topic=topic,
                    subtopic=subtopic,
                    previous_issues=[*format_issues, *last_issues],
                )
                format_issues = find_math_format_issues(exercise) or repair_issues
                if remote_math_repaired:
                    exercise["math_format_repair_applied"] = True
            elif local_math_repaired:
                exercise["math_format_repair_applied"] = True

            if format_issues:
                technical_retry_count = 0
                rejected_attempt = self._snapshot_rejected_attempt(
                    exercise,
                    "Notation mathematique corrompue avant passage chez le juge.",
                    format_issues,
                    "local-format-repair-guard",
                    attempt_number,
                    alignment_status="unknown",
                    alignment_reason="Controle local de notation mathematique avant juge.",
                    flag="wrong",
                    review_stage="format_guard",
                )
                rejected_attempts = self._append_rejected_attempt(rejected_attempts, rejected_attempt)
                self._persist_rejected_attempt_if_needed(rejected_attempt, exercise, audit_context)
                retry_controller.register_failure(
                    attempt_number,
                    format_issues + [rejected_attempt["summary"]],
                    parsed_exercise=exercise,
                )
                previous_bad_outputs.append(str(exercise.get("prompt", "")))
                last_summary = rejected_attempt["summary"]
                last_issues = list(format_issues)
                last_model_name = "local-format-repair-guard"
                quality_feedback = self._build_judge_feedback(
                    rejected_attempt["summary"],
                    [*format_issues, "Regenere avec du LaTeX valide: \\frac{...}{...}, \\int, \\infty, \\mathbb{R}."]
                )
                attempt_number += 1
                continue

            last_exercise = exercise
            completeness_review = assess_exercise_completeness(exercise)
            exercise["prompt"] = completeness_review["clean_prompt"]

            if not completeness_review["is_complete"]:
                technical_retry_count = 0
                rejected_attempt = self._snapshot_rejected_attempt(
                    exercise,
                    completeness_review["summary"],
                    completeness_review["issues"],
                    "local-structural-guard",
                    attempt_number,
                    alignment_status="unknown",
                    alignment_reason="Controle local de completude avant passage chez le juge.",
                    flag="unknown",
                )
                rejected_attempts = self._append_rejected_attempt(rejected_attempts, rejected_attempt)
                self._persist_rejected_attempt_if_needed(rejected_attempt, exercise, audit_context)
                retry_controller.register_failure(
                    attempt_number,
                    completeness_review["issues"] + [completeness_review["summary"]],
                    parsed_exercise=exercise,
                )
                previous_bad_outputs.append(str(exercise.get("prompt", "")))
                last_summary = completeness_review["summary"]
                last_issues = list(completeness_review["issues"])
                last_model_name = "local-structural-guard"
                quality_feedback = self._build_judge_feedback(
                    completeness_review["summary"],
                    completeness_review["issues"],
                )
                attempt_number += 1
                continue

            domain_key = get_domain_validator_key(topic, subtopic, exercise.get("generation_metadata") or {})
            exercise["domain_router_key"] = domain_key or ""
            if domain_key in {"bayes", "finite_probability", "exponential_law"}:
                repaired_probability_exercise = repair_probability_exercise_with_deterministic_solution(exercise)
                if repaired_probability_exercise != exercise:
                    exercise = repaired_probability_exercise
                    exercise["probability_repair_applied"] = True
                for repair_function, repair_flag in (
                    (repair_bayes_solution, "bayes_repair_applied"),
                    (repair_exponential_law_solution, "exponential_law_repair_applied"),
                ):
                    repaired_domain_exercise = repair_function(exercise)
                    if repaired_domain_exercise != exercise:
                        exercise = repaired_domain_exercise
                        exercise[repair_flag] = True

            if domain_key == "regression":
                repaired_regression_exercise = repair_regression_solution(exercise)
                if repaired_regression_exercise != exercise:
                    exercise = repaired_regression_exercise
                    exercise["regression_repair_applied"] = True
            if domain_key == "complex_numbers":
                exercise = repair_complex_number_solution(exercise)
            if domain_key == "linear_systems":
                exercise = repair_linear_system_solution(exercise)

            if bool(exercise.get("too_similar_to_source_case")):
                technical_retry_count = 0
                similarity_issue = (
                    f"too_similar_to_source_case: {exercise.get('too_similar_source_case_id', 'inconnu')} "
                    f"(score={float(exercise.get('too_similar_source_case_score', 0.0)):.3f})"
                )
                rejected_attempt = self._snapshot_rejected_attempt(
                    exercise,
                    "L'enonce genere est trop proche d'un cas source recupere.",
                    [similarity_issue],
                    "local-memory-anti-copy",
                    attempt_number,
                    alignment_status="unknown",
                    alignment_reason="Controle d'anti-copie sur les cas de memoire recuperes.",
                    flag="wrong",
                    review_stage="memory_anti_copy",
                )
                rejected_attempts = self._append_rejected_attempt(rejected_attempts, rejected_attempt)
                self._persist_rejected_attempt_if_needed(rejected_attempt, exercise, audit_context)
                retry_controller.register_failure(
                    attempt_number,
                    [similarity_issue, rejected_attempt["summary"]],
                    parsed_exercise=exercise,
                )
                previous_bad_outputs.append(str(exercise.get("prompt", "")))
                last_summary = rejected_attempt["summary"]
                last_issues = [similarity_issue]
                last_model_name = "local-memory-anti-copy"
                quality_feedback = self._build_judge_feedback(rejected_attempt["summary"], [similarity_issue])
                attempt_number += 1
                continue

            review = judge_generated_exercise(
                exercise,
                level=level,
                section=section,
                topic=topic,
                subtopic=subtopic,
                exercise_type=exercise_type,
            )
            if review.decision == "approved":
                technical_retry_count = 0
                exercise["judge_status"] = "Validé par le juge"
                exercise["judge_summary"] = review.summary
                exercise["judge_alignment_status"] = review.alignment_status
                exercise["judge_alignment_reason"] = review.alignment_reason
                exercise["judge_issues"] = review.issues
                exercise["judge_confidence"] = review.confidence
                exercise["judge_model"] = review.model_name
                if getattr(review, "diagnostics", None):
                    diagnostics = review.diagnostics or {}
                    exercise["judge_response_format_mode"] = diagnostics.get("judge_response_format_mode", "")
                    exercise["judge_raw_response_preview"] = diagnostics.get("judge_raw_response_preview", "")
                    exercise["judge_json_parse_error"] = diagnostics.get("judge_json_parse_error", "")
                    exercise["judge_openrouter_error_type"] = diagnostics.get("judge_openrouter_error_type", "")
                exercise["judge_regeneration_count"] = attempt_number - 1
                exercise["judge_rejected_attempts"] = list(rejected_attempts)
                exercise["judge_validation_flag"] = "approved"
                exercise["judge_blocked"] = False
                exercise["judge_corrections_applied"] = False
                exercise["corrected_fields_applied"] = False
                last_alignment_status = review.alignment_status
                last_alignment_reason = review.alignment_reason
                secondary_validation = self._run_secondary_solution_validation(
                    reviewed_exercise=exercise,
                    level=level,
                    section=section,
                    topic=topic,
                    subtopic=subtopic,
                    exercise_type=exercise_type,
                    attempt_number=attempt_number,
                    review=review,
                )
                if secondary_validation["action"] == "return":
                    return self._finalize_generation_result(secondary_validation["exercise"])
                if secondary_validation.get("technical_error"):
                    technical_retry_count += 1
                rejected_attempt = secondary_validation["rejected_attempt"]
                rejected_attempts = self._append_rejected_attempt(rejected_attempts, rejected_attempt)
                self._persist_rejected_attempt_if_needed(rejected_attempt, exercise, audit_context)
                retry_controller.register_failure(
                    attempt_number,
                    list(rejected_attempt.get("issues", [])) + [str(rejected_attempt.get("summary", ""))],
                    parsed_exercise=exercise,
                )
                previous_bad_outputs.append(str(exercise.get("prompt", "")))
                last_summary = rejected_attempt.get("summary", secondary_validation.get("feedback", "Validation bloquee"))
                last_issues = list(rejected_attempt.get("issues", []))
                last_model_name = secondary_validation.get("model_name", review.model_name)
                if secondary_validation.get("technical_error") and technical_retry_count >= max_technical_retry_budget:
                    blocked = self._build_blocked_generation_result(
                        exercise=exercise,
                        status="Generation bloquee",
                        summary="La validation finale de solution est restee indisponible apres plusieurs tentatives.",
                        issues=[*secondary_validation.get("issues", []), secondary_validation["feedback"]],
                        model_name=secondary_validation.get("model_name", "qwen/qwen-2.5-7b-instruct"),
                        attempt_number=attempt_number,
                        rejected_attempts=rejected_attempts,
                        flag="unknown",
                        alignment_status=review.alignment_status,
                        alignment_reason=review.alignment_reason,
                        warning=secondary_validation.get("warning", ""),
                    )
                    return self._finalize_generation_result(blocked)
                quality_feedback = secondary_validation["feedback"]
                attempt_number += 1
                continue

            if review.decision == "corrected":
                technical_retry_count = 0
                correction_issues = self._assess_judge_correction_package(review.corrected_fields)
                if correction_issues:
                    rejected_attempt = self._snapshot_rejected_attempt(
                        exercise,
                        "Le juge a propose une correction incomplete : enonce, reponse attendue et solution complete doivent etre fournis ensemble.",
                        correction_issues,
                        review.model_name,
                        attempt_number,
                        alignment_status=review.alignment_status,
                        alignment_reason=review.alignment_reason,
                        flag="unknown",
                        review_stage="judge_correction",
                    )
                    rejected_attempts = self._append_rejected_attempt(rejected_attempts, rejected_attempt)
                    self._persist_rejected_attempt_if_needed(rejected_attempt, exercise, audit_context)
                    retry_controller.register_failure(
                        attempt_number,
                        correction_issues + [rejected_attempt["summary"]],
                        parsed_exercise=exercise,
                    )
                    last_summary = rejected_attempt["summary"]
                    last_issues = correction_issues
                    last_model_name = review.model_name
                    quality_feedback = self._build_judge_feedback(
                        rejected_attempt["summary"],
                        correction_issues,
                    )
                    attempt_number += 1
                    continue
                corrected_exercise = self._apply_judge_corrections(exercise, review.corrected_fields)
                if self._contains_problematic_correction_language(corrected_exercise, review):
                    rejected_attempt = self._snapshot_rejected_attempt(
                        corrected_exercise,
                        "Le juge a indique que l'enonce ou la solution restait problematique.",
                        ["La correction signale elle-meme un probleme d'enonce ou de solution."],
                        review.model_name,
                        attempt_number,
                        alignment_status=review.alignment_status,
                        alignment_reason=review.alignment_reason,
                        flag="wrong",
                        review_stage="judge_correction",
                    )
                    rejected_attempts = self._append_rejected_attempt(rejected_attempts, rejected_attempt)
                    self._persist_rejected_attempt_if_needed(rejected_attempt, corrected_exercise, audit_context)
                    retry_controller.register_failure(
                        attempt_number,
                        list(rejected_attempt["issues"]) + [rejected_attempt["summary"]],
                        parsed_exercise=corrected_exercise,
                    )
                    previous_bad_outputs.append(str(corrected_exercise.get("prompt", "")))
                    last_summary = rejected_attempt["summary"]
                    last_issues = list(rejected_attempt["issues"])
                    last_model_name = review.model_name
                    quality_feedback = self._build_judge_feedback(rejected_attempt["summary"], rejected_attempt["issues"])
                    attempt_number += 1
                    continue
                corrected_exercise = enrich_exercise_supports(corrected_exercise)
                corrected_exercise, _ = repair_exercise_math_locally(corrected_exercise)
                corrected_format_issues = find_math_format_issues(corrected_exercise)
                if corrected_format_issues and has_openrouter_config():
                    corrected_exercise, _, repair_issues = repair_exercise_math_with_openrouter(
                        corrected_exercise,
                        level=level,
                        section=section,
                        topic=topic,
                        subtopic=subtopic,
                        previous_issues=corrected_format_issues,
                    )
                    corrected_format_issues = find_math_format_issues(corrected_exercise) or repair_issues
                if corrected_format_issues:
                    rejected_attempt = self._snapshot_rejected_attempt(
                        corrected_exercise,
                        "La correction du juge contient encore une notation mathematique corrompue.",
                        corrected_format_issues,
                        review.model_name,
                        attempt_number,
                        alignment_status=review.alignment_status,
                        alignment_reason=review.alignment_reason,
                        flag="wrong",
                        review_stage="judge_correction_format_guard",
                    )
                    rejected_attempts = self._append_rejected_attempt(rejected_attempts, rejected_attempt)
                    self._persist_rejected_attempt_if_needed(rejected_attempt, corrected_exercise, audit_context)
                    retry_controller.register_failure(
                        attempt_number,
                        corrected_format_issues + [rejected_attempt["summary"]],
                        parsed_exercise=corrected_exercise,
                    )
                    last_summary = rejected_attempt["summary"]
                    last_issues = list(corrected_format_issues)
                    last_model_name = review.model_name
                    quality_feedback = self._build_judge_feedback(rejected_attempt["summary"], corrected_format_issues)
                    attempt_number += 1
                    continue
                corrected_completeness = assess_exercise_completeness(corrected_exercise)
                corrected_exercise["prompt"] = corrected_completeness["clean_prompt"]
                if not corrected_completeness["is_complete"]:
                    rejected_attempt = self._snapshot_rejected_attempt(
                        corrected_exercise,
                        corrected_completeness["summary"],
                        corrected_completeness["issues"],
                        review.model_name,
                        attempt_number,
                        alignment_status=review.alignment_status,
                        alignment_reason=review.alignment_reason,
                        flag="wrong",
                        review_stage="judge_correction",
                    )
                    rejected_attempts = self._append_rejected_attempt(rejected_attempts, rejected_attempt)
                    self._persist_rejected_attempt_if_needed(rejected_attempt, corrected_exercise, audit_context)
                    retry_controller.register_failure(
                        attempt_number,
                        corrected_completeness["issues"] + [corrected_completeness["summary"]],
                        parsed_exercise=corrected_exercise,
                    )
                    previous_bad_outputs.append(str(corrected_exercise.get("prompt", "")))
                    last_summary = corrected_completeness["summary"]
                    last_issues = list(corrected_completeness["issues"])
                    last_model_name = review.model_name
                    quality_feedback = self._build_judge_feedback(
                        corrected_completeness["summary"],
                        corrected_completeness["issues"],
                    )
                    attempt_number += 1
                    continue
                corrected_exercise["judge_status"] = "Corrigé par le juge"
                corrected_exercise["judge_summary"] = review.summary
                corrected_exercise["judge_alignment_status"] = review.alignment_status
                corrected_exercise["judge_alignment_reason"] = review.alignment_reason
                corrected_exercise["judge_issues"] = review.issues
                corrected_exercise["judge_confidence"] = review.confidence
                corrected_exercise["judge_model"] = review.model_name
                corrected_exercise["judge_regeneration_count"] = attempt_number - 1
                corrected_exercise["judge_rejected_attempts"] = list(rejected_attempts)
                corrected_exercise["judge_validation_flag"] = "corrected"
                corrected_exercise["judge_blocked"] = False
                last_alignment_status = review.alignment_status
                last_alignment_reason = review.alignment_reason
                secondary_validation = self._run_secondary_solution_validation(
                    reviewed_exercise=corrected_exercise,
                    level=level,
                    section=section,
                    topic=topic,
                    subtopic=subtopic,
                    exercise_type=exercise_type,
                    attempt_number=attempt_number,
                    review=review,
                )
                if secondary_validation["action"] == "return":
                    return self._finalize_generation_result(secondary_validation["exercise"])
                if secondary_validation.get("technical_error"):
                    technical_retry_count += 1
                rejected_attempt = secondary_validation["rejected_attempt"]
                rejected_attempts = self._append_rejected_attempt(rejected_attempts, rejected_attempt)
                self._persist_rejected_attempt_if_needed(rejected_attempt, corrected_exercise, audit_context)
                retry_controller.register_failure(
                    attempt_number,
                    list(rejected_attempt.get("issues", [])) + [str(rejected_attempt.get("summary", ""))],
                    parsed_exercise=corrected_exercise,
                )
                previous_bad_outputs.append(str(corrected_exercise.get("prompt", "")))
                last_summary = rejected_attempt.get("summary", secondary_validation.get("feedback", "Validation bloquee"))
                last_issues = list(rejected_attempt.get("issues", []))
                last_model_name = secondary_validation.get("model_name", review.model_name)
                if secondary_validation.get("technical_error") and technical_retry_count >= max_technical_retry_budget:
                    blocked = self._build_blocked_generation_result(
                        exercise=corrected_exercise,
                        status="Generation bloquee",
                        summary="La validation finale de solution est restee indisponible apres plusieurs tentatives.",
                        issues=[*secondary_validation.get("issues", []), secondary_validation["feedback"]],
                        model_name=secondary_validation.get("model_name", "qwen/qwen-2.5-7b-instruct"),
                        attempt_number=attempt_number,
                        rejected_attempts=rejected_attempts,
                        flag="unknown",
                        alignment_status=review.alignment_status,
                        alignment_reason=review.alignment_reason,
                        warning=secondary_validation.get("warning", ""),
                    )
                    return self._finalize_generation_result(blocked)
                quality_feedback = secondary_validation["feedback"]
                attempt_number += 1
                continue

            if review.decision == "rejected":
                technical_retry_count = 0
                rejected_attempt = self._snapshot_rejected_attempt(
                    exercise,
                    review.summary,
                    review.issues,
                    review.model_name,
                    attempt_number,
                    alignment_status=review.alignment_status,
                    alignment_reason=review.alignment_reason,
                    flag="misaligned" if review.alignment_status != "aligned" else "wrong",
                )
                rejected_attempts = self._append_rejected_attempt(rejected_attempts, rejected_attempt)
                self._persist_rejected_attempt_if_needed(rejected_attempt, exercise, audit_context)
                retry_controller.register_failure(
                    attempt_number,
                    list(review.issues) + [review.summary],
                    parsed_exercise=exercise,
                )
                previous_bad_outputs.append(str(exercise.get("prompt", "")))
                last_summary = review.summary
                last_issues = list(review.issues)
                last_model_name = review.model_name
                last_alignment_status = review.alignment_status
                last_alignment_reason = review.alignment_reason
                if self._is_missing_alignment_reference(review.alignment_reason):
                    blocked = self._build_blocked_generation_result(
                        exercise=exercise,
                        status="Bloque par l'alignement officiel",
                        summary=review.summary,
                        issues=review.issues,
                        model_name=review.model_name,
                        attempt_number=attempt_number,
                        rejected_attempts=rejected_attempts,
                        flag="misaligned",
                        alignment_status=review.alignment_status,
                        alignment_reason=review.alignment_reason,
                        warning=review.alignment_reason,
                    )
                    exercise["judge_status"] = "Bloqué par l'alignement officiel"
                    exercise["judge_summary"] = review.summary
                    exercise["judge_alignment_status"] = review.alignment_status
                    exercise["judge_alignment_reason"] = review.alignment_reason
                    exercise["judge_issues"] = review.issues
                    exercise["judge_confidence"] = review.confidence
                    exercise["judge_model"] = review.model_name
                    exercise["judge_regeneration_count"] = attempt_number
                    exercise["judge_rejected_attempts"] = list(rejected_attempts)
                    exercise["judge_validation_flag"] = "blocked_missing_alignment"
                    exercise["judge_blocked"] = True
                    exercise["generation_warning"] = self._merge_warning_messages(
                        exercise.get("generation_warning", ""),
                        review.alignment_reason,
                    )
                    return self._finalize_generation_result(blocked)
                quality_feedback = self._build_judge_feedback(review.summary, review.issues)
                attempt_number += 1
                continue

            technical_retry_count += 1
            rejected_attempt = self._snapshot_rejected_attempt(
                exercise,
                review.summary,
                [*review.issues, review.error_message],
                review.model_name,
                attempt_number,
                alignment_status=review.alignment_status or "unknown",
                alignment_reason=review.alignment_reason,
                flag="judge_error",
                judge_status_override="Juge indisponible",
            )
            rejected_attempts = self._append_rejected_attempt(rejected_attempts, rejected_attempt)
            self._persist_rejected_attempt_if_needed(rejected_attempt, exercise, audit_context)
            retry_controller.register_failure(
                attempt_number,
                [*review.issues, review.error_message or review.summary],
                parsed_exercise=exercise,
            )
            last_summary = review.summary
            last_issues = [*review.issues, review.error_message]
            last_model_name = review.model_name
            last_alignment_status = review.alignment_status or "unknown"
            last_alignment_reason = review.alignment_reason
            if technical_retry_count >= max_technical_retry_budget:
                blocked = self._build_blocked_generation_result(
                    exercise=exercise,
                    status="Generation bloquee",
                    summary="Le juge est reste indisponible apres plusieurs tentatives de regeneration.",
                    issues=[*review.issues, review.error_message or review.summary],
                    model_name=review.model_name,
                    attempt_number=attempt_number,
                    rejected_attempts=rejected_attempts,
                    flag="judge_error",
                    alignment_status=review.alignment_status or "unknown",
                    alignment_reason=review.alignment_reason,
                    warning=review.error_message or review.summary,
                )
                return self._finalize_generation_result(blocked)
            quality_feedback = self._build_judge_feedback(
                review.summary,
                [*review.issues, review.error_message],
            )
            attempt_number += 1
            continue

        blocked = self._build_blocked_generation_result(
            exercise=last_exercise,
            status="Generation bloquee",
            summary=(
                "La generation LLM n'a pas produit un exercice valide apres plusieurs tentatives."
                if has_openrouter_config() and not demo_mode_requested
                else "Aucun exercice fiable n'a ete obtenu avant la limite de tentatives."
            ),
            issues=last_issues or [last_summary],
            model_name=last_model_name,
            attempt_number=retry_controller.max_attempts,
            rejected_attempts=rejected_attempts,
            flag="blocked_after_retries",
            alignment_status=last_alignment_status,
            alignment_reason=last_alignment_reason,
            warning=last_summary,
        )
        return self._finalize_generation_result(blocked)

    def _generate_trusted_dataset_demo_exercise(
        self,
        *,
        level: str,
        section: str,
        topic: str,
        subtopic: str,
        difficulty: str,
        exercise_type: str,
        generation_trace_id: str,
        demo_mode_requested: bool = False,
    ) -> dict[str, Any] | None:
        """Return a safe local demo exercise when no OpenRouter key is configured.

        The exercise is not model-generated: it is copied from the curated dataset and
        marked as a trusted dataset demo so the application remains demonstrable offline.
        """
        dataset_cases = retrieve_dataset_cases(
            section=section,
            topic=topic,
            subtopic=subtopic,
            profile=DEFAULT_STUDENT_PROFILE,
            top_k=8,
        )
        if not dataset_cases:
            return None

        selected_case = next(
            (case for case in dataset_cases if not self._statement_requires_missing_visual_support(case.instruction)),
            dataset_cases[0],
        )
        if self._statement_requires_missing_visual_support(selected_case.instruction):
            # Avoid showing dataset exercises that refer to a missing annex/figure in offline demo mode.
            return None

        generated_at = datetime.now().isoformat(timespec="seconds")
        final_answer = str(selected_case.final_answer or "Voir solution détaillée.").strip()
        solution_text = str(selected_case.solution or final_answer).strip()
        steps = self._split_solution_into_steps(solution_text)
        context, questions = split_instruction_into_context_and_questions(selected_case.instruction)
        exercise = {
            "id": f"DEMO-{uuid4().hex[:10].upper()}",
            "section": section,
            "topic": topic,
            "subtopic": subtopic,
            "level": level,
            "generated_at": generated_at,
            "difficulty": difficulty,
            "exercise_type": exercise_type,
            "title": f"Exercice démonstration — {subtopic}",
            "context": context,
            "questions": questions,
            "instruction": selected_case.instruction,
            "prompt": selected_case.instruction,
            "hint": "Identifie d'abord la propriété centrale, puis traite les questions dans l'ordre.",
            "accepted_answers": [final_answer] if final_answer else [],
            "display_answer": final_answer,
            "answer_kind": "text",
            "solution_steps": steps,
            "hidden_solution": solution_text,
            "learning_objective": f"S'entraîner sur : {subtopic}.",
            "tags": [section, topic, subtopic, difficulty, exercise_type, "demo-local"],
            "estimated_time": self._estimate_time_label(difficulty),
            "generation_backend": "trusted-dataset-demo",
            "is_true_llm_generation": False,
            "llm_json_parse_status": "invalid_json",
            "llm_generation_attempts_count": 0,
            "fallback_used": True,
            "fallback_reason": "OpenRouter non configure ou mode demonstration explicitement demande.",
            "display_source_category": "demo_dataset",
            "demo_mode_used": demo_mode_requested or not has_openrouter_config(),
            "generation_warning": (
                "Mode démonstration local : OpenRouter n'est pas configuré, "
                "l'exercice affiché provient directement du dataset validé."
            ),
            "verification_ready": True,
            "verification_message": "Exercice issu du dataset de référence en mode démonstration local.",
            "memory_adaptation_note": "Mode offline : réutilisation contrôlée d'un cas dataset sans génération LLM.",
            "retrieved_case_ids": [case.case_id for case in dataset_cases],
            "retrieved_memory_count": 0,
            "source_case_summaries": [
                {
                    "case_id": case.case_id,
                    "year": case.year,
                    "topic": case.topic,
                    "subtopic": case.subtopic,
                }
                for case in dataset_cases[:3]
            ],
            "judge_status": "Validé localement pour démonstration",
            "judge_summary": "Exercice repris du dataset curé en l'absence de configuration OpenRouter.",
            "judge_alignment_status": "aligned",
            "judge_alignment_reason": "Couple présent dans le référentiel d'alignement et cas issu du dataset.",
            "judge_issues": [],
            "judge_confidence": 1.0,
            "judge_model": "trusted-dataset-demo",
            "judge_regeneration_count": 0,
            "judge_rejected_attempts": [],
            "judge_validation_flag": "approved",
            "judge_blocked": False,
            "judge_corrections_applied": False,
            "corrected_fields_applied": False,
            "solution_validation_status": "Validée localement",
            "solution_validation_summary": "Solution originale du dataset utilisée comme référence.",
            "solution_validation_issues": [],
            "solution_validation_confidence": 1.0,
            "solution_validation_model": "trusted-dataset-demo",
            "solution_validation_flag": "approved",
            "solution_validation_sympy_report": "Non requis en mode dataset demo.",
            "local_validation_flag": "approved",
            "local_validation_summary": "Cas dataset accepté pour démonstration locale.",
            "local_validation_issues": [],
            "local_validation_checks": {"formatting": {"status": "passed", "message": "Énoncé lisible."}},
            "pedagogical_completeness_flag": "complete",
            "pedagogical_completeness_summary": "Énoncé et solution disponibles dans le dataset.",
            "pedagogical_completeness_issues": [],
            "symbolic_checks_ran": False,
            "symbolic_checks_passed": True,
            "symbolic_checks_required": False,
            "fallback_revalidated": True,
            "support_ready": False,
            "student_facing_format_flag": "approved",
            "student_facing_format_issues": [],
        }
        if exercise_type == "QCM":
            exercise["options"] = self._generate_options(final_answer, "text")
        return exercise

    def _statement_requires_missing_visual_support(self, statement: str) -> bool:
        """Detect dataset statements that would need an absent figure/table/annex."""
        normalized = self._normalize_lookup(statement)
        markers = (
            "figure de l annexe",
            "annexe jointe",
            "voir annexe",
            "figure ci contre",
            "figure ci dessous",
            "courbe ci dessous",
            "graphique ci dessous",
            "tableau suivant",
            "diagramme suivant",
        )
        return any(marker in normalized for marker in markers)

    def _split_solution_into_steps(self, solution_text: str) -> list[str]:
        """Extract compact teacher-facing solution steps from a full solution string."""
        compact = " ".join(str(solution_text or "").split())
        if not compact:
            return ["Consulter la correction complète associée à l'exercice."]
        raw_parts = re.split(r"(?:(?<=\.)\s+|\s+)(?=\d+\)|\d+\.|[a-d]\))", compact)
        steps = [part.strip(" -;:") for part in raw_parts if part.strip(" -;:")]
        if len(steps) <= 1:
            sentences = re.split(r"(?<=[.!?])\s+", compact)
            steps = [sentence.strip() for sentence in sentences if sentence.strip()]
        return steps[:5] or [compact[:500]]

    def _run_secondary_solution_validation(
        self,
        *,
        reviewed_exercise: dict[str, Any],
        level: str,
        section: str,
        topic: str,
        subtopic: str,
        exercise_type: str,
        attempt_number: int,
        review: Any,
    ) -> dict[str, Any]:
        """Run the LLM+SymPy solution validator after the main judge."""
        solution_review = validate_exercise_solution(
            reviewed_exercise,
            level=level,
            section=section,
            topic=topic,
            subtopic=subtopic,
            exercise_type=exercise_type,
        )
        if solution_review.decision == "approved":
            reviewed_exercise.update(solution_review.normalized_fields)
            reviewed_exercise["llm_generation_attempts_count"] = attempt_number
            reviewed_exercise["solution_validation_status"] = solution_review.validation_status_label
            reviewed_exercise["solution_validation_summary"] = solution_review.summary
            reviewed_exercise["solution_validation_issues"] = solution_review.issues
            reviewed_exercise["solution_validation_confidence"] = solution_review.confidence
            reviewed_exercise["solution_validation_model"] = solution_review.model_name
            reviewed_exercise["solution_validation_flag"] = "approved"
            reviewed_exercise["solution_validation_sympy_report"] = solution_review.sympy_report
            reviewed_exercise["local_validation_flag"] = solution_review.local_validation_flag
            reviewed_exercise["local_validation_summary"] = solution_review.local_validation_summary
            reviewed_exercise["local_validation_issues"] = solution_review.local_validation_issues
            reviewed_exercise["pedagogical_completeness_flag"] = solution_review.pedagogical_completeness_flag
            reviewed_exercise["pedagogical_completeness_summary"] = solution_review.pedagogical_completeness_summary
            reviewed_exercise["pedagogical_completeness_issues"] = solution_review.pedagogical_completeness_issues
            reviewed_exercise["symbolic_checks_ran"] = solution_review.symbolic_checks_ran
            reviewed_exercise["symbolic_checks_passed"] = solution_review.symbolic_checks_passed
            reviewed_exercise["symbolic_checks_required"] = solution_review.symbolic_checks_required
            reviewed_exercise["fallback_revalidated"] = reviewed_exercise.get("generation_backend") in {
                "dataset-fallback",
                "local-fallback",
                "trusted-dataset-demo",
            }
            gated_exercise = apply_final_display_decision(reviewed_exercise)
            if gated_exercise["final_display_decision"] == "presented":
                return {"action": "return", "exercise": gated_exercise}
            rejected_attempt = self._snapshot_rejected_attempt(
                gated_exercise,
                "Le gate final a bloque l'exercice apres validation.",
                gated_exercise.get("final_display_blocking_reasons", []),
                solution_review.model_name,
                attempt_number,
                alignment_status=review.alignment_status,
                alignment_reason=review.alignment_reason,
                flag="wrong",
                review_stage="final_gate",
            )
            rejected_attempt["solution_validation_sympy_report"] = solution_review.sympy_report
            return {
                "action": "regenerate",
                "rejected_attempt": rejected_attempt,
                "feedback": self._build_judge_feedback(
                    "Le gate final a bloque l'exercice apres validation.",
                    gated_exercise.get("final_display_blocking_reasons", []),
                ),
                "technical_error": False,
                "issues": gated_exercise.get("final_display_blocking_reasons", []),
                "model_name": solution_review.model_name,
                "warning": gated_exercise.get("solution_validation_summary", ""),
            }

        if solution_review.decision == "rejected":
            rejected_attempt = self._snapshot_rejected_attempt(
                reviewed_exercise,
                solution_review.summary,
                solution_review.issues,
                solution_review.model_name,
                attempt_number,
                alignment_status=review.alignment_status,
                alignment_reason=review.alignment_reason,
                flag="wrong",
                review_stage="solution_validator",
            )
            rejected_attempt["solution_validation_sympy_report"] = solution_review.sympy_report
            rejected_attempt["local_validation_flag"] = solution_review.local_validation_flag
            rejected_attempt["local_validation_summary"] = solution_review.local_validation_summary
            rejected_attempt["local_validation_issues"] = solution_review.local_validation_issues
            rejected_attempt["pedagogical_completeness_flag"] = solution_review.pedagogical_completeness_flag
            rejected_attempt["pedagogical_completeness_summary"] = solution_review.pedagogical_completeness_summary
            rejected_attempt["pedagogical_completeness_issues"] = solution_review.pedagogical_completeness_issues
            rejected_attempt["symbolic_checks_ran"] = solution_review.symbolic_checks_ran
            rejected_attempt["symbolic_checks_passed"] = solution_review.symbolic_checks_passed
            rejected_attempt["symbolic_checks_required"] = solution_review.symbolic_checks_required
            feedback = self._build_judge_feedback(
                solution_review.summary,
                [*solution_review.issues, solution_review.sympy_report],
            )
            return {
                "action": "regenerate",
                "rejected_attempt": rejected_attempt,
                "feedback": feedback,
                "technical_error": False,
                "issues": solution_review.issues,
                "model_name": solution_review.model_name,
                "warning": solution_review.error_message,
            }

        rejected_attempt = self._snapshot_rejected_attempt(
            reviewed_exercise,
            solution_review.summary,
            [*solution_review.issues, solution_review.sympy_report],
            solution_review.model_name,
            attempt_number,
            alignment_status=review.alignment_status,
            alignment_reason=review.alignment_reason,
            flag="unknown",
            review_stage="solution_validator",
            judge_status_override="Validateur solution indisponible",
        )
        rejected_attempt["solution_validation_sympy_report"] = solution_review.sympy_report
        rejected_attempt["local_validation_flag"] = solution_review.local_validation_flag
        rejected_attempt["local_validation_summary"] = solution_review.local_validation_summary
        rejected_attempt["local_validation_issues"] = solution_review.local_validation_issues
        rejected_attempt["pedagogical_completeness_flag"] = solution_review.pedagogical_completeness_flag
        rejected_attempt["pedagogical_completeness_summary"] = solution_review.pedagogical_completeness_summary
        rejected_attempt["pedagogical_completeness_issues"] = solution_review.pedagogical_completeness_issues
        rejected_attempt["symbolic_checks_ran"] = solution_review.symbolic_checks_ran
        rejected_attempt["symbolic_checks_passed"] = solution_review.symbolic_checks_passed
        rejected_attempt["symbolic_checks_required"] = solution_review.symbolic_checks_required
        return {
            "action": "regenerate",
            "rejected_attempt": rejected_attempt,
            "feedback": self._build_judge_feedback(
                solution_review.summary,
                [*solution_review.issues, solution_review.sympy_report],
            ),
            "technical_error": True,
            "issues": solution_review.issues,
            "model_name": solution_review.model_name,
            "warning": solution_review.error_message or solution_review.summary,
        }

    def _generate_candidate_exercise(
        self,
        *,
        level: str,
        section: str,
        topic: str,
        subtopic: str,
        difficulty: str,
        exercise_type: str,
        quality_feedback: str = "",
        allow_dataset_demo: bool = False,
        previous_errors: list[str] | None = None,
        previous_bad_outputs: list[str] | None = None,
        generation_strategy: str = "normal_memory_adapted_generation",
    ) -> dict[str, Any]:
        """Generate one candidate exercise before review."""
        try:
            return generate_exercise_with_memory_adaptation(
                level=level,
                section=section,
                topic=topic,
                subtopic=subtopic,
                difficulty=difficulty,
                exercise_type=exercise_type,
                quality_feedback=quality_feedback,
                allow_dataset_demo=allow_dataset_demo,
                previous_errors=previous_errors or [],
                previous_bad_outputs=previous_bad_outputs or [],
                generation_strategy=generation_strategy,
            )
        except Exception as exc:
            if has_openrouter_config() and not allow_dataset_demo:
                diagnostics = dict(getattr(exc, "diagnostics", {}) or {})
                parse_status = getattr(exc, "parse_status", "") or diagnostics.get("openrouter_error_type") or "invalid_json"
                exercise = self._build_generation_shell(
                    level=level,
                    section=section,
                    topic=topic,
                    subtopic=subtopic,
                    difficulty=difficulty,
                    exercise_type=exercise_type,
                    generation_trace_id=uuid4().hex,
                )
                exercise["generation_backend"] = "dataset-fallback-blocked"
                exercise["generation_warning"] = str(exc)
                exercise["fallback_revalidated"] = False
                exercise["llm_json_parse_status"] = parse_status
                exercise["llm_json_extraction_method"] = diagnostics.get("llm_json_extraction_method", "")
                exercise["llm_json_parse_error"] = diagnostics.get("llm_json_parse_error", "")
                exercise["llm_raw_response_preview"] = diagnostics.get("llm_raw_response_preview", getattr(exc, "raw_output", ""))
                exercise["openrouter_http_status"] = diagnostics.get("openrouter_http_status")
                exercise["openrouter_error_type"] = diagnostics.get("openrouter_error_type", parse_status)
                exercise["openrouter_error_message"] = diagnostics.get("openrouter_error_message", str(exc))
                exercise["openrouter_response_format_mode"] = diagnostics.get("openrouter_response_format_mode", "")
                exercise["openrouter_model_used"] = diagnostics.get("openrouter_model_used", "")
                exercise["openrouter_request_id"] = diagnostics.get("openrouter_request_id", "")
                exercise["openrouter_provider"] = diagnostics.get("openrouter_provider", "")
                exercise["openrouter_usage"] = diagnostics.get("openrouter_usage")
                exercise["openrouter_call_attempts"] = diagnostics.get("openrouter_call_attempts", [])
                exercise["prompt_char_count"] = diagnostics.get("prompt_char_count", 0)
                exercise["prompt_token_estimate"] = diagnostics.get("prompt_token_estimate", 0)
                exercise["number_of_memory_cases"] = diagnostics.get("number_of_memory_cases", 0)
                exercise["retrieved_case_ids"] = list(diagnostics.get("retrieved_case_ids", []))
                exercise["source_case_summaries"] = list(diagnostics.get("source_case_summaries", []))
                exercise["source_case_instructions"] = list(diagnostics.get("source_case_instructions", []))
                exercise["is_true_llm_generation"] = False
                exercise["fallback_used"] = False
                exercise["display_source_category"] = "blocked"
                exercise["verification_ready"] = False
                exercise["student_facing_format_flag"] = "wrong"
                exercise["student_facing_format_issues"] = ["Generation LLM indisponible ou JSON inexploitable."]
                return exercise
            exercise = self._resolve_exercise_generator(topic, subtopic)(
                level, topic, subtopic, difficulty, exercise_type
            )
            exercise["section"] = section
            exercise["generation_backend"] = "local-fallback"
            exercise["generation_warning"] = str(exc)
            exercise["fallback_revalidated"] = False
            exercise["llm_json_parse_status"] = getattr(exc, "parse_status", "invalid_json")
            exercise["is_true_llm_generation"] = False
            exercise["fallback_used"] = True
            exercise["display_source_category"] = "blocked"
            exercise.setdefault("generated_at", datetime.now().isoformat(timespec="seconds"))
            exercise.setdefault(
                "hidden_solution",
                self._compose_hidden_solution(exercise.get("solution_steps", []), exercise.get("display_answer", "")),
            )
            exercise["verification_ready"] = True
            return exercise

    def _append_rejected_attempt(
        self,
        rejected_attempts: list[dict[str, Any]],
        rejected_attempt: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Keep the rolling window of refused attempts bounded."""
        updated_attempts = [*rejected_attempts, rejected_attempt]
        return updated_attempts[-25:]

    def _persist_rejected_attempt_if_needed(
        self,
        rejected_attempt: dict[str, Any],
        parent_exercise: dict[str, Any],
        audit_context: dict[str, str] | None,
    ) -> None:
        """Persist one refused attempt only when audit metadata is available."""
        if not audit_context:
            self._register_generation_outcome_memory(
                source=rejected_attempt,
                parent=parent_exercise,
                final_display_decision="blocked",
            )
            return
        persist_rejected_attempt_record(
            rejected_attempt,
            parent_exercise=parent_exercise,
            user_email=audit_context.get("user_email", "inconnu"),
            user_role=audit_context.get("user_role", "Inconnu"),
            user_display_name=audit_context.get("user_display_name", "Utilisateur inconnu"),
        )
        self._register_generation_outcome_memory(
            source=rejected_attempt,
            parent=parent_exercise,
            final_display_decision="blocked",
        )

    def _register_generation_outcome_memory(
        self,
        *,
        source: dict[str, Any],
        parent: dict[str, Any] | None = None,
        final_display_decision: str,
    ) -> None:
        """Feed positive/negative generation outcomes back into the memory layer."""
        parent_exercise = parent or {}
        prompt_signature = self._normalize_lookup(
            "|".join(
                [
                    str(source.get("section") or parent_exercise.get("section") or ""),
                    str(source.get("topic") or parent_exercise.get("topic") or ""),
                    str(source.get("subtopic") or parent_exercise.get("subtopic") or ""),
                    str(source.get("prompt") or parent_exercise.get("prompt") or ""),
                ]
            )
        )[:320]
        issues = [str(item) for item in (source.get("issues") or source.get("judge_issues") or []) if str(item).strip()]
        summary = str(source.get("summary") or source.get("judge_summary") or "").strip()
        failure_categories = classify_failure_categories([summary, *issues])
        register_generation_outcome(
            prompt_signature=prompt_signature,
            section=str(source.get("section") or parent_exercise.get("section") or ""),
            topic=str(source.get("topic") or parent_exercise.get("topic") or ""),
            subtopic=str(source.get("subtopic") or parent_exercise.get("subtopic") or ""),
            retrieved_case_ids=list(source.get("retrieved_case_ids") or parent_exercise.get("retrieved_case_ids") or []),
            generation_backend=str(source.get("generation_backend") or parent_exercise.get("generation_backend") or ""),
            is_true_llm_generation=bool(source.get("is_true_llm_generation", parent_exercise.get("is_true_llm_generation", False))),
            validation_result=str(source.get("judge_validation_flag") or source.get("flag") or parent_exercise.get("judge_validation_flag") or ""),
            judge_issues=issues,
            local_issues=list(source.get("local_validation_issues") or parent_exercise.get("local_validation_issues") or []),
            final_display_decision=final_display_decision,
            student_facing_format_issues=list(source.get("student_facing_format_issues") or parent_exercise.get("student_facing_format_issues") or []),
            failure_categories=failure_categories,
        )

    def _build_generation_shell(
        self,
        *,
        level: str,
        section: str,
        topic: str,
        subtopic: str,
        difficulty: str,
        exercise_type: str,
        generation_trace_id: str,
    ) -> dict[str, Any]:
        """Create a minimal exercise shell for blocked generations."""
        return {
            "id": f"PENDING-{uuid4().hex[:10].upper()}",
            "level": level,
            "section": section,
            "topic": topic,
            "subtopic": subtopic,
            "difficulty": difficulty,
            "exercise_type": exercise_type,
            "title": f"Exercice sur {subtopic}",
            "context": "",
            "questions": [],
            "instruction": "",
            "prompt": "",
            "hint": "",
            "display_answer": "",
            "hidden_solution": "",
            "solution_steps": [],
            "accepted_answers": [],
            "corrected_fields_applied": False,
            "generation_trace_id": generation_trace_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generation_backend": "blocked-before-generation",
            "is_true_llm_generation": False,
            "llm_json_parse_status": "invalid_json",
            "llm_generation_attempts_count": 0,
            "fallback_used": False,
            "fallback_reason": "",
            "display_source_category": "blocked",
            "tags": [section, topic, subtopic, difficulty, exercise_type],
        }

    def _finalize_generation_result(self, exercise: dict[str, Any]) -> dict[str, Any]:
        """Apply the final display gate and store one positive/negative outcome memory trace."""
        gated = apply_final_display_decision(exercise)
        self._register_generation_outcome_memory(
            source=gated,
            parent=gated,
            final_display_decision=str(gated.get("final_display_decision", "blocked")),
        )
        return gated

    def _contains_problematic_correction_language(self, corrected_exercise: dict[str, Any], review: Any) -> bool:
        """Block judge corrections that still describe the statement as faulty."""
        combined = " ".join(
            [
                str(review.summary or ""),
                str(corrected_exercise.get("hidden_solution", "")),
                str(corrected_exercise.get("display_answer", "")),
            ]
        ).lower()
        markers = [
            "l'enonce est problematique",
            "l'énoncé est problématique",
            "il y a une erreur dans l'enonce",
            "il y a une erreur dans l'énoncé",
            "la solution initiale est fausse",
            "il semble y avoir une incomprehension",
            "il semble y avoir une incompréhension",
        ]
        return any(marker in combined for marker in markers)

    def _assess_judge_correction_package(self, corrected_fields: dict[str, Any]) -> list[str]:
        """Require the judge to replace prompt, expected answer, and full solution together."""
        required_fields = {
            "prompt": "L'enonce corrige est absent.",
            "expected_answer": "La reponse attendue corrigee est absente.",
            "full_solution": "La solution complete corrigee est absente.",
        }
        issues: list[str] = []
        for field_name, message in required_fields.items():
            if not str(corrected_fields.get(field_name, "")).strip():
                issues.append(message)
        combined = " ".join(str(corrected_fields.get(field, "")) for field in ("prompt", "expected_answer", "full_solution"))
        if self._contains_problematic_correction_language(
            {
                "hidden_solution": corrected_fields.get("full_solution", ""),
                "display_answer": corrected_fields.get("expected_answer", ""),
            },
            type("JudgeSummary", (), {"summary": combined})(),
        ):
            issues.append("La correction proposee signale encore un probleme d'enonce ou de solution.")
        return issues

    def _build_blocked_generation_result(
        self,
        *,
        exercise: dict[str, Any],
        status: str,
        summary: str,
        issues: list[str],
        model_name: str,
        attempt_number: int,
        rejected_attempts: list[dict[str, Any]],
        flag: str,
        alignment_status: str,
        alignment_reason: str,
        warning: str = "",
    ) -> dict[str, Any]:
        """Return a placeholder payload that can never be shown as a real exercise."""
        blocked_exercise = deepcopy(exercise)
        blocked_exercise["prompt"] = ""
        blocked_exercise["hidden_solution"] = ""
        blocked_exercise["display_answer"] = ""
        blocked_exercise["accepted_answers"] = []
        blocked_exercise["solution_steps"] = []
        blocked_exercise["options"] = []
        blocked_exercise["verification_ready"] = False
        blocked_exercise["verification_message"] = (
            "Aucun enonce n'a ete affiche car la chaine de validation n'a pas confirme un exercice fiable."
        )
        blocked_exercise["judge_status"] = status
        blocked_exercise["judge_summary"] = summary
        blocked_exercise["judge_alignment_status"] = alignment_status or "unknown"
        blocked_exercise["judge_alignment_reason"] = alignment_reason
        blocked_exercise["judge_issues"] = [issue for issue in issues if str(issue).strip()]
        blocked_exercise["judge_confidence"] = 0.0
        blocked_exercise["judge_model"] = model_name
        blocked_exercise["judge_regeneration_count"] = attempt_number
        blocked_exercise["judge_rejected_attempts"] = list(rejected_attempts)
        blocked_exercise["llm_generation_attempts_count"] = max(
            int(blocked_exercise.get("llm_generation_attempts_count", 0) or 0),
            int(attempt_number or 0),
            len(rejected_attempts),
        )
        if rejected_attempts:
            last_attempt = rejected_attempts[-1]
            for key in (
                "openrouter_http_status",
                "openrouter_error_type",
                "openrouter_error_message",
                "openrouter_response_format_mode",
                "openrouter_model_used",
                "openrouter_request_id",
                "openrouter_provider",
                "openrouter_usage",
                "openrouter_call_attempts",
                "llm_raw_response_preview",
                "llm_json_extraction_method",
                "llm_json_parse_error",
                "prompt_char_count",
                "prompt_token_estimate",
                "number_of_memory_cases",
                "retrieved_case_ids",
                "source_case_summaries",
                "source_case_instructions",
                "failure_categories",
            ):
                if key in last_attempt and last_attempt.get(key) not in (None, "", []):
                    blocked_exercise[key] = last_attempt.get(key)
        blocked_exercise["judge_validation_flag"] = flag
        blocked_exercise["judge_blocked"] = True
        blocked_exercise["solution_validation_status"] = "Bloquee"
        blocked_exercise["solution_validation_summary"] = ""
        blocked_exercise["solution_validation_issues"] = []
        blocked_exercise["solution_validation_confidence"] = 0.0
        blocked_exercise["solution_validation_model"] = ""
        blocked_exercise["solution_validation_flag"] = "blocked"
        blocked_exercise["solution_validation_sympy_report"] = ""
        blocked_exercise["local_validation_flag"] = "wrong"
        blocked_exercise["local_validation_summary"] = summary
        blocked_exercise["local_validation_issues"] = [issue for issue in issues if str(issue).strip()]
        blocked_exercise["pedagogical_completeness_flag"] = "not_applicable"
        blocked_exercise["pedagogical_completeness_summary"] = ""
        blocked_exercise["pedagogical_completeness_issues"] = []
        blocked_exercise["symbolic_checks_ran"] = False
        blocked_exercise["symbolic_checks_passed"] = False
        blocked_exercise["symbolic_checks_required"] = False
        blocked_exercise["corrected_fields_applied"] = bool(blocked_exercise.get("corrected_fields_applied", False))
        blocked_exercise["fallback_revalidated"] = False
        blocked_exercise["is_true_llm_generation"] = blocked_exercise.get("generation_backend") in {
            "openrouter-llm",
            "openrouter-llm-repaired-json",
        }
        blocked_exercise["display_source_category"] = "blocked"
        blocked_exercise["generation_warning"] = self._merge_warning_messages(
            blocked_exercise.get("generation_warning", ""),
            warning,
        )
        return blocked_exercise

    def _estimate_time_label(self, difficulty: str) -> str:
        """Map the internal difficulty to a stable display label."""
        difficulty_key = self._normalize_lookup(difficulty)
        return {
            "fondamental": "8 à 10 min",
            "intermediaire": "10 à 14 min",
            "avance": "14 à 18 min",
            "defi": "18 à 25 min",
        }.get(difficulty_key, "10 à 14 min")

    def _apply_judge_corrections(self, exercise: dict[str, Any], corrected_fields: dict[str, Any]) -> dict[str, Any]:
        """Merge judge-supplied corrections into the generated exercise."""
        corrected_exercise = deepcopy(exercise)
        field_map = {
            "title": "title",
            "prompt": "prompt",
            "hint": "hint",
            "learning_objective": "learning_objective",
            "expected_answer": "display_answer",
            "full_solution": "hidden_solution",
            "answer_kind": "answer_kind",
        }
        applied_fields: list[str] = []
        for source_field, target_field in field_map.items():
            value = str(corrected_fields.get(source_field, "")).strip()
            if value:
                corrected_exercise[target_field] = value
                applied_fields.append(target_field)

        expected_answer = str(corrected_fields.get("expected_answer", "")).strip()
        if expected_answer:
            corrected_exercise["accepted_answers"] = [expected_answer]
            corrected_exercise["display_answer"] = expected_answer
            applied_fields.extend(["accepted_answers", "display_answer"])

        corrected_steps = corrected_fields.get("solution_steps") or []
        if isinstance(corrected_steps, list):
            cleaned_steps = [str(step).strip() for step in corrected_steps if str(step).strip()]
            if cleaned_steps:
                corrected_exercise["solution_steps"] = cleaned_steps

        corrected_options = corrected_fields.get("options") or []
        if corrected_exercise.get("exercise_type") == "QCM":
            if isinstance(corrected_options, list):
                cleaned_options = [str(option).strip() for option in corrected_options if str(option).strip()]
                if cleaned_options:
                    corrected_exercise["options"] = cleaned_options
                    applied_fields.append("options")
        else:
            corrected_exercise.pop("options", None)

        if not str(corrected_fields.get("full_solution", "")).strip():
            corrected_exercise["hidden_solution"] = self._compose_hidden_solution(
                corrected_exercise.get("solution_steps", []),
                corrected_exercise.get("display_answer", ""),
            )

        corrected_fields_applied = all(
            str(corrected_exercise.get(field_name, "")).strip()
            for field_name in ("prompt", "display_answer", "hidden_solution")
        )
        corrected_exercise["judge_corrections_applied"] = corrected_fields_applied
        corrected_exercise["corrected_fields_applied"] = corrected_fields_applied
        corrected_exercise["judge_corrected_fields_applied"] = sorted(set(applied_fields))
        return corrected_exercise

    def _snapshot_rejected_attempt(
        self,
        exercise: dict[str, Any],
        summary: str,
        issues: list[str],
        model_name: str,
        attempt_number: int,
        alignment_status: str = "misaligned",
        alignment_reason: str = "",
        flag: str = "wrong",
        review_stage: str = "judge",
        judge_status_override: str = "",
    ) -> dict[str, Any]:
        """Build a stored trace for one exercise refused by the judge."""
        judge_status = judge_status_override.strip() or (
            "Rejete par le validateur de solution" if review_stage == "solution_validator" else "Rejete par le juge"
        )
        return {
            "id": exercise["id"],
            "title": exercise["title"],
            "topic": exercise["topic"],
            "subtopic": exercise["subtopic"],
            "section": exercise.get("section", ""),
            "level": exercise.get("level", ""),
            "difficulty": exercise["difficulty"],
            "exercise_type": exercise["exercise_type"],
            "flag": flag,
            "judge_status": judge_status,
            "judge_validation_flag": flag,
            "summary": summary,
            "issues": list(issues),
            "judge_model": model_name,
            "review_stage": review_stage,
            "attempt_number": attempt_number,
            "alignment_status": alignment_status,
            "alignment_reason": alignment_reason,
            "prompt": exercise.get("prompt", ""),
            "hidden_solution": exercise.get("hidden_solution", ""),
            "display_answer": exercise.get("display_answer", ""),
            "generated_at": exercise.get("generated_at", ""),
            "generation_backend": exercise.get("generation_backend", ""),
            "generation_trace_id": exercise.get("generation_trace_id", ""),
            "retry_strategy": exercise.get("retry_strategy", ""),
            "previous_errors_injected": list(exercise.get("previous_errors_injected", []) or []),
            "failure_categories": list(exercise.get("failure_categories", []) or []),
            "llm_generation_attempts_count": exercise.get("llm_generation_attempts_count", attempt_number),
            "llm_json_parse_status": exercise.get("llm_json_parse_status", ""),
            "llm_json_extraction_method": exercise.get("llm_json_extraction_method", ""),
            "llm_json_parse_error": exercise.get("llm_json_parse_error", ""),
            "llm_raw_response_preview": exercise.get("llm_raw_response_preview", ""),
            "openrouter_http_status": exercise.get("openrouter_http_status"),
            "openrouter_error_type": exercise.get("openrouter_error_type", ""),
            "openrouter_error_message": exercise.get("openrouter_error_message", ""),
            "openrouter_response_format_mode": exercise.get("openrouter_response_format_mode", ""),
            "openrouter_model_used": exercise.get("openrouter_model_used", ""),
            "openrouter_request_id": exercise.get("openrouter_request_id", ""),
            "openrouter_provider": exercise.get("openrouter_provider", ""),
            "openrouter_usage": exercise.get("openrouter_usage"),
            "openrouter_call_attempts": list(exercise.get("openrouter_call_attempts", []) or []),
            "prompt_char_count": exercise.get("prompt_char_count", 0),
            "prompt_token_estimate": exercise.get("prompt_token_estimate", 0),
            "number_of_memory_cases": exercise.get("number_of_memory_cases", 0),
            "retrieved_case_ids": list(exercise.get("retrieved_case_ids", []) or []),
            "source_case_summaries": list(exercise.get("source_case_summaries", []) or []),
            "source_case_instructions": list(exercise.get("source_case_instructions", []) or []),
            "domain_validator_name": exercise.get("domain_validator_name", ""),
            "domain_validator_flag": exercise.get("domain_validator_flag", ""),
            "domain_validator_issues": list(exercise.get("domain_validator_issues", []) or []),
            "deterministic_repair_applied": exercise.get("deterministic_repair_applied", False),
            "values_recomputed": exercise.get("values_recomputed", {}),
            "final_memory_case_ids": list(exercise.get("final_memory_case_ids", []) or []),
            "memory_filter_stage": exercise.get("memory_filter_stage", ""),
            "local_validation_flag": exercise.get("local_validation_flag", ""),
            "local_validation_summary": exercise.get("local_validation_summary", ""),
            "local_validation_issues": list(exercise.get("local_validation_issues", []) or []),
            "symbolic_checks_ran": exercise.get("symbolic_checks_ran"),
            "symbolic_checks_passed": exercise.get("symbolic_checks_passed"),
            "symbolic_checks_required": exercise.get("symbolic_checks_required"),
            "judge_corrections_applied": exercise.get("judge_corrections_applied", False),
            "corrected_fields_applied": exercise.get("corrected_fields_applied", False),
        }

    def _build_judge_feedback(self, summary: str, issues: list[str]) -> str:
        """Convert judge feedback into a compact instruction for regeneration."""
        compact_issues = "; ".join(issue for issue in issues[:3] if issue)
        feedback_parts = [part for part in [summary.strip(), compact_issues.strip()] if part]
        guidance = self._build_targeted_regeneration_guidance(summary, issues)
        if guidance:
            feedback_parts.append(guidance)
        return " ".join(feedback_parts)

    def _build_targeted_regeneration_guidance(self, summary: str, issues: list[str]) -> str:
        """Turn repeated validator failures into explicit instructions for the next prompt."""
        normalized_feedback = self._normalize_lookup(" ".join([summary, *issues]))
        if "derivative validator failed" in normalized_feedback or "derivee" in normalized_feedback:
            return (
                "Your previous derivative was rejected by SymPy. Recompute the derivative step by step "
                "and ensure it passes symbolic validation."
            )
        if "probability validator failed" in normalized_feedback or "probabilite" in normalized_feedback:
            return (
                "Your previous probability computation was rejected by local arithmetic validation. "
                "Recompute every probability exactly before giving the final answer."
            )
        if "complex validator failed" in normalized_feedback or "complexe" in normalized_feedback:
            return (
                "Your previous complex roots were rejected by symbolic validation. "
                "Verify both the quadratic equation and the sum/product constraints."
            )
        if "visual-support validator failed" in normalized_feedback or "support visuel" in normalized_feedback:
            return (
                "Your previous visual support was rejected. Provide chart_data/table_data only when the statement "
                "explicitly requires it, and make it semantically consistent with the statement."
            )
        if "domain validator failed" in normalized_feedback or "f(-x)" in normalized_feedback:
            return (
                "Your previous attempt violated the function domain. Recheck every transformed argument against the declared domain."
            )
        if "area validator failed" in normalized_feedback or "aire" in normalized_feedback:
            return (
                "Your previous area computation had the wrong sign. If the integral is negative and the question asks for an area, flip the sign."
            )
        if "pedagogical-completeness validator failed" in normalized_feedback and "regression" in normalized_feedback:
            return (
                "For statistics/regression, choose one clear method. For classical regression, ask for the correlation coefficient r, "
                "the regression line and estimations; G, G1 and G2 are required only for Mayer exercises. "
                "If you use Mayer, explicitly ask for G, G1, G2 and the Mayer line."
            )
        if "regression numeric validator failed" in normalized_feedback:
            return (
                "Your previous statistics table and solution were inconsistent. Do not change table values unless you recompute r, "
                "the regression line, all predictions and the threshold year from the same table."
            )
        return ""

    def _is_missing_alignment_reference(self, alignment_reason: str) -> bool:
        """Detect when the selected couple is absent from the official alignment file."""
        normalized_reason = self._normalize_lookup(alignment_reason)
        return "aucune entree d alignement officielle" in normalized_reason or "couple non couvert" in normalized_reason

    def _looks_like_probability_topic(self, topic: str, subtopic: str) -> bool:
        normalized = self._normalize_lookup(f"{topic} {subtopic}")
        return any(marker in normalized for marker in ("probabil", "variable aleatoire", "bernoulli", "binomiale", "bayes", "conditionnement", "exponentielle"))

    def _looks_like_statistics_topic(self, topic: str, subtopic: str) -> bool:
        normalized = self._normalize_lookup(f"{topic} {subtopic}")
        return any(marker in normalized for marker in ("statistique", "regression", "correlation", "ajustement"))

    def _merge_warning_messages(self, existing: str, incoming: str) -> str:
        """Merge warning strings without duplicating empty messages."""
        parts = [part.strip() for part in [existing, incoming] if str(part).strip()]
        return " | ".join(parts)

    def _resolve_exercise_generator(self, topic: str, subtopic: str):
        """Choisir le générateur le plus pertinent pour les libellés classiques ou issus du dataset."""
        generator_map = {
            "Équations linéaires": self._linear_equation_exercise,
            "Équations du second degré": self._quadratic_exercise,
            "Fonctions": self._function_exercise,
            "Limites": self._limit_exercise,
            "Dérivées": self._derivative_exercise,
            "Intégrales": self._integral_exercise,
            "Triangles": self._triangle_exercise,
            "Cercles": self._circle_exercise,
            "Géométrie analytique": self._coordinate_geometry_exercise,
            "Probabilités": self._probability_exercise,
            "Statistiques descriptives": self._descriptive_statistics_exercise,
            "Distributions": self._distribution_exercise,
            "Nombres premiers": self._prime_number_exercise,
            "Arithmétique modulaire": self._modular_arithmetic_exercise,
            "Suites": self._sequence_exercise,
        }
        if subtopic in generator_map:
            return generator_map[subtopic]

        normalized = self._normalize_lookup(f"{topic} {subtopic}")
        keyword_rules = [
            (("matrice", "determinant", "systeme"), self._linear_equation_exercise),
            (("exponentielle", "logarithme", "limite", "continuite", "derivation"), self._function_exercise),
            (("integrale", "primitive", "aire", "volume"), self._integral_exercise),
            (("suite",), self._sequence_exercise),
            (("congruence", "bezout", "diophant"), self._modular_arithmetic_exercise),
            (("conique", "espace", "complexe", "similitude", "isometrie", "transformation"), self._coordinate_geometry_exercise),
            (("bayes", "binomiale", "bernoulli", "probabilite", "esperance", "variance"), self._probability_exercise),
            (("statistique", "regression", "correlation"), self._descriptive_statistics_exercise),
        ]

        for keywords, generator in keyword_rules:
            if any(keyword in normalized for keyword in keywords):
                return generator
        return self._generic_exercise

    def verify_answer(self, exercise: dict[str, Any], submitted_answer: str) -> dict[str, Any]:
        """Valider une réponse, avec SymPy quand c'est possible."""
        cleaned_answer = submitted_answer.strip()
        if not cleaned_answer:
            return {
                "is_correct": False,
                "feedback": "Veuillez saisir une réponse avant la vérification.",
                "expected_answer": exercise["display_answer"],
            }

        kind = exercise.get("answer_kind", "expression")
        is_correct = self._compare_answers(cleaned_answer, exercise.get("accepted_answers", []), kind)
        feedback = (
            "Correct. Votre démarche mène bien au résultat mathématique attendu."
            if is_correct
            else "Ce n'est pas encore tout à fait juste. Comparez votre dernière transformation avec la cible attendue et relisez l'indice."
        )
        return {
            "is_correct": is_correct,
            "feedback": feedback,
            "expected_answer": exercise["display_answer"],
            "solution_steps": exercise["solution_steps"],
        }

    def generate_tutor_response(
        self,
        student_message: str,
        mode: str,
        exercise_context: dict[str, Any] | None = None,
        profile: dict[str, Any] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Produire une reponse tutorale via Qwen/OpenRouter avec un repli local."""
        return generate_tutor_reply(
            student_message=student_message,
            mode=mode,
            exercise_context=exercise_context,
            profile=profile,
            conversation_history=conversation_history,
        )

    def get_progress_analytics(
        self,
        student_id: str,
        *,
        user_email: str = "",
        profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retourner les données du tableau de bord analytique depuis MongoDB."""
        return get_user_progress_analytics(
            student_id=student_id,
            user_email=user_email,
            profile=profile,
        )

    def save_settings(self, settings_payload: dict[str, Any]) -> bool:
        """Placeholder de sauvegarde des paramètres."""
        _ = settings_payload
        return True

    def get_teacher_panel_data(self, *, teacher_email: str) -> dict[str, Any]:
        """Retourner les donnees enseignant depuis MongoDB."""
        return get_teacher_panel_snapshot(teacher_email)

    def create_teacher_group(
        self,
        *,
        teacher_email: str,
        teacher_user_id: str,
        teacher_name: str,
        group_name: str,
        section: str = "",
        level: str = "Bac",
    ) -> dict[str, Any]:
        """Creer un groupe enseignant persistant."""
        return create_teacher_group(
            teacher_email=teacher_email,
            teacher_user_id=teacher_user_id,
            teacher_name=teacher_name,
            group_name=group_name,
            section=section,
            level=level,
        )

    def add_student_to_teacher_group(
        self,
        *,
        teacher_email: str,
        group_id: str,
        student_email: str,
    ) -> dict[str, Any]:
        """Ajouter un etudiant a un groupe par son adresse e-mail."""
        return add_student_to_group(
            teacher_email=teacher_email,
            group_id=group_id,
            student_email=student_email,
        )

    def assign_exercise_to_group(
        self,
        *,
        teacher_email: str,
        teacher_user_id: str,
        teacher_name: str,
        group_id: str,
        exercise: dict[str, Any],
        due_date: date | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Assigner un exercice genere a tous les membres d'un groupe."""
        gated_exercise = apply_final_display_decision(exercise)
        if gated_exercise.get("final_display_decision") != "presented":
            reasons = gated_exercise.get("final_display_blocking_reasons", []) or [
                "L'exercice n'a pas passe le gate final."
            ]
            return {
                "ok": False,
                "message": "Impossible d'assigner un exercice bloque par la validation finale.",
                "issues": reasons,
            }
        return assign_teacher_exercise_to_group(
            teacher_email=teacher_email,
            teacher_user_id=teacher_user_id,
            teacher_name=teacher_name,
            group_id=group_id,
            exercise=gated_exercise,
            due_date=due_date,
            note=note,
        )

    def get_teacher_supervision_view(
        self,
        *,
        teacher_email: str,
        assignment_id: str,
        student_email: str = "",
    ) -> dict[str, Any] | None:
        """Charger la vue de supervision d'un exercice assigne."""
        return get_teacher_supervision_view(
            teacher_email=teacher_email,
            assignment_id=assignment_id,
            student_email=student_email,
        )

    def _linear_equation_exercise(self, level: str, topic: str, subtopic: str, difficulty: str, exercise_type: str) -> dict[str, Any]:
        answer = random.choice([4, 5, 6, 7, 8])
        coefficient = random.choice([2, 3, 4, 5])
        offset = random.choice([-9, -5, 3, 7])
        constant = coefficient * answer + offset
        prompt = f"Résoudre pour x : {coefficient}x + ({offset}) = {constant}"
        return self._build_exercise(
            topic,
            subtopic,
            level,
            difficulty,
            exercise_type,
            title="Entraînement sur les équations linéaires",
            prompt=self._adapt_prompt(prompt, exercise_type, topic),
            hint="Annule d'abord le terme constant, puis divise par le coefficient de x.",
            accepted_answers=[str(answer)],
            display_answer=str(answer),
            answer_kind="numeric",
            steps=[
                f"Partir de {coefficient}x + ({offset}) = {constant}.",
                f"Ajouter {-offset} aux deux membres pour isoler le terme en x.",
                f"On obtient {coefficient}x = {coefficient * answer}.",
                f"Diviser par {coefficient} pour trouver x = {answer}.",
            ],
            objective="Isoler l'inconnue à l'aide des opérations inverses.",
        )

    def _quadratic_exercise(self, level: str, topic: str, subtopic: str, difficulty: str, exercise_type: str) -> dict[str, Any]:
        root_a, root_b = random.sample([1, 2, 3, 4, 5, 6], 2)
        coefficient = -(root_a + root_b)
        constant = root_a * root_b
        prompt = f"Résoudre l'équation du second degré x^2 + ({coefficient})x + {constant} = 0."
        return self._build_exercise(
            topic,
            subtopic,
            level,
            difficulty,
            exercise_type,
            title="Vérification des racines d'une équation quadratique",
            prompt=self._adapt_prompt(prompt, exercise_type, topic),
            hint="Cherche deux nombres dont le produit donne le terme constant et la somme le coefficient de x.",
            accepted_answers=[f"{root_a},{root_b}", f"{root_b},{root_a}"],
            display_answer=f"x = {root_a} ou x = {root_b}",
            answer_kind="set",
            steps=[
                f"Factoriser le polynôme sous la forme (x - {root_a})(x - {root_b}) = 0.",
                "Appliquer la propriété du produit nul.",
                f"Les solutions sont x = {root_a} et x = {root_b}.",
            ],
            objective="Relier la factorisation à la propriété du produit nul.",
        )

    def _function_exercise(self, level: str, topic: str, subtopic: str, difficulty: str, exercise_type: str) -> dict[str, Any]:
        coefficient = random.choice([2, 3, 4])
        bias = random.choice([-5, -2, 1, 4])
        input_value = random.choice([2, 3, 5])
        answer = coefficient * input_value + bias
        prompt = f"Soit f(x) = {coefficient}x + ({bias}), calculer f({input_value})."
        return self._build_exercise(
            topic,
            subtopic,
            level,
            difficulty,
            exercise_type,
            title="Évaluation d'une fonction",
            prompt=self._adapt_prompt(prompt, exercise_type, topic),
            hint="Remplace la variable par la valeur donnée avant de simplifier.",
            accepted_answers=[str(answer)],
            display_answer=str(answer),
            answer_kind="numeric",
            steps=[
                f"Remplacer x par {input_value}.",
                f"Calculer {coefficient}({input_value}) + ({bias}).",
                f"Simplifier pour obtenir {answer}.",
            ],
            objective="Évaluer correctement une fonction en un point donné.",
        )

    def _limit_exercise(self, level: str, topic: str, subtopic: str, difficulty: str, exercise_type: str) -> dict[str, Any]:
        point = random.choice([1, 2, 3, 4])
        coefficient = random.choice([2, 3, 5])
        bias = random.choice([-4, 1, 6])
        answer = coefficient * point + bias
        prompt = f"Évaluer lim(x->{point}) ({coefficient}x + ({bias}))."
        return self._build_exercise(
            topic,
            subtopic,
            level,
            difficulty,
            exercise_type,
            title="Évaluation d'une limite",
            prompt=self._adapt_prompt(prompt, exercise_type, topic),
            hint="Pour un polynôme, commence par tester la substitution directe.",
            accepted_answers=[str(answer)],
            display_answer=str(answer),
            answer_kind="numeric",
            steps=[
                "Reconnaître que l'expression est continue au point considéré.",
                f"Remplacer directement x par {point}.",
                f"Calculer {coefficient}({point}) + ({bias}) = {answer}.",
            ],
            objective="Évaluer la limite d'une expression continue par substitution.",
        )

    def _derivative_exercise(self, level: str, topic: str, subtopic: str, difficulty: str, exercise_type: str) -> dict[str, Any]:
        coefficient = random.choice([2, 3, 4])
        power = random.choice([2, 3, 4])
        bias = random.choice([-5, -1, 2, 7])
        x = sp.symbols("x") if sp else None
        if sp:
            expression = coefficient * x**power + bias * x
            derivative = sp.diff(expression, x)
            prompt = f"Déterminer la dérivée de f(x) = {sp.sstr(expression)}."
            display_answer = sp.sstr(derivative)
        else:
            prompt = f"Déterminer la dérivée de f(x) = {coefficient}x^{power} + ({bias})x."
            display_answer = f"{coefficient * power}*x**{power - 1} + ({bias})"
        return self._build_exercise(
            topic,
            subtopic,
            level,
            difficulty,
            exercise_type,
            title="Échauffement sur les dérivées",
            prompt=self._adapt_prompt(prompt, exercise_type, topic),
            hint="Applique la règle de dérivation terme à terme et pense à la dérivée d'un terme linéaire.",
            accepted_answers=[display_answer],
            display_answer=display_answer,
            answer_kind="expression",
            steps=[
                "Dériver chaque terme séparément.",
                "Appliquer la règle de puissance au terme polynomial.",
                "Dériver le terme linéaire.",
                f"Assembler les résultats pour obtenir {display_answer}.",
            ],
            objective="Appliquer correctement la règle de puissance aux polynômes.",
        )

    def _integral_exercise(self, level: str, topic: str, subtopic: str, difficulty: str, exercise_type: str) -> dict[str, Any]:
        coefficient = random.choice([1, 2, 3])
        power = random.choice([1, 2, 3])
        upper_bound = random.choice([2, 3, 4])
        x = sp.symbols("x") if sp else None
        if sp:
            integrand = coefficient * x**power
            answer = sp.integrate(integrand, (x, 0, upper_bound))
            prompt = f"Calculer l'intégrale définie de 0 à {upper_bound} de {sp.sstr(integrand)} dx."
            display_answer = sp.sstr(answer)
        else:
            answer = coefficient * upper_bound ** (power + 1) / (power + 1)
            prompt = f"Calculer l'intégrale définie de 0 à {upper_bound} de {coefficient}x^{power} dx."
            display_answer = str(answer)
        return self._build_exercise(
            topic,
            subtopic,
            level,
            difficulty,
            exercise_type,
            title="Accumulation par intégrale",
            prompt=self._adapt_prompt(prompt, exercise_type, topic),
            hint="Trouve d'abord une primitive, puis évalue-la entre la borne supérieure et la borne inférieure.",
            accepted_answers=[display_answer],
            display_answer=display_answer,
            answer_kind="expression",
            steps=[
                "Déterminer une primitive de l'intégrande avec la règle inverse de puissance.",
                f"Évaluer cette primitive en x = {upper_bound} puis en x = 0.",
                f"Soustraire la valeur à la borne inférieure pour obtenir {display_answer}.",
            ],
            objective="Interpréter une intégrale définie comme une quantité accumulée sur un intervalle.",
        )

    def _triangle_exercise(self, level: str, topic: str, subtopic: str, difficulty: str, exercise_type: str) -> dict[str, Any]:
        leg_a, leg_b = random.choice([(3, 4), (5, 12), (8, 15)])
        hypotenuse = int((leg_a**2 + leg_b**2) ** 0.5)
        prompt = (
            f"Un triangle rectangle a pour côtés de l'angle droit {leg_a} et {leg_b}. "
            "Déterminer la longueur de l'hypoténuse."
        )
        return self._build_exercise(
            topic,
            subtopic,
            level,
            difficulty,
            exercise_type,
            title="Raisonnement sur un triangle rectangle",
            prompt=self._adapt_prompt(prompt, exercise_type, topic),
            hint="Utilise la relation de Pythagore entre les deux côtés de l'angle droit et l'hypoténuse.",
            accepted_answers=[str(hypotenuse)],
            display_answer=str(hypotenuse),
            answer_kind="numeric",
            steps=[
                f"Calculer {leg_a}² + {leg_b}².",
                f"Prendre la racine carrée de {leg_a**2 + leg_b**2}.",
                f"L'hypoténuse vaut {hypotenuse}.",
            ],
            objective="Mobiliser le théorème de Pythagore pour relier les longueurs.",
        )

    def _circle_exercise(self, level: str, topic: str, subtopic: str, difficulty: str, exercise_type: str) -> dict[str, Any]:
        radius = random.choice([3, 4, 5, 7])
        answer = f"{radius * radius}*pi"
        prompt = f"Déterminer l'aire d'un cercle de rayon {radius}. Donner la réponse exacte."
        return self._build_exercise(
            topic,
            subtopic,
            level,
            difficulty,
            exercise_type,
            title="Calcul d'aire d'un cercle",
            prompt=self._adapt_prompt(prompt, exercise_type, topic),
            hint="L'aire dépend du carré du rayon, pas du diamètre.",
            accepted_answers=[answer, f"{radius * radius}pi"],
            display_answer=f"{radius * radius}π",
            answer_kind="expression",
            steps=[
                "Rappeler la formule A = πr².",
                f"Remplacer r par {radius}.",
                f"Calculer d'abord le carré : {radius}² = {radius * radius}.",
                f"L'aire exacte est {radius * radius}π.",
            ],
            objective="Appliquer une formule géométrique en conservant une écriture exacte.",
        )

    def _coordinate_geometry_exercise(self, level: str, topic: str, subtopic: str, difficulty: str, exercise_type: str) -> dict[str, Any]:
        x1, y1 = random.choice([(1, 2), (2, 5), (3, -1)])
        rise, run = random.choice([(4, 2), (3, 6), (-2, 4)])
        x2, y2 = x1 + run, y1 + rise
        answer = rise / run
        prompt = f"Déterminer la pente de la droite passant par ({x1}, {y1}) et ({x2}, {y2})."
        simplified = str(sp.Rational(rise, run)) if sp else str(answer)
        return self._build_exercise(
            topic,
            subtopic,
            level,
            difficulty,
            exercise_type,
            title="Calcul de pente",
            prompt=self._adapt_prompt(prompt, exercise_type, topic),
            hint="Utilise le rapport variation de y sur variation de x en gardant le même ordre dans les différences.",
            accepted_answers=[str(answer), simplified],
            display_answer=simplified,
            answer_kind="expression",
            steps=[
                "Calculer la variation des ordonnées.",
                "Calculer la variation des abscisses.",
                f"Former le quotient (Δy)/(Δx) = {rise}/{run}.",
                f"Simplifier pour obtenir {simplified}.",
            ],
            objective="Interpréter la pente comme un taux de variation vertical sur horizontal.",
        )

    def _probability_exercise(self, level: str, topic: str, subtopic: str, difficulty: str, exercise_type: str) -> dict[str, Any]:
        target = random.choice([4, 5, 6])
        answer = target / 6
        prompt = f"On lance un dé équilibré une seule fois. Quelle est la probabilité d'obtenir un nombre inférieur ou égal à {target} ?"
        return self._build_exercise(
            topic,
            subtopic,
            level,
            difficulty,
            exercise_type,
            title="Vérification de probabilité",
            prompt=self._adapt_prompt(prompt, exercise_type, topic),
            hint="Compte les issues favorables puis divise par le nombre total d'issues équiprobables.",
            accepted_answers=[str(sp.Rational(target, 6)) if sp else str(answer), str(answer)],
            display_answer=str(sp.Rational(target, 6)) if sp else str(answer),
            answer_kind="expression",
            steps=[
                f"Compter les issues favorables : de 1 à {target}.",
                "Compter le nombre total d'issues d'un dé équilibré.",
                f"Former le rapport {target}/6 puis simplifier si nécessaire.",
            ],
            objective="Traduire une question de probabilité en rapport favorable sur total.",
        )

    def _descriptive_statistics_exercise(self, level: str, topic: str, subtopic: str, difficulty: str, exercise_type: str) -> dict[str, Any]:
        data = random.choice([[4, 7, 7, 8, 9], [10, 12, 14, 14, 15], [3, 6, 9, 12, 15]])
        answer = sum(data) / len(data)
        prompt = f"Calculer la moyenne de la série statistique suivante : {data}."
        return self._build_exercise(
            topic,
            subtopic,
            level,
            difficulty,
            exercise_type,
            title="Calcul de moyenne",
            prompt=self._adapt_prompt(prompt, exercise_type, topic),
            hint="Additionne toutes les valeurs avant de diviser par l'effectif.",
            accepted_answers=[str(answer)],
            display_answer=str(answer),
            answer_kind="numeric",
            steps=[
                f"Additionner les valeurs pour obtenir {sum(data)}.",
                f"La série comporte {len(data)} valeurs.",
                f"Diviser {sum(data)} par {len(data)} pour obtenir {answer}.",
            ],
            objective="Calculer une moyenne de manière rigoureuse et lisible.",
        )

    def _distribution_exercise(self, level: str, topic: str, subtopic: str, difficulty: str, exercise_type: str) -> dict[str, Any]:
        values = [0, 1, 2]
        probabilities = [0.2, 0.5, 0.3]
        answer = round(sum(value * probability for value, probability in zip(values, probabilities)), 2)
        prompt = (
            "Une variable aléatoire X prend les valeurs 0, 1 et 2 avec les probabilités 0,2 ; 0,5 ; 0,3. "
            "Déterminer l'espérance E(X)."
        )
        return self._build_exercise(
            topic,
            subtopic,
            level,
            difficulty,
            exercise_type,
            title="Exercice sur l'espérance",
            prompt=self._adapt_prompt(prompt, exercise_type, topic),
            hint="Multiplie chaque issue par sa probabilité puis additionne les contributions.",
            accepted_answers=[str(answer)],
            display_answer=str(answer),
            answer_kind="numeric",
            steps=[
                "Multiplier chaque issue par sa probabilité associée.",
                "Additionner les trois contributions pondérées.",
                f"L'espérance vaut {answer}.",
            ],
            objective="Relier une distribution de probabilité à une moyenne pondérée.",
        )

    def _prime_number_exercise(self, level: str, topic: str, subtopic: str, difficulty: str, exercise_type: str) -> dict[str, Any]:
        number = random.choice([84, 90, 126, 150])
        factors = self._prime_factorization(number)
        prompt = f"Écrire la décomposition en facteurs premiers de {number}."
        return self._build_exercise(
            topic,
            subtopic,
            level,
            difficulty,
            exercise_type,
            title="Décomposition en facteurs premiers",
            prompt=self._adapt_prompt(prompt, exercise_type, topic),
            hint="Commence par les plus petits nombres premiers et poursuis jusqu'à n'obtenir que des facteurs premiers.",
            accepted_answers=[factors.replace(" × ", "*"), factors],
            display_answer=factors,
            answer_kind="text",
            steps=[
                "Diviser successivement par les plus petits nombres premiers possibles.",
                "Continuer jusqu'à ce que tous les facteurs soient premiers.",
                f"La décomposition complète est {factors}.",
            ],
            objective="Décomposer un entier composé de façon unique en facteurs premiers.",
        )

    def _modular_arithmetic_exercise(self, level: str, topic: str, subtopic: str, difficulty: str, exercise_type: str) -> dict[str, Any]:
        base = random.choice([17, 29, 35, 47])
        modulus = random.choice([5, 6, 8, 9])
        answer = base % modulus
        prompt = f"Calculer {base} mod {modulus}."
        return self._build_exercise(
            topic,
            subtopic,
            level,
            difficulty,
            exercise_type,
            title="Échauffement en arithmétique modulaire",
            prompt=self._adapt_prompt(prompt, exercise_type, topic),
            hint="Cherche le reste de la division euclidienne du premier nombre par le second.",
            accepted_answers=[str(answer)],
            display_answer=str(answer),
            answer_kind="numeric",
            steps=[
                f"Diviser {base} par {modulus}.",
                "Identifier le reste de cette division.",
                f"Le reste est {answer}, donc {base} mod {modulus} = {answer}.",
            ],
            objective="Interpréter le calcul modulaire comme un raisonnement sur les restes.",
        )

    def _sequence_exercise(self, level: str, topic: str, subtopic: str, difficulty: str, exercise_type: str) -> dict[str, Any]:
        start = random.choice([2, 3, 5])
        difference = random.choice([3, 4, 6])
        term_number = random.choice([8, 10, 12])
        answer = start + (term_number - 1) * difference
        prompt = f"Une suite arithmétique commence à {start} avec une raison de {difference}. Déterminer le terme numéro {term_number}."
        return self._build_exercise(
            topic,
            subtopic,
            level,
            difficulty,
            exercise_type,
            title="Raisonnement sur les suites",
            prompt=self._adapt_prompt(prompt, exercise_type, topic),
            hint="Utilise la formule a_n = a_1 + (n - 1)d.",
            accepted_answers=[str(answer)],
            display_answer=str(answer),
            answer_kind="numeric",
            steps=[
                "Écrire la formule générale de la suite arithmétique.",
                f"Remplacer a₁ = {start}, d = {difference} et n = {term_number}.",
                f"Simplifier pour obtenir le terme {term_number} = {answer}.",
            ],
            objective="Passer d'un motif répétitif à une formule explicite.",
        )

    def _generic_exercise(self, level: str, topic: str, subtopic: str, difficulty: str, exercise_type: str) -> dict[str, Any]:
        prompt = (
            f"Construire une explication concise et résoudre une question représentative de niveau {difficulty.lower()} "
            f"sur {subtopic.lower()} pour un apprenant de niveau {level.lower()}."
        )
        return self._build_exercise(
            topic,
            subtopic,
            level,
            difficulty,
            exercise_type,
            title=f"Exploration : {subtopic}",
            prompt=self._adapt_prompt(prompt, exercise_type, topic),
            hint="Découpe le problème en un théorème ou une formule, puis une étape d'application.",
            accepted_answers=["Voir la solution guidée."],
            display_answer="Voir la solution guidée.",
            answer_kind="text",
            steps=[
                "Identifier la notion mathématique évaluée.",
                "Choisir le théorème ou la formule adaptée.",
                "L'appliquer soigneusement aux données de l'énoncé.",
            ],
            objective=f"Développer la compréhension conceptuelle de {subtopic.lower()}.",
        )

    def _build_exercise(
        self,
        topic: str,
        subtopic: str,
        level: str,
        difficulty: str,
        exercise_type: str,
        *,
        title: str,
        prompt: str,
        hint: str,
        accepted_answers: list[str],
        display_answer: str,
        answer_kind: str,
        steps: list[str],
        objective: str,
        hidden_solution: str | None = None,
    ) -> dict[str, Any]:
        """Composer la structure standard d'un exercice."""
        exercise_id = f"{topic[:3].upper()}-{subtopic[:3].upper()}-{random.randint(1000, 9999)}"
        payload = {
            "id": exercise_id,
            "topic": topic,
            "subtopic": subtopic,
            "level": level,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "difficulty": difficulty,
            "exercise_type": exercise_type,
            "title": title,
            "prompt": prompt,
            "hint": hint,
            "accepted_answers": accepted_answers,
            "display_answer": display_answer,
            "answer_kind": answer_kind,
            "solution_steps": steps,
            "hidden_solution": hidden_solution or self._compose_hidden_solution(steps, display_answer),
            "learning_objective": objective,
            "tags": [topic, subtopic, difficulty, exercise_type],
        }
        if exercise_type == "QCM":
            payload["options"] = self._generate_options(display_answer, answer_kind)
        return payload

    def _compose_hidden_solution(self, steps: list[str], display_answer: str) -> str:
        """Derive a hidden full solution for internal review and audit logs."""
        cleaned_steps = [str(step).strip() for step in steps if str(step).strip()]
        numbered_steps = " ".join(
            [f"Etape {index} : {step}" for index, step in enumerate(cleaned_steps, start=1)]
        ).strip()
        if display_answer:
            return f"{numbered_steps} Reponse finale : {display_answer}.".strip()
        return numbered_steps or "Solution detaillee indisponible."

    def _generate_options(self, correct_answer: str, answer_kind: str) -> list[str]:
        """Créer quelques distracteurs simples pour les QCM."""
        if answer_kind == "numeric":
            try:
                base_value = float(correct_answer)
                options = [str(base_value - 2), str(base_value), str(base_value + 1), str(base_value + 3)]
                return self._deduplicate_preserving_order(options)
            except ValueError:
                return self._deduplicate_preserving_order([correct_answer, "0", "1", "2"])
        return self._deduplicate_preserving_order([correct_answer, "0", "x", "À retravailler"])

    def _compare_answers(self, provided: str, accepted: list[str], kind: str) -> bool:
        """Comparer la réponse fournie aux formes acceptées avec normalisation robuste."""
        if not accepted:
            return False

        if kind == "set":
            provided_parts = self._normalize_answer_set(provided)
            for candidate in accepted:
                if provided_parts and provided_parts == self._normalize_answer_set(candidate):
                    return True
            return False

        normalized_provided = self._normalize_answer_text(provided)
        if kind == "text":
            return any(
                self._normalize_answer_text(answer) in normalized_provided
                or normalized_provided in self._normalize_answer_text(answer)
                for answer in accepted
                if self._normalize_answer_text(answer)
            )

        for answer in accepted:
            if self._sympy_equivalent(provided, answer):
                return True
            if normalized_provided == self._normalize_answer_text(answer):
                return True
        return False

    def _normalize_answer_set(self, value: str) -> set[str]:
        """Normalize common set-answer variants such as 'x=1 ou x=2'."""
        cleaned = self._normalize_answer_text(value)
        cleaned = re.sub(r"\b(x|y|t|n)\s*=\s*", "", cleaned)
        cleaned = cleaned.replace(" ou ", ",").replace(" or ", ",").replace(";", ",")
        cleaned = cleaned.strip("{}[]() ")
        return {part.strip() for part in cleaned.split(",") if part.strip()}

    def _normalize_answer_text(self, value: str) -> str:
        """Normalize accents, math wrappers, whitespace and punctuation for answer comparison."""
        text = str(value or "").strip().lower()
        text = text.replace("\\(", "").replace("\\)", "").replace("$", "")
        text = text.replace("−", "-").replace("×", "*").replace("^", "**")
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _sympy_equivalent(self, provided: str, expected: str) -> bool:
        """Comparer deux expressions avec simplification symbolique."""
        if sp is None:
            return False
        try:
            provided_expr = sp.sympify(provided.replace("^", "**"))
            expected_expr = sp.sympify(expected.replace("^", "**"))
            return sp.simplify(provided_expr - expected_expr) == 0
        except Exception:
            return False

    def _adapt_prompt(self, base_prompt: str, exercise_type: str, topic: str) -> str:
        """Adapter la formulation selon le type d'exercice."""
        normalized_type = self._normalize_lookup(exercise_type)
        if normalized_type in {"probleme contextualise", "exercice probleme"}:
            return (
                f"Dans un contexte réaliste lié à {topic.lower()}, résoudre le problème suivant : {base_prompt} "
                "Commencez par identifier clairement la quantité mathématique cherchée."
            )
        if normalized_type == "etapes guidees":
            return (
                f"{base_prompt} Présentez le raisonnement intermédiaire en deux ou trois étapes avant d'écrire la réponse finale."
            )
        if exercise_type == "QCM":
            return f"{base_prompt} Puis choisissez la meilleure réponse parmi les propositions."
        return base_prompt

    def _normalize_lookup(self, value: str) -> str:
        """Normaliser une chaîne pour des correspondances de mots-clés plus robustes."""
        normalized = unicodedata.normalize("NFKD", value)
        ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
        return " ".join(ascii_value.lower().split())

    def _prime_factorization(self, number: int) -> str:
        """Générer une écriture simple de la décomposition en facteurs premiers."""
        n = number
        factors = []
        divisor = 2
        while divisor * divisor <= n:
            while n % divisor == 0:
                factors.append(str(divisor))
                n //= divisor
            divisor += 1
        if n > 1:
            factors.append(str(n))
        return " × ".join(factors)

    def _deduplicate_preserving_order(self, values: list[str]) -> list[str]:
        """Supprimer les doublons en conservant l'ordre."""
        seen = set()
        ordered = []
        for value in values:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered


def get_api_client() -> MathTutorApiClient:
    """Retourner une instance sûre du client API frontend."""
    return MathTutorApiClient()
