"""Persistent audit log for generated exercises and judge outcomes."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from typing import Any

from frontend.utils.exercise_presentation_gate import apply_final_display_decision
from frontend.utils.paths import GENERATED_EXERCISES_LOG_PATH


def persist_generated_exercise_records(
    exercise: dict[str, Any],
    *,
    user_email: str,
    user_role: str,
    user_display_name: str,
) -> None:
    """Append rejected attempts and the final exercise to the JSONL audit log."""
    rejected_attempts = exercise.get("judge_rejected_attempts", [])
    for attempt in rejected_attempts:
        persist_rejected_attempt_record(
            attempt,
            parent_exercise=exercise,
            user_email=user_email,
            user_role=user_role,
            user_display_name=user_display_name,
        )
    persist_final_exercise_record(
        exercise,
        user_email=user_email,
        user_role=user_role,
        user_display_name=user_display_name,
    )


def persist_rejected_attempt_record(
    attempt: dict[str, Any],
    *,
    parent_exercise: dict[str, Any],
    user_email: str,
    user_role: str,
    user_display_name: str,
) -> None:
    """Append one rejected generation attempt to the audit log."""
    _append_records(
        [
            _build_audit_record(
                source=attempt,
                parent=parent_exercise,
                record_kind="rejected_attempt",
                user_email=user_email,
                user_role=user_role,
                user_display_name=user_display_name,
                presented_to_student=False,
            )
        ]
    )


def persist_final_exercise_record(
    exercise: dict[str, Any],
    *,
    user_email: str,
    user_role: str,
    user_display_name: str,
) -> None:
    """Append the final returned exercise to the audit log."""
    finalized_exercise = apply_final_display_decision(exercise)
    _append_records(
        [
            _build_audit_record(
                source=finalized_exercise,
                parent=finalized_exercise,
                record_kind="final_presented"
                if finalized_exercise.get("final_display_decision") == "presented"
                else "blocked_final",
                user_email=user_email,
                user_role=user_role,
                user_display_name=user_display_name,
                presented_to_student=finalized_exercise.get("final_display_decision") == "presented",
            )
        ]
    )


def _build_audit_record(
    *,
    source: dict[str, Any],
    parent: dict[str, Any],
    record_kind: str,
    user_email: str,
    user_role: str,
    user_display_name: str,
    presented_to_student: bool,
) -> dict[str, Any]:
    """Normalize one audit entry with enough metadata for later analysis."""
    recorded_at = datetime.now().isoformat(timespec="seconds")
    judge_summary = _value(source, parent, "summary") or _value(source, parent, "judge_summary")
    judge_issues = _ensure_list(_value(source, parent, "issues") or _value(source, parent, "judge_issues"))
    judge_validation_flag = _value(source, parent, "judge_validation_flag") or (
        "wrong" if record_kind == "rejected_attempt" else "unknown"
    )

    return {
        "recorded_at": recorded_at,
        "record_kind": record_kind,
        "presented_to_student": presented_to_student,
        "blocked_before_display": bool(_value(source, parent, "blocked_before_display")),
        "exercise_id": _value(source, parent, "id"),
        "generation_trace_id": _value(source, parent, "generation_trace_id"),
        "generation_attempt_number": _value(source, parent, "attempt_number")
        or _value(source, parent, "generation_attempt_number"),
        "generated_at": _value(source, parent, "generated_at"),
        "user_email": _redact_email(user_email),
        "user_hash": _hash_user_identifier(user_email),
        "user_role": user_role,
        "user_display_name": _redact_display_name(user_display_name),
        "level": _value(source, parent, "level"),
        "section": _value(source, parent, "section"),
        "topic": _value(source, parent, "topic"),
        "subtopic": _value(source, parent, "subtopic"),
        "difficulty": _value(source, parent, "difficulty"),
        "exercise_type": _value(source, parent, "exercise_type"),
        "title": _value(source, parent, "title"),
        "context": _value(source, parent, "context"),
        "questions": _ensure_list(_value(source, parent, "questions")),
        "tags": _ensure_list(_value(source, parent, "tags")),
        "generation_backend": _value(source, parent, "generation_backend"),
        "generation_warning": _value(source, parent, "generation_warning"),
        "is_true_llm_generation": _value(source, parent, "is_true_llm_generation"),
        "llm_json_parse_status": _value(source, parent, "llm_json_parse_status"),
        "llm_json_extraction_method": _value(source, parent, "llm_json_extraction_method"),
        "llm_json_parse_error": _value(source, parent, "llm_json_parse_error"),
        "llm_raw_response_preview": _truncate_text(_value(source, parent, "llm_raw_response_preview"), 1000),
        "llm_generation_attempts_count": _value(source, parent, "llm_generation_attempts_count"),
        "openrouter_http_status": _value(source, parent, "openrouter_http_status"),
        "openrouter_error_type": _value(source, parent, "openrouter_error_type"),
        "openrouter_error_message": _truncate_text(_value(source, parent, "openrouter_error_message"), 1000),
        "openrouter_response_format_mode": _value(source, parent, "openrouter_response_format_mode"),
        "openrouter_model_used": _value(source, parent, "openrouter_model_used"),
        "openrouter_request_id": _value(source, parent, "openrouter_request_id"),
        "openrouter_provider": _value(source, parent, "openrouter_provider"),
        "openrouter_usage": _value(source, parent, "openrouter_usage"),
        "openrouter_call_attempts": _ensure_list(_value(source, parent, "openrouter_call_attempts")),
        "prompt_char_count": _value(source, parent, "prompt_char_count"),
        "prompt_token_estimate": _value(source, parent, "prompt_token_estimate"),
        "number_of_memory_cases": _value(source, parent, "number_of_memory_cases"),
        "fallback_used": _value(source, parent, "fallback_used"),
        "fallback_reason": _value(source, parent, "fallback_reason"),
        "display_source_category": _value(source, parent, "display_source_category"),
        "retry_strategy": _value(source, parent, "retry_strategy"),
        "failure_categories": _ensure_list(_value(source, parent, "failure_categories")),
        "previous_errors_injected": _ensure_list(_value(source, parent, "previous_errors_injected")),
        "demo_mode_used": _value(source, parent, "demo_mode_used"),
        "judge_status": _value(source, parent, "judge_status"),
        "judge_validation_flag": judge_validation_flag,
        "judge_model": _value(source, parent, "judge_model"),
        "judge_summary": judge_summary,
        "judge_alignment_status": _value(source, parent, "alignment_status")
        or _value(source, parent, "judge_alignment_status"),
        "judge_alignment_reason": _value(source, parent, "alignment_reason")
        or _value(source, parent, "judge_alignment_reason"),
        "judge_confidence": _value(source, parent, "judge_confidence"),
        "judge_issues": judge_issues,
        "solution_validation_status": _value(source, parent, "solution_validation_status"),
        "solution_validation_summary": _value(source, parent, "solution_validation_summary"),
        "solution_validation_issues": _ensure_list(_value(source, parent, "solution_validation_issues")),
        "solution_validation_confidence": _value(source, parent, "solution_validation_confidence"),
        "solution_validation_model": _value(source, parent, "solution_validation_model"),
        "solution_validation_flag": _value(source, parent, "solution_validation_flag"),
        "solution_validation_sympy_report": _value(source, parent, "solution_validation_sympy_report"),
        "local_validation_flag": _value(source, parent, "local_validation_flag"),
        "local_validation_summary": _value(source, parent, "local_validation_summary"),
        "local_validation_issues": _ensure_list(_value(source, parent, "local_validation_issues")),
        "pedagogical_completeness_flag": _value(source, parent, "pedagogical_completeness_flag"),
        "pedagogical_completeness_summary": _value(source, parent, "pedagogical_completeness_summary"),
        "pedagogical_completeness_issues": _ensure_list(
            _value(source, parent, "pedagogical_completeness_issues")
        ),
        "symbolic_checks_ran": _value(source, parent, "symbolic_checks_ran"),
        "symbolic_checks_passed": _value(source, parent, "symbolic_checks_passed"),
        "symbolic_checks_required": _value(source, parent, "symbolic_checks_required"),
        "corrected_fields_applied": _value(source, parent, "corrected_fields_applied"),
        "student_facing_format_flag": _value(source, parent, "student_facing_format_flag"),
        "student_facing_format_issues": _ensure_list(_value(source, parent, "student_facing_format_issues")),
        "student_facing_format_after_repair": _value(source, parent, "student_facing_format_after_repair"),
        "domain_validator_name": _value(source, parent, "domain_validator_name"),
        "domain_router_key": _value(source, parent, "domain_router_key"),
        "domain_router_reason": _value(source, parent, "domain_router_reason"),
        "domain_validator_flag": _value(source, parent, "domain_validator_flag"),
        "domain_validator_issues": _ensure_list(_value(source, parent, "domain_validator_issues")),
        "question_coverage_flag": _value(source, parent, "question_coverage_flag"),
        "question_coverage_issues": _ensure_list(_value(source, parent, "question_coverage_issues")),
        "unanswered_question_indices": _ensure_list(_value(source, parent, "unanswered_question_indices")),
        "latex_repair_applied": _value(source, parent, "latex_repair_applied"),
        "latex_repair_issues_remaining": _ensure_list(_value(source, parent, "latex_repair_issues_remaining")),
        "final_gate_steps": _ensure_list(_value(source, parent, "final_gate_steps")),
        "final_gate_failed_step": _value(source, parent, "final_gate_failed_step"),
        "deterministic_repair_applied": _value(source, parent, "deterministic_repair_applied"),
        "values_recomputed": _value(source, parent, "values_recomputed"),
        "memory_filter_stage": _value(source, parent, "memory_filter_stage"),
        "memory_rejected_case_ids": _ensure_list(_value(source, parent, "memory_rejected_case_ids")),
        "memory_rejection_reasons": _ensure_list(_value(source, parent, "memory_rejection_reasons")),
        "final_memory_case_ids": _ensure_list(_value(source, parent, "final_memory_case_ids")),
        "judge_response_format_mode": _value(source, parent, "judge_response_format_mode"),
        "judge_raw_response_preview": _truncate_text(_value(source, parent, "judge_raw_response_preview"), 1000),
        "judge_json_parse_error": _value(source, parent, "judge_json_parse_error"),
        "judge_openrouter_error_type": _value(source, parent, "judge_openrouter_error_type"),
        "validator_response_format_mode": _value(source, parent, "validator_response_format_mode"),
        "validator_raw_response_preview": _truncate_text(_value(source, parent, "validator_raw_response_preview"), 1000),
        "validator_json_parse_error": _value(source, parent, "validator_json_parse_error"),
        "final_display_decision": _value(source, parent, "final_display_decision"),
        "final_display_blocking_reasons": _ensure_list(
            _value(source, parent, "final_display_blocking_reasons")
        ),
        "instruction": _value(source, parent, "prompt"),
        "generation_metadata": _value(source, parent, "generation_metadata"),
        "solution": _value(source, parent, "hidden_solution"),
        "expected_answer": _value(source, parent, "display_answer"),
        "answer_kind": _value(source, parent, "answer_kind"),
        "options": _ensure_list(_value(source, parent, "options")),
        "solution_steps": _ensure_list(_value(source, parent, "solution_steps")),
        "learning_objective": _value(source, parent, "learning_objective"),
        "estimated_time": _value(source, parent, "estimated_time"),
        "memory_adaptation_note": _value(source, parent, "memory_adaptation_note"),
        "retrieved_case_ids": _ensure_list(_value(source, parent, "retrieved_case_ids")),
        "source_case_summaries": _ensure_list(_value(source, parent, "source_case_summaries")),
        "source_case_instructions": _ensure_list(_value(source, parent, "source_case_instructions")),
        "support_summary": _value(source, parent, "support_summary"),
        "support_ready": _value(source, parent, "support_ready"),
        "table_data": _value(source, parent, "table_data"),
        "chart_data": _value(source, parent, "chart_data"),
        "graph_data": _value(source, parent, "graph_data"),
    }


def _append_records(records: list[dict[str, Any]]) -> None:
    """Append audit records as JSON Lines."""
    GENERATED_EXERCISES_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GENERATED_EXERCISES_LOG_PATH.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _value(source: dict[str, Any], parent: dict[str, Any], key: str) -> Any:
    """Read a key from the source record first, then from the final exercise."""
    if key in source and source.get(key) not in (None, ""):
        return source.get(key)
    return parent.get(key)


def _ensure_list(value: Any) -> list[Any]:
    """Normalize optional list-like fields for JSONL persistence."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _truncate_text(value: Any, limit: int) -> str:
    """Keep raw model diagnostics useful without letting the audit file balloon."""
    text = str(value or "").strip()
    return text[:limit]


def _hash_user_identifier(user_email: str) -> str:
    """Return a stable anonymous identifier for audit analysis."""
    normalized = str(user_email or "").strip().lower()
    if not normalized or normalized == "inconnu":
        return "anonymous"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _redact_email(user_email: str) -> str:
    """Avoid writing real e-mail addresses in the local JSONL audit file."""
    normalized = str(user_email or "").strip().lower()
    if "@" not in normalized:
        return "anonymous"
    domain = normalized.split("@", 1)[1]
    return f"anonymous@{domain}" if domain else "anonymous"


def _redact_display_name(display_name: str) -> str:
    """Avoid writing real display names in the local JSONL audit file."""
    return "Utilisateur anonymisé" if str(display_name or "").strip() else "Utilisateur anonyme"
