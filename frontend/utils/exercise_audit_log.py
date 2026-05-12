"""Persistent audit log for generated exercises and judge outcomes."""

from __future__ import annotations

from datetime import datetime
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
        "user_email": user_email,
        "user_role": user_role,
        "user_display_name": user_display_name,
        "level": _value(source, parent, "level"),
        "section": _value(source, parent, "section"),
        "topic": _value(source, parent, "topic"),
        "subtopic": _value(source, parent, "subtopic"),
        "difficulty": _value(source, parent, "difficulty"),
        "exercise_type": _value(source, parent, "exercise_type"),
        "title": _value(source, parent, "title"),
        "tags": _ensure_list(_value(source, parent, "tags")),
        "generation_backend": _value(source, parent, "generation_backend"),
        "generation_warning": _value(source, parent, "generation_warning"),
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
        "final_display_decision": _value(source, parent, "final_display_decision"),
        "final_display_blocking_reasons": _ensure_list(
            _value(source, parent, "final_display_blocking_reasons")
        ),
        "instruction": _value(source, parent, "prompt"),
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
