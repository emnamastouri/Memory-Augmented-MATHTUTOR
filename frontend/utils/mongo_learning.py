"""MongoDB persistence for user learning activity and progress analytics."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import streamlit as st
from pymongo import ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from frontend.utils.constants import (
    DEFAULT_STUDENT_PROFILE,
    MONGO_DB_NAME,
    MONGO_EXERCISE_RECORDS_COLLECTION,
    MONGO_LEARNING_EVENTS_COLLECTION,
    MONGO_USERS_COLLECTION,
)
from frontend.utils.mongo_auth import get_mongo_client

PAGE_VIEW_THROTTLE_SECONDS = 90


def get_learning_events_collection() -> Collection:
    """Return the learning-events collection with basic indexes."""
    collection = get_mongo_client()[MONGO_DB_NAME][MONGO_LEARNING_EVENTS_COLLECTION]
    collection.create_index([("user_email", ASCENDING), ("created_at", DESCENDING)])
    collection.create_index([("user_email", ASCENDING), ("event_type", ASCENDING)])
    collection.create_index([("generation_trace_id", ASCENDING)])
    return collection


def get_exercise_records_collection() -> Collection:
    """Return the persisted exercise-record collection with basic indexes."""
    collection = get_mongo_client()[MONGO_DB_NAME][MONGO_EXERCISE_RECORDS_COLLECTION]
    collection.create_index([("user_email", ASCENDING), ("generated_at", DESCENDING)])
    collection.create_index([("user_email", ASCENDING), ("generation_trace_id", ASCENDING)])
    collection.create_index([("user_email", ASCENDING), ("topic", ASCENDING)])
    return collection


def get_recent_exercise_history(limit: int = 8, *, user_email: str = "") -> list[dict[str, Any]]:
    """Load the most recent persisted exercises for one user."""
    effective_email = user_email.strip().lower() if user_email else get_session_user_context().get("email", "")
    if not effective_email:
        return []

    try:
        cursor = (
            get_exercise_records_collection()
            .find({"user_email": effective_email})
            .sort("generated_at", DESCENDING)
            .limit(limit)
        )
    except PyMongoError:
        return []

    items: list[dict[str, Any]] = []
    for document in cursor:
        if not _document_is_student_visible(document):
            continue
        status = _humanize_status(document.get("latest_status"), document.get("judge_status"))
        generated_at = _coerce_datetime(document.get("generated_at"))
        items.append(
            {
                "id": document.get("exercise_id", document.get("generation_trace_id", "")),
                "generation_trace_id": document.get("generation_trace_id", ""),
                "title": document.get("title", "Exercice"),
                "topic": document.get("topic", "Notion"),
                "subtopic": document.get("subtopic", ""),
                "section": document.get("section", ""),
                "level": document.get("level", ""),
                "exercise_type": document.get("exercise_type", ""),
                "difficulty": document.get("difficulty", "Intermédiaire"),
                "status": status,
                "timestamp": _format_timestamp(generated_at),
            }
        )
    return items


def load_exercise_from_history(generation_trace_id: str) -> dict[str, Any] | None:
    """Restore one full exercise payload from Mongo for the active user."""
    user = get_session_user_context()
    trace_id = str(generation_trace_id).strip()
    if not user.get("email") or not trace_id:
        return None

    try:
        document = get_exercise_records_collection().find_one(
            {"user_email": user["email"], "generation_trace_id": trace_id}
        )
    except PyMongoError:
        return None

    if not document:
        return None
    if not _document_is_student_visible(document):
        return None

    topic = str(document.get("topic", "")).strip()
    subtopic = str(document.get("subtopic", "")).strip()
    difficulty = str(document.get("difficulty", "")).strip() or "Intermédiaire"
    exercise_type = str(document.get("exercise_type", "")).strip() or "Exercice problème"
    display_answer = str(document.get("display_answer", "")).strip()
    accepted_answers = list(document.get("accepted_answers", []) or [])
    if not accepted_answers and display_answer:
        accepted_answers = [display_answer]

    tags = list(document.get("tags", []) or [])
    if not tags:
        tags = [value for value in [topic, subtopic, difficulty, exercise_type] if value]

    return {
        "id": document.get("exercise_id", trace_id),
        "generation_trace_id": trace_id,
        "title": document.get("title", "Exercice"),
        "context": document.get("context", ""),
        "questions": list(document.get("questions", []) or []),
        "instruction": document.get("instruction", document.get("prompt", "")),
        "prompt": document.get("prompt", ""),
        "topic": topic,
        "subtopic": subtopic,
        "section": document.get("section", user.get("section", "")),
        "level": document.get("level", user.get("level", "Bac")),
        "difficulty": difficulty,
        "exercise_type": exercise_type,
        "hint": document.get("hint", ""),
        "learning_objective": document.get("learning_objective", ""),
        "display_answer": display_answer,
        "accepted_answers": accepted_answers,
        "answer_kind": document.get("answer_kind", "expression"),
        "hidden_solution": document.get("hidden_solution", ""),
        "solution_steps": list(document.get("solution_steps", []) or []),
        "support_summary": document.get("support_summary", ""),
        "table_data": document.get("table_data"),
        "chart_data": document.get("chart_data"),
        "graph_data": document.get("graph_data"),
        "assignment_id": document.get("assignment_id", ""),
        "assignment_status": document.get("assignment_status", ""),
        "assignment_due_date": document.get("assignment_due_date", ""),
        "assigned_group_id": document.get("assigned_group_id", ""),
        "assigned_group_name": document.get("assigned_group_name", ""),
        "assigned_by_teacher_email": document.get("assigned_by_teacher_email", ""),
        "assigned_by_teacher_name": document.get("assigned_by_teacher_name", ""),
        "solution_validation_status": document.get("solution_validation_status", ""),
        "solution_validation_summary": document.get("solution_validation_summary", ""),
        "solution_validation_issues": list(document.get("solution_validation_issues", []) or []),
        "solution_validation_confidence": document.get("solution_validation_confidence", 0.0),
        "solution_validation_model": document.get("solution_validation_model", ""),
        "solution_validation_flag": document.get("solution_validation_flag", ""),
        "solution_validation_sympy_report": document.get("solution_validation_sympy_report", ""),
        "local_validation_flag": document.get("local_validation_flag", ""),
        "local_validation_summary": document.get("local_validation_summary", ""),
        "local_validation_issues": list(document.get("local_validation_issues", []) or []),
        "pedagogical_completeness_flag": document.get("pedagogical_completeness_flag", ""),
        "pedagogical_completeness_summary": document.get("pedagogical_completeness_summary", ""),
        "pedagogical_completeness_issues": list(document.get("pedagogical_completeness_issues", []) or []),
        "symbolic_checks_ran": bool(document.get("symbolic_checks_ran", False)),
        "symbolic_checks_passed": document.get("symbolic_checks_passed"),
        "symbolic_checks_required": bool(document.get("symbolic_checks_required", False)),
        "corrected_fields_applied": bool(document.get("corrected_fields_applied", False)),
        "student_facing_format_flag": document.get("student_facing_format_flag", ""),
        "student_facing_format_issues": list(document.get("student_facing_format_issues", []) or []),
        "final_display_decision": document.get("final_display_decision", ""),
        "final_display_blocking_reasons": list(document.get("final_display_blocking_reasons", []) or []),
        "judge_status": document.get("judge_status", ""),
        "judge_summary": document.get("judge_summary", ""),
        "judge_model": document.get("judge_model", ""),
        "judge_validation_flag": document.get("judge_validation_flag", ""),
        "judge_alignment_reason": document.get("judge_alignment_reason", ""),
        "judge_issues": list(document.get("judge_issues", []) or []),
        "judge_rejected_attempts": list(document.get("judge_rejected_attempts", []) or []),
        "generation_backend": document.get("generation_backend", ""),
        "is_true_llm_generation": bool(document.get("is_true_llm_generation", False)),
        "llm_json_parse_status": document.get("llm_json_parse_status", ""),
        "llm_generation_attempts_count": int(document.get("llm_generation_attempts_count", 0) or 0),
        "fallback_used": bool(document.get("fallback_used", False)),
        "fallback_reason": document.get("fallback_reason", ""),
        "display_source_category": document.get("display_source_category", ""),
        "retry_strategy": document.get("retry_strategy", ""),
        "failure_categories": list(document.get("failure_categories", []) or []),
        "previous_errors_injected": list(document.get("previous_errors_injected", []) or []),
        "demo_mode_used": bool(document.get("demo_mode_used", False)),
        "generated_at": document.get("generated_at"),
        "memory_adaptation_note": document.get("memory_adaptation_note", ""),
        "source_case_summaries": list(document.get("source_case_summaries", []) or []),
        "source_case_instructions": list(document.get("source_case_instructions", []) or []),
        "generation_metadata": dict(document.get("generation_metadata", {}) or {}),
        "verification_ready": bool(document.get("verification_ready", True)),
        "verification_message": document.get("verification_message", ""),
        "judge_blocked": bool(document.get("judge_blocked", False)),
        "options": list(document.get("options", []) or []),
        "tags": tags,
    }


def _document_is_student_visible(document: dict[str, Any]) -> bool:
    """Filter out blocked generations from student-facing history and restoration."""
    final_decision = str(document.get("final_display_decision", "")).strip().lower()
    if final_decision:
        return final_decision == "presented"

    judge_flag = str(document.get("judge_validation_flag", "")).strip().lower()
    solution_flag = str(document.get("solution_validation_flag", "")).strip().lower()
    local_flag = str(document.get("local_validation_flag", "")).strip().lower()
    pedagogical_flag = str(document.get("pedagogical_completeness_flag", "")).strip().lower()
    judge_status = str(document.get("judge_status", "")).strip().lower()
    blocked_flags = {
        "judge_error",
        "wrong",
        "misaligned",
        "unknown",
        "blocked_after_retries",
        "blocked_missing_alignment",
    }
    if bool(document.get("judge_blocked", False)):
        return False
    if judge_status == "juge indisponible":
        return False
    if judge_flag in blocked_flags:
        return False
    if solution_flag and solution_flag != "approved":
        return False
    if local_flag and local_flag != "approved":
        return False
    if pedagogical_flag == "incomplete":
        return False
    if str(document.get("student_facing_format_flag", "")).strip().lower() == "wrong":
        return False
    return True


def record_auth_success(source: str) -> None:
    """Persist a successful account creation or login for the active user."""
    user = get_session_user_context()
    if not user.get("email"):
        return

    event_type = "account_created" if source == "register" else "login"
    now = _now_utc()
    _update_user_summary(
        user,
        {
            "last_login_at": now,
            "last_active_at": now,
            "learning_profile_synced_at": now,
            "level": user.get("level", ""),
            "section": user.get("section", ""),
            "role": user.get("role", ""),
            "name": user.get("display_name", ""),
        },
        increment_minutes=1.0 if event_type == "account_created" else 0.25,
    )
    _insert_event(
        user,
        event_type=event_type,
        duration_minutes=1.0 if event_type == "account_created" else 0.25,
        metadata={"source": source},
        created_at=now,
    )


def record_page_consultation(
    page_key: str,
    page_title: str,
    *,
    topic: str = "",
    subtopic: str = "",
    metadata: dict[str, Any] | None = None,
    force: bool = False,
) -> None:
    """Persist one page consultation while throttling Streamlit reruns."""
    user = get_session_user_context()
    if not user.get("email"):
        return

    markers = st.session_state.setdefault("_analytics_page_markers", {})
    now = datetime.now().timestamp()
    last_seen = float(markers.get(page_key, 0.0))
    if not force and now - last_seen < PAGE_VIEW_THROTTLE_SECONDS:
        return
    markers[page_key] = now

    event_time = _now_utc()
    _update_user_summary(
        user,
        {
            "last_active_at": event_time,
            "last_consulted_page": page_title,
            "last_studied_topic": topic or "",
            "last_studied_subtopic": subtopic or "",
        },
        increment_minutes=0.6,
    )
    _insert_event(
        user,
        event_type="page_view",
        duration_minutes=0.6,
        topic=topic,
        subtopic=subtopic,
        page_key=page_key,
        page_title=page_title,
        metadata=metadata or {},
        created_at=event_time,
    )


def record_exercise_generated(exercise: dict[str, Any]) -> None:
    """Persist one generated exercise and initialize its tracked lifecycle."""
    user = get_session_user_context()
    if not user.get("email"):
        return

    generated_at = _coerce_datetime(exercise.get("generated_at")) or _now_utc()
    record_filter = _exercise_record_filter(user, exercise)
    base_record = _exercise_record_payload(user, exercise, generated_at)

    try:
        get_exercise_records_collection().update_one(
            record_filter,
            {
                "$set": base_record,
                "$setOnInsert": {
                    "answer_submissions": [],
                    "hint_events": [],
                    "tutor_turns": 0,
                    "progressive_hint_reveals": 0,
                    "adaptive_hint_requests": 0,
                    "latest_status": "generated",
                    "is_solved": False,
                    "consultation_count": 0,
                    "estimated_study_minutes": 0.0,
                    "created_at": generated_at,
                },
            },
            upsert=True,
        )
    except PyMongoError:
        return

    _update_user_summary(
        user,
        {
            "last_active_at": generated_at,
            "last_studied_topic": exercise.get("topic", ""),
            "last_studied_subtopic": exercise.get("subtopic", ""),
        },
        increment_minutes=2.5,
        extra_increments={
            "learning_counters.generated_exercises": 1,
        },
    )
    _insert_event(
        user,
        event_type="exercise_generated",
        duration_minutes=2.5,
        topic=exercise.get("topic", ""),
        subtopic=exercise.get("subtopic", ""),
        exercise=exercise,
        page_key="exercise_generator",
        page_title="Générateur d'exercices",
        metadata={
            "exercise_type": exercise.get("exercise_type", ""),
            "judge_validation_flag": exercise.get("judge_validation_flag", ""),
            "judge_status": exercise.get("judge_status", ""),
            "support_summary": exercise.get("support_summary", ""),
            "judge_rejected_attempts_count": len(exercise.get("judge_rejected_attempts", []) or []),
        },
        created_at=generated_at,
    )


def record_assigned_exercise_opened(exercise: dict[str, Any], *, source: str = "notification") -> None:
    """Persist that one assigned exercise has been opened by the student."""
    user = get_session_user_context()
    if not user.get("email"):
        return

    opened_at = _now_utc()
    try:
        get_exercise_records_collection().update_one(
            _exercise_record_filter(user, exercise),
            {
                "$set": {
                    "last_activity_at": opened_at,
                    "latest_status": "opened",
                },
                "$inc": {
                    "consultation_count": 1,
                    "estimated_study_minutes": 0.7,
                },
            },
        )
    except PyMongoError:
        return

    _update_user_summary(
        user,
        {
            "last_active_at": opened_at,
            "last_studied_topic": exercise.get("topic", ""),
            "last_studied_subtopic": exercise.get("subtopic", ""),
        },
        increment_minutes=0.7,
        extra_increments={"learning_counters.assigned_exercises_opened": 1},
    )
    _insert_event(
        user,
        event_type="assigned_exercise_opened",
        duration_minutes=0.7,
        topic=exercise.get("topic", ""),
        subtopic=exercise.get("subtopic", ""),
        exercise=exercise,
        page_key="exercise_generator",
        page_title="Générateur d'exercices",
        metadata={"source": source, "assignment_id": exercise.get("assignment_id", "")},
        created_at=opened_at,
    )


def record_exercise_verification(exercise: dict[str, Any], submitted_answer: str, result: dict[str, Any]) -> None:
    """Persist one answer submission and its verification result."""
    user = get_session_user_context()
    if not user.get("email"):
        return

    verified_at = _now_utc()
    is_correct = bool(result.get("is_correct"))
    verification_entry = {
        "submitted_answer": _trim_text(submitted_answer, 3000),
        "expected_answer": _trim_text(result.get("expected_answer", ""), 1200),
        "feedback": _trim_text(result.get("feedback", ""), 2000),
        "is_correct": is_correct,
        "verified_at": verified_at,
    }

    try:
        get_exercise_records_collection().update_one(
            _exercise_record_filter(user, exercise),
            {
                "$push": {"answer_submissions": verification_entry},
                "$set": {
                    "last_activity_at": verified_at,
                    "last_verified_at": verified_at,
                    "last_feedback": verification_entry["feedback"],
                    "last_submitted_answer": verification_entry["submitted_answer"],
                    "latest_status": "correct" if is_correct else "incorrect",
                    "is_solved": is_correct,
                },
                "$inc": {
                    "total_verifications": 1,
                    "correct_verifications": 1 if is_correct else 0,
                    "incorrect_verifications": 0 if is_correct else 1,
                    "estimated_study_minutes": 4.0,
                },
            },
        )
    except PyMongoError:
        return

    _update_user_summary(
        user,
        {
            "last_active_at": verified_at,
            "last_studied_topic": exercise.get("topic", ""),
            "last_studied_subtopic": exercise.get("subtopic", ""),
        },
        increment_minutes=4.0,
        extra_increments={
            "learning_counters.answer_verifications": 1,
            "learning_counters.correct_answers": 1 if is_correct else 0,
            "learning_counters.incorrect_answers": 0 if is_correct else 1,
        },
    )
    _insert_event(
        user,
        event_type="answer_verified",
        duration_minutes=4.0,
        topic=exercise.get("topic", ""),
        subtopic=exercise.get("subtopic", ""),
        exercise=exercise,
        page_key="exercise_generator",
        page_title="Générateur d'exercices",
        metadata=verification_entry,
        created_at=verified_at,
    )


def record_hint_interaction(
    exercise: dict[str, Any],
    *,
    hint_kind: str,
    hint_index: int | None = None,
    hint_text: str = "",
) -> None:
    """Persist the usage of progressive or adaptive hints."""
    user = get_session_user_context()
    if not user.get("email"):
        return

    event_time = _now_utc()
    is_adaptive = hint_kind == "adaptive"
    hint_entry = {
        "kind": hint_kind,
        "index": hint_index,
        "text": _trim_text(hint_text, 1500),
        "created_at": event_time,
    }

    try:
        get_exercise_records_collection().update_one(
            _exercise_record_filter(user, exercise),
            {
                "$push": {"hint_events": hint_entry},
                "$set": {"last_activity_at": event_time},
                "$inc": {
                    "adaptive_hint_requests": 1 if is_adaptive else 0,
                    "progressive_hint_reveals": 0 if is_adaptive else 1,
                    "estimated_study_minutes": 1.2 if is_adaptive else 0.8,
                },
            },
        )
    except PyMongoError:
        return

    _update_user_summary(
        user,
        {
            "last_active_at": event_time,
            "last_studied_topic": exercise.get("topic", ""),
            "last_studied_subtopic": exercise.get("subtopic", ""),
        },
        increment_minutes=1.2 if is_adaptive else 0.8,
        extra_increments={
            "learning_counters.hints_used": 1,
            "learning_counters.adaptive_hints": 1 if is_adaptive else 0,
            "learning_counters.progressive_hints": 0 if is_adaptive else 1,
        },
    )
    _insert_event(
        user,
        event_type="adaptive_hint_requested" if is_adaptive else "hint_revealed",
        duration_minutes=1.2 if is_adaptive else 0.8,
        topic=exercise.get("topic", ""),
        subtopic=exercise.get("subtopic", ""),
        exercise=exercise,
        page_key="exercise_generator",
        page_title="Générateur d'exercices",
        metadata=hint_entry,
        created_at=event_time,
    )


def record_tutor_turn(mode: str, student_message: str, assistant_reply: str, exercise_context: dict[str, Any] | None = None) -> None:
    """Persist one conversational tutoring turn."""
    user = get_session_user_context()
    if not user.get("email"):
        return

    event_time = _now_utc()
    topic = exercise_context.get("topic", "") if exercise_context else ""
    subtopic = exercise_context.get("subtopic", "") if exercise_context else ""
    duration_minutes = _estimate_tutor_minutes(student_message, assistant_reply)

    if exercise_context:
        try:
            get_exercise_records_collection().update_one(
                _exercise_record_filter(user, exercise_context),
                {
                    "$set": {"last_activity_at": event_time},
                    "$inc": {
                        "tutor_turns": 1,
                        "estimated_study_minutes": duration_minutes,
                    },
                },
            )
        except PyMongoError:
            pass

    _update_user_summary(
        user,
        {
            "last_active_at": event_time,
            "last_studied_topic": topic,
            "last_studied_subtopic": subtopic,
        },
        increment_minutes=duration_minutes,
        extra_increments={
            "learning_counters.tutor_turns": 1,
        },
    )
    _insert_event(
        user,
        event_type="tutor_turn",
        duration_minutes=duration_minutes,
        topic=topic,
        subtopic=subtopic,
        exercise=exercise_context,
        page_key="tutoring_chat",
        page_title="Tutorat conversationnel",
        metadata={
            "mode": mode,
            "student_message": _trim_text(student_message, 3000),
            "assistant_reply": _trim_text(assistant_reply, 4500),
        },
        created_at=event_time,
    )


def get_user_progress_analytics(
    *,
    student_id: str = "",
    user_email: str = "",
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate Mongo learning data into charts and metrics for the progress page."""
    effective_profile = profile or st.session_state.get("student_profile", DEFAULT_STUDENT_PROFILE)
    effective_email = user_email.strip().lower() if user_email else get_session_user_context().get("email", "")
    if not effective_email:
        return _build_fallback_analytics(effective_profile)

    try:
        exercises = list(
            get_exercise_records_collection()
            .find({"user_email": effective_email})
            .sort("generated_at", DESCENDING)
            .limit(500)
        )
        events = list(
            get_learning_events_collection()
            .find({"user_email": effective_email})
            .sort("created_at", ASCENDING)
            .limit(4000)
        )
    except PyMongoError:
        return _build_fallback_analytics(effective_profile)

    return _aggregate_progress_data(
        student_id=student_id,
        user_email=effective_email,
        profile=effective_profile,
        exercises=exercises,
        events=events,
    )


def get_user_dashboard_payload(
    *,
    student_id: str = "",
    user_email: str = "",
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the student-dashboard payload from persisted Mongo learning data."""
    effective_profile = deepcopy(profile or st.session_state.get("student_profile", DEFAULT_STUDENT_PROFILE))
    analytics = get_user_progress_analytics(
        student_id=student_id,
        user_email=user_email,
        profile=effective_profile,
    )

    mastery_snapshot = analytics.get("mastery_snapshot", {})
    mastery_evolution = analytics.get("mastery_evolution", {})
    mastery_series = mastery_evolution.get("series", {})
    recent_exercises = get_recent_exercise_history(limit=4, user_email=user_email)
    metrics = analytics.get("metrics", {})

    dashboard_profile = deepcopy(effective_profile)
    dashboard_profile["mastery_score"] = metrics.get("mastery_average", dashboard_profile.get("mastery_score", 0))
    dashboard_profile["weekly_goal_progress"] = min(
        float(dashboard_profile.get("weekly_goal_hours", 6)),
        round(_recent_study_hours(user_email), 1),
    )
    dashboard_profile["streak_days"] = _study_streak_days(user_email)
    dashboard_profile["memory_health"] = _estimate_memory_health(analytics)
    dashboard_profile["current_focus"] = _resolve_current_focus(analytics, dashboard_profile)
    dashboard_profile["strong_topics"] = list(_top_topics_by_score(mastery_snapshot, reverse=True))
    dashboard_profile["weak_topics"] = list(_top_topics_by_score(mastery_snapshot, reverse=False))

    mastery_progress = [
        {"topic": topic, "value": score}
        for topic, score in sorted(mastery_snapshot.items(), key=lambda item: item[1], reverse=True)
    ]

    weak_topics = _build_dashboard_topic_rows(
        topics=sorted(mastery_snapshot.items(), key=lambda item: item[1])[:3],
        mastery_series=mastery_series,
    )
    strong_topics = _build_dashboard_topic_rows(
        topics=sorted(mastery_snapshot.items(), key=lambda item: item[1], reverse=True)[:3],
        mastery_series=mastery_series,
    )

    if not recent_exercises:
        recent_exercises = [
            {
                "title": "Aucun exercice enregistre pour le moment",
                "topic": dashboard_profile["current_focus"],
                "difficulty": "Intermédiaire",
                "status": "Commencer une premiere generation",
                "timestamp": "A l'instant",
            }
        ]

    return {
        "profile": dashboard_profile,
        "mastery_progress": mastery_progress or [{"topic": dashboard_profile["current_focus"], "value": 50}],
        "weak_topics": weak_topics or _build_fallback_topic_rows(dashboard_profile.get("weak_topics", []), 45),
        "strong_topics": strong_topics or _build_fallback_topic_rows(dashboard_profile.get("strong_topics", []), 70),
        "recent_exercises": recent_exercises,
        "recommendations": _build_dashboard_recommendations(
            analytics=analytics,
            profile=dashboard_profile,
            weak_topics=weak_topics,
            strong_topics=strong_topics,
        ),
        "analytics": analytics,
    }


def _aggregate_progress_data(
    *,
    student_id: str,
    user_email: str,
    profile: dict[str, Any],
    exercises: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert raw exercise and event documents into progress analytics."""
    _ = student_id, user_email
    topic_stats: dict[str, dict[str, Any]] = defaultdict(_new_topic_stats)
    distinct_topics: set[str] = set()
    error_signals: Counter[str] = Counter()
    total_study_minutes = 0.0
    consultation_count = 0
    tutor_turns = 0
    hint_events = 0
    correct_answers = 0
    incorrect_answers = 0

    for exercise in exercises:
        topic = str(exercise.get("topic", "")).strip() or str(exercise.get("subtopic", "")).strip() or "Notion"
        stats = topic_stats[topic]
        stats["generated"] += 1
        stats["solved"] += 1 if exercise.get("is_solved") else 0
        stats["verified"] += int(exercise.get("total_verifications", 0))
        stats["correct"] += int(exercise.get("correct_verifications", 0))
        stats["incorrect"] += int(exercise.get("incorrect_verifications", 0))
        stats["hints"] += int(exercise.get("progressive_hint_reveals", 0)) + int(exercise.get("adaptive_hint_requests", 0))
        stats["tutor_turns"] += int(exercise.get("tutor_turns", 0))
        stats["subtopics"].add(str(exercise.get("subtopic", "")).strip())
        stats["section"] = str(exercise.get("section", "")).strip() or stats["section"]
        stats["last_activity"] = max(stats["last_activity"], _coerce_datetime(exercise.get("last_activity_at")) or _coerce_datetime(exercise.get("generated_at")) or _now_utc())
        distinct_topics.add(topic)

        if not exercise.get("is_solved") and exercise.get("incorrect_verifications", 0):
            error_signals[_trim_text(exercise.get("last_feedback", ""), 180)] += 1

    dated_events: list[dict[str, Any]] = []
    for event in events:
        created_at = _coerce_datetime(event.get("created_at")) or _now_utc()
        dated_events.append({**event, "created_at": created_at})
        total_study_minutes += float(event.get("duration_minutes", 0.0) or 0.0)

        event_type = str(event.get("event_type", "")).strip()
        topic = str(event.get("topic", "")).strip() or str(event.get("subtopic", "")).strip() or ""
        if topic:
            distinct_topics.add(topic)

        if event_type == "page_view":
            consultation_count += 1
        elif event_type == "tutor_turn":
            tutor_turns += 1
        elif event_type in {"hint_revealed", "adaptive_hint_requested"}:
            hint_events += 1
        elif event_type == "answer_verified":
            if bool(event.get("metadata", {}).get("is_correct")):
                correct_answers += 1
            else:
                incorrect_answers += 1
                feedback = _trim_text(event.get("metadata", {}).get("feedback", ""), 180)
                if feedback:
                    error_signals[feedback] += 1

    labels, bucket_dates = _build_recent_day_buckets(6)
    topic_ranking = sorted(
        topic_stats.items(),
        key=lambda item: (
            item[1]["generated"] + item[1]["verified"] + item[1]["tutor_turns"],
            item[1]["last_activity"],
        ),
        reverse=True,
    )
    tracked_topics = [topic for topic, _ in topic_ranking[:4]]
    if not tracked_topics:
        tracked_topics = [topic for topic in profile.get("weak_topics", [])[:2]]
    if not tracked_topics:
        tracked_topics = [topic for topic in profile.get("strong_topics", [])[:2]]
    tracked_topics = [topic for topic in tracked_topics if topic]

    weighted_topic_events: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for event in dated_events:
        topic = str(event.get("topic", "")).strip() or str(event.get("subtopic", "")).strip()
        if not topic:
            continue
        weight = _event_weight(event)
        if weight != 0:
            weighted_topic_events[topic].append((event["created_at"], weight))

    mastery_snapshot = {
        topic: _compute_topic_mastery(weighted_topic_events.get(topic, []), up_to=None)
        for topic in (tracked_topics or list(topic_stats.keys()))
    }
    mastery_snapshot = {topic: score for topic, score in mastery_snapshot.items() if topic}
    if not mastery_snapshot:
        mastery_snapshot = {
            topic: 50
            for topic in profile.get("weak_topics", [])[:2] + profile.get("strong_topics", [])[:2]
            if topic
        }

    mastery_series = {}
    for topic in mastery_snapshot:
        topic_events = weighted_topic_events.get(topic, [])
        mastery_series[topic] = [
            _compute_topic_mastery(topic_events, up_to=_end_of_day(bucket_day))
            for bucket_day in bucket_dates
        ]

    weak_topics = [
        {"topic": topic, "mastery": score}
        for topic, score in sorted(mastery_snapshot.items(), key=lambda item: item[1])[:4]
    ]
    solved_exercises = [
        {"topic": topic, "count": stats["solved"] or stats["generated"]}
        for topic, stats in sorted(topic_stats.items(), key=lambda item: item[1]["generated"], reverse=True)[:6]
    ]
    if not solved_exercises:
        solved_exercises = [{"topic": topic, "count": 0} for topic in mastery_snapshot.keys()]

    total_checked = correct_answers + incorrect_answers
    success_rate_pct = round((correct_answers / total_checked) * 100) if total_checked else 0
    mastery_average = round(sum(mastery_snapshot.values()) / len(mastery_snapshot)) if mastery_snapshot else 0
    study_hours = round(total_study_minutes / 60, 1)

    recent_window_start = _now_utc() - timedelta(days=7)
    previous_window_start = _now_utc() - timedelta(days=14)
    recent_checked = sum(
        1
        for event in dated_events
        if event.get("event_type") == "answer_verified"
        and event["created_at"] >= recent_window_start
    )
    previous_checked = sum(
        1
        for event in dated_events
        if event.get("event_type") == "answer_verified"
        and previous_window_start <= event["created_at"] < recent_window_start
    )
    recent_correct = sum(
        1
        for event in dated_events
        if event.get("event_type") == "answer_verified"
        and bool(event.get("metadata", {}).get("is_correct"))
        and event["created_at"] >= recent_window_start
    )
    previous_correct = sum(
        1
        for event in dated_events
        if event.get("event_type") == "answer_verified"
        and bool(event.get("metadata", {}).get("is_correct"))
        and previous_window_start <= event["created_at"] < recent_window_start
    )
    recent_minutes = sum(
        float(event.get("duration_minutes", 0.0) or 0.0)
        for event in dated_events
        if event["created_at"] >= recent_window_start
    )
    previous_minutes = sum(
        float(event.get("duration_minutes", 0.0) or 0.0)
        for event in dated_events
        if previous_window_start <= event["created_at"] < recent_window_start
    )

    previous_snapshot = {
        topic: _compute_topic_mastery(weighted_topic_events.get(topic, []), up_to=_end_of_day((date.today() - timedelta(days=7))))
        for topic in mastery_snapshot
    }
    previous_mastery_average = (
        round(sum(previous_snapshot.values()) / len(previous_snapshot)) if previous_snapshot else mastery_average
    )
    recent_success_rate_pct = round((recent_correct / recent_checked) * 100) if recent_checked else success_rate_pct
    previous_success_rate_pct = (
        round((previous_correct / previous_checked) * 100) if previous_checked else success_rate_pct
    )

    intervention_notes = _build_intervention_notes(
        profile=profile,
        weak_topics=weak_topics,
        error_signals=error_signals,
        study_hours=study_hours,
        consultation_count=consultation_count,
        tutor_turns=tutor_turns,
        hint_events=hint_events,
    )

    return {
        "data_source": "mongo",
        "metrics": {
            "mastery_average": mastery_average,
            "mastery_delta": mastery_average - previous_mastery_average,
            "solved_exercises": sum(stats["solved"] for stats in topic_stats.values()),
            "solved_delta": recent_correct - previous_correct,
            "success_rate_pct": success_rate_pct,
            "success_rate_delta": recent_success_rate_pct - previous_success_rate_pct,
            "study_hours": study_hours,
            "study_hours_delta": round((recent_minutes - previous_minutes) / 60, 1),
            "at_risk_topics": len([item for item in weak_topics if item["mastery"] < 55]),
            "topics_studied": len(distinct_topics),
        },
        "mastery_evolution": {
            "labels": labels,
            "series": mastery_series,
        },
        "solved_exercises": solved_exercises,
        "weak_topics": weak_topics,
        "success_rate": {
            "correct": correct_answers,
            "incorrect": incorrect_answers,
            "hinted": hint_events,
        },
        "mastery_snapshot": mastery_snapshot,
        "intervention_notes": intervention_notes,
        "recent_topics": sorted(distinct_topics)[:10],
    }


def _build_fallback_analytics(profile: dict[str, Any]) -> dict[str, Any]:
    """Return an empty-but-usable analytics payload when Mongo data is absent."""
    labels, _ = _build_recent_day_buckets(6)
    weak_topics = [{"topic": topic, "mastery": 45 + index * 5} for index, topic in enumerate(profile.get("weak_topics", [])[:4])]
    snapshot_topics = profile.get("strong_topics", [])[:2] + profile.get("weak_topics", [])[:2]
    mastery_snapshot = {topic: 50 for topic in snapshot_topics if topic}
    series = {topic: [50 for _ in labels] for topic in mastery_snapshot}

    return {
        "data_source": "bootstrap",
        "metrics": {
            "mastery_average": 0,
            "mastery_delta": 0,
            "solved_exercises": 0,
            "solved_delta": 0,
            "success_rate_pct": 0,
            "success_rate_delta": 0,
            "study_hours": 0.0,
            "study_hours_delta": 0.0,
            "at_risk_topics": len(weak_topics),
            "topics_studied": 0,
        },
        "mastery_evolution": {"labels": labels, "series": series},
        "solved_exercises": [{"topic": topic, "count": 0} for topic in mastery_snapshot],
        "weak_topics": weak_topics,
        "success_rate": {"correct": 0, "incorrect": 0, "hinted": 0},
        "mastery_snapshot": mastery_snapshot,
        "intervention_notes": [
            {
                "title": "Aucune activite enregistree",
                "body": "Les premiers exercices generes, reponses verifiees et echanges de tutorat rempliront automatiquement ce tableau.",
                "footer": "Commencez par generer un exercice",
                "accent": "amber",
            }
        ],
        "recent_topics": [],
    }


def _new_topic_stats() -> dict[str, Any]:
    """Create one mutable topic accumulator."""
    return {
        "generated": 0,
        "solved": 0,
        "verified": 0,
        "correct": 0,
        "incorrect": 0,
        "hints": 0,
        "tutor_turns": 0,
        "subtopics": set(),
        "section": "",
        "last_activity": datetime(1970, 1, 1, tzinfo=UTC),
    }


def _build_intervention_notes(
    *,
    profile: dict[str, Any],
    weak_topics: list[dict[str, Any]],
    error_signals: Counter[str],
    study_hours: float,
    consultation_count: int,
    tutor_turns: int,
    hint_events: int,
) -> list[dict[str, str]]:
    """Create human-readable intervention notes from persisted analytics."""
    notes: list[dict[str, str]] = []
    if weak_topics:
        weakest = weak_topics[0]
        notes.append(
            {
                "title": f"Priorite sur {weakest['topic']}",
                "body": f"Cette notion est actuellement la plus fragile avec une maitrise estimee a {weakest['mastery']}%.",
                "footer": "Action suggeree : nouvel exercice puis tutorat",
                "accent": "amber",
            }
        )

    if error_signals:
        common_error, _ = error_signals.most_common(1)[0]
        notes.append(
            {
                "title": "Erreur recurrente observee",
                "body": common_error,
                "footer": "Source : verifications et retours du tuteur",
                "accent": "teal",
            }
        )

    notes.append(
        {
            "title": "Temps d'etude cumule",
            "body": f"{study_hours:.1f} h estimee(s) a partir des generations, verifications, indices et conversations enregistrees.",
            "footer": f"{consultation_count} consultation(s) de page et {tutor_turns} tour(s) de tutorat",
            "accent": "teal",
        }
    )

    notes.append(
        {
            "title": "Autonomie de resolution",
            "body": (
                f"{hint_events} indice(s) ont ete utilises jusque-la. "
                "Une baisse progressive de ce nombre signale souvent une meilleure autonomie."
            ),
            "footer": f"Focus actuel : {profile.get('current_focus', 'Pratique personnalisee')}",
            "accent": "amber",
        }
    )
    return notes[:4]


def _build_dashboard_topic_rows(
    *,
    topics: list[tuple[str, int]],
    mastery_series: dict[str, list[int]],
) -> list[dict[str, Any]]:
    """Convert mastery topics into rows consumable by the dashboard cards."""
    rows: list[dict[str, Any]] = []
    for topic, mastery in topics:
        series = mastery_series.get(topic, [])
        previous = series[-2] if len(series) >= 2 else mastery
        delta = mastery - previous
        sign = "+" if delta > 0 else ""
        rows.append(
            {
                "topic": topic,
                "mastery": mastery,
                "trend": f"{sign}{delta} pts",
            }
        )
    return rows


def _build_fallback_topic_rows(topics: list[str], base_mastery: int) -> list[dict[str, Any]]:
    """Create lightweight rows when detailed analytics are not available yet."""
    return [
        {
            "topic": topic,
            "mastery": max(10, min(95, base_mastery + index * 5)),
            "trend": "0 pt",
        }
        for index, topic in enumerate(topics[:3])
        if topic
    ]


def _top_topics_by_score(snapshot: dict[str, int], *, reverse: bool) -> list[str]:
    """Return the top or bottom topics as a plain ordered list."""
    ordered = sorted(snapshot.items(), key=lambda item: item[1], reverse=reverse)
    return [topic for topic, _ in ordered[:3] if topic]


def _resolve_current_focus(analytics: dict[str, Any], profile: dict[str, Any]) -> str:
    """Select the most urgent focus topic for the dashboard badge."""
    weak_topics = analytics.get("weak_topics", [])
    if weak_topics:
        return weak_topics[0]["topic"]
    snapshot = analytics.get("mastery_snapshot", {})
    if snapshot:
        return min(snapshot.items(), key=lambda item: item[1])[0]
    return profile.get("current_focus", DEFAULT_STUDENT_PROFILE["current_focus"])


def _estimate_memory_health(analytics: dict[str, Any]) -> int:
    """Derive one memory-health score from correctness, hints and activity."""
    metrics = analytics.get("metrics", {})
    success_rate = int(metrics.get("success_rate_pct", 0))
    hint_count = int(analytics.get("success_rate", {}).get("hinted", 0))
    solved = int(metrics.get("solved_exercises", 0))
    base = 48 + round(success_rate * 0.4) + min(18, solved * 2)
    penalty = min(18, hint_count * 2)
    return max(25, min(98, base - penalty))


def _build_dashboard_recommendations(
    *,
    analytics: dict[str, Any],
    profile: dict[str, Any],
    weak_topics: list[dict[str, Any]],
    strong_topics: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Generate dashboard recommendations from persisted user behavior."""
    recommendations: list[dict[str, str]] = []
    if weak_topics:
        weakest = weak_topics[0]
        recommendations.append(
            {
                "title": f"Renforcer {weakest['topic']}",
                "description": (
                    f"La maitrise estimee sur {weakest['topic']} reste autour de {weakest['mastery']}%. "
                    "Un nouvel exercice suivi d'un court tutorat socratique devrait consolider cette notion."
                ),
                "action": "Generer un exercice cible puis ouvrir le tutorat",
            }
        )

    success_rate = analytics.get("success_rate", {})
    if int(success_rate.get("hinted", 0)) > int(success_rate.get("correct", 0)):
        recommendations.append(
            {
                "title": "Reduire la dependance aux indices",
                "description": (
                    "Les indices sont encore tres utilises. Essayez de formuler d'abord votre prochain pas avant d'ouvrir l'aide progressive."
                ),
                "action": "Tenter une reponse partielle avant l'indice",
            }
        )

    if strong_topics:
        strongest = strong_topics[0]
        recommendations.append(
            {
                "title": f"Capitaliser sur {strongest['topic']}",
                "description": (
                    f"Votre meilleure dynamique actuelle se situe sur {strongest['topic']}. "
                    "Utilisez cette confiance pour enchainer avec un exercice probleme plus long."
                ),
                "action": "Passer a un exercice probleme plus ambitieux",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "title": "Lancer une premiere session",
                "description": "Commencez par generer un exercice, puis verifiez votre reponse pour remplir automatiquement votre tableau de bord.",
                "action": "Ouvrir le generateur d'exercices",
            }
        )
    return recommendations[:3]


def _recent_study_hours(user_email: str) -> float:
    """Compute the study hours accumulated over the last 7 days."""
    if not user_email:
        return 0.0
    recent_window_start = _now_utc() - timedelta(days=7)
    try:
        events = get_learning_events_collection().find(
            {"user_email": user_email, "created_at": {"$gte": recent_window_start}}
        )
    except PyMongoError:
        return 0.0
    total_minutes = sum(float(event.get("duration_minutes", 0.0) or 0.0) for event in events)
    return round(total_minutes / 60, 1)


def _study_streak_days(user_email: str) -> int:
    """Return the current consecutive-day study streak."""
    if not user_email:
        return 0
    try:
        events = list(
            get_learning_events_collection()
            .find({"user_email": user_email})
            .sort("created_at", DESCENDING)
            .limit(120)
        )
    except PyMongoError:
        return 0

    active_days = {
        (_coerce_datetime(event.get("created_at")) or _now_utc()).date()
        for event in events
        if event.get("event_type") in {"exercise_generated", "answer_verified", "tutor_turn", "hint_revealed", "adaptive_hint_requested"}
    }
    if not active_days:
        return 0

    streak = 0
    current_day = date.today()
    if current_day not in active_days and (current_day - timedelta(days=1)) in active_days:
        current_day -= timedelta(days=1)
    while current_day in active_days:
        streak += 1
        current_day -= timedelta(days=1)
    return streak


def _event_weight(event: dict[str, Any]) -> float:
    """Assign one mastery contribution weight to a persisted event."""
    event_type = str(event.get("event_type", "")).strip()
    if event_type == "exercise_generated":
        return 2.0
    if event_type == "answer_verified":
        return 16.0 if bool(event.get("metadata", {}).get("is_correct")) else -6.0
    if event_type == "hint_revealed":
        return -1.0
    if event_type == "adaptive_hint_requested":
        return -1.5
    if event_type == "tutor_turn":
        return 1.0
    return 0.0


def _compute_topic_mastery(topic_events: list[tuple[datetime, float]], *, up_to: datetime | None) -> int:
    """Compute one bounded mastery score from weighted events."""
    baseline = 45.0
    running_score = baseline
    for timestamp, weight in topic_events:
        if up_to is not None and timestamp > up_to:
            continue
        running_score += weight
    return max(10, min(98, round(running_score)))


def _build_recent_day_buckets(count: int) -> tuple[list[str], list[date]]:
    """Return the last `count` calendar days as labels and date objects."""
    days = [date.today() - timedelta(days=offset) for offset in reversed(range(count))]
    return ([item.strftime("%d %b") for item in days], days)


def _exercise_record_payload(user: dict[str, Any], exercise: dict[str, Any], generated_at: datetime) -> dict[str, Any]:
    """Build the persisted metadata for one exercise record."""
    return {
        "user_email": user.get("email", ""),
        "user_id": user.get("user_id", ""),
        "user_display_name": user.get("display_name", ""),
        "user_role": user.get("role", ""),
        "level": exercise.get("level", user.get("level", "")),
        "section": exercise.get("section", user.get("section", "")),
        "exercise_id": exercise.get("id", ""),
        "generation_trace_id": _generation_trace_id(exercise),
        "generated_at": generated_at,
        "last_activity_at": generated_at,
        "title": exercise.get("title", "Exercice"),
        "context": exercise.get("context", ""),
        "questions": list(exercise.get("questions", []) or []),
        "instruction": exercise.get("instruction", exercise.get("prompt", "")),
        "topic": exercise.get("topic", ""),
        "subtopic": exercise.get("subtopic", ""),
        "difficulty": exercise.get("difficulty", ""),
        "exercise_type": exercise.get("exercise_type", ""),
        "generation_backend": exercise.get("generation_backend", ""),
        "is_true_llm_generation": bool(exercise.get("is_true_llm_generation", False)),
        "llm_json_parse_status": exercise.get("llm_json_parse_status", ""),
        "llm_generation_attempts_count": int(exercise.get("llm_generation_attempts_count", 0) or 0),
        "fallback_used": bool(exercise.get("fallback_used", False)),
        "fallback_reason": exercise.get("fallback_reason", ""),
        "display_source_category": exercise.get("display_source_category", ""),
        "retry_strategy": exercise.get("retry_strategy", ""),
        "failure_categories": list(exercise.get("failure_categories", []) or []),
        "previous_errors_injected": list(exercise.get("previous_errors_injected", []) or []),
        "demo_mode_used": bool(exercise.get("demo_mode_used", False)),
        "generation_attempt_number": int(exercise.get("generation_attempt_number", 1) or 1),
        "judge_validation_flag": exercise.get("judge_validation_flag", ""),
        "judge_status": exercise.get("judge_status", ""),
        "judge_model": exercise.get("judge_model", ""),
        "judge_summary": exercise.get("judge_summary", ""),
        "judge_alignment_status": exercise.get("judge_alignment_status", ""),
        "judge_alignment_reason": exercise.get("judge_alignment_reason", ""),
        "judge_issues": list(exercise.get("judge_issues", []) or []),
        "judge_rejected_attempts": list(exercise.get("judge_rejected_attempts", []) or []),
        "solution_validation_status": exercise.get("solution_validation_status", ""),
        "solution_validation_summary": exercise.get("solution_validation_summary", ""),
        "solution_validation_issues": list(exercise.get("solution_validation_issues", []) or []),
        "solution_validation_confidence": exercise.get("solution_validation_confidence", 0.0),
        "solution_validation_model": exercise.get("solution_validation_model", ""),
        "solution_validation_flag": exercise.get("solution_validation_flag", ""),
        "solution_validation_sympy_report": exercise.get("solution_validation_sympy_report", ""),
        "local_validation_flag": exercise.get("local_validation_flag", ""),
        "local_validation_summary": exercise.get("local_validation_summary", ""),
        "local_validation_issues": list(exercise.get("local_validation_issues", []) or []),
        "pedagogical_completeness_flag": exercise.get("pedagogical_completeness_flag", ""),
        "pedagogical_completeness_summary": exercise.get("pedagogical_completeness_summary", ""),
        "pedagogical_completeness_issues": list(exercise.get("pedagogical_completeness_issues", []) or []),
        "symbolic_checks_ran": bool(exercise.get("symbolic_checks_ran", False)),
        "symbolic_checks_passed": exercise.get("symbolic_checks_passed"),
        "symbolic_checks_required": bool(exercise.get("symbolic_checks_required", False)),
        "corrected_fields_applied": bool(exercise.get("corrected_fields_applied", False)),
        "student_facing_format_flag": exercise.get("student_facing_format_flag", ""),
        "student_facing_format_issues": list(exercise.get("student_facing_format_issues", []) or []),
        "final_display_decision": exercise.get("final_display_decision", ""),
        "final_display_blocking_reasons": list(exercise.get("final_display_blocking_reasons", []) or []),
        "prompt": exercise.get("prompt", ""),
        "hint": exercise.get("hint", ""),
        "learning_objective": exercise.get("learning_objective", ""),
        "display_answer": exercise.get("display_answer", ""),
        "hidden_solution": exercise.get("hidden_solution", ""),
        "solution_steps": list(exercise.get("solution_steps", []) or []),
        "memory_adaptation_note": exercise.get("memory_adaptation_note", ""),
        "source_case_summaries": list(exercise.get("source_case_summaries", []) or []),
        "source_case_instructions": list(exercise.get("source_case_instructions", []) or []),
        "generation_metadata": dict(exercise.get("generation_metadata", {}) or {}),
        "support_summary": exercise.get("support_summary", ""),
        "has_table_data": bool(exercise.get("table_data")),
        "has_chart_data": bool(exercise.get("chart_data")),
        "has_graph_data": bool(exercise.get("graph_data")),
        "table_data": exercise.get("table_data"),
        "chart_data": exercise.get("chart_data"),
        "graph_data": exercise.get("graph_data"),
        "accepted_answers": list(exercise.get("accepted_answers", []) or []),
        "options": list(exercise.get("options", []) or []),
        "verification_ready": bool(exercise.get("verification_ready", True)),
        "verification_message": exercise.get("verification_message", ""),
        "judge_blocked": bool(exercise.get("judge_blocked", False)),
        "answer_kind": exercise.get("answer_kind", ""),
        "tags": list(exercise.get("tags", []) or []),
    }


def _exercise_record_filter(user: dict[str, Any], exercise: dict[str, Any]) -> dict[str, Any]:
    """Return the Mongo filter used to address one user's exercise record."""
    return {
        "user_email": user.get("email", ""),
        "generation_trace_id": _generation_trace_id(exercise),
    }


def _generation_trace_id(exercise: dict[str, Any]) -> str:
    """Build a stable per-generation key."""
    if not exercise:
        return ""
    trace_id = str(exercise.get("generation_trace_id", "")).strip()
    if trace_id:
        return trace_id
    exercise_id = str(exercise.get("id", "")).strip() or "exercise"
    generated_at = str(exercise.get("generated_at", "")).strip() or "now"
    return f"{exercise_id}::{generated_at}"


def _insert_event(
    user: dict[str, Any],
    *,
    event_type: str,
    duration_minutes: float,
    topic: str = "",
    subtopic: str = "",
    exercise: dict[str, Any] | None = None,
    page_key: str = "",
    page_title: str = "",
    metadata: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> None:
    """Insert one learning event document into Mongo."""
    event_time = created_at or _now_utc()
    event_payload = {
        "user_email": user.get("email", ""),
        "user_id": user.get("user_id", ""),
        "user_display_name": user.get("display_name", ""),
        "user_role": user.get("role", ""),
        "level": user.get("level", ""),
        "section": user.get("section", ""),
        "event_type": event_type,
        "page_key": page_key,
        "page_title": page_title,
        "topic": topic,
        "subtopic": subtopic,
        "exercise_id": exercise.get("id", "") if exercise else "",
        "generation_trace_id": _generation_trace_id(exercise or {}),
        "duration_minutes": float(duration_minutes),
        "metadata": metadata or {},
        "created_at": event_time,
    }
    try:
        get_learning_events_collection().insert_one(event_payload)
    except PyMongoError:
        return


def _update_user_summary(
    user: dict[str, Any],
    set_fields: dict[str, Any],
    *,
    increment_minutes: float = 0.0,
    extra_increments: dict[str, int | float] | None = None,
) -> None:
    """Update summary counters directly on the user document."""
    if not user.get("email"):
        return

    increments = {"learning_totals.study_minutes": float(increment_minutes)}
    for key, value in (extra_increments or {}).items():
        increments[key] = value

    try:
        get_mongo_client()[MONGO_DB_NAME][MONGO_USERS_COLLECTION].update_one(
            {"email": user["email"]},
            {
                "$set": set_fields,
                "$inc": increments,
            },
        )
    except PyMongoError:
        return


def get_session_user_context() -> dict[str, Any]:
    """Read the authenticated user context from Streamlit session state."""
    auth = st.session_state.get("auth", {})
    profile = st.session_state.get("student_profile", {})
    return {
        "user_id": auth.get("user_id") or profile.get("student_id", ""),
        "email": str(auth.get("email", "")).strip().lower(),
        "display_name": auth.get("display_name") or profile.get("name", DEFAULT_STUDENT_PROFILE["name"]),
        "role": auth.get("role", "Étudiant"),
        "level": profile.get("level", "Bac"),
        "section": profile.get("section", ""),
    }


def _humanize_status(status: Any, judge_status: Any) -> str:
    """Map internal statuses to a short French label."""
    normalized = str(status or "").strip().lower()
    if normalized == "correct":
        return "Correct"
    if normalized == "incorrect":
        return "À revoir"
    if normalized == "generated":
        return "Généré"
    if normalized == "assigned":
        return "Assigne"
    if normalized == "opened":
        return "Ouvert"
    judge_text = str(judge_status or "").strip()
    return judge_text or "Généré"


def _estimate_tutor_minutes(student_message: str, assistant_reply: str) -> float:
    """Estimate study time contributed by one tutoring exchange."""
    total_chars = len(str(student_message)) + len(str(assistant_reply))
    return max(1.5, min(6.0, round(total_chars / 650, 1)))


def _trim_text(value: Any, limit: int) -> str:
    """Safely trim long text before storing it."""
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _format_timestamp(value: datetime | None) -> str:
    """Format timestamps for sidebar/dashboard history."""
    if value is None:
        return "À l'instant"
    return value.astimezone(UTC).strftime("%d %b, %H:%M")


def _coerce_datetime(value: Any) -> datetime | None:
    """Convert stored values to timezone-aware UTC datetimes."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            return None
    return None


def _now_utc() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(UTC)


def _end_of_day(day_value: date) -> datetime:
    """Return the end-of-day instant for one calendar date."""
    return datetime.combine(day_value, time.max, tzinfo=UTC)
