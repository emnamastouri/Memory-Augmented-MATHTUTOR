"""MongoDB persistence for tutoring conversations and resumable threads."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import streamlit as st
from pymongo import ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from frontend.utils.constants import DEFAULT_CHAT_HISTORY, MONGO_DB_NAME, MONGO_TUTORING_THREADS_COLLECTION
from frontend.utils.mongo_auth import get_mongo_client
from frontend.utils.mongo_learning import get_session_user_context


def get_tutoring_threads_collection() -> Collection:
    """Return the tutoring-threads collection with basic indexes."""
    collection = get_mongo_client()[MONGO_DB_NAME][MONGO_TUTORING_THREADS_COLLECTION]
    collection.create_index([("user_email", ASCENDING), ("updated_at", DESCENDING)])
    collection.create_index([("user_email", ASCENDING), ("thread_id", ASCENDING)])
    collection.create_index([("user_email", ASCENDING), ("exercise_ref.generation_trace_id", ASCENDING)])
    return collection


def ensure_tutoring_thread(
    *,
    mode: str,
    exercise_context: dict[str, Any] | None = None,
    session_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return the active thread or create a new one for the current context."""
    user = get_session_user_context()
    if not user.get("email"):
        return None

    active_thread_id = str(st.session_state.get("active_tutoring_thread_id", "")).strip()
    if active_thread_id:
        thread = load_tutoring_thread(active_thread_id)
        if thread and _thread_matches_context(thread, exercise_context):
            return thread

    current_history = deepcopy(session_history or st.session_state.get("chat_history", DEFAULT_CHAT_HISTORY))
    thread = _create_tutoring_thread(
        user=user,
        mode=mode,
        exercise_context=exercise_context,
        initial_messages=current_history,
    )
    if thread:
        st.session_state.active_tutoring_thread_id = thread["thread_id"]
        st.session_state.active_tutoring_thread_title = thread["title"]
    return thread


def append_tutoring_turn(
    *,
    thread_id: str,
    mode: str,
    user_message: str,
    assistant_message: str,
    exercise_context: dict[str, Any] | None = None,
    student_answer_draft: str = "",
) -> None:
    """Append one tutoring turn to a persisted thread."""
    user = get_session_user_context()
    if not user.get("email") or not thread_id:
        return

    timestamp = datetime.now().strftime("%H:%M")
    now = _now_utc()
    user_entry = {
        "role": "user",
        "content": str(user_message).strip(),
        "mode": mode,
        "timestamp": timestamp,
        "created_at": now,
    }
    assistant_entry = {
        "role": "assistant",
        "content": str(assistant_message).strip(),
        "mode": mode,
        "timestamp": timestamp,
        "created_at": now,
    }

    try:
        get_tutoring_threads_collection().update_one(
            {"user_email": user["email"], "thread_id": thread_id},
            {
                "$push": {"messages": {"$each": [user_entry, assistant_entry]}},
                "$set": {
                    "updated_at": now,
                    "last_mode": mode,
                    "last_message_preview": _thread_preview(assistant_message or user_message),
                    "exercise_snapshot": _sanitize_exercise_snapshot(exercise_context),
                    "exercise_ref": _exercise_reference(exercise_context),
                    "last_student_answer_draft": _trim_text(student_answer_draft, 2000),
                },
                "$inc": {"message_count": 2, "turn_count": 1},
            },
        )
    except PyMongoError:
        return

    st.session_state.active_tutoring_thread_id = thread_id


def get_recent_tutoring_threads(limit: int = 12) -> list[dict[str, Any]]:
    """Return the recent tutoring threads for the active user."""
    user = get_session_user_context()
    if not user.get("email"):
        return []

    try:
        cursor = (
            get_tutoring_threads_collection()
            .find({"user_email": user["email"]})
            .sort("updated_at", DESCENDING)
            .limit(limit)
        )
    except PyMongoError:
        return []

    items: list[dict[str, Any]] = []
    for document in cursor:
        exercise_ref = document.get("exercise_ref") or {}
        items.append(
            {
                "thread_id": document.get("thread_id", ""),
                "title": document.get("title", "Conversation"),
                "updated_at_label": _format_thread_time(_coerce_datetime(document.get("updated_at"))),
                "subtopic": exercise_ref.get("subtopic", ""),
                "topic": exercise_ref.get("topic", ""),
                "mode": document.get("last_mode", "Socratique"),
                "message_count": int(document.get("message_count", 0)),
                "preview": document.get("last_message_preview", ""),
            }
        )
    return items


def load_tutoring_thread(thread_id: str) -> dict[str, Any] | None:
    """Load one tutoring thread document for the active user."""
    user = get_session_user_context()
    if not user.get("email") or not thread_id:
        return None

    try:
        return get_tutoring_threads_collection().find_one({"user_email": user["email"], "thread_id": thread_id})
    except PyMongoError:
        return None


def restore_tutoring_thread(thread_id: str) -> bool:
    """Restore one persisted tutoring thread into the current session."""
    thread = load_tutoring_thread(thread_id)
    if not thread:
        return False

    messages = deepcopy(thread.get("messages") or DEFAULT_CHAT_HISTORY)
    st.session_state.chat_history = messages
    st.session_state.active_tutoring_thread_id = thread.get("thread_id", "")
    st.session_state.active_tutoring_thread_title = thread.get("title", "")
    st.session_state.tutoring_state["mode"] = thread.get("last_mode", st.session_state.tutoring_state["mode"])

    exercise_snapshot = thread.get("exercise_snapshot")
    if exercise_snapshot:
        st.session_state.current_exercise = deepcopy(exercise_snapshot)
        st.session_state.tutoring_state["use_exercise_context"] = True
        st.session_state.current_answer = thread.get("last_student_answer_draft", "")
    else:
        st.session_state.current_exercise = None
        st.session_state.tutoring_state["use_exercise_context"] = False
        st.session_state.current_answer = thread.get("last_student_answer_draft", "")
    return True


def get_latest_thread_for_exercise(exercise_context: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the latest stored thread linked to the current exercise, if any."""
    user = get_session_user_context()
    if not user.get("email") or not exercise_context:
        return None

    generation_trace_id = _exercise_reference(exercise_context).get("generation_trace_id", "")
    if not generation_trace_id:
        return None

    try:
        return (
            get_tutoring_threads_collection()
            .find({"user_email": user["email"], "exercise_ref.generation_trace_id": generation_trace_id})
            .sort("updated_at", DESCENDING)
            .limit(1)
            .next()
        )
    except (PyMongoError, StopIteration):
        return None


def get_latest_thread_for_generation_trace(generation_trace_id: str) -> dict[str, Any] | None:
    """Return the latest stored thread linked to one exercise generation trace id."""
    user = get_session_user_context()
    trace_id = str(generation_trace_id).strip()
    if not user.get("email") or not trace_id:
        return None

    try:
        return (
            get_tutoring_threads_collection()
            .find({"user_email": user["email"], "exercise_ref.generation_trace_id": trace_id})
            .sort("updated_at", DESCENDING)
            .limit(1)
            .next()
        )
    except (PyMongoError, StopIteration):
        return None


def _create_tutoring_thread(
    *,
    user: dict[str, Any],
    mode: str,
    exercise_context: dict[str, Any] | None,
    initial_messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Insert one new tutoring thread document."""
    now = _now_utc()
    thread_id = uuid4().hex
    title = _thread_title(exercise_context, initial_messages)
    document = {
        "thread_id": thread_id,
        "user_email": user.get("email", ""),
        "user_id": user.get("user_id", ""),
        "user_display_name": user.get("display_name", ""),
        "created_at": now,
        "updated_at": now,
        "title": title,
        "last_mode": mode,
        "message_count": len(initial_messages),
        "turn_count": 0,
        "messages": [_normalize_message(message) for message in initial_messages],
        "exercise_ref": _exercise_reference(exercise_context),
        "exercise_snapshot": _sanitize_exercise_snapshot(exercise_context),
        "last_message_preview": _thread_preview(initial_messages[-1]["content"]) if initial_messages else "",
        "last_student_answer_draft": str(st.session_state.get("current_answer", "")).strip(),
    }
    try:
        get_tutoring_threads_collection().insert_one(document)
    except PyMongoError:
        return None
    return document


def _thread_title(exercise_context: dict[str, Any] | None, initial_messages: list[dict[str, Any]]) -> str:
    """Build a concise display title for one thread."""
    if exercise_context:
        subtopic = str(exercise_context.get("subtopic", "")).strip()
        title = subtopic or str(exercise_context.get("title", "")).strip()
        if title:
            return _trim_text(title, 72)
    for message in reversed(initial_messages):
        if message.get("role") == "user" and str(message.get("content", "")).strip():
            return _trim_text(message["content"], 72)
    return "Nouvelle conversation de tutorat"


def _thread_matches_context(thread: dict[str, Any], exercise_context: dict[str, Any] | None) -> bool:
    """Check whether a persisted thread still matches the current exercise context."""
    stored_ref = thread.get("exercise_ref") or {}
    current_ref = _exercise_reference(exercise_context)
    return stored_ref == current_ref


def _exercise_reference(exercise_context: dict[str, Any] | None) -> dict[str, str]:
    """Extract the light reference used to match a thread to an exercise."""
    if not exercise_context:
        return {}
    return {
        "exercise_id": str(exercise_context.get("id", "")).strip(),
        "generation_trace_id": str(exercise_context.get("generation_trace_id", "")).strip(),
        "topic": str(exercise_context.get("topic", "")).strip(),
        "subtopic": str(exercise_context.get("subtopic", "")).strip(),
    }


def _sanitize_exercise_snapshot(exercise_context: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep the full exercise context needed to restore the tutoring page."""
    if not exercise_context:
        return None

    allowed_fields = [
        "id",
        "generation_trace_id",
        "title",
        "prompt",
        "topic",
        "subtopic",
        "section",
        "level",
        "exercise_type",
        "difficulty",
        "hint",
        "learning_objective",
        "display_answer",
        "hidden_solution",
        "solution_steps",
        "tags",
        "support_summary",
        "table_data",
        "chart_data",
        "graph_data",
        "judge_status",
        "judge_summary",
        "judge_model",
        "judge_validation_flag",
        "judge_alignment_reason",
        "judge_issues",
        "solution_validation_flag",
        "solution_validation_status",
        "solution_validation_summary",
        "local_validation_flag",
        "local_validation_summary",
        "local_validation_issues",
        "pedagogical_completeness_flag",
        "pedagogical_completeness_summary",
        "pedagogical_completeness_issues",
        "symbolic_checks_ran",
        "symbolic_checks_passed",
        "symbolic_checks_required",
        "corrected_fields_applied",
        "generation_backend",
        "memory_adaptation_note",
        "source_case_summaries",
        "generated_at",
    ]
    return {field: deepcopy(exercise_context.get(field)) for field in allowed_fields if field in exercise_context}


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize one chat message before saving it."""
    return {
        "role": message.get("role", "assistant"),
        "content": str(message.get("content", "")).strip(),
        "mode": message.get("mode", st.session_state.get("tutoring_state", {}).get("mode", "Socratique")),
        "timestamp": message.get("timestamp", datetime.now().strftime("%H:%M")),
        "created_at": _coerce_datetime(message.get("created_at")) or _now_utc(),
    }


def _thread_preview(text: str) -> str:
    """Build a short preview for sidebar history."""
    return _trim_text(text, 96)


def _format_thread_time(value: datetime | None) -> str:
    """Format one thread timestamp for the sidebar list."""
    if value is None:
        return "à l'instant"
    return value.astimezone(UTC).strftime("%d %b, %H:%M")


def _trim_text(value: Any, limit: int) -> str:
    """Trim stored text safely."""
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _coerce_datetime(value: Any) -> datetime | None:
    """Convert stored timestamps into aware UTC datetimes."""
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
