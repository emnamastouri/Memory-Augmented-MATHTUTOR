"""Teacher-side MongoDB services for groups, assignments, dashboards, and supervision."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, date, datetime
from typing import Any
from uuid import uuid4

from pymongo import ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from frontend.utils.constants import (
    DEFAULT_EXERCISE_DIFFICULTY,
    MONGO_DB_NAME,
    MONGO_GROUP_ASSIGNMENTS_COLLECTION,
    MONGO_TEACHER_GROUPS_COLLECTION,
)
from frontend.utils.dataset_catalog import normalize_section_label
from frontend.utils.mongo_auth import get_mongo_client, get_users_collection
from frontend.utils.mongo_learning import (
    get_exercise_records_collection,
    get_user_progress_analytics,
)
from frontend.utils.mongo_notifications import create_user_notification
from frontend.utils.mongo_tutoring import get_tutoring_threads_collection


def get_teacher_groups_collection() -> Collection:
    """Return the teacher-groups collection with its indexes."""
    collection = get_mongo_client()[MONGO_DB_NAME][MONGO_TEACHER_GROUPS_COLLECTION]
    collection.create_index([("teacher_email", ASCENDING), ("group_id", ASCENDING)], unique=True)
    collection.create_index([("teacher_email", ASCENDING), ("name_normalized", ASCENDING)], unique=True)
    return collection


def get_group_assignments_collection() -> Collection:
    """Return the group-assignments collection with its indexes."""
    collection = get_mongo_client()[MONGO_DB_NAME][MONGO_GROUP_ASSIGNMENTS_COLLECTION]
    collection.create_index([("teacher_email", ASCENDING), ("assignment_id", ASCENDING)], unique=True)
    collection.create_index([("teacher_email", ASCENDING), ("created_at", DESCENDING)])
    collection.create_index([("group_id", ASCENDING), ("created_at", DESCENDING)])
    collection.create_index([("recipients.student_email", ASCENDING)])
    return collection


def verify_student_account_by_email(student_email: str) -> dict[str, Any]:
    """Check that one email belongs to an existing student account."""
    normalized_email = _normalize_email(student_email)
    if not normalized_email:
        return {"ok": False, "message": "Veuillez saisir une adresse e-mail valide."}

    try:
        document = get_users_collection().find_one({"email": normalized_email})
    except PyMongoError as exc:
        return {"ok": False, "message": f"Connexion MongoDB indisponible : {exc}"}

    if not document:
        return {"ok": False, "message": "Aucun compte n'existe avec cette adresse e-mail."}
    if not _is_student_role(document.get("role", "")):
        return {"ok": False, "message": "Cette adresse existe, mais elle n'appartient pas a un compte etudiant."}

    return {
        "ok": True,
        "student": {
            "user_id": str(document.get("_id", "")),
            "name": str(document.get("name", "")).strip() or normalized_email.split("@")[0],
            "email": normalized_email,
            "level": str(document.get("level", "Bac")).strip() or "Bac",
            "section": normalize_section_label(str(document.get("section", "")).strip()),
        },
    }


def create_teacher_group(
    *,
    teacher_email: str,
    teacher_user_id: str,
    teacher_name: str,
    group_name: str,
    section: str = "",
    level: str = "Bac",
) -> dict[str, Any]:
    """Create one teacher-owned group."""
    normalized_teacher_email = _normalize_email(teacher_email)
    clean_name = str(group_name).strip()
    if not normalized_teacher_email:
        return {"ok": False, "message": "Compte enseignant introuvable dans la session."}
    if len(clean_name) < 3:
        return {"ok": False, "message": "Le nom du groupe doit contenir au moins 3 caracteres."}

    now = _now_utc()
    document = {
        "group_id": uuid4().hex,
        "teacher_email": normalized_teacher_email,
        "teacher_user_id": str(teacher_user_id).strip(),
        "teacher_name": str(teacher_name).strip(),
        "name": clean_name,
        "name_normalized": clean_name.casefold(),
        "section": normalize_section_label(section) if section else "",
        "level": str(level).strip() or "Bac",
        "student_emails": [],
        "members": [],
        "created_at": now,
        "updated_at": now,
    }

    try:
        get_teacher_groups_collection().insert_one(document)
    except PyMongoError as exc:
        message = "Un groupe avec ce nom existe deja pour cet enseignant."
        if "duplicate key" not in str(exc).lower():
            message = f"Impossible de creer le groupe : {exc}"
        return {"ok": False, "message": message}

    return {"ok": True, "message": "Groupe cree avec succes.", "group": _serialize_group(document)}


def list_teacher_groups(teacher_email: str) -> list[dict[str, Any]]:
    """Return every group created by one teacher."""
    normalized_teacher_email = _normalize_email(teacher_email)
    if not normalized_teacher_email:
        return []

    try:
        cursor = (
            get_teacher_groups_collection()
            .find({"teacher_email": normalized_teacher_email})
            .sort("created_at", DESCENDING)
        )
    except PyMongoError:
        return []

    return [_serialize_group(document) for document in cursor]


def add_student_to_group(
    *,
    teacher_email: str,
    group_id: str,
    student_email: str,
) -> dict[str, Any]:
    """Add one validated student account to one teacher group."""
    normalized_teacher_email = _normalize_email(teacher_email)
    clean_group_id = str(group_id).strip()
    if not normalized_teacher_email or not clean_group_id:
        return {"ok": False, "message": "Groupe ou enseignant introuvable."}

    verification = verify_student_account_by_email(student_email)
    if not verification["ok"]:
        return verification
    student = verification["student"]

    try:
        group = get_teacher_groups_collection().find_one(
            {"teacher_email": normalized_teacher_email, "group_id": clean_group_id}
        )
    except PyMongoError as exc:
        return {"ok": False, "message": f"Impossible de charger le groupe : {exc}"}

    if not group:
        return {"ok": False, "message": "Le groupe selectionne est introuvable."}
    if student["email"] in set(group.get("student_emails", []) or []):
        return {"ok": False, "message": "Cet etudiant est deja membre du groupe."}

    member_payload = {
        "student_email": student["email"],
        "student_name": student["name"],
        "student_user_id": student["user_id"],
        "level": student["level"],
        "section": student["section"],
        "added_at": _now_utc(),
    }

    try:
        get_teacher_groups_collection().update_one(
            {"_id": group["_id"]},
            {
                "$push": {"members": member_payload},
                "$addToSet": {"student_emails": student["email"]},
                "$set": {"updated_at": _now_utc()},
            },
        )
        updated_group = get_teacher_groups_collection().find_one({"_id": group["_id"]})
    except PyMongoError as exc:
        return {"ok": False, "message": f"Impossible d'ajouter l'etudiant : {exc}"}

    return {
        "ok": True,
        "message": f"{student['name']} a ete ajoute au groupe {group['name']}.",
        "group": _serialize_group(updated_group or group),
        "student": student,
    }


def list_group_assignments(teacher_email: str, *, limit: int = 30) -> list[dict[str, Any]]:
    """Return teacher assignments enriched with live student progress."""
    normalized_teacher_email = _normalize_email(teacher_email)
    if not normalized_teacher_email:
        return []

    try:
        cursor = (
            get_group_assignments_collection()
            .find({"teacher_email": normalized_teacher_email})
            .sort("created_at", DESCENDING)
            .limit(limit)
        )
    except PyMongoError:
        return []

    return [_serialize_assignment(document) for document in cursor]


def assign_exercise_to_group(
    *,
    teacher_email: str,
    teacher_user_id: str,
    teacher_name: str,
    group_id: str,
    exercise: dict[str, Any],
    due_date: date | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Clone one teacher-generated exercise to every student in a group and notify them."""
    normalized_teacher_email = _normalize_email(teacher_email)
    clean_group_id = str(group_id).strip()
    if not normalized_teacher_email or not clean_group_id:
        return {"ok": False, "message": "Veuillez reconnecter le compte enseignant avant d'assigner un exercice."}
    if not exercise:
        return {"ok": False, "message": "Generez d'abord un exercice a assigner."}

    try:
        group = get_teacher_groups_collection().find_one(
            {"teacher_email": normalized_teacher_email, "group_id": clean_group_id}
        )
    except PyMongoError as exc:
        return {"ok": False, "message": f"Impossible de charger le groupe : {exc}"}

    if not group:
        return {"ok": False, "message": "Le groupe selectionne est introuvable."}

    members = list(group.get("members", []) or [])
    if not members:
        return {"ok": False, "message": "Ajoutez au moins un etudiant au groupe avant l'affectation."}

    now = _now_utc()
    assignment_id = uuid4().hex
    due_date_label = due_date.isoformat() if due_date else ""
    assignment_recipients: list[dict[str, Any]] = []

    try:
        for member in members:
            student_exercise = deepcopy(exercise)
            student_trace_id = uuid4().hex
            student_exercise["generation_trace_id"] = student_trace_id
            student_exercise["generated_at"] = now.isoformat(timespec="seconds")

            exercise_record = _build_assigned_exercise_record(
                student=member,
                exercise=student_exercise,
                generated_at=now,
                assignment_id=assignment_id,
                due_date_label=due_date_label,
                group=group,
                teacher_email=normalized_teacher_email,
                teacher_name=teacher_name,
            )
            get_exercise_records_collection().update_one(
                {
                    "user_email": exercise_record["user_email"],
                    "generation_trace_id": exercise_record["generation_trace_id"],
                },
                {"$set": exercise_record},
                upsert=True,
            )

            create_user_notification(
                user_email=member.get("student_email", ""),
                title="Nouvel exercice assigne",
                message=(
                    f"{exercise.get('title', 'Exercice')} a ete assigne au groupe {group.get('name', 'Groupe')}."
                ),
                icon="📚",
                kind="assignment",
                assignment_id=assignment_id,
                generation_trace_id=student_trace_id,
                action_page="exercise_generator",
                metadata={
                    "group_name": group.get("name", ""),
                    "topic": exercise.get("topic", ""),
                    "subtopic": exercise.get("subtopic", ""),
                    "due_date": due_date_label,
                },
            )

            assignment_recipients.append(
                {
                    "student_email": member.get("student_email", ""),
                    "student_name": member.get("student_name", ""),
                    "student_user_id": member.get("student_user_id", ""),
                    "level": member.get("level", "Bac"),
                    "section": member.get("section", ""),
                    "generation_trace_id": student_trace_id,
                }
            )

        assignment_document = {
            "assignment_id": assignment_id,
            "teacher_email": normalized_teacher_email,
            "teacher_user_id": str(teacher_user_id).strip(),
            "teacher_name": str(teacher_name).strip(),
            "group_id": group["group_id"],
            "group_name": group.get("name", "Groupe"),
            "group_section": group.get("section", ""),
            "level": exercise.get("level", group.get("level", "Bac")),
            "section": exercise.get("section", group.get("section", "")),
            "topic": exercise.get("topic", ""),
            "subtopic": exercise.get("subtopic", ""),
            "exercise_type": exercise.get("exercise_type", ""),
            "difficulty": exercise.get("difficulty", DEFAULT_EXERCISE_DIFFICULTY),
            "due_date": due_date_label,
            "teacher_note": str(note).strip(),
            "teacher_generation_trace_id": str(exercise.get("generation_trace_id", "")).strip(),
            "exercise_snapshot": deepcopy(exercise),
            "recipients": assignment_recipients,
            "created_at": now,
            "updated_at": now,
        }
        get_group_assignments_collection().insert_one(assignment_document)
    except PyMongoError as exc:
        return {"ok": False, "message": f"Impossible de finaliser l'affectation : {exc}"}

    return {
        "ok": True,
        "message": f"Exercice assigne au groupe {group.get('name', 'Groupe')} ({len(assignment_recipients)} etudiant(s)).",
        "assignment_id": assignment_id,
        "recipient_count": len(assignment_recipients),
    }


def get_teacher_panel_snapshot(teacher_email: str) -> dict[str, Any]:
    """Build the full teacher dashboard payload from Mongo."""
    groups = list_teacher_groups(teacher_email)
    assignments = list_group_assignments(teacher_email, limit=40)
    student_map: dict[str, dict[str, Any]] = {}

    for group in groups:
        for member in group.get("members", []):
            student_map[member["student_email"]] = member

    students = [_build_student_performance_row(student_map[email], teacher_email) for email in sorted(student_map)]
    active_students = [row for row in students if row.get("last_active_label") != "Jamais"]
    average_mastery = round(
        sum(row.get("mastery", 0) for row in students) / len(students)
    ) if students else 0
    at_risk_count = len([row for row in students if row.get("risk") == "Eleve"])
    total_hours = round(sum(float(row.get("study_hours", 0.0)) for row in students), 1)
    open_assignments = sum(1 for item in assignments if item.get("solved_count", 0) < item.get("recipient_count", 0))

    return {
        "metrics": {
            "group_count": len(groups),
            "student_count": len(students),
            "active_students": len(active_students),
            "at_risk_students": at_risk_count,
            "average_mastery": average_mastery,
            "open_assignments": open_assignments,
            "study_hours": total_hours,
        },
        "groups": groups,
        "students": students,
        "assignments": assignments,
    }


def get_teacher_supervision_view(
    *,
    teacher_email: str,
    assignment_id: str,
    student_email: str = "",
) -> dict[str, Any] | None:
    """Load one assignment and one student's related tutoring activity for teacher supervision."""
    normalized_teacher_email = _normalize_email(teacher_email)
    clean_assignment_id = str(assignment_id).strip()
    if not normalized_teacher_email or not clean_assignment_id:
        return None

    try:
        assignment = get_group_assignments_collection().find_one(
            {"teacher_email": normalized_teacher_email, "assignment_id": clean_assignment_id}
        )
    except PyMongoError:
        return None

    if not assignment:
        return None

    recipients = list(assignment.get("recipients", []) or [])
    if not recipients:
        return {
            "assignment": _serialize_assignment(assignment),
            "students": [],
            "active_student": None,
        }

    normalized_student_email = _normalize_email(student_email)
    available_emails = {recipient.get("student_email", "") for recipient in recipients}
    if normalized_student_email not in available_emails:
        normalized_student_email = recipients[0].get("student_email", "")

    recipient = next(
        item for item in recipients if item.get("student_email", "") == normalized_student_email
    )
    exercise_record = _load_student_exercise_record(
        student_email=normalized_student_email,
        generation_trace_id=recipient.get("generation_trace_id", ""),
    )
    threads = _load_student_threads_for_exercise(
        student_email=normalized_student_email,
        generation_trace_id=recipient.get("generation_trace_id", ""),
    )
    student_profile = _load_student_profile(normalized_student_email)
    student_analytics = get_user_progress_analytics(
        user_email=normalized_student_email,
        profile=student_profile,
    )

    return {
        "assignment": _serialize_assignment(assignment),
        "students": [
            {
                "student_email": item.get("student_email", ""),
                "student_name": item.get("student_name", ""),
                "generation_trace_id": item.get("generation_trace_id", ""),
            }
            for item in recipients
        ],
        "active_student": {
            "student_email": normalized_student_email,
            "student_name": recipient.get("student_name", ""),
            "generation_trace_id": recipient.get("generation_trace_id", ""),
            "exercise_record": exercise_record,
            "threads": threads,
            "analytics": student_analytics,
        },
    }


def _build_student_performance_row(student: dict[str, Any], teacher_email: str) -> dict[str, Any]:
    """Convert one student account plus analytics into a dashboard row."""
    profile = _load_student_profile(student.get("student_email", ""))
    analytics = get_user_progress_analytics(
        user_email=student.get("student_email", ""),
        profile=profile,
    )
    metrics = analytics.get("metrics", {})
    mastery = int(metrics.get("mastery_average", 0))
    success_rate = int(metrics.get("success_rate_pct", 0))
    study_hours = float(metrics.get("study_hours", 0.0))
    open_assignments = _count_open_assignments_for_student(teacher_email, student.get("student_email", ""))

    account_document = None
    try:
        account_document = get_users_collection().find_one({"email": student.get("student_email", "")})
    except PyMongoError:
        account_document = None

    last_active = _coerce_datetime((account_document or {}).get("last_active_at"))
    current_focus = str((account_document or {}).get("last_studied_topic", "")).strip()
    if not current_focus:
        weak_topics = analytics.get("weak_topics", [])
        current_focus = weak_topics[0]["topic"] if weak_topics else "Aucun focus detecte"

    return {
        "name": student.get("student_name", ""),
        "email": student.get("student_email", ""),
        "level": student.get("level", "Bac"),
        "section": student.get("section", ""),
        "mastery": mastery,
        "success_rate": success_rate,
        "study_hours": study_hours,
        "focus": current_focus,
        "topics_studied": int(metrics.get("topics_studied", 0)),
        "solved_exercises": int(metrics.get("solved_exercises", 0)),
        "open_assignments": open_assignments,
        "risk": _risk_label(mastery, success_rate),
        "last_active_label": _format_relative_time(last_active),
    }


def _count_open_assignments_for_student(teacher_email: str, student_email: str) -> int:
    """Count assignments from this teacher that are not yet solved by the student."""
    try:
        cursor = get_group_assignments_collection().find(
            {"teacher_email": _normalize_email(teacher_email), "recipients.student_email": _normalize_email(student_email)}
        )
    except PyMongoError:
        return 0

    count = 0
    for assignment in cursor:
        for recipient in assignment.get("recipients", []) or []:
            if recipient.get("student_email", "") != _normalize_email(student_email):
                continue
            record = _load_student_exercise_record(
                student_email=student_email,
                generation_trace_id=recipient.get("generation_trace_id", ""),
            )
            if not record or not bool(record.get("is_solved", False)):
                count += 1
            break
    return count


def _load_student_exercise_record(*, student_email: str, generation_trace_id: str) -> dict[str, Any] | None:
    """Load one student's exercise record."""
    try:
        return get_exercise_records_collection().find_one(
            {
                "user_email": _normalize_email(student_email),
                "generation_trace_id": str(generation_trace_id).strip(),
            }
        )
    except PyMongoError:
        return None


def _load_student_threads_for_exercise(*, student_email: str, generation_trace_id: str) -> list[dict[str, Any]]:
    """Load tutoring threads linked to one student's assigned exercise."""
    try:
        cursor = (
            get_tutoring_threads_collection()
            .find(
                {
                    "user_email": _normalize_email(student_email),
                    "exercise_ref.generation_trace_id": str(generation_trace_id).strip(),
                }
            )
            .sort("updated_at", DESCENDING)
        )
    except PyMongoError:
        return []

    threads: list[dict[str, Any]] = []
    for document in cursor:
        threads.append(
            {
                "thread_id": document.get("thread_id", ""),
                "title": document.get("title", "Conversation"),
                "updated_at_label": _format_relative_time(_coerce_datetime(document.get("updated_at"))),
                "message_count": int(document.get("message_count", 0)),
                "mode": document.get("last_mode", "Socratique"),
                "messages": list(document.get("messages", []) or []),
                "last_student_answer_draft": document.get("last_student_answer_draft", ""),
            }
        )
    return threads


def _load_student_profile(student_email: str) -> dict[str, Any]:
    """Build a compact profile dict for analytics."""
    try:
        document = get_users_collection().find_one({"email": _normalize_email(student_email)})
    except PyMongoError:
        document = None

    if not document:
        return {"name": student_email, "level": "Bac", "section": "", "weak_topics": [], "strong_topics": []}
    return {
        "name": str(document.get("name", "")).strip() or student_email,
        "level": str(document.get("level", "Bac")).strip() or "Bac",
        "section": normalize_section_label(str(document.get("section", "")).strip()),
        "weak_topics": [],
        "strong_topics": [],
    }


def _serialize_group(document: dict[str, Any]) -> dict[str, Any]:
    """Normalize one group document for the UI."""
    members = list(document.get("members", []) or [])
    return {
        "group_id": document.get("group_id", ""),
        "name": document.get("name", "Groupe"),
        "section": document.get("section", ""),
        "level": document.get("level", "Bac"),
        "member_count": len(members),
        "members": [
            {
                "student_email": item.get("student_email", ""),
                "student_name": item.get("student_name", ""),
                "level": item.get("level", "Bac"),
                "section": item.get("section", ""),
            }
            for item in members
        ],
        "created_at_label": _format_relative_time(_coerce_datetime(document.get("created_at"))),
    }


def _serialize_assignment(document: dict[str, Any]) -> dict[str, Any]:
    """Normalize one assignment document for the teacher UI."""
    recipients = list(document.get("recipients", []) or [])
    solved_count = 0
    opened_count = 0
    waiting_count = 0

    for recipient in recipients:
        record = _load_student_exercise_record(
            student_email=recipient.get("student_email", ""),
            generation_trace_id=recipient.get("generation_trace_id", ""),
        )
        latest_status = str((record or {}).get("latest_status", "")).strip().lower()
        if bool((record or {}).get("is_solved", False)) or latest_status == "correct":
            solved_count += 1
        elif latest_status in {"opened", "incorrect"} or int((record or {}).get("consultation_count", 0)) > 0:
            opened_count += 1
        else:
            waiting_count += 1

    return {
        "assignment_id": document.get("assignment_id", ""),
        "group_id": document.get("group_id", ""),
        "group_name": document.get("group_name", ""),
        "title": (document.get("exercise_snapshot") or {}).get("title", "Exercice"),
        "topic": document.get("topic", ""),
        "subtopic": document.get("subtopic", ""),
        "exercise_type": document.get("exercise_type", ""),
        "due_date": document.get("due_date", ""),
        "teacher_note": document.get("teacher_note", ""),
        "recipient_count": len(recipients),
        "solved_count": solved_count,
        "opened_count": opened_count,
        "waiting_count": waiting_count,
        "created_at_label": _format_relative_time(_coerce_datetime(document.get("created_at"))),
        "exercise_snapshot": deepcopy(document.get("exercise_snapshot") or {}),
        "recipients": recipients,
    }


def _build_assigned_exercise_record(
    *,
    student: dict[str, Any],
    exercise: dict[str, Any],
    generated_at: datetime,
    assignment_id: str,
    due_date_label: str,
    group: dict[str, Any],
    teacher_email: str,
    teacher_name: str,
) -> dict[str, Any]:
    """Build the persisted exercise record inserted for one assigned student."""
    topic = str(exercise.get("topic", "")).strip()
    subtopic = str(exercise.get("subtopic", "")).strip()
    difficulty = str(exercise.get("difficulty", "")).strip() or DEFAULT_EXERCISE_DIFFICULTY
    exercise_type = str(exercise.get("exercise_type", "")).strip() or "Exercice probleme"
    tags = list(exercise.get("tags", []) or [])
    if not tags:
        tags = [value for value in [exercise.get("section", ""), topic, subtopic, difficulty, exercise_type] if value]

    return {
        "user_email": student.get("student_email", ""),
        "user_id": student.get("student_user_id", ""),
        "user_display_name": student.get("student_name", ""),
        "user_role": "Etudiant",
        "level": exercise.get("level", student.get("level", "Bac")),
        "section": exercise.get("section", student.get("section", "")),
        "exercise_id": exercise.get("id", ""),
        "generation_trace_id": exercise.get("generation_trace_id", ""),
        "generated_at": generated_at,
        "last_activity_at": generated_at,
        "title": exercise.get("title", "Exercice"),
        "topic": topic,
        "subtopic": subtopic,
        "difficulty": difficulty,
        "exercise_type": exercise_type,
        "generation_backend": exercise.get("generation_backend", ""),
        "generation_attempt_number": int(exercise.get("generation_attempt_number", 1) or 1),
        "judge_validation_flag": exercise.get("judge_validation_flag", ""),
        "judge_status": exercise.get("judge_status", ""),
        "judge_model": exercise.get("judge_model", ""),
        "judge_summary": exercise.get("judge_summary", ""),
        "judge_alignment_status": exercise.get("judge_alignment_status", ""),
        "judge_alignment_reason": exercise.get("judge_alignment_reason", ""),
        "judge_issues": list(exercise.get("judge_issues", []) or []),
        "judge_rejected_attempts": list(exercise.get("judge_rejected_attempts", []) or []),
        "solution_validation_flag": exercise.get("solution_validation_flag", ""),
        "solution_validation_status": exercise.get("solution_validation_status", ""),
        "solution_validation_summary": exercise.get("solution_validation_summary", ""),
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
        "prompt": exercise.get("prompt", ""),
        "hint": exercise.get("hint", ""),
        "learning_objective": exercise.get("learning_objective", ""),
        "display_answer": exercise.get("display_answer", ""),
        "hidden_solution": exercise.get("hidden_solution", ""),
        "solution_steps": list(exercise.get("solution_steps", []) or []),
        "memory_adaptation_note": exercise.get("memory_adaptation_note", ""),
        "source_case_summaries": list(exercise.get("source_case_summaries", []) or []),
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
        "tags": tags,
        "assignment_id": assignment_id,
        "assignment_status": "assigned",
        "assigned_at": generated_at,
        "assignment_due_date": due_date_label,
        "assigned_group_id": group.get("group_id", ""),
        "assigned_group_name": group.get("name", ""),
        "assigned_by_teacher_email": teacher_email,
        "assigned_by_teacher_name": teacher_name,
        "answer_submissions": [],
        "hint_events": [],
        "tutor_turns": 0,
        "progressive_hint_reveals": 0,
        "adaptive_hint_requests": 0,
        "latest_status": "assigned",
        "is_solved": False,
        "consultation_count": 0,
        "estimated_study_minutes": 0.0,
        "created_at": generated_at,
    }


def _normalize_email(value: str) -> str:
    """Normalize one email address."""
    return str(value or "").strip().lower()


def _is_student_role(value: str) -> bool:
    """Return whether the stored role corresponds to a student account."""
    normalized = str(value or "").strip().lower()
    return normalized in {"etudiant", "étudiant"}


def _risk_label(mastery: int, success_rate: int) -> str:
    """Build a short qualitative risk label."""
    if mastery < 50 or success_rate < 40:
        return "Eleve"
    if mastery < 68 or success_rate < 60:
        return "Modere"
    return "Faible"


def _format_relative_time(value: datetime | None) -> str:
    """Format a relative time label for dashboards."""
    if value is None:
        return "Jamais"
    now = _now_utc()
    delta = now - value
    if delta.total_seconds() < 3600:
        minutes = max(1, round(delta.total_seconds() / 60))
        return f"Il y a {minutes} min"
    if delta.total_seconds() < 86400:
        hours = max(1, round(delta.total_seconds() / 3600))
        return f"Il y a {hours} h"
    days = max(1, delta.days)
    return f"Il y a {days} j"


def _coerce_datetime(value: Any) -> datetime | None:
    """Convert stored Mongo values to timezone-aware datetimes."""
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
    """Return the current UTC datetime."""
    return datetime.now(UTC)
