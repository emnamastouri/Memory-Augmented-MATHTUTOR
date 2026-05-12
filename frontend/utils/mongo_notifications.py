"""MongoDB-backed notifications for student and teacher sidebars."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from frontend.utils.constants import MONGO_DB_NAME, MONGO_USER_NOTIFICATIONS_COLLECTION
from frontend.utils.mongo_auth import get_mongo_client


def get_user_notifications_collection() -> Collection:
    """Return the persisted user-notifications collection with basic indexes."""
    collection = get_mongo_client()[MONGO_DB_NAME][MONGO_USER_NOTIFICATIONS_COLLECTION]
    collection.create_index([("user_email", ASCENDING), ("created_at", DESCENDING)])
    collection.create_index([("user_email", ASCENDING), ("is_read", ASCENDING)])
    collection.create_index([("assignment_id", ASCENDING)])
    return collection


def create_user_notification(
    *,
    user_email: str,
    title: str,
    message: str,
    icon: str = "🔔",
    kind: str = "info",
    assignment_id: str = "",
    generation_trace_id: str = "",
    action_page: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Insert one user notification in MongoDB."""
    normalized_email = user_email.strip().lower()
    if not normalized_email:
        return

    payload = {
        "user_email": normalized_email,
        "title": str(title).strip() or "Notification",
        "message": str(message).strip(),
        "icon": str(icon).strip() or "🔔",
        "kind": str(kind).strip() or "info",
        "assignment_id": str(assignment_id).strip(),
        "generation_trace_id": str(generation_trace_id).strip(),
        "action_page": str(action_page).strip(),
        "metadata": metadata or {},
        "is_read": False,
        "created_at": _now_utc(),
    }
    try:
        get_user_notifications_collection().insert_one(payload)
    except PyMongoError:
        return


def get_user_notifications(user_email: str, *, limit: int = 12) -> list[dict[str, Any]]:
    """Return the latest persisted notifications for one user."""
    normalized_email = user_email.strip().lower()
    if not normalized_email:
        return []

    try:
        cursor = (
            get_user_notifications_collection()
            .find({"user_email": normalized_email})
            .sort("created_at", DESCENDING)
            .limit(limit)
        )
    except PyMongoError:
        return []

    items: list[dict[str, Any]] = []
    for document in cursor:
        created_at = _coerce_datetime(document.get("created_at"))
        items.append(
            {
                "id": str(document.get("_id", "")),
                "title": document.get("title", "Notification"),
                "message": document.get("message", ""),
                "icon": document.get("icon", "🔔"),
                "kind": document.get("kind", "info"),
                "assignment_id": document.get("assignment_id", ""),
                "generation_trace_id": document.get("generation_trace_id", ""),
                "action_page": document.get("action_page", ""),
                "metadata": document.get("metadata", {}) or {},
                "is_read": bool(document.get("is_read", False)),
                "timestamp": _format_timestamp(created_at),
            }
        )
    return items


def mark_user_notification_read(notification_id: str) -> None:
    """Mark one notification as read after the user opened it."""
    object_id = _coerce_object_id(notification_id)
    if object_id is None:
        return

    try:
        get_user_notifications_collection().update_one(
            {"_id": object_id},
            {"$set": {"is_read": True, "read_at": _now_utc()}},
        )
    except PyMongoError:
        return


def _coerce_object_id(value: str) -> ObjectId | None:
    """Safely convert a string into a BSON object id."""
    try:
        return ObjectId(str(value).strip())
    except Exception:
        return None


def _format_timestamp(value: datetime | None) -> str:
    """Format one notification timestamp for compact sidebar display."""
    if value is None:
        return "A l'instant"
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
