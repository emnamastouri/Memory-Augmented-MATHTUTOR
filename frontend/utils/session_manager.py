"""Gestion du state Streamlit pour le frontend MathTutorAI."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime

import streamlit as st
from streamlit.errors import StreamlitAPIException

from frontend.utils.constants import (
    DEFAULT_AUTH_STATE,
    DEFAULT_CHAT_HISTORY,
    DEFAULT_NOTIFICATIONS,
    DEFAULT_SETTINGS,
    DEFAULT_STUDENT_PROFILE,
    DEFAULT_TEACHER_ASSIGNMENTS,
)
from frontend.utils.dataset_catalog import normalize_section_label
from frontend.utils.page_router import switch_to_page


def initialize_session_state() -> None:
    """Initialiser les valeurs par defaut de la session."""
    defaults = {
        "student_profile": deepcopy(DEFAULT_STUDENT_PROFILE),
        "chat_history": deepcopy(DEFAULT_CHAT_HISTORY),
        "active_tutoring_thread_id": "",
        "active_tutoring_thread_title": "",
        "current_exercise": None,
        "pending_assignment_trace_id": "",
        "current_answer": "",
        "last_verification": None,
        "selected_topic": DEFAULT_STUDENT_PROFILE["weak_topics"][0],
        "tutoring_state": {
            "mode": DEFAULT_STUDENT_PROFILE["preferred_mode"],
            "use_exercise_context": True,
        },
        "exercise_history": [],
        "flagged_exercises": [],
        "generation_memory_bank": [],
        "notifications": deepcopy(DEFAULT_NOTIFICATIONS),
        "settings": deepcopy(DEFAULT_SETTINGS),
        "show_exercise_hint": False,
        "exercise_hint_level": 0,
        "adaptive_hint_message": "",
        "teacher_assignments": deepcopy(DEFAULT_TEACHER_ASSIGNMENTS),
        "teacher_preview_exercise": None,
        "auth": deepcopy(DEFAULT_AUTH_STATE),
        "_analytics_page_markers": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "theme_migrated_to_light" not in st.session_state:
        theme_aliases = {
            "Midnight Chalkboard": "Tableau nocturne",
            "Ocean Graphite": "Ocean graphite",
            "Ivory Slate": "Ardoise ivoire",
            "Océan graphite": "Ocean graphite",
        }
        current_theme = st.session_state.settings.get("theme")
        st.session_state.settings["theme"] = theme_aliases.get(current_theme, current_theme or "Ardoise ivoire")
        if st.session_state.settings["theme"] == "Tableau nocturne":
            st.session_state.settings["theme"] = "Ardoise ivoire"
        st.session_state.theme_migrated_to_light = True


def append_chat_message(role: str, content: str, mode: str | None = None) -> None:
    """Ajouter un message dans l'historique du tutorat."""
    st.session_state.chat_history.append(
        {
            "role": role,
            "content": content,
            "mode": mode or st.session_state.tutoring_state["mode"],
            "timestamp": datetime.now().strftime("%H:%M"),
        }
    )


def push_notification(message: str, icon: str = "\u2705") -> None:
    """Ajouter une notification et l'afficher en toast si disponible."""
    safe_icon = _sanitize_notification_icon(icon)
    st.session_state.notifications.insert(0, {"message": message, "icon": safe_icon})
    st.session_state.notifications = st.session_state.notifications[:6]
    toast = getattr(st, "toast", None)
    if callable(toast):
        try:
            toast(message, icon=safe_icon)
        except StreamlitAPIException:
            toast(message, icon="⚠")


def _sanitize_notification_icon(icon: str) -> str:
    """Normalize toast icons to a Streamlit-compatible single emoji."""
    raw_icon = str(icon or "").strip()
    if not raw_icon:
        return "✅"

    keyword_map = {
        "alerte": "⚠",
        "warning": "⚠",
        "warn": "⚠",
        "error": "⚠",
        "danger": "⚠",
        "success": "✅",
        "ok": "✅",
        "info": "ℹ",
        "teacher": "📚",
        "enseignant": "📚",
    }
    lowered = raw_icon.lower()
    if lowered in keyword_map:
        return keyword_map[lowered]

    candidates = [raw_icon]
    if any(marker in raw_icon for marker in ("â", "Ã", "ï", "ð")):
        recovered = _recover_mojibake(raw_icon)
        if recovered and recovered not in candidates:
            candidates.insert(0, recovered)

    fallback_map = {
        "🧑‍🏫": "📚",
        "🧑🏽‍🎓": "👤",
        "⚠️": "⚠",
        "âš ï¸": "⚠",
    }
    for candidate in list(candidates):
        simplified = fallback_map.get(candidate)
        if simplified and simplified not in candidates:
            candidates.append(simplified)

    for candidate in candidates:
        simplified = _simplify_icon_candidate(candidate)
        if simplified:
            return simplified

    return "⚠"


def _recover_mojibake(value: str) -> str:
    """Attempt one cp1252->utf8 recovery for mojibake icons."""
    try:
        return value.encode("cp1252").decode("utf-8")
    except Exception:
        return value


def _simplify_icon_candidate(value: str) -> str:
    """Keep only simple single-codepoint icons accepted reliably by Streamlit toasts."""
    candidate = str(value or "").strip()
    if not candidate:
        return ""
    if len(candidate) == 1:
        return candidate
    if len(candidate) == 2 and candidate.endswith("\ufe0f"):
        return candidate[0]
    return ""


def set_current_exercise(exercise: dict) -> None:
    """Sauvegarder l'exercice actif."""
    st.session_state.current_exercise = exercise
    st.session_state.current_answer = ""
    st.session_state.last_verification = None
    st.session_state.show_exercise_hint = False
    st.session_state.exercise_hint_level = 0
    st.session_state.adaptive_hint_message = ""


def clear_current_exercise_context() -> None:
    """Effacer l'exercice actif et les elements de travail qui lui sont lies."""
    st.session_state.current_exercise = None
    st.session_state.current_answer = ""
    st.session_state.last_verification = None
    st.session_state.show_exercise_hint = False
    st.session_state.exercise_hint_level = 0
    st.session_state.adaptive_hint_message = ""


def log_generated_exercise(exercise: dict) -> None:
    """Ajouter un exercice genere a l'historique recent."""
    history_item = {
        "id": exercise["id"],
        "title": exercise["title"],
        "topic": exercise["topic"],
        "subtopic": exercise["subtopic"],
        "difficulty": exercise["difficulty"],
        "status": "Genere",
        "timestamp": datetime.now().strftime("%d %b, %H:%M"),
    }

    filtered = [item for item in st.session_state.exercise_history if item["id"] != exercise["id"]]
    st.session_state.exercise_history = [history_item, *filtered][:12]


def log_flagged_exercise(exercise: dict, *, reason: str, issues: list[str] | None = None, source: str = "Juge OpenRouter") -> None:
    """Conserver un exercice refuse par le juge avec un flag wrong."""
    flagged_item = {
        "id": exercise["id"],
        "title": exercise["title"],
        "topic": exercise["topic"],
        "subtopic": exercise["subtopic"],
        "difficulty": exercise["difficulty"],
        "flag": "wrong",
        "reason": reason,
        "issues": list(issues or []),
        "source": source,
        "timestamp": datetime.now().strftime("%d %b, %H:%M"),
    }
    filtered = [
        item
        for item in st.session_state.flagged_exercises
        if not (item["id"] == flagged_item["id"] and item["timestamp"] == flagged_item["timestamp"])
    ]
    st.session_state.flagged_exercises = [flagged_item, *filtered][:20]


def update_exercise_result(exercise_id: str, status: str) -> None:
    """Mettre a jour le statut d'un exercice apres verification."""
    for item in st.session_state.exercise_history:
        if item["id"] == exercise_id:
            item["status"] = status
            item["timestamp"] = datetime.now().strftime("%d %b, %H:%M")
            break


def clear_learning_session(preserve_settings: bool = True) -> None:
    """Reinitialiser la session pedagogique sans perdre le profil."""
    settings = deepcopy(st.session_state.settings)
    profile = deepcopy(st.session_state.student_profile)
    assignments = deepcopy(st.session_state.teacher_assignments)
    auth_state = deepcopy(st.session_state.auth)

    for key in [
        "chat_history",
        "active_tutoring_thread_id",
        "active_tutoring_thread_title",
        "current_exercise",
        "pending_assignment_trace_id",
        "current_answer",
        "last_verification",
        "exercise_history",
        "flagged_exercises",
        "generation_memory_bank",
        "notifications",
        "show_exercise_hint",
        "exercise_hint_level",
        "adaptive_hint_message",
        "selected_topic",
        "tutoring_state",
        "chat_use_exercise_context",
        "generator_level",
        "generator_section",
        "generator_topic",
        "generator_subtopic",
        "generator_difficulty",
        "generator_type",
        "teacher_assignment_section",
        "teacher_assignment_topic",
        "teacher_assignment_subtopic",
        "teacher_preview_exercise",
        "_analytics_page_markers",
    ]:
        if key in st.session_state:
            del st.session_state[key]

    initialize_session_state()
    st.session_state.student_profile = profile
    st.session_state.teacher_assignments = assignments
    st.session_state.auth = auth_state
    st.session_state.notifications = []
    if preserve_settings:
        st.session_state.settings = settings


def update_settings(new_settings: dict) -> None:
    """Mettre a jour les parametres de la session."""
    st.session_state.settings.update(new_settings)


def authenticate_user(user: dict, *, is_new_account: bool = False) -> None:
    """Placer l'utilisateur authentifie dans la session."""
    display_name = user["name"].strip() or user["email"].split("@")[0].replace(".", " ").title()
    role = user["role"]
    user_level = user.get("level", "Bac") or "Bac"

    profile = deepcopy(DEFAULT_STUDENT_PROFILE)
    profile["student_id"] = user.get("user_id", profile["student_id"])
    profile["name"] = display_name

    if role == "Enseignant":
        profile["avatar"] = "\U0001f9d1\U0001f3eb"
        profile["grade_band"] = "Enseignant"
        profile["level"] = "Corps enseignant"
        profile["section"] = "Corps enseignant"
        profile["current_focus"] = "Pilotage de cohorte"
    else:
        profile["avatar"] = "\U0001f9d1\U0001f3fd\u200d\U0001f393"
        profile["level"] = user_level
        profile["section"] = normalize_section_label(user.get("section", "Mathématiques"))
        profile["grade_band"] = _grade_band_from_level(user_level)
        profile["current_focus"] = "Pratique personnalisée"

    st.session_state.student_profile = profile
    st.session_state.auth = {
        "authenticated": True,
        "role": role,
        "user_id": user.get("user_id", ""),
        "email": user["email"].strip(),
        "display_name": display_name,
        "is_new_account": is_new_account,
    }


def sync_authenticated_user(user: dict) -> None:
    """Synchroniser les donnees d'identite du compte sans reinitialiser la session d'apprentissage."""
    display_name = user["name"].strip() or user["email"].split("@")[0].replace(".", " ").title()
    role = user["role"]

    st.session_state.auth.update(
        {
            "authenticated": True,
            "role": role,
            "user_id": user.get("user_id", st.session_state.auth.get("user_id", "")),
            "email": user["email"].strip(),
            "display_name": display_name,
        }
    )

    st.session_state.student_profile["student_id"] = user.get(
        "user_id",
        st.session_state.student_profile.get("student_id", ""),
    )
    st.session_state.student_profile["name"] = display_name
    if role == "Enseignant":
        st.session_state.student_profile["avatar"] = "\U0001f9d1\U0001f3eb"
        st.session_state.student_profile["grade_band"] = "Enseignant"
        st.session_state.student_profile["level"] = "Corps enseignant"
        st.session_state.student_profile["section"] = "Corps enseignant"
        if not st.session_state.student_profile.get("current_focus"):
            st.session_state.student_profile["current_focus"] = "Pilotage de cohorte"
    else:
        user_level = user.get("level", "Bac") or "Bac"
        st.session_state.student_profile["avatar"] = "\U0001f9d1\U0001f3fd\u200d\U0001f393"
        st.session_state.student_profile["level"] = user_level
        st.session_state.student_profile["section"] = normalize_section_label(
            user.get("section", st.session_state.student_profile.get("section", "Mathématiques"))
        )
        st.session_state.student_profile["grade_band"] = _grade_band_from_level(user_level)
        if st.session_state.student_profile.get("current_focus") == "Pilotage de cohorte":
            st.session_state.student_profile["current_focus"] = "Pratique personnalisée"


def logout_user() -> None:
    """Revenir a l'ecran d'accueil en conservant le theme choisi."""
    preserved_settings = deepcopy(st.session_state.settings)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()
    st.session_state.settings = preserved_settings
    st.session_state.auth = deepcopy(DEFAULT_AUTH_STATE)
    st.session_state.notifications = [
        {
            "message": "Déconnexion effectuée. Vous pouvez vous reconnecter depuis la page d'accueil.",
            "icon": "\U0001f44b",
        }
    ]


def require_authentication(page_name: str) -> None:
    """Bloquer l'acces a une page si l'utilisateur n'est pas connecte."""
    if st.session_state.auth["authenticated"]:
        return

    st.warning(f"La page « {page_name} » est accessible apres connexion ou creation de compte.")
    if st.button("Retour a l'accueil", key=f"go_welcome_{page_name.lower().replace(' ', '_')}"):
        switch_to_page("welcome")
    st.stop()


def is_teacher_account() -> bool:
    """Indiquer si le compte connecte est un compte enseignant."""
    return st.session_state.auth.get("role") == "Enseignant"


def require_teacher_access(page_name: str = "Panneau enseignant") -> None:
    """Interdire les fonctions enseignant aux comptes etudiants."""
    require_authentication(page_name)
    if is_teacher_account():
        return

    st.error("Acces refuse. Cette fonctionnalite est reservee aux comptes enseignant.")
    st.caption("Votre compte etudiant ne peut pas acceder au panneau enseignant ni aux fonctions de pilotage.")
    if st.button("Retourner au tableau de bord", key=f"go_student_dashboard_{page_name.lower().replace(' ', '_')}"):
        switch_to_page("student_dashboard")
    st.stop()


def _grade_band_from_level(level: str) -> str:
    """Associer un niveau a une mention plus lisible."""
    mapping = {
        "Bac": "Sections du baccalauréat",
    }
    return mapping.get(level, level)
