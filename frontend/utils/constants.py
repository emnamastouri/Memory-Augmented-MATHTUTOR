"""Configuration statique et donnees initiales pour MathTutorAI."""

from __future__ import annotations

import os
from typing import Any


def _read_streamlit_secret(section: str, key: str) -> Any:
    """Read one optional Streamlit secret without failing in CLI/test mode."""
    try:
        import streamlit as st

        raw_section = st.secrets.get(section, {})
        if isinstance(raw_section, dict):
            return raw_section.get(key)
        return getattr(raw_section, key, None)
    except Exception:
        return None


def _runtime_setting(section: str, key: str, env_name: str, default: str) -> str:
    """Resolve a setting from environment, then Streamlit secrets, then fallback."""
    env_value = os.getenv(env_name)
    if env_value:
        return env_value.strip()
    secret_value = _read_streamlit_secret(section, key)
    if secret_value:
        return str(secret_value).strip()
    return default

APP_NAME = "MathTutorAI"
APP_ICON = "\U0001f9e0"
APP_TAGLINE = "Tutorat mathematique augmente par la memoire"

MONGO_URI = _runtime_setting("mongo", "uri", "MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = _runtime_setting("mongo", "db_name", "MONGO_DB_NAME", "mathtutorai")
MONGO_USERS_COLLECTION = "utilisateurs"
MONGO_LEARNING_EVENTS_COLLECTION = "learning_events"
MONGO_EXERCISE_RECORDS_COLLECTION = "exercise_records"
MONGO_TUTORING_THREADS_COLLECTION = "tutoring_threads"
MONGO_TEACHER_GROUPS_COLLECTION = "teacher_groups"
MONGO_GROUP_ASSIGNMENTS_COLLECTION = "group_assignments"
MONGO_USER_NOTIFICATIONS_COLLECTION = "user_notifications"

AUTH_ROLES = ["Étudiant", "Enseignant"]
LEVELS = ["Bac"]
DIFFICULTIES = ["Fondamental", "Intermédiaire", "Avancé", "Défi"]
DEFAULT_EXERCISE_DIFFICULTY = "Intermédiaire"
EXERCISE_TYPES = ["Exercice problème", "QCM"]

TUTORING_MODES = {
    "Socratique": "Guider l'eleve avec des questions progressives et des points de controle.",
    "Mode indice": "Donner des indices courts sans reveler toute la solution.",
    "Explication d'erreur": "Expliquer les erreurs frequentes et leur origine mathematique.",
}

THEME_PRESETS = {
    "Tableau nocturne": {
        "bg_main": "#07111f",
        "bg_panel": "rgba(11, 22, 36, 0.88)",
        "bg_soft": "rgba(17, 30, 49, 0.95)",
        "bg_gradient_start": "#07111f",
        "bg_gradient_end": "#020811",
        "glow_primary": "rgba(47, 181, 169, 0.16)",
        "glow_secondary": "rgba(255, 200, 87, 0.14)",
        "sidebar_gradient_end": "#09131f",
        "text_primary": "#f5f7fb",
        "text_secondary": "#9eb2ca",
        "accent": "#2fb5a9",
        "accent_alt": "#ffc857",
        "success": "#39d98a",
        "danger": "#ff7a7a",
        "border": "rgba(255, 255, 255, 0.08)",
        "shadow": "0 24px 60px rgba(0, 0, 0, 0.28)",
    },
    "Ocean graphite": {
        "bg_main": "#08131a",
        "bg_panel": "rgba(13, 25, 32, 0.9)",
        "bg_soft": "rgba(20, 33, 41, 0.96)",
        "bg_gradient_start": "#08131a",
        "bg_gradient_end": "#051017",
        "glow_primary": "rgba(73, 166, 255, 0.15)",
        "glow_secondary": "rgba(126, 224, 195, 0.14)",
        "sidebar_gradient_end": "#0c161f",
        "text_primary": "#eff6f7",
        "text_secondary": "#97aeb5",
        "accent": "#49a6ff",
        "accent_alt": "#7ee0c3",
        "success": "#42d392",
        "danger": "#ff826e",
        "border": "rgba(255, 255, 255, 0.09)",
        "shadow": "0 24px 60px rgba(0, 0, 0, 0.26)",
    },
    "Ardoise ivoire": {
        "bg_main": "#edf1f5",
        "bg_panel": "rgba(255, 255, 255, 0.92)",
        "bg_soft": "rgba(245, 248, 252, 0.98)",
        "bg_gradient_start": "#f5f7fb",
        "bg_gradient_end": "#e7edf5",
        "glow_primary": "rgba(15, 139, 141, 0.13)",
        "glow_secondary": "rgba(255, 159, 28, 0.12)",
        "sidebar_gradient_end": "#e8eef6",
        "text_primary": "#172233",
        "text_secondary": "#5b6b80",
        "accent": "#0f8b8d",
        "accent_alt": "#ff9f1c",
        "success": "#2d9d78",
        "danger": "#d1495b",
        "border": "rgba(22, 34, 51, 0.08)",
        "shadow": "0 24px 60px rgba(30, 48, 71, 0.12)",
    },
}

DEFAULT_STUDENT_PROFILE = {
    "student_id": "STD-024",
    "name": "Lina Haddad",
    "avatar": "\U0001f9d1\U0001f3fd\u200d\U0001f393",
    "level": "Bac",
    "grade_band": "Sections du baccalauréat",
    "section": "Mathématiques",
    "weekly_goal_hours": 6,
    "weekly_goal_progress": 4.5,
    "streak_days": 11,
    "mastery_score": 78,
    "memory_health": 84,
    "current_focus": "Suites numériques",
    "preferred_mode": "Socratique",
    "strong_topics": ["Fonction exponentielle", "Probabilités conditionnelles", "Suites numériques"],
    "weak_topics": ["Intégrales et primitives", "Fonction logarithme népérien", "Nombres complexes"],
}

DEFAULT_CHAT_HISTORY = [
    {
        "role": "assistant",
        "content": (
            "Bienvenue. Je me souviens de ton dernier travail en mathematiques, "
            "nous pouvons reprendre avec des indices cibles, un accompagnement guidé ou une analyse d'erreurs."
        ),
        "mode": "Socratique",
    }
]

DEFAULT_SETTINGS = {
    "theme": "Ardoise ivoire",
    "llm_backend": "llama.cpp",
    "model_name": "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "context_window": 4096,
    "temperature": 0.35,
    "max_tokens": 512,
    "stream_responses": True,
    "retrieval_top_k": 5,
    "retrieval_strategy": "Memoire + recherche vectorielle",
    "similarity_threshold": 0.72,
    "memory_window": 8,
    "enable_long_term_memory": True,
    "auto_profile_updates": True,
    "retain_incorrect_attempts": True,
}

DEFAULT_AUTH_STATE = {
    "authenticated": False,
    "role": "Étudiant",
    "user_id": "",
    "email": "",
    "display_name": "",
    "is_new_account": False,
}

WELCOME_STATS = [
    {"label": "Generation d'exercices", "value": "Adaptative"},
    {"label": "Tutorat", "value": "Memoire active"},
    {"label": "Verification", "value": "SymPy"},
]

WELCOME_HIGHLIGHTS = [
    {
        "title": "Exercices adaptes",
        "description": "Creer une pratique ciblee selon le niveau, le chapitre, la difficulte et le format demande.",
        "footer": "Pret pour une API FastAPI",
    },
    {
        "title": "Tutorat conversationnel",
        "description": "Passer d'un mode socratique a des indices ou a une explication d'erreur en conservant l'historique.",
        "footer": "Contexte de session persistant",
    },
    {
        "title": "Memoire et validation",
        "description": "Combiner memoire d'apprentissage, LLM local, FAISS et validation symbolique des reponses.",
        "footer": "Architecture prete pour un projet academique",
    },
]

DEFAULT_TEACHER_ASSIGNMENTS = [
    {
        "student": "Lina Haddad",
        "section": "Mathématiques",
        "topic": "Analyse",
        "subtopic": "intégrales, primitives, aires et volumes",
        "due_date": "2026-05-10",
        "status": "Assigné",
    },
    {
        "student": "Adam Ben Salah",
        "section": "Sciences expérimentales",
        "topic": "Probabilités",
        "subtopic": "loi binomiale et schéma de Bernoulli",
        "due_date": "2026-05-11",
        "status": "En révision",
    },
]

DEFAULT_NOTIFICATIONS = [
    {
        "message": "Nouvelle recommandation : reviser les limites avant le prochain quiz de calcul.",
        "icon": "\U0001f514",
    }
]
