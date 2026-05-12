"""Point d'entree principal de l'application Streamlit MathTutorAI."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from frontend.components.navbar import (
    render_sidebar_auth_status,
    render_sidebar_brand,
    render_sidebar_guest_hint,
    render_sidebar_notifications,
    render_sidebar_profile,
)
from frontend.utils.constants import APP_ICON, APP_NAME, THEME_PRESETS
from frontend.utils.mongo_notifications import get_user_notifications
from frontend.utils.page_router import register_pages
from frontend.utils.session_manager import initialize_session_state, is_teacher_account, logout_user

BASE_DIR = Path(__file__).resolve().parent


def _inject_theme(theme_name: str) -> None:
    """Injecter les variables CSS du theme actif."""
    theme = THEME_PRESETS.get(theme_name, THEME_PRESETS["Ardoise ivoire"])
    variable_block = "\n".join([f"  --{key.replace('_', '-')}: {value};" for key, value in theme.items()])
    st.markdown(f"<style>:root {{\n{variable_block}\n}}</style>", unsafe_allow_html=True)


def _load_custom_css() -> None:
    """Charger la feuille de style globale."""
    css_path = BASE_DIR / "styles" / "custom.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _build_navigation(is_authenticated: bool, *, teacher_access: bool):
    """Construire une navigation adaptee au role du compte."""
    pages_dir = BASE_DIR / "pages"
    page_registry = {
        "welcome": st.Page(str(pages_dir / "0_Welcome.py"), title="Accueil", icon="✨"),
        "student_dashboard": st.Page(
            str(pages_dir / "1_Student_Dashboard.py"),
            title="Tableau de bord",
            icon="🏠",
            default=is_authenticated,
        ),
        "exercise_generator": st.Page(
            str(pages_dir / "2_Exercise_Generator.py"),
            title="Générateur d'exercices",
            icon="🧩",
        ),
        "tutoring_chat": st.Page(
            str(pages_dir / "3_Tutoring_Chat.py"),
            title="Tutorat conversationnel",
            icon="💬",
        ),
        "progress_tracking": st.Page(
            str(pages_dir / "4_Progress_Tracking.py"),
            title="Suivi de progression",
            icon="📈",
        ),
        "teacher_panel": st.Page(
            str(pages_dir / "5_Teacher_Panel.py"),
            title="Panneau enseignant",
            icon="🧑‍🏫",
        ),
        "settings": st.Page(
            str(pages_dir / "6_Settings.py"),
            title="Paramètres",
            icon="⚙️",
        ),
    }
    register_pages(page_registry)

    if not is_authenticated:
        return st.navigation([page_registry["welcome"]], position="sidebar")

    sections = {
        "Apprentissage": [
            page_registry["student_dashboard"],
            page_registry["exercise_generator"],
            page_registry["tutoring_chat"],
            page_registry["progress_tracking"],
        ],
        "Compte": [page_registry["settings"]],
    }

    if teacher_access:
        sections["Pilotage enseignant"] = [page_registry["teacher_panel"]]

    return st.navigation(sections, position="sidebar")


def main() -> None:
    """Lancer l'application Streamlit."""
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()
    _inject_theme(st.session_state.settings["theme"])
    _load_custom_css()
    navigation = _build_navigation(
        st.session_state.auth["authenticated"],
        teacher_access=is_teacher_account(),
    )

    render_sidebar_brand()
    if st.session_state.auth["authenticated"]:
        render_sidebar_auth_status(st.session_state.auth)
        render_sidebar_profile(st.session_state.student_profile)
        if st.sidebar.button("Se déconnecter", use_container_width=True):
            logout_user()
            st.rerun()
        st.sidebar.markdown("---")
        persisted_notifications = get_user_notifications(st.session_state.auth.get("email", ""), limit=8)
        render_sidebar_notifications([*persisted_notifications, *st.session_state.notifications])
    else:
        render_sidebar_guest_hint()
    navigation.run()
