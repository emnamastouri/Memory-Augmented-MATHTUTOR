"""Éléments partagés de navigation et d'en-tête."""

from __future__ import annotations

from html import escape

import streamlit as st

from frontend.utils.constants import APP_ICON, APP_NAME, APP_TAGLINE
from frontend.utils.mongo_notifications import mark_user_notification_read
from frontend.utils.page_router import switch_to_page


def render_sidebar_brand() -> None:
    """Afficher l'identité de l'application dans la barre latérale."""
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand__icon">{APP_ICON}</div>
            <div>
                <div class="sidebar-brand__title">{escape(APP_NAME)}</div>
                <div class="sidebar-brand__tagline">{escape(APP_TAGLINE)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_guest_hint() -> None:
    """Afficher une aide rapide pour les visiteurs non connectés."""
    st.sidebar.markdown(
        """
        <div class="glass-card sidebar-helper">
            <div class="card-title">Première étape</div>
            <div class="card-body">
                Connectez-vous ou créez un compte pour accéder au tutorat, au suivi de progression et aux outils enseignant.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_profile(profile: dict) -> None:
    """Afficher le profil actif dans la barre latérale."""
    st.sidebar.markdown(
        f"""
        <div class="glass-card sidebar-profile">
            <div class="sidebar-profile__avatar">{escape(profile["avatar"])}</div>
            <div class="sidebar-profile__meta">
                <div class="sidebar-profile__name">{escape(profile["name"])}</div>
                <div class="sidebar-profile__level">{escape(profile["grade_band"])} · {escape(profile["level"])}</div>
                <div class="sidebar-profile__focus">Focus actuel : {escape(profile["current_focus"])}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_auth_status(auth: dict) -> None:
    """Afficher le rôle du compte connecté."""
    st.sidebar.markdown(
        f"""
        <div class="sidebar-role-chip">
            Connecté en tant que {escape(auth["role"])}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_notifications(notifications: list[dict]) -> None:
    """Afficher les notifications sans surcharger l'interface."""
    with st.sidebar.expander("Notifications", expanded=False):
        if not notifications:
            st.caption("Aucune notification pour le moment.")
            return

        for index, notice in enumerate(notifications[:8], start=1):
            icon = str(notice.get("icon", "🔔")).strip() or "🔔"
            title = str(notice.get("title", "Notification")).strip() or "Notification"
            message = str(notice.get("message", "")).strip()
            timestamp = str(notice.get("timestamp", "")).strip()
            is_read = bool(notice.get("is_read", False))

            st.markdown(
                f"""
                <div class="notification-card">
                    <div><strong>{escape(icon)} {escape(title)}</strong></div>
                    <div>{escape(message)}</div>
                    <div class="card-caption">{escape(timestamp or ('Deja lu' if is_read else 'Nouveau'))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            action_page = str(notice.get("action_page", "")).strip()
            generation_trace_id = str(notice.get("generation_trace_id", "")).strip()
            if action_page or generation_trace_id:
                label = "Ouvrir l'exercice" if generation_trace_id else "Ouvrir"
                action_key = f"notice_action_{notice.get('id', index)}"
                if st.button(label, key=action_key, use_container_width=True):
                    if notice.get("id"):
                        mark_user_notification_read(str(notice["id"]))
                    if generation_trace_id:
                        st.session_state.pending_assignment_trace_id = generation_trace_id
                    switch_to_page(action_page or "exercise_generator")


def render_page_hero(title: str, subtitle: str, badge: str | None = None) -> None:
    """Afficher la bannière principale d'une page."""
    badge_markup = f'<span class="hero-banner__badge">{escape(badge)}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-banner__content">
                <div class="hero-banner__eyebrow">Espace MathTutorAI</div>
                <h1>{escape(title)}</h1>
                <p>{escape(subtitle)}</p>
            </div>
            <div class="hero-banner__meta">
                {badge_markup}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
