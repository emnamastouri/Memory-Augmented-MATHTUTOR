"""Composants visuels réutilisables pour l'interface Streamlit."""

from __future__ import annotations

from html import escape

import streamlit as st


def render_section_header(title: str, description: str, icon: str = "") -> None:
    """Afficher un en-tête de section léger."""
    label = f"{icon} {title}".strip()
    st.markdown(
        f"""
        <div class="section-header">
            <h3>{escape(label)}</h3>
            <p>{escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_highlight_card(title: str, body: str, footer: str | None = None, accent: str = "teal") -> None:
    """Afficher une carte de mise en avant."""
    footer_markup = f'<div class="card-footer">{escape(footer)}</div>' if footer else ""
    st.markdown(
        f"""
        <div class="glass-card accent-{escape(accent)}">
            <div class="card-title">{escape(title)}</div>
            <div class="card-body">{escape(body)}</div>
            {footer_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_topic_cluster(title: str, topics: list[dict], accent: str = "teal") -> None:
    """Afficher un groupe de points forts ou de fragilités."""
    topic_items = "".join(
        [
            (
                '<div class="topic-row">'
                f'<div><div class="topic-row__name">{escape(item["topic"])}</div>'
                f'<div class="topic-row__meta">Maîtrise {item["mastery"]}%</div></div>'
                f'<div class="topic-row__trend">{escape(item["trend"])}</div>'
                "</div>"
            )
            for item in topics
        ]
    )
    st.markdown(
        f"""
        <div class="glass-card accent-{escape(accent)}">
            <div class="card-title">{escape(title)}</div>
            {topic_items}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_card(recommendation: dict) -> None:
    """Afficher une recommandation personnalisée."""
    st.markdown(
        f"""
        <div class="glass-card recommendation-card">
            <div class="card-title">{escape(recommendation["title"])}</div>
            <div class="card-body">{escape(recommendation["description"])}</div>
            <div class="recommendation-action">{escape(recommendation["action"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_profile_summary(profile: dict) -> None:
    """Afficher un résumé compact du profil."""
    st.markdown(
        f"""
        <div class="glass-card profile-summary">
            <div class="profile-summary__avatar">{escape(profile["avatar"])}</div>
            <div>
                <div class="card-title">{escape(profile["name"])}</div>
                <div class="card-body">Section : {escape(profile["section"])}</div>
                <div class="profile-summary__meta">{escape(profile["grade_band"])} · Objectif hebdomadaire {profile["weekly_goal_hours"]}h</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
