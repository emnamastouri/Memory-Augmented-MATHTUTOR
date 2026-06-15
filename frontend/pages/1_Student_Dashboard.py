"""Tableau de bord etudiant pour MathTutorAI."""

from __future__ import annotations

import streamlit as st

from frontend.components.cards import (
    render_highlight_card,
    render_profile_summary,
    render_recommendation_card,
    render_section_header,
    render_topic_cluster,
)
from frontend.components.navbar import render_page_hero
from frontend.utils.api_client import get_api_client
from frontend.utils.mongo_learning import record_page_consultation
from frontend.utils.page_router import switch_to_page
from frontend.utils.session_manager import initialize_session_state, require_authentication


def render_page() -> None:
    """Afficher le tableau de bord etudiant."""
    initialize_session_state()
    require_authentication("Tableau de bord")
    record_page_consultation(
        "student_dashboard",
        "Tableau de bord",
        topic=st.session_state.student_profile.get("current_focus", ""),
    )
    api_client = get_api_client()
    dashboard = api_client.get_student_dashboard(
        st.session_state.student_profile["student_id"],
        user_email=st.session_state.auth.get("email", ""),
        profile=st.session_state.student_profile,
    )
    profile = dashboard["profile"]
    st.session_state.student_profile.update(profile)
    metrics = dashboard.get("analytics", {}).get("metrics", {})
    intervention_notes = dashboard.get("analytics", {}).get("intervention_notes", [])

    render_page_hero(
        "Tableau de bord etudiant",
        "Suivez vos recommandations, vos points forts, vos fragilites et la prochaine meilleure action d'apprentissage.",
        badge=profile["current_focus"],
    )

    lead_col, profile_col = st.columns([1.8, 1], gap="large")
    with lead_col:
        render_section_header(
            "Bon retour",
            "Votre espace met en avant les priorites pedagogiques issues de votre memoire d'apprentissage persistante.",
            "🎯",
        )
        action_col, focus_col = st.columns([1, 1], gap="medium")
        with action_col:
            if st.button("Continuer l'apprentissage", type="primary"):
                switch_to_page("exercise_generator")
        with focus_col:
            top_recommendation = dashboard["recommendations"][0] if dashboard["recommendations"] else None
            render_highlight_card(
                "Recommandation du moment",
                (top_recommendation or {}).get("description", "Commencez par generer un exercice cible puis discutez-le avec le tuteur."),
                (top_recommendation or {}).get("action", "Meilleure prochaine etape pour aujourd'hui"),
            )
    with profile_col:
        render_profile_summary(profile)

    metric_cols = st.columns(4, gap="medium")
    with metric_cols[0]:
        st.metric("Score de maitrise", f"{profile['mastery_score']}%", _format_delta(metrics.get("mastery_delta", 0), "pts"))
    with metric_cols[1]:
        st.metric("Serie d'etude", f"{profile['streak_days']} jours", _streak_caption(profile["streak_days"]))
    with metric_cols[2]:
        st.metric("Sante memoire", f"{profile['memory_health']}%", _memory_caption(profile["memory_health"]))
    with metric_cols[3]:
        st.metric(
            "Objectif hebdomadaire",
            f"{profile['weekly_goal_progress']}/{profile['weekly_goal_hours']} h",
            _weekly_goal_caption(profile["weekly_goal_progress"], profile["weekly_goal_hours"]),
        )

    overview_col, recommendations_col = st.columns([1.2, 1], gap="large")
    with overview_col:
        render_section_header("Signaux par theme", "Visualisez vos zones de confort et les notions a renforcer.", "📚")
        strength_col, weakness_col = st.columns(2, gap="medium")
        with strength_col:
            render_topic_cluster("Points forts", dashboard["strong_topics"], accent="teal")
        with weakness_col:
            render_topic_cluster("Points a renforcer", dashboard["weak_topics"], accent="amber")

        render_section_header("Progression de maitrise", "Estimation actuelle de votre niveau sur les grands domaines.", "📈")
        for item in dashboard["mastery_progress"]:
            left_col, right_col = st.columns([1, 3], gap="small")
            with left_col:
                st.markdown(f"**{item['topic']}**")
            with right_col:
                st.progress(item["value"] / 100, text=f"{item['value']}%")

        render_section_header("Exercices recents", "Historique recent de pratique et de revision.", "📝")
        for item in dashboard["recent_exercises"]:
            with st.expander(f"{item['title']} · {item['topic']}", expanded=False):
                st.markdown(f"**Statut :** {item.get('status', item.get('score', 'En cours'))}")
                st.caption(item.get("timestamp", item.get("time", "A l'instant")))

    with recommendations_col:
        render_section_header("Recommandations personnalisees", "Actions suggerees selon votre memoire et votre niveau.", "🧠")
        for recommendation in dashboard["recommendations"]:
            render_recommendation_card(recommendation)

        render_section_header("Habitudes d'apprentissage", "Observations utiles pour progresser plus regulierement.", "🌱")
        if intervention_notes:
            for note in intervention_notes[:2]:
                render_highlight_card(
                    note.get("title", "Observation"),
                    note.get("body", ""),
                    note.get("footer", ""),
                    accent=note.get("accent", "teal"),
                )
        else:
            render_highlight_card(
                "Premiers signaux en attente",
                "Generez un exercice, verifiez une reponse et echangez avec le tuteur pour alimenter ce bloc automatiquement.",
                "Les habitudes d'apprentissage apparaitront ici",
            )


def _format_delta(value: int | float, suffix: str) -> str:
    """Format one metric delta with sign."""
    rounded = round(float(value))
    sign = "+" if rounded > 0 else ""
    return f"{sign}{rounded} {suffix}".strip()


def _streak_caption(streak_days: int) -> str:
    """Describe the current streak for the metric card."""
    if streak_days <= 0:
        return "Nouvelle serie a lancer"
    if streak_days == 1:
        return "1 jour consecutif"
    return f"{streak_days} jours consecutifs"


def _memory_caption(memory_health: int) -> str:
    """Describe the memory-health score in a short subtitle."""
    if memory_health >= 80:
        return "Retention solide"
    if memory_health >= 60:
        return "A consolider"
    return "Renforcement conseille"


def _weekly_goal_caption(progress_hours: float, goal_hours: float) -> str:
    """Describe weekly-goal completion."""
    if not goal_hours:
        return "Objectif non defini"
    ratio = min(1.0, float(progress_hours) / float(goal_hours))
    return f"{round(ratio * 100)}% atteint"


render_page()
