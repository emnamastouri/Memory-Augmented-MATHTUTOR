"""Page de suivi de progression pour MathTutorAI."""

from __future__ import annotations

import streamlit as st

from frontend.components.cards import render_highlight_card, render_section_header
from frontend.components.navbar import render_page_hero
from frontend.components.progress_charts import (
    build_mastery_evolution_chart,
    build_mastery_snapshot_radar,
    build_solved_exercises_chart,
    build_success_rate_chart,
    build_weak_topics_chart,
)
from frontend.utils.api_client import get_api_client
from frontend.utils.mongo_learning import record_page_consultation
from frontend.utils.session_manager import initialize_session_state, require_authentication


def render_page() -> None:
    """Afficher le tableau de suivi de progression."""
    initialize_session_state()
    require_authentication("Suivi de progression")
    record_page_consultation(
        "progress_tracking",
        "Suivi de progression",
        topic=st.session_state.student_profile.get("current_focus", ""),
    )
    api_client = get_api_client()
    analytics = api_client.get_progress_analytics(
        st.session_state.student_profile["student_id"],
        user_email=st.session_state.auth.get("email", ""),
        profile=st.session_state.student_profile,
    )
    metrics = analytics["metrics"]

    render_page_hero(
        "Suivi de progression",
        "Analysez l'evolution de maitrise, le volume d'exercices, les fragilites et les habitudes d'etude enregistrees dans MongoDB.",
        badge="Analyses Mongo + Plotly",
    )

    render_section_header(
        "Pilotage des progres",
        "Tableaux de bord interactifs alimentes par vos generations, verifications, indices et discussions avec le tuteur.",
        "📈",
    )

    metric_cols = st.columns(5, gap="medium")
    with metric_cols[0]:
        st.metric("Maitrise moyenne", f"{metrics['mastery_average']}%", _format_delta(metrics["mastery_delta"], "pts"))
    with metric_cols[1]:
        st.metric("Exercices resolus", str(metrics["solved_exercises"]), _format_delta(metrics["solved_delta"], "cette semaine"))
    with metric_cols[2]:
        st.metric("Taux de reussite", f"{metrics['success_rate_pct']}%", _format_delta(metrics["success_rate_delta"], "pts"))
    with metric_cols[3]:
        st.metric("Temps d'etude", f"{metrics['study_hours']:.1f} h", _format_hours_delta(metrics["study_hours_delta"]))
    with metric_cols[4]:
        st.metric("Themes a risque", str(metrics["at_risk_topics"]), f"{metrics['topics_studied']} theme(s) etudie(s)")

    if analytics.get("data_source") != "mongo":
        st.info(
            "Aucune activite persistante n'a encore ete retrouvee pour ce compte. "
            "Les courbes se rempliront automatiquement a partir des prochains exercices, indices, verifications et echanges tutoraux."
        )

    overview_tab, diagnostics_tab = st.tabs(["Vue d'ensemble", "Diagnostic par theme"])

    with overview_tab:
        chart_col_1, chart_col_2 = st.columns(2, gap="large")
        with chart_col_1:
            render_section_header("Evolution de maitrise", "Variation de la maitrise estimee sur les notions les plus travaillees.", "📉")
            st.plotly_chart(
                build_mastery_evolution_chart(analytics["mastery_evolution"]),
                use_container_width=True,
            )
        with chart_col_2:
            render_section_header("Volume d'exercices", "Nombre d'exercices resolus ou travailles par domaine.", "📚")
            st.plotly_chart(
                build_solved_exercises_chart(analytics["solved_exercises"]),
                use_container_width=True,
            )

        chart_col_3, chart_col_4 = st.columns(2, gap="large")
        with chart_col_3:
            render_section_header("Notions fragiles", "Chapitres a renforcer d'apres les resultats et l'usage des indices.", "🧭")
            st.plotly_chart(
                build_weak_topics_chart(analytics["weak_topics"]),
                use_container_width=True,
            )
        with chart_col_4:
            render_section_header("Repartition des resultats", "Poids des reponses justes, fausses ou aidees par indice.", "🎯")
            st.plotly_chart(
                build_success_rate_chart(analytics["success_rate"]),
                use_container_width=True,
            )

    with diagnostics_tab:
        radar_col, mastery_col = st.columns([1.15, 1], gap="large")
        with radar_col:
            render_section_header("Equilibre des competences", "Comparer d'un seul regard les notions les plus travaillees.", "🕸️")
            st.plotly_chart(
                build_mastery_snapshot_radar(analytics["mastery_snapshot"]),
                use_container_width=True,
            )
        with mastery_col:
            render_section_header("Notes d'intervention", "Constats derives de vos interactions persistantes.", "📝")
            for note in analytics.get("intervention_notes", []):
                render_highlight_card(
                    note.get("title", "Note"),
                    note.get("body", ""),
                    note.get("footer", ""),
                    accent=note.get("accent", "teal"),
                )
            for topic, mastery in analytics["mastery_snapshot"].items():
                st.markdown(f"**{topic}**")
                st.progress(mastery / 100, text=f"{mastery}%")


def _format_delta(value: int | float, suffix: str) -> str:
    """Format metric deltas with signs."""
    rounded = round(value)
    sign = "+" if rounded > 0 else ""
    return f"{sign}{rounded} {suffix}".strip()


def _format_hours_delta(value: int | float) -> str:
    """Format study-hours delta text."""
    rounded = round(float(value), 1)
    sign = "+" if rounded > 0 else ""
    return f"{sign}{rounded} h sur 7 jours"


render_page()
