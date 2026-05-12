"""Page d'accueil et d'authentification de MathTutorAI."""

from __future__ import annotations

import streamlit as st

from frontend.components.auth_forms import render_auth_panels
from frontend.components.cards import render_highlight_card, render_section_header
from frontend.utils.constants import APP_NAME, WELCOME_HIGHLIGHTS, WELCOME_STATS
from frontend.utils.page_router import switch_to_page
from frontend.utils.session_manager import initialize_session_state, logout_user, push_notification


def _render_welcome_hero() -> None:
    """Afficher le bloc d'introduction public."""
    st.markdown(
        f"""
        <div class="welcome-stage">
            <h1>Apprenez les mathematiques avec un tuteur IA qui memorise votre maniere de raisonner.</h1>
            <p>
                {APP_NAME} reunit la generation d'exercices, le tutorat conversationnel,
                le suivi de progression, la memoire d'apprentissage et la verification symbolique
                dans une interface Streamlit claire et prete pour une integration backend.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page() -> None:
    """Afficher la page d'accueil avec connexion et creation de compte."""
    initialize_session_state()

    if st.session_state.auth["authenticated"]:
        st.markdown(
            f"""
            <div class="welcome-stage">
                <h1>Votre espace est pret.</h1>
                <p>
                    Vous etes connecte en tant que {st.session_state.auth["display_name"] or st.session_state.student_profile["name"]}.
                    Vous pouvez ouvrir votre espace de travail ou changer de compte.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        action_col, logout_col = st.columns(2, gap="medium")
        with action_col:
            if st.button("Ouvrir l'espace", type="primary"):
                destination = "teacher_panel" if st.session_state.auth["role"] == "Enseignant" else "student_dashboard"
                switch_to_page(destination)
        with logout_col:
            if st.button("Changer de compte"):
                logout_user()
                push_notification("Vous pouvez maintenant utiliser un autre compte.", "🔐")
                st.rerun()
        return

    lead_col, auth_col = st.columns([1.45, 1], gap="large")

    with lead_col:
        _render_welcome_hero()
        stats = st.columns(3, gap="medium")
        for column, stat in zip(stats, WELCOME_STATS):
            with column:
                st.metric(stat["label"], stat["value"])

        render_section_header(
            "Ce que propose la plateforme",
            "Un apercu rapide du parcours etudiant et des outils de pilotage enseignant.",
            "✨",
        )
        feature_cols = st.columns(3, gap="medium")
        for column, feature in zip(feature_cols, WELCOME_HIGHLIGHTS):
            with column:
                render_highlight_card(feature["title"], feature["description"], feature["footer"])

        render_section_header(
            "Fonctionnement",
            "Le projet est structure pour etre reellement exploitable, pas seulement demonstratif.",
            "🧩",
        )
        process_cols = st.columns(3, gap="medium")
        process_content = [
            (
                "1. Generer et resoudre",
                "L'etudiant demarre avec des exercices cibles selon le niveau, la section, le chapitre et la difficulte.",
                "Generateur d'exercices et espace de reponse",
            ),
            (
                "2. Etre guide",
                "Le tuteur conversationnel s'adapte avec des indices, des questions socratiques ou une explication d'erreur.",
                "Historique de discussion persistant",
            ),
            (
                "3. Mesurer et ameliorer",
                "L'etudiant et l'enseignant suivent la progression avec des tableaux de bord interactifs.",
                "Analyses Plotly et memoire pedagogique",
            ),
        ]
        for column, item in zip(process_cols, process_content):
            with column:
                render_highlight_card(item[0], item[1], item[2], accent="amber")

    with auth_col:
        render_section_header(
            "Commencer",
            "Connectez-vous, reinitialisez votre mot de passe ou creez un compte pour acceder a votre espace pedagogique personnalise.",
            "🔐",
        )
        destination = render_auth_panels()
        st.info(
            "Le compte est enregistre dans MongoDB local, la recuperation de mot de passe fonctionne en mode local par code temporaire, et le flux est pret pour une future API FastAPI."
        )
        st.markdown(
            """
            <div class="form-footnote">
                Conseil : un compte <strong>Enseignant</strong> ouvre directement le panneau enseignant,
                tandis qu'un compte <strong>Étudiant</strong> mene au tableau de bord apprenant.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if destination:
            action = "Compte cree avec succes." if st.session_state.auth["is_new_account"] else "Connexion reussie."
            push_notification(action, "✅")
            switch_to_page(destination)


render_page()
