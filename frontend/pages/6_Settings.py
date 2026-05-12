"""Page de gestion du compte utilisateur pour MathTutorAI."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from frontend.components.cards import render_highlight_card, render_section_header
from frontend.components.navbar import render_page_hero
from frontend.utils.dataset_catalog import get_sections
from frontend.utils.mongo_auth import (
    change_user_password,
    evaluate_password_strength,
    get_user_account,
    update_user_profile,
)
from frontend.utils.mongo_learning import record_page_consultation
from frontend.utils.session_manager import (
    clear_learning_session,
    initialize_session_state,
    logout_user,
    push_notification,
    require_authentication,
    sync_authenticated_user,
)


def render_page() -> None:
    """Afficher un espace de parametres recentre sur le compte personnel."""
    initialize_session_state()
    require_authentication("Parametres")
    record_page_consultation("settings", "Parametres")

    account = _load_account_snapshot()
    role = account.get("role", st.session_state.auth.get("role", "Etudiant"))
    is_teacher = role == "Enseignant"

    render_page_hero(
        "Mon compte",
        "Modifiez vos informations personnelles et la securite du compte. Les reglages techniques internes de l'application ne sont pas exposes ici.",
        badge=role,
    )

    render_section_header(
        "Parametres personnels",
        "Cette rubrique est reservee a la gestion de votre profil, de votre mot de passe et de votre session personnelle.",
        "👤",
    )

    intro_col, guard_col = st.columns([1.3, 1], gap="large")
    with intro_col:
        render_highlight_card(
            "Compte actif",
            f"{account.get('name', st.session_state.auth.get('display_name', 'Utilisateur'))} utilise actuellement l'application avec le role {role.lower()}.",
            f"Adresse associee : {account.get('email', st.session_state.auth.get('email', ''))}",
        )
    with guard_col:
        render_highlight_card(
            "Protection du systeme",
            "Les controles techniques du LLM, de la memoire, de la recuperation et du moteur interne ont ete retires de cette page.",
            "Aucune manipulation systeme n'est accessible depuis ce compte",
            accent="amber",
        )

    profile_tab, security_tab, session_tab = st.tabs(
        ["Profil personnel", "Securite", "Session personnelle"]
    )

    with profile_tab:
        _render_profile_tab(account, role=role, is_teacher=is_teacher)

    with security_tab:
        _render_security_tab(account)

    with session_tab:
        _render_session_tab()


def _render_profile_tab(account: dict, *, role: str, is_teacher: bool) -> None:
    """Render the editable account profile panel."""
    sections = get_sections()
    section_value = account.get("section", "")

    overview_cols = st.columns(4, gap="medium")
    with overview_cols[0]:
        st.metric("Role", role)
    with overview_cols[1]:
        st.metric("Niveau", "Corps enseignant" if is_teacher else (account.get("level", "Bac") or "Bac"))
    with overview_cols[2]:
        st.metric("Section", "Corps enseignant" if is_teacher else (section_value or "Non definie"))
    with overview_cols[3]:
        st.metric("Compte cree", _format_account_date(account.get("created_at")))

    render_section_header(
        "Informations modifiables",
        "Vous pouvez mettre a jour votre identite d'affichage et, pour un compte etudiant, la section de rattachement.",
        "🪪",
    )

    with st.form("profile_settings_form", clear_on_submit=False):
        name_input = st.text_input(
            "Nom complet",
            value=account.get("name", st.session_state.auth.get("display_name", "")),
            placeholder="Nom et prenom",
        )
        st.text_input(
            "Adresse e-mail",
            value=account.get("email", st.session_state.auth.get("email", "")),
            disabled=True,
            help="L'adresse e-mail reste la cle principale du compte et n'est pas modifiable depuis cette page.",
        )

        meta_col, level_col = st.columns(2, gap="medium")
        with meta_col:
            st.text_input("Type de compte", value=role, disabled=True)
        with level_col:
            st.text_input(
                "Niveau d'etude",
                value="Corps enseignant" if is_teacher else (account.get("level", "Bac") or "Bac"),
                disabled=True,
            )

        selected_section = ""
        if is_teacher:
            st.info("Les comptes enseignant ne renseignent pas de section eleve.")
        else:
            section_index = sections.index(section_value) if section_value in sections else 0
            selected_section = st.selectbox(
                "Section",
                sections,
                index=section_index,
                help="La section oriente les exercices et les recommandations pedagodiques.",
            )

        submitted = st.form_submit_button("Enregistrer mes informations", type="primary")

    if submitted:
        result = update_user_profile(
            email=account.get("email", ""),
            name=name_input,
            section=selected_section,
        )
        if result["ok"]:
            sync_authenticated_user(result["user"])
            push_notification("Profil personnel mis a jour.", "👤")
            st.success(result["message"])
            st.rerun()
        else:
            st.error(result["message"])
            for error in result.get("errors", []):
                st.caption(f"- {error}")

    render_highlight_card(
        "Lecture seule pour les donnees sensibles",
        "Le role du compte, l'adresse e-mail et les reglages techniques de l'application sont volontairement proteges depuis cette interface.",
        "Modification reservee aux donnees personnelles utiles a l'apprentissage",
    )


def _render_security_tab(account: dict) -> None:
    """Render the password-change panel."""
    render_section_header(
        "Mot de passe",
        "Renouvelez votre mot de passe sans toucher aux mecanismes internes de l'application.",
        "🔒",
    )

    with st.form("password_change_form", clear_on_submit=False):
        current_password = st.text_input(
            "Mot de passe actuel",
            type="password",
            placeholder="Saisissez votre mot de passe actuel",
        )
        new_password = st.text_input(
            "Nouveau mot de passe",
            type="password",
            placeholder="Choisissez un mot de passe fort",
        )
        confirm_password = st.text_input(
            "Confirmer le nouveau mot de passe",
            type="password",
            placeholder="Retapez le nouveau mot de passe",
        )

        strength = evaluate_password_strength(
            new_password,
            email=account.get("email", ""),
            name=account.get("name", ""),
        )
        _render_password_strength_panel(strength)

        submitted = st.form_submit_button("Mettre a jour le mot de passe", type="primary")

    if submitted:
        result = change_user_password(
            email=account.get("email", ""),
            current_password=current_password,
            new_password=new_password,
            confirm_password=confirm_password,
        )
        if result["ok"]:
            push_notification("Mot de passe mis a jour avec succes.", "🔐")
            st.success(result["message"])
            st.rerun()
        else:
            st.error(result["message"])
            for error in result.get("errors", []):
                st.caption(f"- {error}")

    last_password_update = account.get("password_updated_at")
    render_highlight_card(
        "Derniere mise a jour du mot de passe",
        _format_account_date(last_password_update, empty_label="Aucune date disponible"),
        "Les mots de passe sont stockes avec hachage PBKDF2 dans MongoDB local.",
        accent="amber",
    )


def _render_session_tab() -> None:
    """Render personal session actions only."""
    render_section_header(
        "Actions personnelles",
        "Ces commandes agissent uniquement sur votre session et vos donnees de travail locales.",
        "🧹",
    )

    render_highlight_card(
        "Aucune commande systeme",
        "Vous pouvez nettoyer votre session pedagogique ou vous deconnecter, mais vous n'avez pas acces aux parametres du moteur de generation, de la memoire ou de la recuperation.",
        "Le nettoyage concerne la session locale : exercice actif, chat courant, brouillons, indices et filtres temporaires",
        accent="amber",
    )

    clear_col, logout_col = st.columns(2, gap="medium")
    with clear_col:
        if st.button("Vider ma session d'apprentissage", type="primary"):
            clear_learning_session(preserve_settings=True)
            push_notification(
                "La session locale a ete reinitialisee. Votre compte et votre historique Mongo restent conserves.",
                "🧼",
            )
            st.rerun()
    with logout_col:
        if st.button("Se deconnecter"):
            logout_user()
            st.rerun()


def _load_account_snapshot() -> dict:
    """Load the freshest account state from Mongo, with session fallback."""
    result = get_user_account(st.session_state.auth.get("email", ""))
    if result["ok"]:
        account = result["account"]
        sync_authenticated_user(account)
        return account

    st.warning(
        "Le compte n'a pas pu etre relu depuis MongoDB. Les informations affichees proviennent de la session locale."
    )
    return {
        "user_id": st.session_state.auth.get("user_id", ""),
        "name": st.session_state.auth.get("display_name", "Utilisateur"),
        "email": st.session_state.auth.get("email", ""),
        "role": st.session_state.auth.get("role", "Etudiant"),
        "level": st.session_state.student_profile.get("level", "Bac"),
        "section": st.session_state.student_profile.get("section", ""),
        "created_at": None,
        "password_updated_at": None,
        "profile_updated_at": None,
    }


def _render_password_strength_panel(strength: dict) -> None:
    """Display a compact password-strength checklist."""
    if not strength:
        return

    progress_value = 0.0 if strength["max_score"] == 0 else strength["score"] / strength["max_score"]
    st.progress(
        progress_value,
        text=f"Solidite du nouveau mot de passe : {strength['label']} ({strength['score']}/{strength['max_score']})",
    )
    for requirement in strength["requirements"]:
        icon = "✅" if requirement["ok"] else "○"
        st.caption(f"{icon} {requirement['label']}")


def _format_account_date(value: datetime | None, *, empty_label: str = "Non disponible") -> str:
    """Format one account timestamp for the settings page."""
    if value is None:
        return empty_label
    return value.astimezone().strftime("%d/%m/%Y")


render_page()
