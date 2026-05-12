"""Composants de connexion, reinitialisation et creation de compte."""

from __future__ import annotations

import streamlit as st

from frontend.utils.constants import AUTH_ROLES, LEVELS
from frontend.utils.dataset_catalog import get_default_section, get_sections
from frontend.utils.mongo_learning import record_auth_success
from frontend.utils.mongo_auth import (
    evaluate_password_strength,
    login_user,
    register_user,
    request_password_reset,
    reset_password,
    validate_registration_data,
)
from frontend.utils.session_manager import authenticate_user, push_notification


def render_auth_panels() -> str | None:
    """Afficher les formulaires d'authentification et retourner la page cible."""
    _ensure_auth_defaults()
    login_tab, register_tab = st.tabs(["Connexion", "Creer un compte"])
    destination = None

    with login_tab:
        destination = _render_login_tab()

    with register_tab:
        if not destination:
            destination = _render_register_tab()

    return destination


def _render_login_tab() -> str | None:
    """Afficher la connexion et la recuperation de mot de passe."""
    destination = None

    with st.form("login_form", clear_on_submit=False):
        st.text_input("Adresse e-mail", key="login_email", placeholder="etudiant@mathtutor.ai")
        st.text_input("Mot de passe", key="login_password", type="password", placeholder="Entrez votre mot de passe")
        submitted = st.form_submit_button("Se connecter", type="primary")

    st.caption("Connexion reliee au serveur MongoDB local : mongodb://localhost:27017")
    if submitted:
        if not st.session_state.login_email.strip() or not st.session_state.login_password.strip():
            st.error("Veuillez saisir l'adresse e-mail et le mot de passe.")
        else:
            result = login_user(st.session_state.login_email, st.session_state.login_password)
            if not result["ok"]:
                st.error(result["message"])
            else:
                authenticate_user(result["user"], is_new_account=False)
                record_auth_success("login")
                destination = _destination_for_role(result["user"]["role"])

    with st.expander("Mot de passe oublie ?", expanded=False):
        _render_password_reset_panel()

    return destination


def _render_password_reset_panel() -> None:
    """Afficher le flux local de reinitialisation du mot de passe."""
    st.caption(
        "En mode local, un code temporaire est genere dans cette interface. En production, il sera remplace par un envoi e-mail via l'API."
    )

    identity_cols = st.columns(2, gap="medium")
    with identity_cols[0]:
        st.text_input(
            "Adresse e-mail du compte",
            key="reset_request_email",
            placeholder="etudiant@mathtutor.ai",
        )
    with identity_cols[1]:
        st.text_input(
            "Nom complet du compte",
            key="reset_request_name",
            placeholder="Amina Rahal",
        )

    if st.button("Generer un code de reinitialisation", key="request_reset_button"):
        result = request_password_reset(
            st.session_state.reset_request_email,
            st.session_state.reset_request_name,
        )
        if result["ok"]:
            st.session_state.reset_feedback = {"kind": "success", "message": result["message"]}
            dev_token = result.get("dev_token")
            if dev_token:
                st.session_state.reset_dev_token = dev_token
                st.session_state.reset_email_confirm = st.session_state.reset_request_email.strip().lower()
                st.session_state.reset_name_reference = st.session_state.reset_request_name.strip()
                push_notification("Code de reinitialisation genere pour la session locale.", "🔑")
        else:
            st.session_state.reset_feedback = {"kind": "error", "message": result["message"]}

    feedback = st.session_state.get("reset_feedback")
    if feedback:
        if feedback["kind"] == "success":
            st.success(feedback["message"])
        else:
            st.error(feedback["message"])

    if st.session_state.get("reset_dev_token"):
        st.info(
            "Code temporaire local : "
            f"`{st.session_state.reset_dev_token}`. "
            "Conservez-le pendant 15 minutes pour finaliser la reinitialisation."
        )

    st.divider()
    st.markdown("**Finaliser la reinitialisation**")
    st.text_input(
        "Adresse e-mail a reinitialiser",
        key="reset_email_confirm",
        placeholder="etudiant@mathtutor.ai",
    )
    st.text_input(
        "Code temporaire",
        key="reset_token_input",
        placeholder="AB12CD34",
    )
    st.text_input(
        "Nouveau mot de passe",
        key="reset_new_password",
        type="password",
        placeholder="Choisissez un mot de passe fort",
    )
    st.text_input(
        "Confirmer le nouveau mot de passe",
        key="reset_new_password_confirm",
        type="password",
        placeholder="Retapez le nouveau mot de passe",
    )

    strength = evaluate_password_strength(
        st.session_state.reset_new_password,
        email=st.session_state.reset_email_confirm,
        name=st.session_state.get("reset_name_reference", ""),
    )
    _render_password_strength_panel(strength, title="Solidite du nouveau mot de passe")

    reset_disabled = not (
        st.session_state.reset_email_confirm.strip()
        and st.session_state.reset_token_input.strip()
        and st.session_state.reset_new_password
        and st.session_state.reset_new_password_confirm
    )
    if st.button(
        "Mettre a jour le mot de passe",
        type="primary",
        key="confirm_reset_button",
        disabled=reset_disabled,
    ):
        result = reset_password(
            email=st.session_state.reset_email_confirm,
            reset_token=st.session_state.reset_token_input,
            new_password=st.session_state.reset_new_password,
            confirm_password=st.session_state.reset_new_password_confirm,
        )
        if result["ok"]:
            st.success(result["message"])
            st.session_state.login_prefill_email = st.session_state.reset_email_confirm.strip().lower()
            st.session_state.reset_clear_pending = True
            push_notification("Mot de passe reinitialise avec succes.", "✅")
            st.rerun()
        else:
            st.error(result["message"])
            for error in result.get("errors", []):
                st.caption(f"• {error}")


def _render_register_tab() -> str | None:
    """Afficher un parcours de creation de compte avec validation forte."""
    sections = get_sections()
    destination = None

    identity_cols = st.columns(2, gap="medium")
    with identity_cols[0]:
        st.text_input("Nom complet", key="register_name", placeholder="Amina Rahal")
    with identity_cols[1]:
        st.text_input("Adresse e-mail", key="register_email", placeholder="amina@example.com")

    role_col, details_col = st.columns([1, 2], gap="medium")
    with role_col:
        st.selectbox("Type de compte", AUTH_ROLES, key="register_role")

    is_teacher_registration = st.session_state.register_role == "Enseignant"
    selected_level = st.session_state.register_level
    selected_section = st.session_state.register_section

    with details_col:
        if is_teacher_registration:
            st.info("Pour un compte enseignant, il n'est pas necessaire de renseigner le niveau d'etude ni la section.")
            selected_level = ""
            selected_section = ""
        else:
            st.caption("Le niveau actuellement disponible est le bac. Les autres niveaux sont en cours de construction.")
            level_col, section_col = st.columns(2, gap="medium")
            with level_col:
                st.selectbox(
                    "Niveau d'etude",
                    LEVELS,
                    key="register_level",
                    help="Les autres niveaux seront ajoutes prochainement.",
                )
                selected_level = st.session_state.register_level
            with section_col:
                st.selectbox("Section", sections, key="register_section")
                selected_section = st.session_state.register_section

    password_cols = st.columns(2, gap="medium")
    with password_cols[0]:
        st.text_input(
            "Creer un mot de passe",
            key="register_password",
            type="password",
            placeholder="Minimum 10 caracteres",
        )
    with password_cols[1]:
        st.text_input(
            "Confirmer le mot de passe",
            key="register_password_confirm",
            type="password",
            placeholder="Retapez le mot de passe",
        )

    validation = validate_registration_data(
        name=st.session_state.register_name,
        email=st.session_state.register_email,
        password=st.session_state.register_password,
        confirm_password=st.session_state.register_password_confirm,
        role=st.session_state.register_role,
        level=selected_level,
        section=selected_section,
    )
    _render_password_strength_panel(validation["password_strength"], title="Solidite du mot de passe")

    if any(
        [
            st.session_state.register_name,
            st.session_state.register_email,
            st.session_state.register_password,
            st.session_state.register_password_confirm,
        ]
    ):
        if validation["ok"]:
            st.success("Le formulaire est complet et respecte les exigences de securite.")
        else:
            st.warning("Le compte ne peut pas encore etre cree. Corrigez les points suivants :")
            for error in validation["errors"]:
                st.caption(f"• {error}")

    st.checkbox(
        "Je confirme que ces informations sont exactes et que ce compte sera utilise dans un cadre pedagogique.",
        key="register_acknowledge",
    )

    can_submit = validation["ok"] and st.session_state.register_acknowledge
    if st.button("Creer le compte", type="primary", key="create_account_button", disabled=not can_submit):
        result = register_user(
            name=st.session_state.register_name,
            email=st.session_state.register_email,
            password=st.session_state.register_password,
            role=st.session_state.register_role,
            level=selected_level,
            section=selected_section,
        )
        if not result["ok"]:
            st.error(result["message"])
            for error in result.get("errors", []):
                st.caption(f"• {error}")
        else:
            authenticate_user(result["user"], is_new_account=True)
            record_auth_success("register")
            destination = _destination_for_role(result["user"]["role"])

    return destination


def _render_password_strength_panel(strength: dict, *, title: str) -> None:
    """Afficher un indicateur clair de force du mot de passe."""
    if not strength:
        return

    progress_value = 0.0 if strength["max_score"] == 0 else strength["score"] / strength["max_score"]
    st.progress(progress_value, text=f"{title} : {strength['label']} ({strength['score']}/{strength['max_score']})")
    for requirement in strength["requirements"]:
        icon = "✅" if requirement["ok"] else "○"
        st.caption(f"{icon} {requirement['label']}")


def _ensure_auth_defaults() -> None:
    """Initialiser les cles de session utilisees par les formulaires."""
    sections = get_sections()
    defaults = {
        "login_email": "",
        "login_password": "",
        "register_name": "",
        "register_email": "",
        "register_password": "",
        "register_password_confirm": "",
        "register_role": AUTH_ROLES[0],
        "register_level": LEVELS[0],
        "register_section": get_default_section(),
        "register_acknowledge": False,
        "reset_request_email": "",
        "reset_request_name": "",
        "reset_email_confirm": "",
        "reset_token_input": "",
        "reset_new_password": "",
        "reset_new_password_confirm": "",
        "reset_dev_token": "",
        "reset_feedback": None,
        "reset_name_reference": "",
        "reset_clear_pending": False,
        "login_prefill_email": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.login_prefill_email:
        st.session_state.login_email = st.session_state.login_prefill_email
        st.session_state.login_password = ""
        st.session_state.login_prefill_email = ""

    if st.session_state.reset_clear_pending:
        _clear_reset_state()
        st.session_state.reset_clear_pending = False

    if st.session_state.register_section not in sections:
        st.session_state.register_section = get_default_section()


def _clear_reset_state() -> None:
    """Nettoyer les champs lies a la reinitialisation du mot de passe."""
    for key in [
        "reset_request_email",
        "reset_request_name",
        "reset_email_confirm",
        "reset_token_input",
        "reset_new_password",
        "reset_new_password_confirm",
        "reset_dev_token",
        "reset_feedback",
        "reset_name_reference",
    ]:
        st.session_state[key] = "" if key != "reset_feedback" else None


def _destination_for_role(role: str) -> str:
    """Choisir la premiere page apres authentification."""
    if role == "Enseignant":
        return "teacher_panel"
    return "student_dashboard"
