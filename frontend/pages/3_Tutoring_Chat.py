"""Page de tutorat conversationnel pour MathTutorAI."""

from __future__ import annotations

from copy import deepcopy

import streamlit as st

from frontend.components.cards import render_highlight_card, render_section_header
from frontend.components.chat_ui import render_chat_history, render_typing_indicator, stream_assistant_reply
from frontend.components.exercise_widgets import render_exercise_card, render_exercise_supports
from frontend.components.navbar import render_page_hero
from frontend.utils.api_client import get_api_client
from frontend.utils.constants import DEFAULT_CHAT_HISTORY, TUTORING_MODES
from frontend.utils.mongo_learning import (
    get_recent_exercise_history,
    load_exercise_from_history,
    record_page_consultation,
    record_tutor_turn,
)
from frontend.utils.mongo_tutoring import (
    append_tutoring_turn,
    ensure_tutoring_thread,
    get_latest_thread_for_exercise,
    get_latest_thread_for_generation_trace,
    get_recent_tutoring_threads,
    restore_tutoring_thread,
)
from frontend.utils.page_router import switch_to_page
from frontend.utils.session_manager import (
    append_chat_message,
    clear_current_exercise_context,
    initialize_session_state,
    push_notification,
    require_authentication,
    set_current_exercise,
)


def _active_exercise_context() -> dict | None:
    """Return the exercise context only when the user enabled it."""
    if st.session_state.tutoring_state["use_exercise_context"]:
        return st.session_state.current_exercise
    return None


def _reset_current_conversation() -> None:
    """Start a new local conversation without deleting persisted history."""
    st.session_state.chat_history = deepcopy(DEFAULT_CHAT_HISTORY)
    st.session_state.active_tutoring_thread_id = ""
    st.session_state.active_tutoring_thread_title = ""


def _go_to_exercise_generator_for_new_conversation() -> None:
    """Reset tutoring-local state and navigate to the exercise generator page."""
    _reset_current_conversation()
    clear_current_exercise_context()
    st.session_state.tutoring_state["use_exercise_context"] = False
    push_notification(
        "Le tutorat a ete reinitialise. Generez maintenant un nouvel exercice pour demarrer une nouvelle discussion.",
        "🧩",
    )
    switch_to_page("exercise_generator")


def _maybe_auto_resume_exercise_thread(current_exercise: dict | None) -> None:
    """Automatically resume the last saved thread for the current exercise when relevant."""
    if not current_exercise:
        return
    if not st.session_state.tutoring_state["use_exercise_context"]:
        return
    if st.session_state.get("active_tutoring_thread_id"):
        return
    if len(st.session_state.chat_history) > 1:
        return

    latest_thread = get_latest_thread_for_exercise(current_exercise)
    if not latest_thread:
        return

    thread_id = latest_thread.get("thread_id", "")
    if not thread_id:
        return
    if restore_tutoring_thread(thread_id):
        push_notification("Derniere conversation sur cet exercice restauree.", "🕘")
        st.rerun()


def _open_exercise_in_tutoring(generation_trace_id: str) -> None:
    """Open one exercise from history inside tutoring, with or without prior discussion."""
    linked_thread = get_latest_thread_for_generation_trace(generation_trace_id)
    if linked_thread and restore_tutoring_thread(linked_thread.get("thread_id", "")):
        push_notification("Discussion precedente restauree pour cet exercice.", "🧠")
        st.rerun()

    exercise = load_exercise_from_history(generation_trace_id)
    if not exercise:
        st.sidebar.error("Impossible de charger cet exercice depuis l'historique.")
        return

    set_current_exercise(exercise)
    st.session_state.tutoring_state["use_exercise_context"] = True
    st.session_state.chat_history = deepcopy(DEFAULT_CHAT_HISTORY)
    st.session_state.active_tutoring_thread_id = ""
    st.session_state.active_tutoring_thread_title = exercise.get("title", "Nouvelle conversation")
    push_notification("Exercice charge dans le tutorat. Vous pouvez commencer la discussion.", "📘")
    st.rerun()


def _run_tutor_turn(prompt: str) -> None:
    """Generer et afficher la reponse du tuteur."""
    api_client = get_api_client()
    current_exercise = _active_exercise_context()
    thread = ensure_tutoring_thread(
        mode=st.session_state.tutoring_state["mode"],
        exercise_context=current_exercise,
        session_history=st.session_state.chat_history,
    )

    append_chat_message("user", prompt, mode=st.session_state.tutoring_state["mode"])

    with st.chat_message("user", avatar="🧑🏽‍🎓"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧠"):
        render_typing_indicator()
        response = api_client.generate_tutor_response(
            student_message=prompt,
            mode=st.session_state.tutoring_state["mode"],
            exercise_context=current_exercise,
            profile=st.session_state.student_profile,
            conversation_history=st.session_state.chat_history,
        )
        streamed = stream_assistant_reply(response)
        record_tutor_turn(
            st.session_state.tutoring_state["mode"],
            prompt,
            streamed,
            current_exercise,
        )
        if thread:
            append_tutoring_turn(
                thread_id=thread["thread_id"],
                mode=st.session_state.tutoring_state["mode"],
                user_message=prompt,
                assistant_message=streamed,
                exercise_context=current_exercise,
                student_answer_draft=st.session_state.get("current_answer", ""),
            )
    append_chat_message("assistant", streamed, mode=st.session_state.tutoring_state["mode"])


def render_page() -> None:
    """Afficher l'interface de discussion avec le tuteur."""
    initialize_session_state()
    require_authentication("Tutorat conversationnel")
    current_exercise = _active_exercise_context()
    _maybe_auto_resume_exercise_thread(current_exercise)
    record_page_consultation(
        "tutoring_chat",
        "Tutorat conversationnel",
        topic=(current_exercise or {}).get("topic", ""),
        subtopic=(current_exercise or {}).get("subtopic", ""),
        metadata={"mode": st.session_state.tutoring_state["mode"]},
    )
    render_page_hero(
        "Tutorat conversationnel",
        "Discutez avec le tuteur en mode socratique, indice ou explication d'erreur, puis reprenez ensuite chaque conversation depuis votre historique.",
        badge=st.session_state.tutoring_state["mode"],
    )

    st.sidebar.caption("Une nouvelle conversation vous renvoie vers le generateur d'exercices sans effacer l'historique Mongo deja enregistre.")
    if st.sidebar.button("Nouvelle conversation"):
        _go_to_exercise_generator_for_new_conversation()
    if st.session_state.get("active_tutoring_thread_title"):
        st.sidebar.caption(f"Fil actif : {st.session_state.active_tutoring_thread_title}")

    with st.sidebar.expander("Historique des conversations", expanded=False):
        recent_threads = get_recent_tutoring_threads(limit=10)
        if recent_threads:
            for item in recent_threads:
                st.markdown(f"**{item['title']}**")
                footer = item["topic"] or item["mode"]
                st.caption(f"{footer} · {item['updated_at_label']} · {item['message_count']} message(s)")
                if item.get("preview"):
                    st.caption(item["preview"])
                if st.button("Reprendre", key=f"resume_thread_{item['thread_id']}"):
                    if restore_tutoring_thread(item["thread_id"]):
                        push_notification("Conversation restauree depuis l'historique utilisateur.", "🧠")
                        st.rerun()
                st.divider()
        else:
            st.caption("Les prochaines conversations seront memorisees ici.")

    with st.sidebar.expander("Historique des exercices", expanded=False):
        exercise_history = get_recent_exercise_history(limit=10)
        if exercise_history:
            for item in exercise_history:
                st.markdown(f"**{item['title']}**")
                st.caption(f"{item['topic']} · {item['status']} · {item['timestamp']}")
                trace_id = item.get("generation_trace_id", "")
                linked_thread = get_latest_thread_for_generation_trace(trace_id) if trace_id else None
                action_label = "Reprendre la discussion" if linked_thread else "Discuter cet exercice"
                if trace_id and st.button(action_label, key=f"open_exercise_chat_{trace_id}"):
                    _open_exercise_in_tutoring(trace_id)
                st.divider()
        else:
            st.caption("Les exercices generes apparaitront ici pour etre discutes dans le tutorat.")

    render_section_header(
        "Assistant pedagogique",
        "Une experience de tutorat proche de ChatGPT, enrichie par la memoire d'apprentissage et par un historique recuperable.",
        "💬",
    )

    controls_col, context_col = st.columns([1.6, 1], gap="large")
    with controls_col:
        mode = st.radio(
            "Mode de tutorat",
            list(TUTORING_MODES.keys()),
            index=list(TUTORING_MODES.keys()).index(st.session_state.tutoring_state["mode"]),
            horizontal=True,
        )
        st.session_state.tutoring_state["mode"] = mode
        st.caption(TUTORING_MODES[mode])
    with context_col:
        st.checkbox(
            "Utiliser l'exercice actif comme contexte",
            key="chat_use_exercise_context",
            value=st.session_state.tutoring_state["use_exercise_context"],
        )
        st.session_state.tutoring_state["use_exercise_context"] = st.session_state.chat_use_exercise_context
        current_exercise = _active_exercise_context()
        active_context = current_exercise["subtopic"] if current_exercise else "Aucun exercice actif"
        render_highlight_card(
            "Contexte courant",
            active_context,
            st.session_state.get("active_tutoring_thread_title") or "Ancrage memoire du tuteur",
        )

    quick_prompt = None
    chat_col, exercise_col = st.columns([1.45, 1], gap="large")

    with chat_col:
        quick_col_1, quick_col_2, quick_col_3 = st.columns(3, gap="medium")
        with quick_col_1:
            if st.button("J'ai besoin d'un indice"):
                quick_prompt = "Donne-moi un petit indice sans reveler toute la reponse."
        with quick_col_2:
            if st.button("Verifie mon raisonnement"):
                quick_prompt = "Voici mon raisonnement. Peux-tu me dire si la structure est correcte ?"
        with quick_col_3:
            if st.button("Explique l'erreur typique"):
                quick_prompt = "Quelle erreur frequente dois-je surveiller dans cette notion ?"

        with st.container():
            render_chat_history(st.session_state.chat_history)

    with exercise_col:
        render_section_header(
            "Exercice de reference",
            "L'enonce actif ou restaure depuis l'historique reste visible pendant la discussion.",
            "📘",
        )
        if current_exercise:
            render_exercise_card(current_exercise)
            render_exercise_supports(current_exercise)
            render_highlight_card(
                "Objectif du tutorat",
                "Le tuteur repond en gardant cet exercice comme ancrage de conversation, y compris apres restauration depuis l'historique.",
                current_exercise.get("subtopic", "Contexte actif"),
            )
        else:
            render_highlight_card(
                "Aucun exercice actif",
                "Vous pouvez discuter librement sur une notion, ou ouvrir un exercice de l'historique pour demarrer une discussion.",
                "Contexte libre",
            )

    prompt = st.chat_input("Posez une question au tuteur sur une notion, un exercice ou une erreur...")
    submitted_prompt = prompt or quick_prompt

    if submitted_prompt:
        _run_tutor_turn(submitted_prompt)
        push_notification(f"Nouvelle reponse ajoutee en mode {st.session_state.tutoring_state['mode'].lower()}.", "🧠")
        st.rerun()


render_page()
