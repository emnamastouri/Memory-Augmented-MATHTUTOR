"""Éléments d'interface pour le tutorat conversationnel."""

from __future__ import annotations

import time

import streamlit as st


def render_chat_history(messages: list[dict]) -> None:
    """Rejouer l'historique de conversation dans l'interface chat."""
    for message in messages:
        avatar = "🧠" if message["role"] == "assistant" else "🧑🏽‍🎓"
        with st.chat_message(message["role"], avatar=avatar):
            mode = message.get("mode")
            if mode and message["role"] == "assistant":
                st.caption(f"Mode {mode}")
            st.markdown(message["content"])


def stream_assistant_reply(full_text: str, delay: float = 0.015) -> str:
    """Simuler un affichage progressif de la réponse de l'assistant."""
    placeholder = st.empty()
    assembled = []
    for word in full_text.split():
        assembled.append(word)
        placeholder.markdown(" ".join(assembled) + " ▌")
        time.sleep(delay)
    final_text = " ".join(assembled)
    placeholder.markdown(final_text)
    return final_text


def render_typing_indicator(label: str = "Le tuteur prépare une réponse") -> None:
    """Afficher un indicateur de rédaction."""
    st.markdown(
        f"""
        <div class="typing-indicator">
            <span>{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
