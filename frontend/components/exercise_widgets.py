"""Composants de rendu pour les exercices."""

from __future__ import annotations

from html import escape, unescape
import re
from textwrap import dedent

import plotly.graph_objects as go
import streamlit as st


def render_filter_summary(selection: dict) -> None:
    """Afficher le résumé des filtres actifs."""
    pills = "".join(
        [f'<span class="tag-pill">{escape(str(value))}</span>' for value in selection.values()]
    )
    st.markdown(
        dedent(
            f"""
            <div class="glass-card">
                <div class="card-title">Configuration du générateur</div>
                <div class="card-body">Profil actif pour la prochaine demande d'exercice.</div>
                <div class="pill-row">{pills}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_exercise_card(exercise: dict) -> None:
    """Afficher un exercice généré avec ses métadonnées."""
    hidden_tags = {"Fondamental", "Intermédiaire", "Avancé", "Défi", "Intermediaire", "Avance", "Defi"}
    visible_tags = [tag for tag in exercise.get("tags", []) if str(tag) not in hidden_tags]
    tag_markup = "".join([f'<span class="tag-pill">{escape(tag)}</span>' for tag in visible_tags])
    options_markup = ""
    prompt_text = _format_exercise_prompt(_build_student_statement(exercise))
    learning_objective = _clean_card_text(exercise.get("learning_objective", ""))
    meta_label = escape(exercise["level"]) if exercise.get("level") else ""
    meta_markup = f'<div class="exercise-card__meta">{meta_label}</div>' if meta_label else ""
    if exercise.get("options"):
        option_items = "".join([f"<li>{escape(option)}</li>" for option in exercise["options"]])
        options_markup = f"<div class='card-body'><strong>Choix proposés</strong><ul>{option_items}</ul></div>"
    st.markdown(
        dedent(
            f"""
            <div class="glass-card exercise-card">
                <div class="exercise-card__header">
                    <div>
                        <div class="card-title">{escape(exercise["title"])}</div>
                        {meta_markup}
                    </div>
                </div>
                <div class="pill-row">{tag_markup}</div>
                <div class="card-body exercise-card__prompt">{escape(prompt_text)}</div>
                {options_markup}
                <div class="exercise-objective">Objectif pédagogique : {escape(learning_objective)}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_exercise_supports(exercise: dict) -> None:
    """Afficher les tableaux et graphiques complementaires d'un exercice."""
    table_data = exercise.get("table_data")
    chart_data = exercise.get("chart_data")
    graph_data = exercise.get("graph_data")

    if not table_data and not chart_data and not graph_data:
        return

    st.markdown("#### Annexe de données")
    if exercise.get("support_summary"):
        st.caption(exercise["support_summary"])

    if table_data:
        st.markdown(_build_table_markup(table_data), unsafe_allow_html=True)

    if chart_data:
        figure = _build_support_chart(chart_data)
        st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
        caption = str(chart_data.get("caption", "")).strip()
        if caption:
            st.caption(caption)
    elif graph_data and isinstance(graph_data, dict) and graph_data.get("series"):
        figure = _build_support_chart(graph_data)
        st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
        caption = str(graph_data.get("caption", "")).strip()
        if caption:
            st.caption(caption)


def render_solution_steps(steps: list[str]) -> None:
    """Afficher les étapes de solution dans un panneau repliable."""
    with st.expander("Étapes de solution", expanded=False):
        for index, step in enumerate(steps, start=1):
            st.markdown(f"**Étape {index}.** {step}")


def render_empty_exercise_state() -> None:
    """Afficher un état vide avant la première génération."""
    st.markdown(
        dedent(
            """
                <div class="glass-card empty-state">
                    <div class="card-title">Aucun exercice généré pour le moment</div>
                    <div class="card-body">
                    Choisissez une section, un thème, un sous-thème et un type d'exercice, puis lancez la génération pour démarrer.
                    </div>
                </div>
            """
        ),
        unsafe_allow_html=True,
    )


def _build_table_markup(table_data: dict) -> str:
    """Render one exercise table as HTML to avoid extra dataframe dependencies."""
    headers = "".join([f"<th>{escape(str(header))}</th>" for header in table_data.get("headers", [])])
    body_rows = []
    for row in table_data.get("rows", []):
        cells = "".join([f"<td>{escape(str(cell))}</td>" for cell in row])
        body_rows.append(f"<tr>{cells}</tr>")
    caption = str(table_data.get("caption", "")).strip()
    caption_markup = f"<div class='card-body'><strong>{escape(caption)}</strong></div>" if caption else ""
    return dedent(
        f"""
        <div class="glass-card">
            {caption_markup}
            <div class="exercise-support-table">
                <table>
                    <thead><tr>{headers}</tr></thead>
                    <tbody>{''.join(body_rows)}</tbody>
                </table>
            </div>
        </div>
        """
    )


def _clean_card_text(value: object) -> str:
    """Keep the rendered card free of internal metadata fragments."""
    text = str(value or "").strip()
    for _ in range(3):
        decoded = unescape(text)
        if decoded == text:
            break
        text = decoded
    if not text:
        return ""
    text = _truncate_internal_fragment(text)
    text = re.sub(
        r"<div[^>]*exercise-objective[^>]*>.*$",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"<div[^>]*class=[\"']exercise-objective[\"'][^>]*>.*?</div>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"&lt;div[^&]*exercise-objective[^&]*&gt;.*$",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.split(
        r"\bObjectif p.{0,4}dagogique\s*:",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    text = re.sub(r"exercise-objective.*$", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</div>\s*$", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&lt;[^&]+&gt;", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _format_exercise_prompt(text: str) -> str:
    """Insert readable line breaks between the main parts of the statement."""
    if not text:
        return ""

    formatted = re.sub(r"\s+(?=\d+\))", "\n", text)
    formatted = re.sub(r"\s+(?=[a-d]\))", "\n", formatted)
    formatted = re.sub(r"(?<=\.)\s+(?=Pour\s)", "\n", formatted)
    formatted = re.sub(r"\n{2,}", "\n", formatted)
    return formatted.strip()


def _build_student_statement(exercise: dict) -> str:
    """Prefer the structured v7 context/questions statement when available."""
    context = _clean_card_text(exercise.get("context", ""))
    questions = [_clean_card_text(item) for item in (exercise.get("questions") or []) if str(item).strip()]
    if questions:
        numbered = "\n".join(f"{index}) {question}" for index, question in enumerate(questions, start=1))
        return f"{context}\n{numbered}".strip()
    return _clean_card_text(exercise.get("prompt", ""))


def _truncate_internal_fragment(text: str) -> str:
    """Cut leaked internal markup or objective metadata from the first marker onward."""
    markers = [
        "exercise-objective",
        "Objectif pédagogique",
        "Objectif pedagogique",
        "&lt;div",
        "<div",
    ]
    cut_positions = [text.find(marker) for marker in markers if marker in text]
    if not cut_positions:
        return text
    cut_index = min(position for position in cut_positions if position >= 0)
    return text[:cut_index]


def _build_support_chart(chart_data: dict) -> go.Figure:
    """Create a lightweight Plotly chart for exercise annexes."""
    figure = go.Figure()
    color_cycle = ["#2fb5a9", "#49a6ff", "#ffc857", "#ff7a7a"]

    for index, series in enumerate(chart_data.get("series", [])):
        common_kwargs = {
            "name": str(series.get("name", f"Série {index + 1}")),
            "x": series.get("x", []),
            "y": series.get("y", []),
            "marker": {"size": 9, "color": color_cycle[index % len(color_cycle)]},
        }
        chart_type = chart_data.get("type", "scatter")
        if chart_type == "line":
            figure.add_trace(go.Scatter(mode="lines+markers", line={"width": 3}, **common_kwargs))
        elif chart_type == "bar":
            figure.add_trace(go.Bar(marker={"color": color_cycle[index % len(color_cycle)]}, **common_kwargs))
        else:
            figure.add_trace(go.Scatter(mode="markers", **common_kwargs))

    figure.update_layout(
        title={"text": chart_data.get("title", ""), "x": 0.02},
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.12},
        xaxis={"title": chart_data.get("x_label", "")},
        yaxis={"title": chart_data.get("y_label", "")},
    )
    return figure
