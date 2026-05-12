"""Graphiques Plotly pour le suivi de progression."""

from __future__ import annotations

import plotly.graph_objects as go


def build_mastery_evolution_chart(analytics: dict) -> go.Figure:
    """Créer la courbe d'évolution de maîtrise."""
    figure = go.Figure()
    labels = analytics["labels"]
    colors = ["#2fb5a9", "#49a6ff", "#ffc857", "#ff7a7a"]
    for index, (topic, values) in enumerate(analytics["series"].items()):
        figure.add_trace(
            go.Scatter(
                x=labels,
                y=values,
                mode="lines+markers",
                name=topic,
                line={"width": 3, "color": colors[index % len(colors)]},
                marker={"size": 8},
            )
        )
    figure.update_layout(
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.15},
        yaxis={"title": "Maîtrise %", "range": [0, 100]},
        xaxis={"title": "Jalon"},
    )
    return figure


def build_solved_exercises_chart(solved_exercises: list[dict]) -> go.Figure:
    """Afficher le nombre d'exercices résolus par thème."""
    figure = go.Figure(
        data=[
            go.Bar(
                x=[item["topic"] for item in solved_exercises],
                y=[item["count"] for item in solved_exercises],
                marker={"color": ["#2fb5a9", "#49a6ff", "#ffc857", "#7ee0c3", "#ff7a7a"]},
            )
        ]
    )
    figure.update_layout(
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"title": "Exercices résolus"},
        xaxis={"title": "Thème"},
    )
    return figure


def build_weak_topics_chart(weak_topics: list[dict]) -> go.Figure:
    """Afficher les thèmes les plus fragiles."""
    figure = go.Figure(
        data=[
            go.Bar(
                x=[item["mastery"] for item in weak_topics],
                y=[item["topic"] for item in weak_topics],
                orientation="h",
                marker={"color": "#ff7a7a"},
            )
        ]
    )
    figure.update_layout(
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title": "Maîtrise %", "range": [0, 100]},
        yaxis={"title": ""},
    )
    figure.update_yaxes(autorange="reversed")
    return figure


def build_success_rate_chart(success_rate: dict) -> go.Figure:
    """Afficher la répartition des résultats."""
    figure = go.Figure(
        data=[
            go.Pie(
                labels=["Correct", "Incorrect", "Avec indice"],
                values=[
                    success_rate["correct"],
                    success_rate["incorrect"],
                    success_rate["hinted"],
                ],
                hole=0.62,
                marker={"colors": ["#39d98a", "#ff7a7a", "#ffc857"]},
                textinfo="label+percent",
            )
        ]
    )
    figure.update_layout(
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return figure


def build_mastery_snapshot_radar(snapshot: dict[str, int]) -> go.Figure:
    """Afficher la maîtrise actuelle sous forme radar."""
    categories = list(snapshot.keys())
    values = list(snapshot.values())
    figure = go.Figure(
        data=[
            go.Scatterpolar(
                r=values + values[:1],
                theta=categories + categories[:1],
                fill="toself",
                line={"color": "#49a6ff", "width": 3},
                fillcolor="rgba(73, 166, 255, 0.22)",
            )
        ]
    )
    figure.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 100]}},
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return figure
