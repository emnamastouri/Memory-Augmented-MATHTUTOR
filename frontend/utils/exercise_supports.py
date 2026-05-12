from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any
import unicodedata

from frontend.utils.openrouter_client import (
    extract_openrouter_text,
    get_openrouter_client,
    get_openrouter_settings,
    has_openrouter_config,
)


def enrich_exercise_supports(exercise: dict[str, Any]) -> dict[str, Any]:
    """Attach only the annex supports that are explicitly required by the statement."""
    enriched = deepcopy(exercise)
    support_flags = _detect_support_requirements(enriched)

    normalized_table = _normalize_table_data(enriched.get("table_data"))
    normalized_chart = _normalize_chart_data(enriched.get("chart_data"))

    if normalized_table:
        enriched["table_data"] = normalized_table
    else:
        enriched.pop("table_data", None)

    if normalized_chart:
        enriched["chart_data"] = normalized_chart
    else:
        enriched.pop("chart_data", None)

    if not support_flags["needs_table"]:
        enriched.pop("table_data", None)
    if not support_flags["needs_chart"]:
        enriched.pop("chart_data", None)

    if support_flags["needs_table"] and not enriched.get("table_data") and enriched.get("chart_data"):
        derived_table = _table_from_chart(enriched["chart_data"])
        if derived_table:
            enriched["table_data"] = derived_table

    if support_flags["needs_chart"] and not enriched.get("chart_data") and enriched.get("table_data"):
        derived_chart = _chart_from_table(enriched["table_data"], enriched)
        if derived_chart:
            enriched["chart_data"] = derived_chart

    if (
        (support_flags["needs_table"] and not enriched.get("table_data"))
        or (support_flags["needs_chart"] and not enriched.get("chart_data"))
    ) and has_openrouter_config():
        generated_supports = _generate_missing_supports(enriched, support_flags)
        generated_table = _normalize_table_data(generated_supports.get("table_data"))
        generated_chart = _normalize_chart_data(generated_supports.get("chart_data"))

        if support_flags["needs_table"] and generated_table:
            enriched["table_data"] = generated_table
        if support_flags["needs_chart"] and generated_chart:
            enriched["chart_data"] = generated_chart

        if support_flags["needs_table"] and not enriched.get("table_data") and enriched.get("chart_data"):
            derived_table = _table_from_chart(enriched["chart_data"])
            if derived_table:
                enriched["table_data"] = derived_table

        if support_flags["needs_chart"] and not enriched.get("chart_data") and enriched.get("table_data"):
            derived_chart = _chart_from_table(enriched["table_data"], enriched)
            if derived_chart:
                enriched["chart_data"] = derived_chart

    enriched["support_requirements"] = support_flags
    enriched["support_ready"] = (
        (not support_flags["needs_table"] or bool(enriched.get("table_data")))
        and (not support_flags["needs_chart"] or bool(enriched.get("chart_data")))
    )
    enriched["support_summary"] = _build_support_summary(enriched, support_flags)
    return enriched


def describe_supports_for_judge(exercise: dict[str, Any]) -> str:
    """Summarize the attached supports for the judge prompt."""
    support_flags = exercise.get("support_requirements") or _detect_support_requirements(exercise)
    parts = [
        f"Support tableau explicitement demande : {'oui' if support_flags.get('needs_table') else 'non'}",
        f"Support graphique explicitement demande : {'oui' if support_flags.get('needs_chart') else 'non'}",
    ]

    table_data = exercise.get("table_data")
    chart_data = exercise.get("chart_data")

    if table_data:
        headers = ", ".join([str(item) for item in table_data.get("headers", [])])
        preview_rows = table_data.get("rows", [])[:3]
        parts.append(
            "Tableau attache : "
            f"colonnes = {headers or 'non precisees'} ; "
            f"lignes d'apercu = {json.dumps(preview_rows, ensure_ascii=False)}"
        )
    else:
        parts.append("Tableau attache : aucun")

    if chart_data:
        parts.append(
            "Graphique attache : "
            f"type = {chart_data.get('type', 'inconnu')} ; "
            f"titre = {chart_data.get('title', '') or 'sans titre'} ; "
            f"series = {len(chart_data.get('series', []))}"
        )
    else:
        parts.append("Graphique attache : aucun")

    return "\n".join([f"- {part}" for part in parts])


def _detect_support_requirements(exercise: dict[str, Any]) -> dict[str, bool]:
    """Detect only supports that the statement says are already provided to the student."""
    text = _normalize_support_text(
        " ".join(
            [
                str(exercise.get("title", "")),
                str(exercise.get("prompt", "")),
                str(exercise.get("topic", "")),
                str(exercise.get("subtopic", "")),
                str(exercise.get("learning_objective", "")),
            ]
        )
    )

    table_patterns = [
        r"\ble tableau (?:suivant|ci-dessous|ci dessous|donne|fourni)\b",
        r"\btableau (?:suivant|ci-dessous|ci dessous|donne|fourni|de donnees|des donnees)\b",
        r"\bdans le tableau\b",
        r"\ba partir du tableau\b",
        r"\bdonnees? suivantes\b",
        r"\bserie statistique suivante\b",
        r"\bloi de probabilite (?:suivante|ci-dessous|ci dessous|donnee|fournie)\b",
        r"\bdistribution (?:suivante|ci-dessous|ci dessous|donnee|fournie)\b",
        r"\bx_i\b",
        r"\by_i\b",
        r"\bvaleurs? de x\b",
        r"\bvaleurs? de y\b",
    ]
    chart_patterns = [
        r"\ble graphique (?:suivant|ci-dessous|ci dessous|donne|fourni)\b",
        r"\bgraphique (?:suivant|ci-dessous|ci dessous|donne|fourni)\b",
        r"\ble graphe (?:suivant|ci-dessous|ci dessous|donne|fourni)\b",
        r"\bgraphe (?:suivant|ci-dessous|ci dessous|donne|fourni)\b",
        r"\bla courbe (?:suivante|ci-dessous|ci dessous|donnee|fournie)\b",
        r"\bcourbe (?:suivante|ci-dessous|ci dessous|donnee|fournie)\b",
        r"\bfigure (?:suivante|ci-dessous|ci dessous|donnee|fournie)\b",
        r"\bnuage de points (?:suivant|ci-dessous|ci dessous|donne|fourni)\b",
        r"\bvoir annexe\b",
        r"\bannexe (?:ci-dessous|ci dessous|jointe|fournie|suivante)\b",
        r"\ba partir du graphique\b",
        r"\ba partir du graphe\b",
        r"\ba partir de la courbe\b",
        r"\blecture graphique\b",
    ]

    needs_table = any(re.search(pattern, text) for pattern in table_patterns)
    needs_chart = any(re.search(pattern, text) for pattern in chart_patterns)
    return {"needs_table": needs_table, "needs_chart": needs_chart}


def _generate_missing_supports(exercise: dict[str, Any], support_flags: dict[str, bool]) -> dict[str, Any]:
    """Ask OpenRouter to generate missing annex data only when explicitly required."""
    settings = get_openrouter_settings()
    if settings is None:
        return {}

    client = get_openrouter_client()
    prompt = (
        "Tu completes un exercice de mathematiques pour qu'il soit autoporteur dans une application Streamlit. "
        "Ajoute un tableau de donnees ou un graphique seulement si l'enonce dit explicitement qu'un support est fourni a l'eleve "
        "(par exemple tableau suivant, graphique ci-dessous, voir annexe, donnees suivantes). "
        "N'ajoute jamais un support que l'eleve est cense construire lui-meme, comme un tableau de variation, un tableau de signe, "
        "une courbe a tracer ou un nuage de points a representer. "
        "Les donnees doivent etre coherentes avec l'enonce, la solution cachee et le type d'exercice. "
        "N'ajoute aucun texte hors JSON.\n\n"
        f"Besoin tableau: {'oui' if support_flags['needs_table'] else 'non'}\n"
        f"Besoin graphique: {'oui' if support_flags['needs_chart'] else 'non'}\n"
        f"Titre: {exercise.get('title', '')}\n"
        f"Enonce: {exercise.get('prompt', '')}\n"
        f"Solution cachee: {exercise.get('hidden_solution', '')}\n"
        f"Objectif pedagogique: {exercise.get('learning_objective', '')}\n"
        "Format attendu:\n"
        "{\n"
        '  "table_data": {\n'
        '    "caption": "court texte",\n'
        '    "headers": ["colonne 1", "colonne 2"],\n'
        '    "rows": [[1, 2], [3, 4]]\n'
        "  },\n"
        '  "chart_data": {\n'
        '    "type": "scatter ou line ou bar",\n'
        '    "title": "titre",\n'
        '    "caption": "court texte",\n'
        '    "x_label": "axe x",\n'
        '    "y_label": "axe y",\n'
        '    "series": [{"name": "serie 1", "x": [1, 2], "y": [3, 4]}]\n'
        "  }\n"
        "}\n"
        "Si un support n'est pas necessaire, renvoie null pour ce support."
    )

    request_kwargs = {
        "model": settings.exercise_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu es l'agent d'enrichissement de MathTutorAI. "
                    "Tu fournis seulement des tableaux et graphiques coherents en JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1000,
    }

    try:
        try:
            response = client.chat.completions.create(
                response_format={"type": "json_object"},
                **request_kwargs,
            )
        except Exception:
            response = client.chat.completions.create(**request_kwargs)
    except Exception:
        return {}

    content = extract_openrouter_text(response)
    return _extract_json_object(content)


def _normalize_table_data(raw_table: Any) -> dict[str, Any] | None:
    """Sanitize table payloads into a stable structure."""
    if not isinstance(raw_table, dict):
        return None

    headers = raw_table.get("headers") or []
    rows = raw_table.get("rows") or []
    if not isinstance(headers, list) or not isinstance(rows, list):
        return None

    cleaned_headers = [str(item).strip() for item in headers if str(item).strip()]
    cleaned_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            continue
        cleaned_row = [_coerce_cell_value(cell) for cell in row]
        if cleaned_row:
            cleaned_rows.append(cleaned_row)

    if not cleaned_headers or not cleaned_rows:
        return None

    width = len(cleaned_headers)
    normalized_rows = [row[:width] + [""] * max(0, width - len(row)) for row in cleaned_rows]
    return {
        "caption": str(raw_table.get("caption", "")).strip(),
        "headers": cleaned_headers,
        "rows": normalized_rows,
    }


def _normalize_chart_data(raw_chart: Any) -> dict[str, Any] | None:
    """Sanitize chart payloads into a stable structure."""
    if not isinstance(raw_chart, dict):
        return None

    chart_type = str(raw_chart.get("type", "")).strip().lower()
    if chart_type not in {"scatter", "line", "bar"}:
        return None

    raw_series = raw_chart.get("series") or []
    if not isinstance(raw_series, list):
        return None

    cleaned_series = []
    for series in raw_series:
        if not isinstance(series, dict):
            continue
        x_values = series.get("x") or []
        y_values = series.get("y") or []
        if not isinstance(x_values, list) or not isinstance(y_values, list) or not x_values or not y_values:
            continue
        limit = min(len(x_values), len(y_values))
        cleaned_series.append(
            {
                "name": str(series.get("name", "Serie 1")).strip() or "Serie 1",
                "x": [_coerce_cell_value(value) for value in x_values[:limit]],
                "y": [_coerce_numeric_value(value) for value in y_values[:limit]],
            }
        )

    if not cleaned_series:
        return None

    return {
        "type": chart_type,
        "title": str(raw_chart.get("title", "")).strip(),
        "caption": str(raw_chart.get("caption", "")).strip(),
        "x_label": str(raw_chart.get("x_label", "")).strip(),
        "y_label": str(raw_chart.get("y_label", "")).strip(),
        "series": cleaned_series,
    }


def _table_from_chart(chart_data: dict[str, Any]) -> dict[str, Any] | None:
    """Derive a table from a chart series when only the chart is available."""
    series = chart_data.get("series", [])
    if not series:
        return None

    first_series = series[0]
    x_values = first_series.get("x", [])
    y_values = first_series.get("y", [])
    if not x_values or not y_values:
        return None

    rows = [[x_value, y_value] for x_value, y_value in zip(x_values, y_values)]
    return {
        "caption": chart_data.get("caption", "") or "Tableau de valeurs associe au graphique.",
        "headers": [chart_data.get("x_label", "x"), chart_data.get("y_label", "y")],
        "rows": rows,
    }


def _chart_from_table(table_data: dict[str, Any], exercise: dict[str, Any]) -> dict[str, Any] | None:
    """Derive a simple chart from a 2-column numeric table."""
    headers = table_data.get("headers", [])
    rows = table_data.get("rows", [])
    if len(headers) < 2 or not rows:
        return None

    x_values = [row[0] for row in rows if len(row) >= 2]
    y_values = [_coerce_numeric_value(row[1]) for row in rows if len(row) >= 2]
    if not x_values or not y_values:
        return None

    text = _normalize_support_text(
        " ".join(
            [
                str(exercise.get("prompt", "")),
                str(exercise.get("topic", "")),
                str(exercise.get("subtopic", "")),
            ]
        )
    )
    chart_type = "scatter" if any(keyword in text for keyword in ["nuage", "correlation", "regression"]) else "line"
    return {
        "type": chart_type,
        "title": f"Visualisation : {exercise.get('title', 'Exercice')}",
        "caption": "Graphique derive automatiquement du tableau de donnees.",
        "x_label": str(headers[0]),
        "y_label": str(headers[1]),
        "series": [{"name": "Serie principale", "x": x_values, "y": y_values}],
    }


def _build_support_summary(exercise: dict[str, Any], support_flags: dict[str, bool]) -> str:
    """Provide a short human-readable support status."""
    pieces: list[str] = []
    if support_flags["needs_table"]:
        pieces.append("tableau fourni" if exercise.get("table_data") else "tableau manquant")
    if support_flags["needs_chart"]:
        pieces.append("graphique fourni" if exercise.get("chart_data") else "graphique manquant")
    if not pieces:
        return "Aucun support visuel supplementaire n'etait requis pour cet exercice."
    return "Support annexe : " + " ; ".join(pieces) + "."


def _extract_json_object(content: str) -> dict[str, Any]:
    """Parse the first JSON object contained in a model response."""
    raw = (content or "").strip()
    if not raw:
        return {}
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


def _coerce_cell_value(value: Any) -> Any:
    """Normalize one cell value for table or axis usage."""
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    number_match = re.fullmatch(r"-?\d+(?:[.,]\d+)?", text)
    if number_match:
        return _coerce_numeric_value(text)
    return text


def _coerce_numeric_value(value: Any) -> float | int | str:
    """Convert numeric-like values while preserving labels when conversion fails."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 6)
    text = str(value).strip().replace(",", ".")
    try:
        numeric = float(text)
    except ValueError:
        return str(value).strip()
    return int(numeric) if numeric.is_integer() else round(numeric, 6)


def _normalize_support_text(value: Any) -> str:
    """Normalize one text block for robust keyword matching."""
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower().replace("\n", " ")
    return re.sub(r"\s+", " ", ascii_text).strip()
