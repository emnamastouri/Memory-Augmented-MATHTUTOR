"""OpenRouter-backed tutoring agent for the conversational tutoring page."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from frontend.utils.constants import DEFAULT_STUDENT_PROFILE
from frontend.utils.openrouter_client import (
    extract_openrouter_text,
    get_openrouter_client,
    get_openrouter_settings,
    has_openrouter_config,
    summarize_openrouter_response_issue,
)


def generate_tutor_reply(
    *,
    student_message: str,
    mode: str,
    exercise_context: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
) -> str:
    """Generate one tutoring answer using OpenRouter with a local fallback."""
    cleaned_message = str(student_message).strip()
    if not cleaned_message:
        return "Precise ta question ou partage ton raisonnement, et je t'aide pas a pas."

    learner_profile = deepcopy(profile or DEFAULT_STUDENT_PROFILE)

    if has_openrouter_config():
        try:
            return _generate_openrouter_reply(
                student_message=cleaned_message,
                mode=mode,
                exercise_context=exercise_context,
                profile=learner_profile,
                conversation_history=conversation_history or [],
            )
        except Exception:
            pass

    return _generate_local_fallback(
        student_message=cleaned_message,
        mode=mode,
        exercise_context=exercise_context,
        profile=learner_profile,
    )


def _generate_openrouter_reply(
    *,
    student_message: str,
    mode: str,
    exercise_context: dict[str, Any] | None,
    profile: dict[str, Any],
    conversation_history: list[dict[str, Any]],
) -> str:
    """Call the remote Qwen tutoring model through OpenRouter."""
    settings = get_openrouter_settings()
    if settings is None:
        raise RuntimeError("OpenRouter n'est pas configure.")

    client = get_openrouter_client()
    messages = [
        {"role": "system", "content": _build_system_prompt(mode=mode, profile=profile, exercise_context=exercise_context)},
        {"role": "system", "content": _build_context_brief(exercise_context)},
    ]
    messages.extend(_build_history_messages(conversation_history, student_message))
    messages.append({"role": "user", "content": _build_user_turn(student_message, mode, exercise_context)})

    response = client.chat.completions.create(
        model=settings.tutor_model,
        messages=messages,
        temperature=0.45,
        max_tokens=650,
    )
    content = extract_openrouter_text(response)
    if not content:
        raise RuntimeError(summarize_openrouter_response_issue(response))
    return _postprocess_reply(content)


def _build_system_prompt(*, mode: str, profile: dict[str, Any], exercise_context: dict[str, Any] | None) -> str:
    """Create the tutor persona and tutoring constraints."""
    student_name = str(profile.get("name", DEFAULT_STUDENT_PROFILE["name"])).strip().split()[0]
    level = str(profile.get("level", "Bac")).strip() or "Bac"
    section = str(profile.get("section", "Bac")).strip() or "Bac"
    weak_topics = ", ".join(profile.get("weak_topics", [])[:3]) or "aucune faiblesse precisee"
    strong_topics = ", ".join(profile.get("strong_topics", [])[:3]) or "aucun point fort precise"

    mode_rules = {
        "Socratique": (
            "Tu guides surtout par questions courtes, relances et petites validations. "
            "Ne donne pas directement la solution complete sauf si l'eleve la demande explicitement."
        ),
        "Mode indice": (
            "Tu donnes un indice court, progressif, concret, sans reveler toute la solution finale. "
            "Tu peux proposer la prochaine etape utile."
        ),
        "Explication d'erreur": (
            "Tu identifies l'erreur probable, expliques pourquoi elle pose probleme, "
            "puis proposes une correction methodologique claire."
        ),
    }

    exercise_rule = (
        "Un exercice actif est fourni: reste centre sur cet enonce et ses annexes. "
        "Tu peux exploiter la solution cachee comme reference interne, mais ne l'affiche pas directement."
        if exercise_context
        else "Aucun exercice actif n'est fourni: aide sur la notion et le raisonnement partage."
    )

    return (
        "Tu es MathTutorAI, un tuteur de mathematiques francophone, clair, patient et rigoureux. "
        f"Tu accompagnes {student_name}, eleve de niveau {level}, section {section}. "
        f"Ses points forts actuels: {strong_topics}. Ses points a renforcer: {weak_topics}. "
        f"Mode courant: {mode}. {mode_rules.get(mode, mode_rules['Socratique'])} "
        f"{exercise_rule} "
        "Quand tu aides, privilegie un francais simple, des etapes lisibles et du Markdown leger. "
        "Si l'eleve partage un raisonnement, evalue-le sans inventer des erreurs absentes. "
        "Si un calcul depend d'un support annexe present, cite-le explicitement."
    )


def _build_context_brief(exercise_context: dict[str, Any] | None) -> str:
    """Summarize the active exercise for the tutor model."""
    if not exercise_context:
        return "Aucun exercice de reference actif."

    table_data = exercise_context.get("table_data")
    chart_data = exercise_context.get("chart_data")
    solution_steps = exercise_context.get("solution_steps") or []
    visible_steps = [str(step).strip() for step in solution_steps if str(step).strip()][:6]

    context_lines = [
        "Exercice de reference actif:",
        f"- Titre: {exercise_context.get('title', '')}",
        f"- Niveau: {exercise_context.get('level', '')}",
        f"- Section: {exercise_context.get('section', '')}",
        f"- Theme: {exercise_context.get('topic', '')}",
        f"- Sous-theme: {exercise_context.get('subtopic', '')}",
        f"- Type: {exercise_context.get('exercise_type', '')}",
        f"- Enonce: {exercise_context.get('prompt', '')}",
        f"- Indice principal: {exercise_context.get('hint', '')}",
        f"- Objectif pedagogique: {exercise_context.get('learning_objective', '')}",
        f"- Reponse attendue interne: {exercise_context.get('display_answer', '')}",
    ]

    if visible_steps:
        context_lines.append(f"- Etapes de solution internes: {' | '.join(visible_steps)}")

    hidden_solution = str(exercise_context.get("hidden_solution", "")).strip()
    if hidden_solution:
        context_lines.append(f"- Solution complete interne: {_truncate(hidden_solution, 1200)}")

    if isinstance(table_data, dict):
        headers = table_data.get("headers") or []
        rows = table_data.get("rows") or []
        context_lines.append(
            f"- Tableau annexe: colonnes = {headers}; apercu = {rows[:4]}"
        )

    if isinstance(chart_data, dict):
        chart_series = chart_data.get("series") or []
        context_lines.append(
            f"- Graphique annexe: type = {chart_data.get('type', '')}; titre = {chart_data.get('title', '')}; series = {len(chart_series)}"
        )

    return "\n".join(context_lines)


def _build_history_messages(conversation_history: list[dict[str, Any]], student_message: str) -> list[dict[str, str]]:
    """Keep a short, relevant chat history window for the tutoring model."""
    history_messages: list[dict[str, str]] = []
    filtered_history = [
        entry
        for entry in conversation_history
        if entry.get("role") in {"user", "assistant"} and str(entry.get("content", "")).strip()
    ]

    if filtered_history and filtered_history[-1].get("role") == "user":
        last_content = str(filtered_history[-1].get("content", "")).strip()
        if last_content == student_message.strip():
            filtered_history = filtered_history[:-1]

    for entry in filtered_history[-6:]:
        role = "assistant" if entry.get("role") == "assistant" else "user"
        content = str(entry.get("content", "")).strip()
        if content:
            history_messages.append({"role": role, "content": _truncate(content, 900)})
    return history_messages


def _build_user_turn(student_message: str, mode: str, exercise_context: dict[str, Any] | None) -> str:
    """Build the final user turn sent to the tutor model."""
    request_lines = [
        f"Mode de tutorat demande: {mode}",
        f"Message de l'eleve: {student_message}",
    ]
    if exercise_context:
        request_lines.append(
            "L'eleve travaille sur l'exercice actif visible a l'ecran. "
            "Sers-toi-en comme reference principale."
        )
    else:
        request_lines.append("Aucun exercice actif n'est disponible: reponds sur la notion generale.")
    request_lines.append(
        "Reponds directement en francais pour l'interface. "
        "Si utile, termine par une prochaine etape concrete."
    )
    return "\n".join(request_lines)


def _postprocess_reply(content: str) -> str:
    """Normalize one model answer before rendering it in the chat UI."""
    cleaned = str(content).strip()
    cleaned = cleaned.replace("\r\n", "\n")
    return cleaned or "Je suis pret a t'aider. Dis-moi precisement ou tu bloques."


def _generate_local_fallback(
    *,
    student_message: str,
    mode: str,
    exercise_context: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
) -> str:
    """Fallback tutoring answer used when OpenRouter is unavailable."""
    learner_name = (profile or DEFAULT_STUDENT_PROFILE)["name"].split()[0]
    focus = exercise_context["subtopic"] if exercise_context else "votre notion actuelle"

    if mode == "Mode indice":
        base = (
            f"{learner_name}, concentre-toi sur la structure de {focus.lower()}. "
            "Commence par l'etape la plus simple, puis avance transformation par transformation."
        )
        if exercise_context:
            base += f" Un premier appui utile ici: {exercise_context.get('hint', '')}"
        return base

    if mode == "Explication d'erreur":
        return (
            f"Dans {focus.lower()}, l'erreur frequente consiste a appliquer une formule trop tot sans identifier exactement ce qu'on cherche. "
            f"Dans ton message « {student_message} », je verifierais d'abord la quantite visee, puis les signes, les conditions de domaine ou les constantes oubliees."
        )

    prompt_hook = (
        f"Puisque tu travailles sur {focus.lower()}, que sais-tu deja de la premiere transformation ou du premier theoreme a mobiliser ? "
        "Explique-moi ton prochain pas et je verifierai le raisonnement plutot que de donner directement le resultat."
    )
    if exercise_context:
        prompt_hook += f" Tu peux t'appuyer sur cet objectif: {exercise_context.get('learning_objective', '')}."
    return prompt_hook


def _truncate(text: str, limit: int) -> str:
    """Trim long context strings to keep prompts under control."""
    cleaned = str(text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."
