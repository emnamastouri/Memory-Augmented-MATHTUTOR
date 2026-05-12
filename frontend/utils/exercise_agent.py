"""OpenRouter-backed, Memento-inspired exercise generator."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from html import unescape
import hashlib
import json
import re
from typing import Any
import unicodedata

import streamlit as st

from frontend.utils.constants import DEFAULT_STUDENT_PROFILE
from frontend.utils.exercise_memory import (
    DatasetExerciseCase,
    retain_generation_memory,
    retrieve_dataset_cases,
    retrieve_generation_memories,
)
from frontend.utils.openrouter_client import (
    extract_openrouter_text,
    get_openrouter_client,
    get_openrouter_settings,
    has_openrouter_config,
    summarize_openrouter_response_issue,
)


def generate_exercise_with_memory_adaptation(
    *,
    level: str,
    section: str,
    topic: str,
    subtopic: str,
    difficulty: str,
    exercise_type: str,
    profile: dict[str, Any] | None = None,
    quality_feedback: str = "",
) -> dict[str, Any]:
    """Generate a new exercise using case-based memory adaptation."""
    active_profile = deepcopy(profile or st.session_state.get("student_profile", DEFAULT_STUDENT_PROFILE))
    dataset_cases = retrieve_dataset_cases(
        section=section,
        topic=topic,
        subtopic=subtopic,
        profile=active_profile,
        top_k=3,
    )
    student_memories = retrieve_generation_memories(
        section=section,
        topic=topic,
        subtopic=subtopic,
        top_k=2,
    )

    generation_backend = "openrouter-qwen"
    generation_warning = ""
    payload: dict[str, Any]

    if has_openrouter_config():
        try:
            payload = _call_openrouter_for_exercise(
                level=level,
                section=section,
                topic=topic,
                subtopic=subtopic,
                difficulty=difficulty,
                exercise_type=exercise_type,
                profile=active_profile,
                dataset_cases=dataset_cases,
                student_memories=student_memories,
                quality_feedback=quality_feedback,
            )
        except Exception as exc:
            generation_backend = "dataset-fallback"
            generation_warning = str(exc)
            payload = _build_fallback_payload(
                topic=topic,
                subtopic=subtopic,
                difficulty=difficulty,
                exercise_type=exercise_type,
                dataset_cases=dataset_cases,
            )
    else:
        generation_backend = "dataset-fallback"
        generation_warning = "OpenRouter n'est pas configure."
        payload = _build_fallback_payload(
            topic=topic,
            subtopic=subtopic,
            difficulty=difficulty,
            exercise_type=exercise_type,
            dataset_cases=dataset_cases,
        )

    exercise = _build_exercise_payload(
        payload=payload,
        level=level,
        section=section,
        topic=topic,
        subtopic=subtopic,
        difficulty=difficulty,
        exercise_type=exercise_type,
        dataset_cases=dataset_cases,
        student_memories=student_memories,
        generation_backend=generation_backend,
        generation_warning=generation_warning,
    )
    completeness_review = assess_exercise_completeness(exercise)
    exercise["prompt"] = completeness_review["clean_prompt"]

    if completeness_review["is_complete"]:
        retain_generation_memory(
            section=section,
            topic=topic,
            subtopic=subtopic,
            difficulty=difficulty,
            exercise_type=exercise_type,
            generated_exercise=exercise,
            profile=active_profile,
            retrieved_cases=dataset_cases,
        )
    return exercise


def _call_openrouter_for_exercise(
    *,
    level: str,
    section: str,
    topic: str,
    subtopic: str,
    difficulty: str,
    exercise_type: str,
    profile: dict[str, Any],
    dataset_cases: list[DatasetExerciseCase],
    student_memories: list[dict[str, Any]],
    quality_feedback: str,
) -> dict[str, Any]:
    """Call Qwen through OpenRouter with a case-based memory prompt."""
    settings = get_openrouter_settings()
    if settings is None:
        raise RuntimeError("Configuration OpenRouter indisponible.")

    client = get_openrouter_client()
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {
            "role": "user",
            "content": _build_user_prompt(
                level=level,
                section=section,
                topic=topic,
                subtopic=subtopic,
                difficulty=difficulty,
                exercise_type=exercise_type,
                profile=profile,
                dataset_cases=dataset_cases,
                student_memories=student_memories,
                quality_feedback=quality_feedback,
            ),
        },
    ]

    request_kwargs = {
        "model": settings.exercise_model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 2600,
    }

    try:
        response = client.chat.completions.create(
            response_format={"type": "json_object"},
            **request_kwargs,
        )
    except Exception:
        response = client.chat.completions.create(**request_kwargs)

    content = extract_openrouter_text(response)
    if not content:
        raise RuntimeError(summarize_openrouter_response_issue(response))
    payload = _extract_json_payload(content)
    if not payload and content:
        payload = _repair_payload_with_model(
            client=client,
            model_name=settings.exercise_model,
            messages=messages,
            raw_content=content,
        )
    if not payload:
        raise RuntimeError("Le modele n'a pas renvoye un JSON exploitable pour l'exercice.")
    return payload


def _build_system_prompt() -> str:
    """Instruction set for the Memento-inspired exercise generator."""
    return (
        "Tu es l'agent de generation d'exercices de MathTutorAI pour les mathematiques du bac. "
        "Travaille selon une methode inspiree de Memento : "
        "1) memory reading : lis les cas analogues issus du dataset et de la memoire etudiante ; "
        "2) case selection : privilegie les cas les plus proches de la section, du theme, du sous-theme et des difficultes de l'apprenant ; "
        "3) reuse and revise : reutilise le schema pedagogique sans recopier l'exercice source ; "
        "4) memory writing preparation : renvoie des champs propres et structurables pour que le systeme puisse stocker ce nouvel exercice comme nouvelle experience. "
        "Retourne uniquement un objet JSON valide. "
        "N'utilise pas de markdown. N'utilise pas de LaTeX. "
        "Emploie un francais clair et des notations texte simples comme x^2, sqrt(5), P(A), f(x)."
    )


def _build_user_prompt(
    *,
    level: str,
    section: str,
    topic: str,
    subtopic: str,
    difficulty: str,
    exercise_type: str,
    profile: dict[str, Any],
    dataset_cases: list[DatasetExerciseCase],
    student_memories: list[dict[str, Any]],
    quality_feedback: str,
) -> str:
    """Assemble the generation prompt from the request and retrieved memories."""
    dataset_block = "\n".join(
        [
            (
                f"- Cas {index}: annee={case.year or 'inconnue'} | theme={case.topic} | sous-theme={case.subtopic} | "
                f"enonce={_truncate(_sanitize_display_text(case.instruction), 700)} | "
                f"reponse_finale={_truncate(_sanitize_display_text(case.final_answer), 240)}"
            )
            for index, case in enumerate(dataset_cases, start=1)
        ]
    ) or "- Aucun cas dataset pertinent disponible."

    student_memory_block = "\n".join(
        [
            (
                f"- Memoire {index}: section={item.get('section')} | theme={item.get('topic')} | sous-theme={item.get('subtopic')} | "
                f"objectif={_sanitize_display_text(item.get('learning_objective', ''))} | "
                f"extrait={_sanitize_display_text(item.get('prompt_excerpt', ''))}"
            )
            for index, item in enumerate(student_memories, start=1)
        ]
    ) or "- Aucune memoire de generation anterieure pour ce profil."

    return (
        "Contexte de generation\n"
        f"- Niveau: {level}\n"
        f"- Section: {section}\n"
        f"- Theme: {topic}\n"
        f"- Sous-theme: {subtopic}\n"
        f"- Difficulte: {difficulty}\n"
        f"- Type d'exercice: {exercise_type}\n\n"
        "Profil apprenant\n"
        f"- Nom: {profile.get('name', 'Etudiant')}\n"
        f"- Focus actuel: {profile.get('current_focus', '')}\n"
        f"- Points a renforcer: {', '.join(profile.get('weak_topics', []))}\n"
        f"- Points forts: {', '.join(profile.get('strong_topics', []))}\n\n"
        "Memory reading - cas dataset recuperes\n"
        f"{dataset_block}\n\n"
        "Memory reading - experiences recentes de generation\n"
        f"{student_memory_block}\n\n"
        + (
            "Retour qualite du juge precedent\n"
            f"- Feedback a prendre en compte: {quality_feedback}\n\n"
            if quality_feedback.strip()
            else ""
        )
        + (
        "Tache\n"
        "Genere un nouvel exercice original, adapte au niveau du bac, aligne sur la section et le sous-theme demandes. "
        "Le nouvel exercice doit rester proche du style des cas recuperes, mais ne doit copier ni les nombres ni l'enonce source. "
        "Fais un exercice exploitable immediatement dans une interface educative.\n\n"
        "Format JSON attendu\n"
        "{\n"
        '  "title": "titre court",\n'
        '  "prompt": "enonce complet en une ou plusieurs phrases",\n'
        '  "hint": "indice utile sans donner toute la solution",\n'
        '  "learning_objective": "objectif pedagogique principal",\n'
        '  "expected_answer": "reponse finale attendue en texte simple",\n'
        '  "full_solution": "solution complete redigee en texte simple, sans markdown",\n'
        '  "answer_kind": "text, numeric, expression ou set",\n'
        '  "solution_steps": ["etape 1", "etape 2", "etape 3"],\n'
        '  "options": ["option A", "option B", "option C", "option D"],\n'
        '  "table_data": {"caption": "texte court", "headers": ["colonne 1", "colonne 2"], "rows": [[1, 2], [3, 4]]},\n'
        '  "chart_data": {"type": "scatter ou line ou bar", "title": "titre", "caption": "texte court", "x_label": "axe x", "y_label": "axe y", "series": [{"name": "serie 1", "x": [1, 2], "y": [3, 4]}]},\n'
        '  "memory_rationale": "phrase courte expliquant comment les cas recuperes ont inspire l exercice",\n'
        '  "estimated_time": "estimation en minutes"\n'
        "}\n"
        "Regles supplementaires\n"
        "- Si le type n'est pas QCM, renvoie une liste options vide.\n"
        "- Garde 3 a 5 etapes de solution.\n"
        "- La reponse finale doit rester concise.\n"
        "- La solution complete doit etre coherente avec l'enonce et la reponse finale.\n"
        "- Pour un QCM, la solution complete doit identifier la bonne option et la justifier.\n"
        "- L'enonce final doit contenir de vraies taches a faire par l'eleve. Un simple contexte sans question ni consigne est interdit.\n"
        "- Pour un exercice probleme, donne au moins deux demandes explicites formulees comme des questions ou des verbes d'action numerotes.\n"
        "- Pour un QCM, pose une question claire avant les options.\n"
        "- Remplis table_data et chart_data seulement si l'enonce dit explicitement qu'un tableau, un graphique, une courbe fournie ou une annexe est deja donne(e) a l'eleve.\n"
        "- N'ajoute jamais un support que l'eleve doit lui-meme construire, par exemple un tableau de variation, un tableau de signe, une courbe a tracer ou un nuage de points a representer.\n"
        "- N'ecris jamais une consigne du type 'voir annexe' sans fournir les donnees correspondantes dans le JSON.\n"
        "- N'ajoute aucun texte avant ou apres le JSON."
        )
    )


def _build_exercise_payload(
    *,
    payload: dict[str, Any],
    level: str,
    section: str,
    topic: str,
    subtopic: str,
    difficulty: str,
    exercise_type: str,
    dataset_cases: list[DatasetExerciseCase],
    student_memories: list[dict[str, Any]],
    generation_backend: str,
    generation_warning: str,
) -> dict[str, Any]:
    """Normalize the model output into the exercise structure used by the UI."""
    generated_at = datetime.now().isoformat(timespec="seconds")
    exercise_id_seed = f"{section}|{topic}|{subtopic}|{generated_at}"
    exercise_id = hashlib.sha1(exercise_id_seed.encode("utf-8")).hexdigest()[:10].upper()

    solution_steps = payload.get("solution_steps") or []
    if not isinstance(solution_steps, list):
        solution_steps = [str(solution_steps)]
    solution_steps = [_sanitize_display_text(step) for step in solution_steps if str(step).strip()][:5]
    if not solution_steps:
        solution_steps = [
            "Identifier la notion centrale de l'enonce.",
            "Appliquer la propriete ou la formule adaptee.",
            "Simplifier jusqu'a la reponse finale.",
        ]

    options = payload.get("options") or []
    if not isinstance(options, list):
        options = [str(options)]
    options = [_sanitize_display_text(option) for option in options if str(option).strip()]
    if exercise_type != "QCM":
        options = []

    expected_answer = _sanitize_display_text(payload.get("expected_answer", "Reponse en attente"))
    answer_kind = str(payload.get("answer_kind", "text")).strip().lower() or "text"
    title = _sanitize_display_text(payload.get("title", f"Exercice sur {subtopic}")) or f"Exercice sur {subtopic}"
    prompt = _clean_exercise_statement(payload.get("prompt", "")) or (
        f"Traitez un exercice de niveau {difficulty.lower()} sur {subtopic.lower()}."
    )
    hint = _sanitize_display_text(payload.get("hint", "Repere la notion mathematique principale avant de calculer."))
    learning_objective = _sanitize_display_text(
        payload.get("learning_objective", f"Renforcer la maitrise de {subtopic.lower()}.")
    ) or f"Renforcer la maitrise de {subtopic.lower()}."
    hidden_solution = _sanitize_display_text(payload.get("full_solution", ""))
    if not hidden_solution:
        hidden_solution = _compose_full_solution(solution_steps, expected_answer)

    exercise = {
        "id": f"LLM-{exercise_id}",
        "section": section,
        "topic": topic,
        "subtopic": subtopic,
        "level": level,
        "generated_at": generated_at,
        "difficulty": difficulty,
        "exercise_type": exercise_type,
        "title": title,
        "prompt": prompt,
        "hint": hint,
        "accepted_answers": [expected_answer] if expected_answer else [],
        "display_answer": expected_answer or "Reponse en attente",
        "answer_kind": answer_kind,
        "solution_steps": solution_steps,
        "hidden_solution": hidden_solution,
        "learning_objective": learning_objective,
        "tags": [section, topic, subtopic, difficulty, exercise_type],
        "estimated_time": _normalize_estimated_time(payload.get("estimated_time"), difficulty),
        "generation_backend": generation_backend,
        "generation_warning": generation_warning,
        "verification_ready": False,
        "verification_message": "La verification automatique des exercices generes par OpenRouter sera ajoutee dans l'etape suivante.",
        "memory_adaptation_note": _sanitize_display_text(
            payload.get(
                "memory_rationale",
                "L'exercice a ete genere en reutilisant des cas analogues du dataset et de la memoire de session.",
            )
        ),
        "retrieved_case_ids": [case.case_id for case in dataset_cases],
        "retrieved_memory_count": len(student_memories),
        "source_case_summaries": [
            {
                "case_id": case.case_id,
                "year": case.year,
                "topic": case.topic,
                "subtopic": case.subtopic,
            }
            for case in dataset_cases
        ],
    }
    if payload.get("table_data") is not None:
        exercise["table_data"] = payload.get("table_data")
    if payload.get("chart_data") is not None:
        exercise["chart_data"] = payload.get("chart_data")
    if options:
        exercise["options"] = options
    return exercise


def _build_fallback_payload(
    *,
    topic: str,
    subtopic: str,
    difficulty: str,
    exercise_type: str,
    dataset_cases: list[DatasetExerciseCase],
) -> dict[str, Any]:
    """Provide a dataset-aware fallback when the remote model is unavailable."""
    exemplar = dataset_cases[0] if dataset_cases else None
    base_prompt = _sanitize_display_text(
        exemplar.instruction if exemplar else f"Construis un exercice de niveau {difficulty} sur {subtopic}."
    )
    base_prompt = _clean_exercise_statement(base_prompt)
    if base_prompt and not re.search(r"[.!?]$", base_prompt):
        base_prompt = f"{base_prompt}."
    return {
        "title": f"Exercice inspire du dataset : {subtopic}",
        "prompt": base_prompt
        or (
            f"Resoudre un exercice de type {exercise_type.lower()} sur {subtopic.lower()}. "
            "Justifier les etapes essentielles puis conclure clairement."
        ),
        "hint": "Repere d'abord la propriete, la formule ou le theoreme central avant de lancer les calculs.",
        "learning_objective": f"Mobiliser les connaissances de {subtopic.lower()} dans un nouvel enonce.",
        "expected_answer": (
            _sanitize_display_text(exemplar.final_answer)
            if exemplar and exemplar.final_answer
            else "Voir la correction a venir."
        ),
        "full_solution": (
            _sanitize_display_text(exemplar.solution)
            if exemplar and exemplar.solution
            else (
                "Analyser les donnees de l'enonce, choisir la propriete adaptee, "
                "developper le calcul et conclure par une reponse finale explicite."
            )
        ),
        "answer_kind": "text",
        "solution_steps": [
            "Analyser les donnees et identifier la notion mathematique visee.",
            "Structurer la resolution a partir du theoreme ou de la formule utile.",
            "Conclure par une reponse finale claire et justifiee.",
        ],
        "options": [],
        "memory_rationale": "Fallback local : construction a partir du cas dataset le plus proche faute d'appel distant disponible.",
        "estimated_time": _default_estimated_time(difficulty),
    }


def _compose_full_solution(solution_steps: list[str], expected_answer: str) -> str:
    """Build a hidden full solution when the model omits it."""
    numbered_steps = " ".join(
        [f"Etape {index} : {step}" for index, step in enumerate(solution_steps, start=1) if step]
    ).strip()
    if expected_answer and expected_answer != "Reponse en attente":
        return f"{numbered_steps} Reponse finale : {expected_answer}.".strip()
    return numbered_steps or "Solution detaillee indisponible."


def _extract_json_payload(raw_content: str) -> dict[str, Any]:
    """Parse the first valid JSON object returned by the model."""
    content = (raw_content or "").strip()
    if not content:
        return {}

    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()

    parsed = _load_json_candidate(content)
    if parsed:
        return parsed

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    return _load_json_candidate(content[start : end + 1])


def _repair_payload_with_model(
    *,
    client: Any,
    model_name: str,
    messages: list[dict[str, str]],
    raw_content: str,
) -> dict[str, Any]:
    """Ask the model to rewrite its previous answer as strict JSON before falling back."""
    repair_messages = [
        *messages,
        {"role": "assistant", "content": raw_content},
        {
            "role": "user",
            "content": (
                "Reformate strictement ta reponse precedente en un objet JSON valide. "
                "Ne mets aucun texte avant ou apres le JSON. "
                "Supprime le LaTeX et garde des notations texte simples."
            ),
        },
    ]

    request_kwargs = {
        "model": model_name,
        "messages": repair_messages,
        "temperature": 0,
        "max_tokens": 1100,
    }

    try:
        repair_response = client.chat.completions.create(
            response_format={"type": "json_object"},
            **request_kwargs,
        )
    except Exception:
        repair_response = client.chat.completions.create(**request_kwargs)

    repair_content = extract_openrouter_text(repair_response)
    return _extract_json_payload(repair_content)


def _default_estimated_time(difficulty: str) -> str:
    """Return a stable estimated time for the selected difficulty."""
    return {
        "Fondamental": "8 a 10 min",
        "Intermédiaire": "10 a 14 min",
        "Avancé": "14 a 18 min",
        "Défi": "18 a 25 min",
    }.get(difficulty, "10 a 14 min")


def _truncate(value: str, max_length: int) -> str:
    """Shorten long context fields before sending them to the model."""
    compact = " ".join(str(value).split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3].rstrip() + "..."


def _load_json_candidate(candidate: str) -> dict[str, Any]:
    """Load a JSON candidate, repairing common invalid backslashes if needed."""
    for payload in (candidate, _repair_invalid_json_backslashes(candidate)):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _repair_invalid_json_backslashes(value: str) -> str:
    """Escape stray backslashes such as LaTeX markers so JSON parsing can recover."""
    repaired = re.sub(r"\\(?=[A-Za-z]{2,})", r"\\\\", value)
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", repaired)


def _normalize_estimated_time(value: Any, difficulty: str) -> str:
    """Normalize numeric or textual duration values into a display-friendly label."""
    if isinstance(value, (int, float)):
        return f"{int(value)} min"

    cleaned = _sanitize_display_text(value)
    if not cleaned:
        return _default_estimated_time(difficulty)
    if cleaned.isdigit():
        return f"{cleaned} min"
    return cleaned


def assess_exercise_completeness(exercise: dict[str, Any]) -> dict[str, Any]:
    """Inspect one candidate before display and reject obviously truncated statements."""
    prompt = _clean_exercise_statement(exercise.get("prompt", ""))
    normalized_prompt = _normalize_for_match(prompt)
    issues: list[str] = []

    if not prompt:
        issues.append("L'enonce est vide apres nettoyage.")
    if len(prompt) < 90:
        issues.append("L'enonce est trop court pour etre autoporteur.")
    if "..." in prompt or "…" in prompt:
        issues.append("L'enonce contient des points de suspension qui signalent une coupure.")
    if "a partir de l idee suivante" in normalized_prompt or "redigez et resolvez un exercice" in normalized_prompt:
        issues.append("L'enonce affiche encore une consigne interne destinee au modele.")
    if "objectif pedagogique" in normalized_prompt:
        issues.append("L'enonce contient encore un bloc de metadonnees interne.")
    if any(marker in normalized_prompt for marker in ["<div", "</div", "class=", "card-body"]):
        issues.append("L'enonce contient encore des fragments HTML techniques.")
    if _looks_cut_off(prompt):
        issues.append("L'enonce semble se terminer de maniere tronquee.")
    if not _has_explicit_student_task(prompt, exercise.get("exercise_type", "")):
        issues.append("L'enonce presente seulement un contexte sans question ni consigne explicite.")

    needs_options = str(exercise.get("exercise_type", "")).strip().upper() == "QCM"
    if needs_options and len(exercise.get("options") or []) < 2:
        issues.append("Le QCM ne contient pas assez de propositions.")

    support_ready = exercise.get("support_ready")
    if support_ready is False and ("annexe" in normalized_prompt or "tableau" in normalized_prompt or "graphe" in normalized_prompt):
        issues.append("L'enonce renvoie vers une annexe manquante ou incomplete.")

    return {
        "is_complete": not issues,
        "summary": (
            "L'enonce est complet et peut etre soumis au juge."
            if not issues
            else "L'enonce est refuse avant affichage car il reste incomplet ou pollue par des artefacts internes."
        ),
        "issues": issues,
        "clean_prompt": prompt,
    }


def _sanitize_display_text(value: Any) -> str:
    """Convert model or dataset text into clean plain text for the Streamlit UI."""
    text = unescape(str(value or "")).replace("\r", "").strip()
    if not text:
        return ""

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\$\$(.+?)\$\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\$(.+?)\$", r"\1", text)

    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
        text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)

    replacements = {
        "\\(": "(",
        "\\)": ")",
        "\\[": "[",
        "\\]": "]",
        "\\mapsto": " -> ",
        "\\to": " -> ",
        "\\in": " in ",
        "\\left": "",
        "\\right": "",
        "\\times": " x ",
        "\\cdot": " * ",
        "\\leq": " <= ",
        "\\geq": " >= ",
        "\\le": " <= ",
        "\\ge": " >= ",
        "\\neq": " != ",
        "\\infty": " infini ",
        "\\mathbb{N}": " N ",
        "\\mathbb{R}": " R ",
        "\\ln": "ln",
        "\\alpha": "alpha",
        "\\beta": "beta",
        "\\eta": "eta",
        "\\lambda": "lambda",
        "\\pi": "pi",
        "\\sqrt": "sqrt",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_exercise_statement(value: Any) -> str:
    """Remove internal prompt wrappers and leaked metadata from one student-facing statement."""
    raw_text = str(value or "").replace("\r", "").strip()
    for _ in range(3):
        decoded = unescape(raw_text)
        if decoded == raw_text:
            break
        raw_text = decoded
    raw_text = _truncate_internal_prompt_fragment(raw_text)
    raw_text = re.sub(
        r"<div[^>]*exercise-objective[^>]*>.*$",
        " ",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    raw_text = re.sub(
        r"&lt;div[^&]*exercise-objective[^&]*&gt;.*$",
        " ",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = _sanitize_display_text(raw_text)
    if not text:
        return ""

    text = re.sub(
        r"^A partir de l['’]id[ée]e suivante,\s*r[ée]digez et r[ée]solvez un exercice de type.+?:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^R[ée]digez et r[ée]solvez un exercice de type.+?:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"exercise-objective.*$", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.split(r"\bObjectif p[ée]dagogique\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.split(r"\bMemory adaptation\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.split(r"\bSource\s*:\s*", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -:\n\t")


def _looks_cut_off(text: str) -> bool:
    """Heuristic detection for statements that end mid-sentence."""
    compact = text.strip()
    if not compact:
        return True
    if compact.endswith(("...", "…", ":", ";", ",")):
        return True

    terminal_match = re.findall(r"[A-Za-zÀ-ÿ0-9_+\-*/^=]+", compact)
    if not terminal_match:
        return False

    last_token = _normalize_for_match(terminal_match[-1])
    if last_token in {"pour", "avec", "et", "ou", "de", "du", "des", "la", "le", "les", "un", "une", "sur", "dans", "par", "au", "aux", "en"}:
        return True
    if compact.count("(") > compact.count(")") + 1:
        return True
    return False


def _has_explicit_student_task(text: str, exercise_type: Any) -> bool:
    """Check that the statement contains at least one concrete task for the student."""
    normalized_text = _normalize_for_match(text)
    if not normalized_text:
        return False

    if re.search(r"[A-Za-z0-9\)\]]\?(?:\s|$)", text):
        return True

    action_patterns = [
        r"\b[0-9]+\s*[\)\.]",
        r"(?:^|\s)[a-d]\)",
        r"\bmontrer\b",
        r"\bcalculer\b",
        r"\bdeterminer\b",
        r"\betudier\b",
        r"\bresoudre\b",
        r"\bjustifier\b",
        r"\brepresenter\b",
        r"\btracer\b",
        r"\bdresser\b",
        r"\bdeduire\b",
        r"\bverifier\b",
        r"\bcompleter\b",
        r"\bprouver\b",
        r"\bdonner\b",
        r"\bexprimer\b",
        r"\bconstruire\b",
        r"\bencadrer\b",
        r"\btrouver\b",
        r"\bcomparer\b",
    ]
    if any(re.search(pattern, normalized_text) for pattern in action_patterns):
        return True

    if str(exercise_type).strip().upper() == "QCM":
        return any(
            marker in normalized_text
            for marker in ["choisir", "cocher", "quelle est", "laquelle", "parmi les propositions"]
        )

    return False


def _normalize_for_match(value: Any) -> str:
    """Fold accents and punctuation for resilient local keyword checks."""
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text).strip().lower()


def _truncate_internal_prompt_fragment(text: str) -> str:
    """Cut internal leaked HTML or objective metadata from the first marker onward."""
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
