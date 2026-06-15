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
from frontend.utils.exercise_schema import compose_instruction, has_explicit_questions
from frontend.utils.exercise_memory import (
    DatasetExerciseCase,
    build_memory_adapted_generation_prompt,
    find_too_similar_source_case,
    get_negative_memory_patterns,
    get_positive_memory_patterns,
    retain_generation_memory,
    retrieve_dataset_cases,
    retrieve_generation_memories,
)
from frontend.utils.math_format_guard import repair_corrupted_latex_commands, repair_math_text_locally
from frontend.utils.openrouter_client import (
    call_openrouter_chat,
    extract_json_object,
    get_openrouter_settings,
    has_openrouter_config,
    is_dataset_demo_allowed,
    normalize_generated_exercise_payload,
    parse_json_object_detailed,
    validate_generated_exercise_schema,
)


class ExerciseGenerationError(RuntimeError):
    """Raised when the remote LLM generation path fails before a valid exercise exists."""

    def __init__(
        self,
        message: str,
        *,
        issues: list[str] | None = None,
        parse_status: str = "invalid_json",
        raw_output: str = "",
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.issues = list(issues or [])
        self.parse_status = parse_status
        self.raw_output = raw_output
        self.diagnostics = dict(diagnostics or {})


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
    allow_dataset_demo: bool = False,
    previous_errors: list[str] | None = None,
    previous_bad_outputs: list[str] | None = None,
    generation_strategy: str = "normal_memory_adapted_generation",
) -> dict[str, Any]:
    """Generate a new exercise using case-based memory adaptation."""
    active_profile = deepcopy(profile or st.session_state.get("student_profile", DEFAULT_STUDENT_PROFILE))
    dataset_cases = retrieve_dataset_cases(
        section=section,
        topic=topic,
        subtopic=subtopic,
        profile=active_profile,
        difficulty=difficulty,
        exercise_type=exercise_type,
        top_k=3,
    )
    student_memories = retrieve_generation_memories(
        section=section,
        topic=topic,
        subtopic=subtopic,
        top_k=2,
    )

    generation_backend = "openrouter-llm"
    generation_warning = ""
    llm_json_parse_status = "valid_json"
    openrouter_diagnostics: dict[str, Any] = {}
    payload: dict[str, Any]

    if has_openrouter_config():
        try:
            positive_patterns = get_positive_memory_patterns(section, topic, subtopic)
            negative_patterns = get_negative_memory_patterns(section, topic, subtopic)
            payload, llm_json_parse_status, openrouter_diagnostics = _call_openrouter_for_exercise(
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
                previous_errors=previous_errors or [],
                previous_bad_outputs=previous_bad_outputs or [],
                positive_patterns=positive_patterns,
                negative_patterns=negative_patterns,
                generation_strategy=generation_strategy,
            )
            generation_backend = (
                "openrouter-llm"
                if llm_json_parse_status == "valid_json"
                else "openrouter-llm-repaired-json"
            )
        except Exception as exc:
            if allow_dataset_demo and is_dataset_demo_allowed():
                generation_backend = "trusted-dataset-demo"
                generation_warning = (
                    "Mode demonstration actif : l'appel OpenRouter a echoue, exercice repris depuis le dataset."
                )
                llm_json_parse_status = getattr(exc, "parse_status", "invalid_json")
                openrouter_diagnostics = getattr(exc, "diagnostics", {})
                payload = _build_fallback_payload(
                    topic=topic,
                    subtopic=subtopic,
                    difficulty=difficulty,
                    exercise_type=exercise_type,
                    dataset_cases=dataset_cases,
                )
            else:
                raise
    else:
        generation_backend = "trusted-dataset-demo"
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
        llm_json_parse_status=llm_json_parse_status,
        demo_mode_used=(generation_backend == "trusted-dataset-demo"),
        openrouter_diagnostics=openrouter_diagnostics,
    )
    exercise["retry_strategy"] = generation_strategy
    similar, similar_case_id, similarity_score = find_too_similar_source_case(exercise.get("prompt", ""), dataset_cases)
    exercise["too_similar_to_source_case"] = similar
    exercise["too_similar_source_case_id"] = similar_case_id
    exercise["too_similar_source_case_score"] = similarity_score
    completeness_review = assess_exercise_completeness(exercise)
    exercise["prompt"] = completeness_review["clean_prompt"]
    exercise["instruction"] = completeness_review["clean_prompt"]

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
    previous_errors: list[str],
    previous_bad_outputs: list[str],
    positive_patterns: list[dict[str, Any]],
    negative_patterns: list[dict[str, Any]],
    generation_strategy: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Call Qwen through OpenRouter with a case-based memory prompt."""
    settings = get_openrouter_settings()
    if settings is None:
        raise RuntimeError("Configuration OpenRouter indisponible.")

    messages = [
        {"role": "system", "content": _build_system_prompt()},
        *build_memory_adapted_generation_prompt(
            section=section,
            topic=topic,
            subtopic=subtopic,
            difficulty=difficulty,
            exercise_type=exercise_type,
            retrieved_cases=dataset_cases,
            positive_patterns=positive_patterns,
            negative_patterns=negative_patterns,
            previous_errors=[*previous_errors, quality_feedback] if quality_feedback else previous_errors,
        ),
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
                previous_errors=previous_errors,
                previous_bad_outputs=previous_bad_outputs,
                generation_strategy=generation_strategy,
            ),
        },
    ]

    diagnostics_base = _build_prompt_diagnostics(messages=messages, dataset_cases=dataset_cases)
    call_result = call_openrouter_chat(
        model=settings.exercise_model,
        messages=messages,
        temperature=0.1,
        top_p=0.9,
        max_tokens=4000 if generation_strategy != "simple_exercise_generation" else 2500,
        purpose="exercise",
    )
    diagnostics = {**diagnostics_base, **_call_result_diagnostics(call_result)}
    if not call_result.ok:
        raise ExerciseGenerationError(
            call_result.error_message or "L'appel OpenRouter a echoue avant reception d'une sortie exploitable.",
            issues=[call_result.error_message or call_result.error_type or "OpenRouter call failed."],
            parse_status=call_result.error_type or "unknown_error",
            raw_output=call_result.raw_response_preview,
            diagnostics=diagnostics,
        )

    content = call_result.content
    parse_result = parse_json_object_detailed(content)
    payload = parse_result.data
    parse_status = "valid_json" if parse_result.ok else "invalid_json"
    diagnostics.update(
        {
            "llm_raw_response_preview": parse_result.raw_preview or call_result.raw_response_preview,
            "llm_json_extraction_method": parse_result.extraction_method,
            "llm_json_parse_error": parse_result.error or "",
        }
    )
    if not payload and content:
        payload = _repair_payload_with_model(
            model_name=settings.exercise_model,
            messages=messages,
            raw_content=content,
        )
        if payload:
            parse_status = "repaired_json"
            diagnostics["llm_json_extraction_method"] = "repaired_json"
    if not payload:
        raise ExerciseGenerationError(
            "Le modele n'a pas renvoye un JSON exploitable pour l'exercice.",
            parse_status="invalid_json",
            raw_output=content,
            diagnostics=diagnostics,
        )

    normalized_payload = _repair_payload_latex_fields(normalize_generated_exercise_payload(payload))
    is_valid, schema_issues = validate_generated_exercise_schema(normalized_payload)
    if not is_valid:
        raise ExerciseGenerationError(
            "Le JSON de l'exercice ne respecte pas le schema attendu.",
            issues=schema_issues,
            parse_status=parse_status,
            raw_output=content,
            diagnostics=diagnostics,
        )
    return normalized_payload, parse_status, diagnostics


def _build_system_prompt() -> str:
    """Instruction set for the Memento-inspired exercise generator."""
    return (
        "You are MathTutorAI, a strict mathematical exercise generator for Tunisian Baccalaureate students. "
        "LLM generation is the main path. Deterministic validators only verify or repair arithmetic. "
        "Return ONLY one valid JSON object, with no markdown fence and no prose outside JSON. "
        "The exercise must be fully solvable from the statement and contain a context, at least two explicit numbered questions, and all required data. "
        "All math expressions must use valid LaTeX inside \\( ... \\) or \\[ ... \\]. "
        "Never output malformed tokens such as frace, fracpi, mathbb R, in fty, e^{0U}_0 or \\\\int_\\alpha^{{{0 f}}}. "
        "Use \\frac{...}{...}, \\mathbb{R}, \\mathbb{N}, \\infty and \\int_a^b f(x)\\,dx."
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
    previous_errors: list[str],
    previous_bad_outputs: list[str],
    generation_strategy: str,
) -> str:
    """Assemble the generation prompt from the request and retrieved memories."""
    prompt_cases = dataset_cases[:2]
    dataset_block = "\n".join(
        [
            (
                f"- Cas {index}: annee={case.year or 'inconnue'} | theme={case.topic} | sous-theme={case.subtopic} | "
                f"enonce={_truncate(_sanitize_display_text(case.instruction), 700)} | "
                f"solution={_truncate(_sanitize_display_text(case.solution), 700)} | "
                f"reponse_finale={_truncate(_sanitize_display_text(case.final_answer), 240)}"
            )
            for index, case in enumerate(prompt_cases, start=1)
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
    previous_error_block = "\n".join(f"- {item}" for item in previous_errors if str(item).strip()) or "- Aucune"
    previous_output_block = "\n".join(
        f"- Sortie rejetee {index}: {_truncate(item, 240)}"
        for index, item in enumerate(previous_bad_outputs[-2:], start=1)
        if str(item).strip()
    ) or "- Aucune sortie brute rejetee."

    return (
        "Contexte de generation\n"
        f"- Niveau: {level}\n"
        f"- Section: {section}\n"
        f"- Theme: {topic}\n"
        f"- Sous-theme: {subtopic}\n"
        f"- Difficulte: {difficulty}\n"
        f"- Type d'exercice: {exercise_type}\n"
        f"- Strategie courante: {generation_strategy}\n\n"
        "Profil apprenant\n"
        f"- Nom: {profile.get('name', 'Etudiant')}\n"
        f"- Focus actuel: {profile.get('current_focus', '')}\n"
        f"- Points a renforcer: {', '.join(profile.get('weak_topics', []))}\n"
        f"- Points forts: {', '.join(profile.get('strong_topics', []))}\n\n"
        "Cas dataset recuperes\n"
        f"{dataset_block}\n\n"
        "Memoire recente\n"
        f"{student_memory_block}\n\n"
        "Erreurs precedentes exactes\n"
        f"{previous_error_block}\n\n"
        "Sorties brutes precedemment rejetees\n"
        f"{previous_output_block}\n\n"
        + (
            "Retour qualite du juge precedent\n"
            f"- Feedback a prendre en compte: {quality_feedback}\n"
            "Tu dois corriger ces erreurs et ne jamais repeter la meme structure rejetee.\n\n"
            if quality_feedback.strip()
            else ""
        )
        + _build_hard_requirements_block(subtopic)
        + _build_strategy_instructions(generation_strategy, previous_errors, subtopic)
        + (
            "Schema JSON obligatoire\n"
            "{\n"
            '  "title": "string",\n'
            '  "context": "string",\n'
            '  "questions": ["string", "string", "string"],\n'
            '  "instruction": "string",\n'
            '  "solution": "string",\n'
            '  "expected_answer": "string",\n'
            '  "answer_kind": "text | expression | numeric | table",\n'
            '  "solution_steps": ["string"],\n'
            '  "learning_objective": "string",\n'
            '  "estimated_time": "string",\n'
            '  "table_data": null,\n'
            '  "chart_data": null,\n'
            '  "graph_data": null,\n'
            '  "generation_metadata": {\n'
            '    "target_section": "string",\n'
            '    "target_topic": "string",\n'
            '    "target_subtopic": "string",\n'
            '    "exercise_family": "string",\n'
            '    "requires_symbolic_check": true,\n'
            '    "requires_numeric_check": false\n'
            "  },\n"
            '  "hint": "string",\n'
            '  "memory_rationale": "string",\n'
            '  "options": []\n'
            "}\n"
            "Contraintes absolues\n"
            "- context decrit seulement la situation.\n"
            "- questions contient au moins deux vraies questions explicites. Si une tentative precedente a ete rejetee pour contexte sans question, donne au moins trois questions numerotees.\n"
            "- instruction doit etre compose de context puis des questions numerotees.\n"
            "- La solution doit repondre a toutes les questions.\n"
            "- expected_answer doit etre recalcule depuis la solution et ne jamais la contredire.\n"
            "- Si le type n'est pas QCM, options doit etre vide.\n"
            "- Toute expression mathematique doit etre en LaTeX propre entre \\( ... \\) ou \\[ ... \\].\n"
            "- N'ecris jamais frace, fracpi, mathbb R, in fty, e^{0U}_0 ou {{{0 f}}}.\n"
            "- Pour les probabilites, donne toutes les donnees utiles, garde les probabilites dans [0,1] et impose une loi qui somme a 1.\n"
            "- Si le sous-theme concerne variables aleatoires, inclure au moins: loi de probabilite, E(X), V(X) ou sigma(X).\n"
            "- Pour les suites numeriques, ecris explicitement U_0, U_{n+1} et au moins deux questions portant sur la recurrence ou sa consequence.\n"
            "- Pour la regression, tous les calculs doivent etre compatibles avec table_data.\n"
            "- N'ajoute aucun texte avant ou apres le JSON."
        )
    )


TOPIC_HARD_REQUIREMENTS: dict[str, list[str]] = {
    "suites numeriques": [
        "L'enonce doit ecrire explicitement la recurrence sous une forme lisible et parsable, par exemple \\( U_{n+1}=e^{-n}U_n \\) ou \\( u_{n+1}=au_n+b \\).",
        "Les conditions initiales doivent etre ecrites explicitement, par exemple \\( U_0=1 \\).",
        "La solution doit deriver correctement les premiers termes, une relation sur \\( V_n=\\ln(U_n) \\) si elle est introduite, une forme fermee si elle existe, puis la limite.",
        "Interdiction absolue des notations corrompues du type e^{0U}_0, e^{-nU}_n, mathbb N sans antislash ou ln mal ecrit.",
    ],
    "suites definies par une integrale": [
        "L'enonce doit definir explicitement une suite indexee par n a l'aide d'une integrale, par exemple \\( I_n=\\int_a^b ...\\,dx \\).",
        "Au moins deux questions doivent porter explicitement sur \\( I_n \\), \\( u_n \\) ou une suite indexee par \\( n \\).",
        "La solution doit justifier une limite, une monotonie, un encadrement, une recurrence ou un equivalent lie a cette suite.",
    ],
    "equations differentielles": [
        "L'enonce doit ecrire clairement l'equation differentielle, les conditions eventuelles et la fonction inconnue.",
        "La solution doit verifier la forme proposee par substitution et traiter les conditions initiales ou aux bornes si elles existent.",
    ],
    "probabilites variables aleatoires": [
        "Toutes les probabilites numeriques utiles doivent etre donnees explicitement dans l'enonce ou dans un tableau.",
        "Si l'exercice porte sur une loi binomiale, il faut ecrire explicitement n, p et l'evenement demande avec une formulation non ambigue du type au moins, au plus, exactement, plus de, moins de.",
    ],
    "fonction exponentielle logarithme": [
        "Le domaine de definition de \\( \\ln \\) ou de l'exponentielle doit etre respecte dans toutes les transformations.",
        "Les derivees, limites ou equivalences doivent rester compatibles avec le niveau Bac.",
    ],
    "séries à deux caractères, régression et corrélation": [
        "L'enonce doit choisir clairement une seule famille: regression classique (coefficient r, droite de regression, estimation) ou methode de Mayer (G, G1, G2, droite de Mayer).",
        "Si l'exercice est en regression classique, il n'est pas obligatoire de demander G, G1 ou G2.",
        "Si un tableau de donnees est fourni ou demande, la solution doit etre numeriquement coherente avec ce tableau.",
        "Ne modifie jamais un tableau statistique sans recalculer le coefficient de correlation, la droite de regression et toutes les estimations.",
    ],
    "suites définies par une intégrale": [
        "L'enonce doit definir explicitement une suite indexee par n a l'aide d'une integrale, par exemple \\( I_n=\\int_a^b ...\\,dx \\).",
        "Au moins deux questions doivent porter explicitement sur \\( I_n \\), \\( u_n \\) ou une suite indexee par \\( n \\).",
        "La solution doit justifier une limite, une monotonie, un encadrement, une recurrence ou un equivalent/asymptotique lie a cette suite.",
        "Une simple etude de fonction, meme avec une primitive, est interdite si elle ne contient pas une vraie suite definie par integrale.",
    ],
    "fonction logarithme népérien": [
        "L'enonce doit mobiliser explicitement \\( \\ln \\) et son domaine de definition.",
        "Les limites, derivees ou integrales demandees doivent rester compatibles avec le niveau Bac.",
    ],
    "fonctions réciproques et bijection": [
        "L'enonce doit demander de prouver une bijection sur un intervalle precis et de definir ou utiliser la reciproque.",
        "La solution doit verifier la monotonie, l'image de l'intervalle et la formule de la reciproque si elle est demandee.",
    ],
    "conditionnement, probabilités totales et Bayes": [
        "L'enonce doit fournir toutes les probabilites numeriques necessaires.",
        "La solution doit utiliser explicitement une probabilite conditionnelle, la formule des probabilites totales ou la formule de Bayes.",
    ],
    "nombres complexes": [
        "Ne dis jamais que \\( e^{i\\pi/3} \\) est une cinquieme racine de l'unite: il faut que \\(5\\theta\\) soit un multiple de \\(2\\pi\\).",
        "La condition \\(\\arg(z')\\equiv-\\arg(z)\\) ne prouve pas \\(z'=\\overline z\\) sans egalite des modules.",
        "Si l'exercice est Vrai/Faux, expected_answer doit contenir exactement une reponse par item.",
    ],
    "matrices determinants systemes lineaires": [
        "Utilise de preference un systeme 2x2 a solution entiere.",
        "Verifie que expected_answer satisfait toutes les equations.",
        "Ecris les systemes avec \\begin{cases} ... \\end{cases} en LaTeX propre.",
    ],
    "graphes": [
        "Fournis toujours graph_data avec vertices, edges et directed.",
        "Ecris aussi le graphe dans l'enonce sous la forme V={...} et E={...}.",
    ],
}


def _build_hard_requirements_block(subtopic: str) -> str:
    """Return subtopic-specific constraints injected into the generator prompt."""
    normalized = _normalize_for_match(subtopic)
    requirements: list[str] = []
    for key, value in TOPIC_HARD_REQUIREMENTS.items():
        if _normalize_for_match(key) == normalized:
            requirements = value
            break
    if not requirements:
        return (
            "Contraintes du sous-theme\n"
            "- L'exercice doit rester strictement centre sur le sous-theme demande.\n"
            "- Les questions doivent citer explicitement la notion du sous-theme dans l'enonce ou dans les objets mathematiques utilises.\n\n"
        )
    items = "\n".join(f"- {item}" for item in requirements)
    return f"Contraintes du sous-theme\n{items}\n\n"


def _build_strategy_instructions(generation_strategy: str, previous_errors: list[str], subtopic: str) -> str:
    """Inject reason-aware retry instructions into the generator prompt."""
    normalized_strategy = _normalize_for_match(generation_strategy)
    normalized_errors = _normalize_for_match(" ".join(previous_errors))

    base = ["Instructions de retry"]
    if "context_only" in normalized_errors or "sans question" in normalized_errors:
        base.append("- La tentative precedente a ete refusee car l'enonce n'avait pas de vraies questions. Tu dois fournir au moins 3 questions numerotees.")
    if "probabil" in normalized_errors:
        base.append("- La tentative precedente avait une incoherence de probabilites. Utilise un modele fini simple et verifie que la loi somme a 1.")
    if "expected_answer" in normalized_errors or "reponse attendue" in normalized_errors or "contrad" in normalized_errors:
        base.append("- expected_answer doit etre rederive de la solution complete et contenir exactement les resultats finaux.")
    if "json" in normalized_errors or "schema" in normalized_errors:
        base.append("- Retourne un JSON minimal, strict et sans aucun texte parasite.")

    if normalized_strategy == "strict_schema_generation":
        base.append("- Priorite absolue au schema: contexte court puis exactement 3 questions numerotees.")
    elif normalized_strategy == "simple_exercise_generation":
        base.append("- Genere un exercice plus simple, avec donnees numeriques modestes et une structure tres claire.")
    elif normalized_strategy == "topic_template_guided_generation":
        base.append(f"- Utilise un gabarit pedagogique standard du sous-theme {subtopic}, sans copier un cas source.")
    elif normalized_strategy == "deterministic_arithmetic_repair":
        base.append("- Si le theme est probabiliste ou arithmetique, choisis une structure dont les calculs peuvent etre recomputes localement.")

    return "\n".join(base) + "\n\n"


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
    llm_json_parse_status: str,
    demo_mode_used: bool,
    openrouter_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize the model output into the exercise structure used by the UI."""
    diagnostics = openrouter_diagnostics or {}
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

    normalized_payload = normalize_generated_exercise_payload(payload)
    expected_answer = _sanitize_display_text(normalized_payload.get("expected_answer", "Reponse en attente"))
    answer_kind = str(normalized_payload.get("answer_kind", "text")).strip().lower() or "text"
    title = _sanitize_display_text(normalized_payload.get("title", f"Exercice sur {subtopic}")) or f"Exercice sur {subtopic}"
    prompt = _clean_exercise_statement(normalized_payload.get("instruction", "")) or (
        f"Traitez un exercice de niveau {difficulty.lower()} sur {subtopic.lower()}."
    )
    context = _sanitize_display_text(normalized_payload.get("context", ""))
    questions = [_sanitize_display_text(item) for item in (normalized_payload.get("questions") or []) if str(item).strip()]
    instruction = compose_instruction(context, questions) if context or questions else prompt
    prompt = _clean_exercise_statement(instruction)
    hint = _sanitize_display_text(normalized_payload.get("hint", "Repere la notion mathematique principale avant de calculer."))
    learning_objective = _sanitize_display_text(
        normalized_payload.get("learning_objective", f"Renforcer la maitrise de {subtopic.lower()}.")
    ) or f"Renforcer la maitrise de {subtopic.lower()}."
    hidden_solution = _sanitize_display_text(normalized_payload.get("solution", ""))
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
        "context": context,
        "questions": questions,
        "instruction": prompt,
        "prompt": prompt,
        "hint": hint,
        "accepted_answers": [expected_answer] if expected_answer else [],
        "display_answer": expected_answer or "Reponse en attente",
        "answer_kind": answer_kind,
        "solution_steps": solution_steps,
        "hidden_solution": hidden_solution,
        "learning_objective": learning_objective,
        "tags": [section, topic, subtopic, difficulty, exercise_type],
        "estimated_time": _normalize_estimated_time(normalized_payload.get("estimated_time"), difficulty),
        "generation_backend": generation_backend,
        "generation_warning": generation_warning,
        "is_true_llm_generation": generation_backend in {"openrouter-llm", "openrouter-llm-repaired-json"},
        "llm_json_parse_status": llm_json_parse_status,
        "llm_json_extraction_method": diagnostics.get("llm_json_extraction_method", ""),
        "llm_json_parse_error": diagnostics.get("llm_json_parse_error", ""),
        "llm_raw_response_preview": diagnostics.get("llm_raw_response_preview", ""),
        "openrouter_http_status": diagnostics.get("openrouter_http_status"),
        "openrouter_error_type": diagnostics.get("openrouter_error_type", ""),
        "openrouter_error_message": diagnostics.get("openrouter_error_message", ""),
        "openrouter_response_format_mode": diagnostics.get("openrouter_response_format_mode", ""),
        "openrouter_model_used": diagnostics.get("openrouter_model_used", ""),
        "openrouter_request_id": diagnostics.get("openrouter_request_id", ""),
        "openrouter_provider": diagnostics.get("openrouter_provider", ""),
        "openrouter_usage": diagnostics.get("openrouter_usage"),
        "openrouter_call_attempts": diagnostics.get("openrouter_call_attempts", []),
        "prompt_char_count": diagnostics.get("prompt_char_count", 0),
        "prompt_token_estimate": diagnostics.get("prompt_token_estimate", 0),
        "number_of_memory_cases": diagnostics.get("number_of_memory_cases", len(dataset_cases)),
        "fallback_used": generation_backend in {"trusted-dataset-demo", "dataset-fallback-blocked", "local-fallback"},
        "fallback_reason": generation_warning if generation_backend in {"trusted-dataset-demo", "dataset-fallback-blocked", "local-fallback"} else "",
        "display_source_category": (
            "llm_generated"
            if generation_backend in {"openrouter-llm", "openrouter-llm-repaired-json"}
            else "demo_dataset"
        ),
        "demo_mode_used": demo_mode_used,
        "verification_ready": False,
        "verification_message": "La verification automatique des exercices generes par OpenRouter sera ajoutee dans l'etape suivante.",
        "memory_adaptation_note": _sanitize_display_text(
            normalized_payload.get(
                "memory_rationale",
                "L'exercice a ete genere en reutilisant des cas analogues du dataset et de la memoire de session.",
            )
        ),
        "retrieved_case_ids": [case.case_id for case in dataset_cases],
        "final_memory_case_ids": [case.case_id for case in dataset_cases[:2]],
        "memory_filter_stage": "strict_topic_filtered",
        "memory_rejected_case_ids": [],
        "memory_rejection_reasons": [],
        "retrieved_memory_count": len(student_memories),
        "source_case_summaries": [
            {
                "case_id": case.case_id,
                "year": case.year,
                "topic": case.topic,
                "subtopic": case.subtopic,
                "instruction": _truncate(_sanitize_display_text(case.instruction), 220),
            }
            for case in dataset_cases
        ],
        "source_case_instructions": [_sanitize_display_text(case.instruction) for case in dataset_cases],
        "generation_metadata": normalized_payload.get("generation_metadata", {}),
    }
    if normalized_payload.get("table_data") is not None:
        exercise["table_data"] = normalized_payload.get("table_data")
    if normalized_payload.get("chart_data") is not None:
        exercise["chart_data"] = normalized_payload.get("chart_data")
    if normalized_payload.get("graph_data") is not None:
        exercise["graph_data"] = normalized_payload.get("graph_data")
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
    if _normalize_for_match(subtopic) == _normalize_for_match("séries à deux caractères, régression et corrélation"):
        return _build_regression_fallback_payload(difficulty=difficulty, exercise_type=exercise_type)

    exemplar = dataset_cases[0] if dataset_cases else None
    base_prompt = _sanitize_display_text(
        exemplar.instruction if exemplar else f"Construis un exercice de niveau {difficulty} sur {subtopic}."
    )
    base_prompt = _clean_exercise_statement(base_prompt)
    if base_prompt and not re.search(r"[.!?]$", base_prompt):
        base_prompt = f"{base_prompt}."
    fallback_questions = [
        f"1) Identifier la notion principale mobilisee en {subtopic.lower()}.",
        "2) Resoudre l'exercice en justifiant les etapes essentielles.",
    ]
    return {
        "title": f"Exercice inspire du dataset : {subtopic}",
        "context": base_prompt
        or (
            f"Resoudre un exercice de type {exercise_type.lower()} sur {subtopic.lower()}. "
            "Justifier les etapes essentielles puis conclure clairement."
        ),
        "questions": fallback_questions,
        "instruction": compose_instruction(
            base_prompt
            or (
                f"Resoudre un exercice de type {exercise_type.lower()} sur {subtopic.lower()}. "
                "Justifier les etapes essentielles puis conclure clairement."
            ),
            fallback_questions,
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
        "generation_metadata": {
            "target_section": "",
            "target_topic": topic,
            "target_subtopic": subtopic,
            "exercise_family": exercise_type,
            "requires_symbolic_check": True,
            "requires_numeric_check": False,
        },
    }


def _build_regression_fallback_payload(*, difficulty: str, exercise_type: str) -> dict[str, Any]:
    """Build a deterministic, internally consistent classical-regression exercise.

    The numbers are computed locally so the table, regression line, coefficient r and
    estimations cannot drift apart when the LLM fails to return usable JSON.
    """
    years = list(range(2008, 2018))
    ranks = list(range(1, len(years) + 1))
    production = [1220, 1255, 1287, 1321, 1357, 1392, 1424, 1463, 1495, 1534]
    stats = _compute_simple_regression(ranks, production)
    if stats is None:
        # Defensive fallback: this branch should never happen with the static table above.
        return {
            "title": "Exercice de regression statistique",
            "prompt": "On considere une serie statistique double. Calculer le coefficient de correlation, determiner une droite de regression et l'utiliser pour une estimation.",
            "hint": "Commence par calculer les moyennes, puis la covariance et les variances.",
            "learning_objective": "Mobiliser la regression lineaire et la correlation.",
            "expected_answer": "Voir la correction detaillee.",
            "full_solution": "La correction sera etablie a partir des donnees du tableau.",
            "answer_kind": "text",
            "solution_steps": ["Calculer les grandeurs statistiques.", "Determiner la droite de regression.", "Utiliser la droite pour estimer."],
            "options": [],
            "estimated_time": _default_estimated_time(difficulty),
        }

    slope = stats["slope"]
    intercept = stats["intercept"]
    r_value = stats["r"]
    predicted_2023 = slope * 16 + intercept
    predicted_2024 = slope * 17 + intercept
    threshold = 1800
    threshold_rank = int((threshold - intercept) // slope) + 1
    threshold_year = 2007 + threshold_rank
    need_2024 = 12.5 * 135
    line = f"P={_fr_num(slope)}I+{_fr_num(intercept)}"
    enough_2024 = predicted_2024 >= need_2024
    prompt = (
        "Le tableau ci-dessous donne, pour les annees 2008 a 2017, la production de lait cru en Tunisie. "
        "On considere la serie statistique \\((I,P)\\), ou \\(I\\) est le rang de l'annee et \\(P\\) la production annuelle en millions de litres. "
        "1) Calculer le coefficient de correlation lineaire de la serie \\((I,P)\\) et interpreter le resultat. "
        "2) Determiner une equation de la droite de regression de \\(P\\) en \\(I\\), sous la forme \\(P=aI+b\\). "
        "3) a) Estimer la production de lait cru en 2023 ; "
        "b) determiner a partir de quelle annee la production depassera 1800 millions de litres ; "
        "c) sachant que la population tunisienne atteindra 12,5 millions d'habitants en 2024 et qu'en moyenne un Tunisien consomme 135 litres de lait cru par an, "
        "dire si la production estimee repondra au besoin de la Tunisie en 2024."
    )
    solution = (
        f"On prend les rangs \\(I=1,2,\\ldots,10\\) pour les annees 2008 a 2017. "
        f"Les calculs donnent un coefficient de correlation \\(r\\approx {_fr_num(r_value, 4)}\\), ce qui indique une tres forte correlation lineaire positive. "
        f"La droite de regression de \\(P\\) en \\(I\\) est \\({line}\\). "
        f"Pour 2023, le rang est \\(I=16\\), donc \\(P\\approx {_fr_num(slope)}\\times16+{_fr_num(intercept)}={_fr_num(predicted_2023)}\\) millions de litres. "
        f"La condition \\(P>1800\\) donne \\(I>{_fr_num((threshold - intercept) / slope)}\\), donc le plus petit rang entier convenable est \\(I={threshold_rank}\\), soit l'annee {threshold_year}. "
        f"En 2024, \\(I=17\\), donc la production estimee vaut \\({_fr_num(predicted_2024)}\\) millions de litres. "
        f"Le besoin est \\(12,5\\times135={_fr_num(need_2024)}\\) millions de litres ; "
        f"la production estimee {'suffit' if enough_2024 else 'ne suffit pas'} donc pour couvrir ce besoin."
    )
    expected = (
        f"\\(r\\approx {_fr_num(r_value, 4)}\\), \\({line}\\), production 2023 \\(\\approx {_fr_num(predicted_2023)}\\) millions de litres, "
        f"depassement de 1800 a partir de {threshold_year}, production 2024 {'suffisante' if enough_2024 else 'insuffisante'}."
    )
    return {
        "title": "Regression lineaire et correlation",
        "context": "Le tableau ci-dessous donne, pour les annees 2008 a 2017, la production de lait cru en Tunisie. On considere la serie statistique \\((I,P)\\), ou \\(I\\) est le rang de l'annee et \\(P\\) la production annuelle en millions de litres.",
        "questions": [
            "Calculer le coefficient de correlation lineaire de la serie \\((I,P)\\) et interpreter le resultat.",
            "Determiner une equation de la droite de regression de \\(P\\) en \\(I\\), sous la forme \\(P=aI+b\\).",
            "Estimer la production de lait cru en 2023 puis determiner a partir de quelle annee la production depassera 1800 millions de litres.",
        ],
        "instruction": prompt,
        "hint": "Utilise les rangs \\(I=1,2,\\ldots,10\\), puis calcule \\(r\\), la droite \\(P=aI+b\\) et les estimations demandees.",
        "learning_objective": "Exploiter une serie statistique a deux caracteres par correlation, regression et estimation.",
        "expected_answer": expected,
        "full_solution": solution,
        "answer_kind": "text",
        "solution_steps": [
            "Associer a chaque annee son rang \\(I\\).",
            "Calculer le coefficient de correlation et interpreter son signe et sa valeur.",
            "Determiner la droite de regression \\(P=aI+b\\).",
            "Utiliser la droite pour estimer les productions et comparer au besoin donne.",
        ],
        "options": [],
        "table_data": {
            "caption": "Production de lait cru en Tunisie de 2008 a 2017",
            "headers": ["Annee", "Production (en millions de litres)"],
            "rows": [[year, value] for year, value in zip(years, production)],
        },
        "memory_rationale": "Fallback deterministe : generation controlee d'un exercice de regression classique avec calculs recomputes localement.",
        "estimated_time": _default_estimated_time(difficulty),
        "generation_metadata": {
            "target_section": "",
            "target_topic": "Statistiques",
            "target_subtopic": "series a deux caracteres, regression et correlation",
            "exercise_family": exercise_type,
            "requires_symbolic_check": False,
            "requires_numeric_check": True,
        },
    }


def _compute_simple_regression(x_values: list[float], y_values: list[float]) -> dict[str, float] | None:
    n = len(x_values)
    if n < 3 or n != len(y_values):
        return None
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n
    sxx = sum((x - mean_x) ** 2 for x in x_values)
    syy = sum((y - mean_y) ** 2 for y in y_values)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    return {
        "slope": sxy / sxx,
        "intercept": mean_y - (sxy / sxx) * mean_x,
        "r": sxy / ((sxx * syy) ** 0.5),
    }


def _fr_num(value: float, decimals: int = 2) -> str:
    formatted = f"{value:.{decimals}f}"
    formatted = formatted.rstrip("0").rstrip(".")
    return formatted.replace(".", "{,}")


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
    parsed = extract_json_object(raw_content)
    return normalize_generated_exercise_payload(parsed) if parsed else {}


def _repair_payload_with_model(
    *,
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
                "Garde uniquement un LaTeX simple et valide pour les expressions mathematiques. "
                "Le JSON final doit contenir au minimum : title, context, questions, instruction, solution, expected_answer, "
                "answer_kind, solution_steps, learning_objective, estimated_time, table_data, chart_data, graph_data, generation_metadata."
            ),
        },
    ]

    repair_result = call_openrouter_chat(
        model=model_name,
        messages=repair_messages,
        temperature=0,
        top_p=0.1,
        max_tokens=2600,
        purpose="exercise_repair",
        response_mode="json_object",
    )
    if not repair_result.ok:
        return {}
    parse_result = parse_json_object_detailed(repair_result.content)
    return normalize_generated_exercise_payload(parse_result.data) if parse_result.data else {}


def _build_prompt_diagnostics(
    *,
    messages: list[dict[str, str]],
    dataset_cases: list[DatasetExerciseCase],
) -> dict[str, Any]:
    prompt_text = "\n".join(str(message.get("content", "")) for message in messages)
    return {
        "prompt_char_count": len(prompt_text),
        "prompt_token_estimate": max(1, len(prompt_text) // 4),
        "number_of_memory_cases": min(2, len(dataset_cases)),
        "retrieved_case_ids": [case.case_id for case in dataset_cases],
        "source_case_instructions": [_truncate(_sanitize_display_text(case.instruction), 700) for case in dataset_cases],
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


def _call_result_diagnostics(call_result: Any) -> dict[str, Any]:
    return {
        "openrouter_http_status": call_result.http_status,
        "openrouter_error_type": call_result.error_type or "",
        "openrouter_error_message": call_result.error_message or "",
        "openrouter_response_format_mode": call_result.response_format_mode,
        "openrouter_model_used": call_result.model,
        "openrouter_request_id": call_result.request_id or "",
        "openrouter_provider": call_result.provider or "",
        "openrouter_usage": call_result.usage,
        "openrouter_call_attempts": call_result.attempts,
        "llm_raw_response_preview": call_result.raw_response_preview,
    }


def _repair_payload_latex_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Repair known LaTeX command corruption immediately after JSON parsing."""
    repaired = dict(payload)
    for field in ("title", "context", "instruction", "solution", "expected_answer", "learning_objective", "estimated_time", "hint"):
        if isinstance(repaired.get(field), str):
            repaired[field] = repair_corrupted_latex_commands(repaired[field])
    for field in ("questions", "solution_steps", "options"):
        if isinstance(repaired.get(field), list):
            repaired[field] = [
                repair_corrupted_latex_commands(str(item))
                for item in repaired[field]
            ]
    return repaired


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
    questions = [str(item).strip() for item in (exercise.get("questions") or []) if str(item).strip()]
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
    if not has_explicit_questions(prompt, questions):
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
    """Convert model or dataset text into clean student-facing text while preserving LaTeX."""
    text = unescape(str(value or "")).replace("\r", "").strip()
    if not text:
        return ""

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\$\$(.+?)\$\$", r"\\[\1\\]", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", r"\\(\1\\)", text)

    # Keep LaTeX commands instead of removing backslashes. The previous version
    # transformed valid expressions such as \frac{...}{...} into corrupted tokens
    # like 'frace' after model mistakes; this guard repairs common corruptions
    # and preserves the mathematical notation for the judge and the UI.
    text = repair_math_text_locally(text)
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
