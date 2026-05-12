"""Page de generation d'exercices pour MathTutorAI."""

from __future__ import annotations

import streamlit as st

from frontend.components.cards import render_highlight_card, render_section_header
from frontend.components.exercise_widgets import (
    render_empty_exercise_state,
    render_exercise_card,
    render_exercise_supports,
    render_filter_summary,
    render_solution_steps,
)
from frontend.components.navbar import render_page_hero
from frontend.utils.alignment_catalog import (
    get_supported_sections,
    get_supported_subtopics_for_section_topic,
    get_supported_topics_for_section,
)
from frontend.utils.api_client import get_api_client
from frontend.utils.constants import DEFAULT_EXERCISE_DIFFICULTY, EXERCISE_TYPES, LEVELS
from frontend.utils.dataset_catalog import normalize_section_label
from frontend.utils.exercise_audit_log import persist_final_exercise_record
from frontend.utils.exercise_presentation_gate import can_present_exercise
from frontend.utils.hint_engine import build_adaptive_hint, build_progressive_hints, get_revealed_hints
from frontend.utils.mongo_learning import (
    load_exercise_from_history,
    record_assigned_exercise_opened,
    record_exercise_generated,
    record_exercise_verification,
    record_hint_interaction,
    record_page_consultation,
)
from frontend.utils.page_router import switch_to_page
from frontend.utils.session_manager import (
    initialize_session_state,
    log_flagged_exercise,
    log_generated_exercise,
    push_notification,
    require_authentication,
    set_current_exercise,
    update_exercise_result,
)


def _is_exercise_deliverable(exercise: dict | None) -> bool:
    """Allow display only for exercises that cleared both validation stages."""
    allowed, _reasons = can_present_exercise(exercise or {})
    return allowed


def _generate_exercise() -> None:
    """Generer et stocker un exercice a partir des filtres actifs."""
    api_client = get_api_client()
    with st.spinner("Generation et validation de l'exercice personnalise..."):
        exercise = api_client.generate_exercise(
            level=st.session_state.generator_level,
            section=st.session_state.generator_section,
            topic=st.session_state.generator_topic,
            subtopic=st.session_state.generator_subtopic,
            difficulty=st.session_state.generator_difficulty,
            exercise_type=st.session_state.generator_type,
            audit_context={
                "user_email": st.session_state.auth.get("email", "inconnu"),
                "user_role": st.session_state.auth.get("role", "Inconnu"),
                "user_display_name": st.session_state.auth.get("display_name", "Utilisateur inconnu"),
            },
        )

    for flagged_attempt in exercise.get("judge_rejected_attempts", []):
        log_flagged_exercise(
            flagged_attempt,
            reason=flagged_attempt.get("summary", "Refuse par le juge."),
            issues=flagged_attempt.get("issues", []),
            source=flagged_attempt.get("judge_model", "Juge OpenRouter"),
        )

    set_current_exercise(exercise)
    log_generated_exercise(exercise)
    record_exercise_generated(exercise)
    persist_final_exercise_record(
        exercise,
        user_email=st.session_state.auth.get("email", "inconnu"),
        user_role=st.session_state.auth.get("role", "Inconnu"),
        user_display_name=st.session_state.auth.get("display_name", "Utilisateur inconnu"),
    )

    if not _is_exercise_deliverable(exercise):
        push_notification("La chaine de validation a bloque cette generation avant affichage a l'etudiant.", "⚠")
        return

    if exercise.get("judge_blocked"):
        push_notification("Le juge a bloque cette generation avant affichage a l'etudiant.", "⚠")
    elif exercise.get("judge_validation_flag") == "corrected":
        push_notification(f"Exercice corrige par le juge sur {exercise['subtopic'].lower()}.", "🧪")
    else:
        push_notification(f"Nouvel exercice valide sur {exercise['subtopic'].lower()}.", "🧩")


def _matches_current_selection(exercise: dict | None) -> bool:
    """Check whether the displayed exercise still matches the active generator filters."""
    if not exercise:
        return False

    return all(
        [
            exercise.get("level") == st.session_state.generator_level,
            exercise.get("section") == st.session_state.generator_section,
            exercise.get("topic") == st.session_state.generator_topic,
            exercise.get("subtopic") == st.session_state.generator_subtopic,
            exercise.get("exercise_type") == st.session_state.generator_type,
        ]
    )


def _maybe_restore_pending_assigned_exercise() -> None:
    """Open one assigned exercise pushed by a sidebar notification."""
    trace_id = str(st.session_state.get("pending_assignment_trace_id", "")).strip()
    if not trace_id:
        return

    st.session_state.pending_assignment_trace_id = ""
    exercise = load_exercise_from_history(trace_id)
    if not exercise:
        push_notification("Impossible de charger l'exercice assigne depuis la notification.", "⚠")
        return

    st.session_state.generator_level = exercise.get("level", st.session_state.get("generator_level", "Bac"))
    st.session_state.generator_section = exercise.get("section", st.session_state.get("generator_section", ""))
    st.session_state.generator_topic = exercise.get("topic", st.session_state.get("generator_topic", ""))
    st.session_state.generator_subtopic = exercise.get("subtopic", st.session_state.get("generator_subtopic", ""))
    st.session_state.generator_type = exercise.get(
        "exercise_type",
        st.session_state.get("generator_type", "Exercice problÃ¨me"),
    )
    set_current_exercise(exercise)
    record_assigned_exercise_opened(exercise, source="sidebar_notification")
    push_notification("Exercice assigne ouvert depuis vos notifications.", "📘")


def render_page() -> None:
    """Afficher l'espace de generation d'exercices."""
    initialize_session_state()
    require_authentication("Generateur d'exercices")
    _maybe_restore_pending_assigned_exercise()
    record_page_consultation("exercise_generator", "Generateur d'exercices")
    render_page_hero(
        "Generateur d'exercices",
        "Creez des exercices de mathematiques selon le niveau, la section, le theme et le format souhaite.",
        badge=st.session_state.student_profile["preferred_mode"],
    )

    sections = get_supported_sections()
    preferred_section = normalize_section_label(st.session_state.student_profile.get("section", ""))
    if not sections:
        st.error("Aucune notion officiellement couverte n'est disponible pour le generateur.")
        st.stop()

    st.session_state.setdefault("generator_level", st.session_state.student_profile["level"])
    default_section = preferred_section if preferred_section in sections else (sections[0] if sections else "")
    st.session_state.setdefault("generator_section", default_section)
    if st.session_state.generator_section not in sections and sections:
        st.session_state.generator_section = default_section

    topics = get_supported_topics_for_section(st.session_state.generator_section)
    if st.session_state.get("generator_topic") not in topics:
        st.session_state.generator_topic = topics[0] if topics else ""

    subtopics = get_supported_subtopics_for_section_topic(
        st.session_state.generator_section,
        st.session_state.generator_topic,
    )
    if st.session_state.get("generator_subtopic") not in subtopics:
        st.session_state.generator_subtopic = subtopics[0] if subtopics else ""

    st.session_state.generator_difficulty = DEFAULT_EXERCISE_DIFFICULTY
    st.session_state.setdefault("generator_type", "Exercice problème")
    if st.session_state.generator_type not in EXERCISE_TYPES:
        st.session_state.generator_type = "Exercice problème"

    render_section_header(
        "Parametrage",
        "Preparez une demande realiste alignee sur le niveau, la section, la notion et le type d'exercice attendu.",
        "🧩",
    )

    controls_col, insights_col = st.columns([1.6, 1], gap="large")
    with controls_col:
        top_row = st.columns(3, gap="medium")
        with top_row[0]:
            st.selectbox("Niveau", LEVELS, key="generator_level")
        with top_row[1]:
            st.selectbox("Section", sections, key="generator_section")
        with top_row[2]:
            refreshed_topics = get_supported_topics_for_section(st.session_state.generator_section)
            if st.session_state.generator_topic not in refreshed_topics:
                st.session_state.generator_topic = refreshed_topics[0] if refreshed_topics else ""
            st.selectbox("Theme", refreshed_topics, key="generator_topic")

        refreshed_subtopics = get_supported_subtopics_for_section_topic(
            st.session_state.generator_section,
            st.session_state.generator_topic,
        )
        if st.session_state.generator_subtopic not in refreshed_subtopics:
            st.session_state.generator_subtopic = refreshed_subtopics[0] if refreshed_subtopics else ""

        bottom_row = st.columns(2, gap="medium")
        with bottom_row[0]:
            st.selectbox("Sous-theme", refreshed_subtopics, key="generator_subtopic")
        with bottom_row[1]:
            st.selectbox("Type d'exercice", EXERCISE_TYPES, key="generator_type")

        st.caption(
            "Le niveau disponible actuellement est le bac. Les autres niveaux arriveront prochainement. "
            "Les couples affiches ici sont limites aux notions couvertes par le programme officiel de reference."
        )

        action_row = st.columns([1, 1], gap="medium")
        with action_row[0]:
            if st.button("Generer l'exercice", type="primary"):
                _generate_exercise()
        with action_row[1]:
            if st.button("Generer un autre") and st.session_state.current_exercise:
                _generate_exercise()

    with insights_col:
        render_filter_summary(
            {
                "Niveau": st.session_state.generator_level,
                "Section": st.session_state.generator_section,
                "Theme": st.session_state.generator_topic,
                "Sous-theme": st.session_state.generator_subtopic,
                "Type": st.session_state.generator_type,
            }
        )
        render_highlight_card(
            "Ciblage intelligent",
            "Le generateur lit des cas analogues du dataset, reutilise les plus proches, puis revise l'enonce via l'agent OpenRouter.",
            "Memoire de cas + generation distante",
        )

    exercise_col, support_col = st.columns([1.65, 1], gap="large")
    stored_exercise = st.session_state.current_exercise
    current_exercise = stored_exercise if _matches_current_selection(stored_exercise) else None
    exercise_deliverable = _is_exercise_deliverable(current_exercise)
    progressive_hints = build_progressive_hints(current_exercise) if exercise_deliverable else []
    revealed_hints = get_revealed_hints(current_exercise, st.session_state.exercise_hint_level) if exercise_deliverable else []

    with exercise_col:
        render_section_header("Exercice genere", "Carte de l'exercice, zone de reponse et verification de la solution.", "✍️")
        if stored_exercise and not current_exercise:
            st.info(
                "Les parametres affiches ont change depuis la derniere generation. "
                "Generez un nouvel exercice pour voir un enonce correspondant a cette selection."
            )
        if current_exercise:
            if not exercise_deliverable:
                _allowed, blocking_reasons = can_present_exercise(current_exercise)
                st.error(current_exercise.get("judge_summary", "Le juge a bloque cet exercice avant affichage a l'eleve."))
                st.info("Aucun enonce n'a ete montre a l'etudiant pour cette tentative. Relancez une generation.")
                if blocking_reasons:
                    st.caption("Blocages detectes : " + " | ".join(blocking_reasons))
            else:
                render_exercise_card(current_exercise)
                render_exercise_supports(current_exercise)
                st.text_area(
                    "Reponse de l'etudiant",
                    key="current_answer",
                    height=160,
                    placeholder="Ecrivez ici votre raisonnement complet ou votre reponse finale...",
                )

                action_col, hint_col, adaptive_col = st.columns(3, gap="medium")
                with action_col:
                    if current_exercise.get("verification_ready", True):
                        if st.button("Verifier la reponse", type="primary"):
                            api_client = get_api_client()
                            with st.spinner("Verification symbolique en cours..."):
                                result = api_client.verify_answer(current_exercise, st.session_state.current_answer)
                            st.session_state.last_verification = result
                            record_exercise_verification(current_exercise, st.session_state.current_answer, result)
                            status = "Correct" if result["is_correct"] else "A revoir"
                            update_exercise_result(current_exercise["id"], status)
                            push_notification(f"Verification terminee : {status.lower()}.", "🧪")
                    else:
                        if st.button("Discuter avec le tuteur", type="primary"):
                            st.session_state.tutoring_state["use_exercise_context"] = True
                            push_notification(
                                "Le tutorat conversationnel a ouvert l'exercice actif comme contexte.",
                                "💬",
                            )
                            switch_to_page("tutoring_chat")
                with hint_col:
                    next_hint_disabled = st.session_state.exercise_hint_level >= len(progressive_hints)
                    if st.button(
                        "Afficher l'indice suivant",
                        disabled=next_hint_disabled,
                    ):
                        st.session_state.exercise_hint_level += 1
                        st.session_state.adaptive_hint_message = ""
                        hint_index = st.session_state.exercise_hint_level
                        hint_text = progressive_hints[hint_index - 1] if 0 < hint_index <= len(progressive_hints) else ""
                        record_hint_interaction(
                            current_exercise,
                            hint_kind="progressive",
                            hint_index=hint_index,
                            hint_text=hint_text,
                        )
                with adaptive_col:
                    if st.button("Indice adaptatif"):
                        st.session_state.adaptive_hint_message = build_adaptive_hint(
                            current_exercise,
                            st.session_state.current_answer,
                        )
                        record_hint_interaction(
                            current_exercise,
                            hint_kind="adaptive",
                            hint_text=st.session_state.adaptive_hint_message,
                        )

                if revealed_hints:
                    for index, hint_text in enumerate(revealed_hints, start=1):
                        with st.expander(f"Indice {index}", expanded=index == len(revealed_hints)):
                            st.write(hint_text)

                if st.session_state.adaptive_hint_message:
                    st.warning(st.session_state.adaptive_hint_message)

                if not current_exercise.get("verification_ready", True):
                    st.info(current_exercise.get("verification_message", "La verification automatique sera ajoutee dans la prochaine etape."))

                verification = st.session_state.last_verification
                if verification:
                    if verification["is_correct"]:
                        st.success(verification["feedback"])
                    else:
                        st.error(verification["feedback"])
                        st.caption(f"Reponse attendue : {verification['expected_answer']}")
                    render_solution_steps(verification["solution_steps"])
        else:
            render_empty_exercise_state()

    with support_col:
        render_section_header("Panneau d'aide", "Indices, metadonnees et objectif pedagogique de l'exercice.", "🪟")
        if current_exercise:
            render_highlight_card(
                "Validation du juge",
                current_exercise.get("judge_summary", "Aucun verdict du juge disponible."),
                (
                    f"Statut : {current_exercise.get('judge_status', 'Non evalue')} | "
                    f"Modele : {current_exercise.get('judge_model', 'deepseek/deepseek-r1:free')}"
                ),
                accent="amber" if current_exercise.get("judge_validation_flag") == "corrected" else "teal",
            )
            solution_validation_flag = current_exercise.get("solution_validation_flag", "")
            if current_exercise.get("solution_validation_summary"):
                render_highlight_card(
                    "Validation solution",
                    current_exercise.get(
                        "solution_validation_summary",
                        "Aucun verdict de validation LLM + SymPy disponible.",
                    ),
                    (
                        f"Statut : {current_exercise.get('solution_validation_status', 'Non evalue')} | "
                        f"Modele : {current_exercise.get('solution_validation_model', 'qwen/qwen-2.5-7b-instruct')}"
                    ),
                    accent="teal" if solution_validation_flag == "approved" else "amber",
                )
            if exercise_deliverable:
                render_highlight_card(
                    "Indice principal",
                    current_exercise["hint"],
                    "Piste initiale",
                )
                render_highlight_card(
                    "Aide progressive",
                    (
                        f"{len(revealed_hints)} indice(s) reveles sur {len(progressive_hints)}. "
                        "Le bouton d'indice avance maintenant par paliers et l'indice adaptatif reformule une piste a partir de votre brouillon."
                    ),
                    "Mode socratique",
                )
                render_highlight_card(
                    "Objectif pedagogique",
                    current_exercise["learning_objective"],
                    f"ID exercice : {current_exercise['id']}",
                    accent="amber",
                )
                render_highlight_card(
                    "Adaptation memoire",
                    current_exercise.get("memory_adaptation_note", "Aucune note memoire disponible."),
                    (
                        f"Source : {current_exercise.get('generation_backend', 'inconnue')} | "
                        f"Cas dataset : {len(current_exercise.get('source_case_summaries', []))}"
                    ),
                )
                render_highlight_card(
                    "Annexe d'exercice",
                    current_exercise.get("support_summary", "Aucun support annexe renseigne."),
                    "Tableaux et graphiques autoporteurs",
                )
            if current_exercise.get("judge_alignment_reason"):
                st.caption("Alignement officiel : " + current_exercise["judge_alignment_reason"])
            if current_exercise.get("judge_issues"):
                st.caption("Points du juge : " + " | ".join(current_exercise["judge_issues"]))
            if current_exercise.get("solution_validation_issues"):
                st.caption(
                    "Points de validation solution : "
                    + " | ".join(current_exercise["solution_validation_issues"])
                )
            if current_exercise.get("solution_validation_sympy_report"):
                st.caption("Rapport SymPy : " + current_exercise["solution_validation_sympy_report"])
            if current_exercise.get("judge_regeneration_count", 0):
                st.caption(
                    f"Tentatives refusees puis regenerees : {current_exercise['judge_regeneration_count']}"
                )
            if current_exercise.get("generation_warning"):
                st.warning(
                    "Generation de secours utilisee : "
                    f"{current_exercise['generation_warning']}"
                )
        else:
            render_highlight_card(
                "Pret a demarrer",
                "Des qu'un exercice est genere, ce panneau affichera les indices utiles et les reperes pedagogiques.",
                "Aucun exercice actif",
            )


render_page()
