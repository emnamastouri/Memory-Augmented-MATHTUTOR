"""Teacher control panel for groups, assignments, and tutoring supervision."""

from __future__ import annotations

from datetime import date

import streamlit as st

from frontend.components.cards import render_highlight_card, render_section_header
from frontend.components.exercise_widgets import render_exercise_card, render_exercise_supports, render_filter_summary
from frontend.components.navbar import render_page_hero
from frontend.utils.alignment_catalog import (
    get_supported_sections,
    get_supported_subtopics_for_section_topic,
    get_supported_topics_for_section,
)
from frontend.utils.api_client import get_api_client
from frontend.utils.constants import DEFAULT_EXERCISE_DIFFICULTY, EXERCISE_TYPES, LEVELS
from frontend.utils.exercise_presentation_gate import can_present_exercise
from frontend.utils.mongo_learning import record_page_consultation
from frontend.utils.session_manager import (
    initialize_session_state,
    push_notification,
    require_teacher_access,
)


def _queue_widget_reset(*keys: str) -> None:
    """Schedule widget-bound session keys to be reset on the next rerun."""
    pending = set(st.session_state.get("_teacher_pending_widget_resets", []))
    pending.update(key for key in keys if key)
    st.session_state._teacher_pending_widget_resets = sorted(pending)


def _apply_deferred_widget_resets() -> None:
    """Apply queued widget resets before any teacher widgets are instantiated."""
    pending = list(st.session_state.get("_teacher_pending_widget_resets", []))
    if not pending:
        return

    for key in pending:
        st.session_state[key] = ""
    st.session_state._teacher_pending_widget_resets = []


def _initialize_teacher_filters(groups: list[dict]) -> None:
    """Keep teacher-side assignment filters aligned with the selected group."""
    sections = get_supported_sections()
    default_section = groups[0]["section"] if groups and groups[0].get("section") in sections else (sections[0] if sections else "")
    st.session_state.setdefault("teacher_assignment_group_id", groups[0]["group_id"] if groups else "")
    st.session_state.setdefault("teacher_assignment_level", LEVELS[0])
    st.session_state.setdefault("teacher_assignment_section", default_section)
    st.session_state.setdefault("teacher_assignment_topic", "")
    st.session_state.setdefault("teacher_assignment_subtopic", "")
    st.session_state.setdefault("teacher_assignment_type", "Exercice problÃ¨me")
    st.session_state.setdefault("teacher_assignment_note", "")

    if groups and st.session_state.teacher_assignment_group_id not in {group["group_id"] for group in groups}:
        st.session_state.teacher_assignment_group_id = groups[0]["group_id"]

    selected_group = next((item for item in groups if item["group_id"] == st.session_state.teacher_assignment_group_id), None)
    if selected_group and selected_group.get("section") in sections:
        st.session_state.teacher_assignment_section = selected_group["section"]
    elif st.session_state.teacher_assignment_section not in sections and sections:
        st.session_state.teacher_assignment_section = sections[0]

    topics = get_supported_topics_for_section(st.session_state.teacher_assignment_section)
    if st.session_state.teacher_assignment_topic not in topics:
        st.session_state.teacher_assignment_topic = topics[0] if topics else ""

    subtopics = get_supported_subtopics_for_section_topic(
        st.session_state.teacher_assignment_section,
        st.session_state.teacher_assignment_topic,
    )
    if st.session_state.teacher_assignment_subtopic not in subtopics:
        st.session_state.teacher_assignment_subtopic = subtopics[0] if subtopics else ""

    if st.session_state.teacher_assignment_type not in EXERCISE_TYPES:
        st.session_state.teacher_assignment_type = EXERCISE_TYPES[0]


def _generate_teacher_preview() -> None:
    """Generate one exercise preview reserved for the teacher panel."""
    api_client = get_api_client()
    with st.spinner("Generation et validation de l'exercice enseignant..."):
        exercise = api_client.generate_exercise(
            level=st.session_state.teacher_assignment_level,
            section=st.session_state.teacher_assignment_section,
            topic=st.session_state.teacher_assignment_topic,
            subtopic=st.session_state.teacher_assignment_subtopic,
            difficulty=DEFAULT_EXERCISE_DIFFICULTY,
            exercise_type=st.session_state.teacher_assignment_type,
            audit_context={
                "user_email": st.session_state.auth.get("email", ""),
                "user_role": st.session_state.auth.get("role", ""),
                "user_display_name": st.session_state.auth.get("display_name", ""),
            },
        )
    st.session_state.teacher_preview_exercise = exercise
    preview_ok, blocking_reasons = can_present_exercise(exercise)
    if preview_ok:
        push_notification("Un nouvel exercice enseignant a ete genere pour previsualisation.", "🧑‍🏫")
        return

    push_notification(
        "La previsualisation generee a ete bloquee avant affectation. Consultez les raisons dans le panneau enseignant.",
        "⚠",
    )
    st.warning("\n".join([f"- {reason}" for reason in blocking_reasons]) or "Previsualisation bloquee.")


def _teacher_preview_matches_filters(preview: dict | None) -> bool:
    """Ensure the preview still matches the current assignment form."""
    if not preview:
        return False
    return all(
        [
            preview.get("level") == st.session_state.teacher_assignment_level,
            preview.get("section") == st.session_state.teacher_assignment_section,
            preview.get("topic") == st.session_state.teacher_assignment_topic,
            preview.get("subtopic") == st.session_state.teacher_assignment_subtopic,
            preview.get("exercise_type") == st.session_state.teacher_assignment_type,
        ]
    )


def _teacher_preview_is_assignable(preview: dict | None) -> tuple[bool, list[str]]:
    """Return whether the current teacher preview can safely be assigned."""
    if not preview:
        return False, ["Aucune previsualisation n'est disponible."]
    if not _teacher_preview_matches_filters(preview):
        return False, ["La previsualisation ne correspond plus aux filtres actuels."]
    return can_present_exercise(preview)


def _submit_group_creation() -> None:
    """Create one new teacher group from the sidebar form."""
    api_client = get_api_client()
    result = api_client.create_teacher_group(
        teacher_email=st.session_state.auth.get("email", ""),
        teacher_user_id=st.session_state.auth.get("user_id", ""),
        teacher_name=st.session_state.auth.get("display_name", ""),
        group_name=st.session_state.teacher_new_group_name,
        section=st.session_state.teacher_new_group_section,
        level=LEVELS[0],
    )
    if result["ok"]:
        push_notification(result["message"], "👥")
        _queue_widget_reset("teacher_new_group_name")
        st.rerun()
    st.error(result["message"])


def _submit_student_assignment_to_group() -> None:
    """Attach one student email to the selected teacher group."""
    api_client = get_api_client()
    result = api_client.add_student_to_teacher_group(
        teacher_email=st.session_state.auth.get("email", ""),
        group_id=st.session_state.teacher_group_membership_target,
        student_email=st.session_state.teacher_student_email,
    )
    if result["ok"]:
        push_notification(result["message"], "✅")
        _queue_widget_reset("teacher_student_email")
        st.rerun()
    st.error(result["message"])


def _assign_preview_to_group() -> None:
    """Assign the current preview to the selected teacher group."""
    preview = st.session_state.get("teacher_preview_exercise")
    preview_assignable, blocking_reasons = _teacher_preview_is_assignable(preview)
    if not preview_assignable:
        st.error("Cette previsualisation ne peut pas etre assignee tant qu'elle reste bloquee.")
        for reason in blocking_reasons:
            st.caption(f"Blocage : {reason}")
        return

    api_client = get_api_client()
    result = api_client.assign_exercise_to_group(
        teacher_email=st.session_state.auth.get("email", ""),
        teacher_user_id=st.session_state.auth.get("user_id", ""),
        teacher_name=st.session_state.auth.get("display_name", ""),
        group_id=st.session_state.teacher_assignment_group_id,
        exercise=preview,
        due_date=st.session_state.teacher_assignment_due_date,
        note=st.session_state.teacher_assignment_note,
    )
    if result["ok"]:
        push_notification(result["message"], "📚")
        st.session_state.teacher_preview_exercise = None
        _queue_widget_reset("teacher_assignment_note")
        st.rerun()
    st.error(result["message"])


def _render_dashboard_tab(data: dict[str, object]) -> None:
    """Render the teacher dashboard tab."""
    metrics = data.get("metrics", {})
    metric_cols = st.columns(5, gap="medium")
    with metric_cols[0]:
        st.metric("Groupes", str(metrics.get("group_count", 0)))
    with metric_cols[1]:
        st.metric("Etudiants suivis", str(metrics.get("student_count", 0)))
    with metric_cols[2]:
        st.metric("Etudiants actifs", str(metrics.get("active_students", 0)))
    with metric_cols[3]:
        st.metric("Maitrise moyenne", f"{metrics.get('average_mastery', 0)}%")
    with metric_cols[4]:
        st.metric("Affectations ouvertes", str(metrics.get("open_assignments", 0)))

    student_rows = data.get("students", [])
    assignment_rows = data.get("assignments", [])
    left_col, right_col = st.columns([1.45, 1], gap="large")
    with left_col:
        render_section_header(
            "Performance etudiante",
            "Maitrise, activite recente, heures d'etude et focus des eleves rattaches a vos groupes.",
            "📊",
        )
        if student_rows:
            st.dataframe(
                [
                    {
                        "Etudiant": row["name"],
                        "Email": row["email"],
                        "Section": row["section"],
                        "Maitrise": f"{row['mastery']}%",
                        "Reussite": f"{row['success_rate']}%",
                        "Heures": row["study_hours"],
                        "Focus": row["focus"],
                        "Affectations ouvertes": row["open_assignments"],
                        "Risque": row["risk"],
                        "Derniere activite": row["last_active_label"],
                    }
                    for row in student_rows
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Creez un premier groupe puis ajoutez des etudiants pour activer le tableau de bord.")

    with right_col:
        render_highlight_card(
            "Lecture rapide de la cohorte",
            (
                f"{metrics.get('student_count', 0)} etudiant(s) suivis sur {metrics.get('group_count', 0)} groupe(s). "
                f"Le volume cumule d'etude atteint {metrics.get('study_hours', 0.0)} h."
            ),
            "Synthese Mongo",
        )
        if assignment_rows:
            latest = assignment_rows[0]
            render_highlight_card(
                "Derniere affectation",
                (
                    f"{latest['title']} a ete envoye au groupe {latest['group_name']} "
                    f"pour la notion {latest['subtopic']}."
                ),
                f"{latest['solved_count']}/{latest['recipient_count']} resolu(s)",
                accent="amber",
            )
        else:
            render_highlight_card(
                "Affectations",
                "Aucune affectation n'a encore ete envoyee depuis cet espace enseignant.",
                "Pret pour le premier exercice",
                accent="amber",
            )


def _render_groups_tab(data: dict[str, object]) -> None:
    """Render group creation and membership management."""
    groups = data.get("groups", [])
    sections = get_supported_sections()
    st.session_state.setdefault("teacher_new_group_name", "")
    st.session_state.setdefault("teacher_new_group_section", sections[0] if sections else "")
    st.session_state.setdefault("teacher_student_email", "")
    st.session_state.setdefault(
        "teacher_group_membership_target",
        groups[0]["group_id"] if groups else "",
    )
    if groups and st.session_state.teacher_group_membership_target not in {group["group_id"] for group in groups}:
        st.session_state.teacher_group_membership_target = groups[0]["group_id"]

    form_col, list_col = st.columns([1, 1.15], gap="large")
    with form_col:
        render_section_header(
            "Creer et alimenter les groupes",
            "Verifiez les e-mails etudiants avant de les rattacher a un groupe de travail.",
            "👥",
        )
        with st.form("teacher_create_group_form", clear_on_submit=False):
            st.text_input("Nom du groupe", key="teacher_new_group_name", placeholder="Ex. Bac Maths - Groupe A")
            st.selectbox(
                "Section de reference",
                sections,
                key="teacher_new_group_section",
                disabled=not sections,
            )
            create_group = st.form_submit_button("Creer le groupe", type="primary")
        if create_group:
            _submit_group_creation()

        st.markdown("---")
        with st.form("teacher_add_student_to_group_form", clear_on_submit=False):
            group_options = {group["name"]: group["group_id"] for group in groups}
            selected_group_name = st.selectbox(
                "Groupe cible",
                list(group_options.keys()) or ["Aucun groupe disponible"],
                key="teacher_group_membership_target_name",
                disabled=not group_options,
            )
            if group_options:
                st.session_state.teacher_group_membership_target = group_options[selected_group_name]
            st.text_input(
                "Adresse e-mail etudiant",
                key="teacher_student_email",
                placeholder="etudiant@exemple.com",
                disabled=not group_options,
            )
            add_student = st.form_submit_button(
                "Verifier puis ajouter l'etudiant",
                type="secondary",
                disabled=not group_options,
            )
        if add_student:
            _submit_student_assignment_to_group()

    with list_col:
        render_section_header(
            "Groupes existants",
            "Chaque groupe memorise ses membres et pourra recevoir des exercices en bloc.",
            "🧩",
        )
        if not groups:
            st.info("Aucun groupe enseignant n'est encore cree.")
            return

        for group in groups:
            with st.container():
                st.markdown(f"**{group['name']}**")
                st.caption(f"{group['member_count']} membre(s) · {group['section'] or 'Section libre'} · {group['created_at_label']}")
                if group["members"]:
                    st.dataframe(
                        [
                            {
                                "Etudiant": member["student_name"],
                                "E-mail": member["student_email"],
                                "Section": member["section"],
                                "Niveau": member["level"],
                            }
                            for member in group["members"]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("Ce groupe ne contient encore aucun etudiant.")
                st.divider()


def _render_assignment_tab(data: dict[str, object]) -> None:
    """Render exercise generation and assignment for teachers."""
    groups = data.get("groups", [])
    preview = st.session_state.get("teacher_preview_exercise")

    if not groups:
        st.info("Créez d'abord un groupe pour pouvoir générer puis assigner un exercice.")
        return

    _initialize_teacher_filters(groups)
    matching_preview = preview if _teacher_preview_matches_filters(preview) else None
    preview_assignable, preview_blocking_reasons = _teacher_preview_is_assignable(matching_preview)
    selected_group = next((item for item in groups if item["group_id"] == st.session_state.teacher_assignment_group_id), None)
    group_has_members = bool(selected_group and selected_group.get("member_count", 0))

    config_col, preview_col = st.columns([1.15, 1], gap="large")
    with config_col:
        render_section_header(
            "Generer puis assigner",
            "Le meme exercice est clone pour tous les membres du groupe, avec une notification individuelle.",
            "📚",
        )
        group_options = {group["name"]: group["group_id"] for group in groups}
        selected_group_name = st.selectbox(
            "Groupe destinataire",
            list(group_options.keys()),
            index=list(group_options.values()).index(st.session_state.teacher_assignment_group_id),
        )
        st.session_state.teacher_assignment_group_id = group_options[selected_group_name]

        top_row = st.columns(3, gap="medium")
        with top_row[0]:
            st.selectbox("Niveau", LEVELS, key="teacher_assignment_level")
        with top_row[1]:
            sections = get_supported_sections()
            st.selectbox("Section", sections, key="teacher_assignment_section")
        with top_row[2]:
            topics = get_supported_topics_for_section(st.session_state.teacher_assignment_section)
            if st.session_state.teacher_assignment_topic not in topics:
                st.session_state.teacher_assignment_topic = topics[0] if topics else ""
            st.selectbox("Theme", topics, key="teacher_assignment_topic")

        subtopics = get_supported_subtopics_for_section_topic(
            st.session_state.teacher_assignment_section,
            st.session_state.teacher_assignment_topic,
        )
        bottom_row = st.columns(3, gap="medium")
        with bottom_row[0]:
            if st.session_state.teacher_assignment_subtopic not in subtopics:
                st.session_state.teacher_assignment_subtopic = subtopics[0] if subtopics else ""
            st.selectbox("Sous-theme", subtopics, key="teacher_assignment_subtopic")
        with bottom_row[1]:
            st.selectbox("Type d'exercice", EXERCISE_TYPES, key="teacher_assignment_type")
        with bottom_row[2]:
            st.date_input(
                "Date limite",
                key="teacher_assignment_due_date",
                value=date.today(),
            )

        st.text_area(
            "Note enseignant (optionnelle)",
            key="teacher_assignment_note",
            height=90,
            placeholder="Consigne additionnelle visible seulement depuis l'espace enseignant.",
        )

        action_row = st.columns(2, gap="medium")
        with action_row[0]:
            if st.button("Generer l'exercice enseignant", type="primary"):
                _generate_teacher_preview()
                st.rerun()
        with action_row[1]:
            if st.button(
                "Assigner au groupe",
                disabled=not matching_preview or not group_has_members or not preview_assignable,
            ):
                _assign_preview_to_group()

        if not group_has_members:
            st.caption("Ajoutez au moins un etudiant au groupe avant l'envoi.")
        if matching_preview and not preview_assignable:
            st.caption("Cette previsualisation reste visible pour diagnostic enseignant mais ne sera pas assignee.")

    with preview_col:
        render_filter_summary(
            {
                "Niveau": st.session_state.teacher_assignment_level,
                "Section": st.session_state.teacher_assignment_section,
                "Theme": st.session_state.teacher_assignment_topic,
                "Sous-theme": st.session_state.teacher_assignment_subtopic,
                "Type": st.session_state.teacher_assignment_type,
            }
        )
        if preview and not matching_preview:
            st.info("La previsualisation actuelle ne correspond plus aux filtres affiches. Regénérez avant d'assigner.")
        if matching_preview:
            if preview_assignable:
                render_exercise_card(matching_preview)
                render_exercise_supports(matching_preview)
                with st.expander("Solution complete (visible enseignant uniquement)", expanded=False):
                    st.markdown(matching_preview.get("hidden_solution", "Solution non disponible."))
                    for index, step in enumerate(matching_preview.get("solution_steps", []) or [], start=1):
                        st.markdown(f"**Etape {index}.** {step}")
            else:
                st.error("Cette previsualisation a ete bloquee avant affectation.")
                for reason in preview_blocking_reasons:
                    st.caption(f"Blocage : {reason}")
                with st.expander("Details internes de la previsualisation bloquee", expanded=False):
                    render_exercise_card(matching_preview)
                    render_exercise_supports(matching_preview)
                    st.markdown("**Solution complete (visible enseignant uniquement)**")
                    st.markdown(matching_preview.get("hidden_solution", "Solution non disponible."))
                    for index, step in enumerate(matching_preview.get("solution_steps", []) or [], start=1):
                        st.markdown(f"**Etape {index}.** {step}")
        else:
            render_highlight_card(
                "Previsualisation enseignant",
                "Generez un exercice pour visualiser l'enonce, les annexes et la solution complete avant l'affectation.",
                "Acces reserve enseignant",
                accent="amber",
            )


def _render_supervision_tab(data: dict[str, object]) -> None:
    """Render read-only supervision of student tutoring threads."""
    assignments = data.get("assignments", [])
    if not assignments:
        st.info("Les affectations envoyees apparaîtront ici avec les conversations de tutorat associees.")
        return

    assignment_labels = {
        f"{item['group_name']} · {item['title']} · {item['created_at_label']} · {item['assignment_id'][:6]}": item["assignment_id"]
        for item in assignments
    }
    st.session_state.setdefault("teacher_supervision_assignment_id", assignments[0]["assignment_id"])
    if st.session_state.teacher_supervision_assignment_id not in {item["assignment_id"] for item in assignments}:
        st.session_state.teacher_supervision_assignment_id = assignments[0]["assignment_id"]

    selected_assignment_label = st.selectbox(
        "Affectation a superviser",
        list(assignment_labels.keys()),
        index=list(assignment_labels.values()).index(st.session_state.teacher_supervision_assignment_id),
    )
    st.session_state.teacher_supervision_assignment_id = assignment_labels[selected_assignment_label]

    api_client = get_api_client()
    base_payload = api_client.get_teacher_supervision_view(
        teacher_email=st.session_state.auth.get("email", ""),
        assignment_id=st.session_state.teacher_supervision_assignment_id,
    )
    if not base_payload:
        st.warning("Impossible de charger cette affectation.")
        return

    student_options = {
        f"{item['student_name']} · {item['student_email']}": item["student_email"]
        for item in base_payload.get("students", [])
    }
    if not student_options:
        st.info("Cette affectation n'a aucun destinataire.")
        return

    st.session_state.setdefault("teacher_supervision_student_email", next(iter(student_options.values())))
    if st.session_state.teacher_supervision_student_email not in set(student_options.values()):
        st.session_state.teacher_supervision_student_email = next(iter(student_options.values()))

    selected_student_label = st.selectbox(
        "Etudiant a observer",
        list(student_options.keys()),
        index=list(student_options.values()).index(st.session_state.teacher_supervision_student_email),
    )
    st.session_state.teacher_supervision_student_email = student_options[selected_student_label]

    payload = api_client.get_teacher_supervision_view(
        teacher_email=st.session_state.auth.get("email", ""),
        assignment_id=st.session_state.teacher_supervision_assignment_id,
        student_email=st.session_state.teacher_supervision_student_email,
    )
    if not payload:
        st.warning("La supervision n'a pas pu etre chargee.")
        return

    assignment = payload["assignment"]
    active_student = payload["active_student"]
    analytics = active_student.get("analytics", {})
    metrics = analytics.get("metrics", {})
    exercise_record = active_student.get("exercise_record") or {}
    threads = active_student.get("threads", [])
    exercise_snapshot = assignment.get("exercise_snapshot", {})

    top_col, side_col = st.columns([1.5, 1], gap="large")
    with top_col:
        render_section_header(
            "Exercice assigne et travail eleve",
            "L'enseignant voit l'enonce, les annexes et la solution complete, puis peut suivre le tutorat lie a cet exercice.",
            "🧑‍🏫",
        )
        render_exercise_card(exercise_snapshot)
        render_exercise_supports(exercise_snapshot)
        with st.expander("Solution complete (visible enseignant uniquement)", expanded=False):
            st.markdown(exercise_snapshot.get("hidden_solution", "Solution non disponible."))
            for index, step in enumerate(exercise_snapshot.get("solution_steps", []) or [], start=1):
                st.markdown(f"**Etape {index}.** {step}")

        st.markdown("#### Travail de l'etudiant")
        st.caption(
            f"Statut actuel : {exercise_record.get('latest_status', 'assigné')} · "
            f"Consultations : {exercise_record.get('consultation_count', 0)} · "
            f"Tours de tutorat : {exercise_record.get('tutor_turns', 0)}"
        )
        if exercise_record.get("last_submitted_answer"):
            st.text_area(
                "Derniere reponse soumise",
                value=str(exercise_record.get("last_submitted_answer", "")),
                height=120,
                disabled=True,
            )
        if exercise_record.get("last_feedback"):
            st.info(exercise_record["last_feedback"])

        st.markdown("#### Conversations de tutorat")
        if not threads:
            st.caption("Aucune conversation n'a encore ete demarree pour cet exercice.")
        for thread in threads:
            with st.expander(
                f"{thread['title']} · {thread['updated_at_label']} · {thread['message_count']} message(s)",
                expanded=False,
            ):
                st.caption(f"Mode : {thread['mode']}")
                for message in thread.get("messages", []):
                    role_label = "Eleve" if message.get("role") == "user" else "Tuteur"
                    st.markdown(f"**{role_label}** · {message.get('mode', '')}")
                    st.write(message.get("content", ""))
                if thread.get("last_student_answer_draft"):
                    st.caption("Dernier brouillon connu de l'etudiant")
                    st.code(thread["last_student_answer_draft"])

    with side_col:
        render_highlight_card(
            "Vue etudiant",
            (
                f"{active_student.get('student_name', '')}\n\n"
                f"Maitrise moyenne : {metrics.get('mastery_average', 0)}% · "
                f"Reussite : {metrics.get('success_rate_pct', 0)}%."
            ),
            active_student.get("student_email", ""),
        )
        render_highlight_card(
            "Affectation",
            (
                f"Groupe : {assignment.get('group_name', '')}\n\n"
                f"Echeance : {assignment.get('due_date', 'Non definie') or 'Non definie'}"
            ),
            f"{assignment.get('solved_count', 0)}/{assignment.get('recipient_count', 0)} resolu(s)",
            accent="amber",
        )
        render_highlight_card(
            "Lecture tutorat",
            "Cette colonne reste en lecture seule : seul l'etudiant peut converser, l'enseignant peut seulement superviser.",
            "Acces conversationnel observe",
        )


def render_page() -> None:
    """Display the teacher panel."""
    initialize_session_state()
    require_teacher_access("Panneau enseignant")
    _apply_deferred_widget_resets()
    record_page_consultation("teacher_panel", "Panneau enseignant")

    api_client = get_api_client()
    data = api_client.get_teacher_panel_data(teacher_email=st.session_state.auth.get("email", ""))

    render_page_hero(
        "Panneau enseignant",
        "Creez des groupes, affectez des exercices, suivez les performances reelles et supervisez le tutorat des eleves.",
        badge="Espace enseignant",
    )

    dashboard_tab, groups_tab, assignment_tab, supervision_tab = st.tabs(
        ["Tableau de bord", "Groupes", "Affectation d'exercices", "Supervision du tutorat"]
    )

    with dashboard_tab:
        _render_dashboard_tab(data)
    with groups_tab:
        _render_groups_tab(data)
    with assignment_tab:
        _render_assignment_tab(data)
    with supervision_tab:
        _render_supervision_tab(data)


render_page()
