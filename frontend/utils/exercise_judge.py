"""OpenRouter-based judge for validating generated exercise instructions."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from frontend.utils.alignment_catalog import AlignmentRecord, get_alignment_record
from frontend.utils.exercise_supports import describe_supports_for_judge
from frontend.utils.openrouter_client import (
    extract_openrouter_text,
    get_openrouter_client,
    get_openrouter_settings,
    has_openrouter_config,
    summarize_openrouter_response_issue,
)

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approved", "corrected", "rejected"],
        },
        "summary": {"type": "string"},
        "alignment_status": {
            "type": "string",
            "enum": ["aligned", "misaligned"],
        },
        "alignment_reason": {"type": "string"},
        "issues": {
            "type": "array",
            "items": {"type": "string"},
        },
        "confidence": {"type": "number"},
        "corrected_title": {"type": "string"},
        "corrected_prompt": {"type": "string"},
        "corrected_hint": {"type": "string"},
        "corrected_learning_objective": {"type": "string"},
        "corrected_expected_answer": {"type": "string"},
        "corrected_full_solution": {"type": "string"},
        "corrected_answer_kind": {"type": "string"},
        "corrected_options": {
            "type": "array",
            "items": {"type": "string"},
        },
        "corrected_solution_steps": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["decision", "summary", "alignment_status", "alignment_reason", "issues", "confidence"],
}


@dataclass(frozen=True)
class ExerciseJudgeDecision:
    """Structured result returned by the exercise judge."""

    decision: str
    summary: str
    alignment_status: str
    alignment_reason: str
    issues: list[str]
    confidence: float
    corrected_fields: dict[str, Any]
    model_name: str
    error_message: str = ""


def judge_generated_exercise(
    exercise: dict[str, Any],
    *,
    level: str,
    section: str,
    topic: str,
    subtopic: str,
    exercise_type: str,
) -> ExerciseJudgeDecision:
    """Review one generated exercise before exposing it to the student."""
    settings = get_openrouter_settings()
    default_model = settings.judge_model if settings is not None else "qwen/qwen-2.5-7b-instruct"

    if not has_openrouter_config() or settings is None:
        return ExerciseJudgeDecision(
            decision="error",
            summary="Le juge OpenRouter n'est pas configure.",
            alignment_status="misaligned",
            alignment_reason="Configuration OpenRouter absente.",
            issues=["Ajoutez la section [openrouter] avec api_key dans .streamlit/secrets.toml."],
            confidence=0.0,
            corrected_fields={},
            model_name=default_model,
            error_message="Configuration OpenRouter absente.",
        )

    alignment_record = get_alignment_record(section, topic, subtopic)
    if alignment_record is None:
        return ExerciseJudgeDecision(
            decision="rejected",
            summary="Le couple section-theme-sous-theme n'apparait pas dans le programme officiel de reference.",
            alignment_status="misaligned",
            alignment_reason=(
                "Aucune entree d'alignement officielle n'a ete trouvee pour ce couple. "
                "Le juge bloque donc cet exercice."
            ),
            issues=[
                "Couple non couvert par le referentiel officiel fourni.",
                "La regeneration ne peut pas produire un exercice aligne tant que ce couple reste absent du fichier d'alignement.",
            ],
            confidence=1.0,
            corrected_fields={},
            model_name=settings.judge_model,
            error_message="",
        )

    prompt = _build_judge_prompt(
        exercise=exercise,
        level=level,
        section=section,
        topic=topic,
        subtopic=subtopic,
        exercise_type=exercise_type,
        alignment_record=alignment_record,
    )

    try:
        payload = _call_openrouter_judge(prompt, settings.judge_model)
    except RuntimeError as exc:
        return ExerciseJudgeDecision(
            decision="error",
            summary="Le juge OpenRouter est temporairement indisponible.",
            alignment_status="misaligned",
            alignment_reason="Le service du juge n'a pas pu verifier l'alignement officiel.",
            issues=[str(exc)],
            confidence=0.0,
            corrected_fields={},
            model_name=settings.judge_model,
            error_message=str(exc),
        )

    decision = str(payload.get("decision", "")).strip().lower()
    if decision not in {"approved", "corrected", "rejected"}:
        decision = "error"
    alignment_status = str(payload.get("alignment_status", "")).strip().lower()
    if alignment_status not in {"aligned", "misaligned"}:
        alignment_status = "misaligned"
    alignment_reason = str(payload.get("alignment_reason", "")).strip() or "Motif d'alignement indisponible."
    if alignment_status != "aligned":
        decision = "rejected"

    issues = payload.get("issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)]
    cleaned_issues = [str(item).strip() for item in issues if str(item).strip()]

    corrected_fields = {
        "title": str(payload.get("corrected_title", "")).strip(),
        "prompt": str(payload.get("corrected_prompt", "")).strip(),
        "hint": str(payload.get("corrected_hint", "")).strip(),
        "learning_objective": str(payload.get("corrected_learning_objective", "")).strip(),
        "expected_answer": str(payload.get("corrected_expected_answer", "")).strip(),
        "full_solution": str(payload.get("corrected_full_solution", "")).strip(),
        "answer_kind": str(payload.get("corrected_answer_kind", "")).strip(),
        "options": payload.get("corrected_options") or [],
        "solution_steps": payload.get("corrected_solution_steps") or [],
    }

    return ExerciseJudgeDecision(
        decision=decision,
        summary=str(payload.get("summary", "")).strip() or "Verdict du juge indisponible.",
        alignment_status=alignment_status,
        alignment_reason=alignment_reason,
        issues=cleaned_issues,
        confidence=_coerce_confidence(payload.get("confidence")),
        corrected_fields=corrected_fields,
        model_name=settings.judge_model,
        error_message="" if decision != "error" else "Le format de sortie du juge est invalide.",
    )


def _build_judge_prompt(
    *,
    exercise: dict[str, Any],
    level: str,
    section: str,
    topic: str,
    subtopic: str,
    exercise_type: str,
    alignment_record: AlignmentRecord,
) -> str:
    """Create the quality-control prompt for the judge."""
    options = exercise.get("options", [])
    option_block = "\n".join([f"- {option}" for option in options]) if options else "- Aucun choix propose"
    solution_steps = "\n".join([f"- {step}" for step in exercise.get("solution_steps", [])]) or "- Aucune etape fournie"
    full_solution = str(exercise.get("hidden_solution", "")).strip() or "- Solution detaillee absente"
    warnings_block = "\n".join([f"- {warning}" for warning in alignment_record.warnings]) or "- Aucun avertissement specifique"
    support_block = describe_supports_for_judge(exercise)

    return (
        "Tu es le juge qualite de MathTutorAI. "
        "Tu controles si un exercice de mathematiques peut etre montre a un eleve de terminale. "
        "Verifie la correction mathematique, la coherence pedagogique, l'alignement avec la demande et la solvabilite. "
        "La solution complete fournie plus bas est reservee au controle interne et ne sera pas affichee a l'eleve.\n\n"
        "Regle dure d'alignement : si l'exercice sort du programme officiel du couple cible, "
        "tu dois obligatoirement retourner alignment_status='misaligned' et decision='rejected'. "
        "Tu n'as pas le droit de choisir corrected pour un exercice non aligne.\n\n"
        "Regles de decision :\n"
        "- approved : l'exercice est bon et peut etre montre tel quel.\n"
        "- corrected : l'exercice est deja aligne au programme officiel mais contient des problemes locaux corrigibles rapidement sans changer le theme.\n"
        "- rejected : l'exercice doit etre regenere car il est faux, incoherent, hors sujet, ambigu ou insuffisamment exploitable.\n\n"
        "Controle notamment :\n"
        "- l'enonce est comprehensible, complet et sans contradiction ;\n"
        "- la consigne correspond au niveau Bac ;\n"
        "- le theme, le sous-theme et le type demande sont respectes ;\n"
        "- l'exercice reste strictement dans le programme officiel fourni ci-dessous ;\n"
        "- l'exercice respecte le focus attendu pour ce couple officiel ;\n"
        "- la reponse attendue, les etapes proposees et la solution complete sont compatibles avec l'enonce ;\n"
        "- si l'enonce dit explicitement qu'un tableau, un graphique, une courbe fournie, des donnees x_i / y_i ou une annexe sont donnes a l'eleve, les supports attaches doivent etre presents et exploitables ;\n"
        "- si l'enonce demande au contraire a l'eleve de tracer une courbe, dresser un tableau de variation ou representer un nuage de points, il ne faut pas exiger un support deja construit ;\n"
        "- un QCM doit avoir des choix coherents ;\n"
        "- un exercice probleme doit rester resolvable et mathematiquement propre.\n\n"
        f"Demande cible\n- Niveau : {level}\n- Section : {section}\n- Theme : {topic}\n- Sous-theme : {subtopic}\n- Type : {exercise_type}\n\n"
        "Programme officiel de reference\n"
        f"- Couple officiel : {alignment_record.section_label} / {alignment_record.topic_label}\n"
        f"- Portee officielle : {alignment_record.official_program_scope}\n"
        f"- Focus de verification : {alignment_record.topic_focus}\n"
        "Avertissements du referentiel\n"
        f"{warnings_block}\n\n"
        "Exercice genere\n"
        f"- Titre : {exercise.get('title', '')}\n"
        f"- Enonce : {exercise.get('prompt', '')}\n"
        f"- Indice : {exercise.get('hint', '')}\n"
        f"- Objectif pedagogique : {exercise.get('learning_objective', '')}\n"
        f"- Reponse attendue : {exercise.get('display_answer', '')}\n"
        f"- Nature de reponse : {exercise.get('answer_kind', '')}\n"
        "Choix proposes\n"
        f"{option_block}\n"
        "Supports attaches\n"
        f"{support_block}\n"
        "Etapes de solution\n"
        f"{solution_steps}\n"
        "Solution complete interne\n"
        f"{full_solution}\n\n"
        "Reponds par un seul objet JSON valide conforme a ce schema conceptuel : "
        "{decision, summary, alignment_status, alignment_reason, issues, confidence, corrected_title, corrected_prompt, corrected_hint, corrected_learning_objective, corrected_expected_answer, corrected_full_solution, corrected_answer_kind, corrected_options, corrected_solution_steps}. "
        "Si tu choisis corrected, renvoie obligatoirement corrected_prompt, corrected_expected_answer et corrected_full_solution, "
        "plus tout autre champ corrige necessaire pour rendre l'exercice montrable a l'eleve sans sortir du programme officiel. "
        "Ces trois champs doivent etre coherents entre eux et remplacer la version initiale. "
        "Si tu choisis approved ou rejected, n'invente pas de correction superflue. "
        "N'ajoute aucun texte avant ou apres le JSON."
    )


def _call_openrouter_judge(prompt: str, model_name: str) -> dict[str, Any]:
    """Call the OpenRouter judge model and parse the resulting JSON object."""
    client = get_openrouter_client()
    messages = [
        {
            "role": "system",
            "content": (
                "Tu es le juge qualite de MathTutorAI. "
                "Retourne uniquement un objet JSON valide, sans markdown et sans texte additionnel."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    request_kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1600,
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
        payload = _repair_payload_with_model(client=client, model_name=model_name, messages=messages, raw_content=content)
    if not payload:
        raise RuntimeError("Le juge OpenRouter n'a pas renvoye un JSON exploitable.")
    return payload


def _repair_payload_with_model(
    *,
    client: Any,
    model_name: str,
    messages: list[dict[str, str]],
    raw_content: str,
) -> dict[str, Any]:
    """Ask the judge model to rewrite its previous answer as strict JSON."""
    repair_messages = [
        *messages,
        {"role": "assistant", "content": raw_content},
        {
            "role": "user",
            "content": (
                "Reformate strictement ta reponse precedente en un objet JSON valide conforme au schema attendu. "
                "Ne mets aucun texte avant ou apres le JSON."
            ),
        },
    ]
    request_kwargs = {
        "model": model_name,
        "messages": repair_messages,
        "temperature": 0,
        "max_tokens": 1600,
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


def _load_json_candidate(candidate: str) -> dict[str, Any]:
    """Load one candidate JSON object, repairing common backslash issues if needed."""
    for payload in (candidate, _repair_invalid_json_backslashes(candidate)):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _repair_invalid_json_backslashes(value: str) -> str:
    """Escape stray backslashes so JSON parsing can recover."""
    repaired = re.sub(r"\\(?=[A-Za-z]{2,})", r"\\\\", value)
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", repaired)


def _coerce_confidence(value: Any) -> float:
    """Clamp confidence values into the [0, 1] interval."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))
