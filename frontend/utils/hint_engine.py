"""Utilities for progressive and adaptive exercise hints."""

from __future__ import annotations

from typing import Any


def build_progressive_hints(exercise: dict[str, Any]) -> list[str]:
    """Derive a stable 3-level hint sequence from one exercise payload."""
    base_hint = str(exercise.get("hint", "")).strip()
    solution_steps = [str(step).strip() for step in exercise.get("solution_steps", []) if str(step).strip()]
    exercise_type = str(exercise.get("exercise_type", "")).strip()
    learning_objective = str(exercise.get("learning_objective", "")).strip()

    hints: list[str] = []

    if base_hint:
        hints.append(
            "Indice 1. Commence par identifier la notion centrale avant tout calcul. "
            f"{base_hint}"
        )

    if solution_steps:
        hints.append(
            "Indice 2. Pose-toi cette question : quelle est la toute premiere action mathematique utile ici ? "
            f"Une bonne piste est : {solution_steps[0]}"
        )

    if len(solution_steps) >= 2:
        hints.append(
            "Indice 3. Organise ton raisonnement en deux temps sans ecrire encore la reponse finale : "
            f"d'abord {solution_steps[0].lower()}, puis {solution_steps[1].lower()}."
        )
    elif solution_steps:
        hints.append(
            "Indice 3. Transforme maintenant ton idee en demarche complete : "
            f"appuie-toi sur cette etape-cle {solution_steps[0].lower()}."
        )
    elif learning_objective:
        hints.append(
            "Indice 3. Garde ce cap pedagogique : "
            f"{learning_objective}"
        )

    if exercise_type == "QCM" and len(hints) >= 3:
        hints[2] += " Pour un QCM, elimine d'abord les propositions impossibles avant de comparer les deux plus plausibles."

    return _deduplicate(hints)[:3]


def build_adaptive_hint(exercise: dict[str, Any], student_answer: str) -> str:
    """Create a Socratic hint adapted to the student's current answer draft."""
    answer = " ".join(student_answer.split()).strip()
    solution_steps = [str(step).strip() for step in exercise.get("solution_steps", []) if str(step).strip()]
    answer_kind = str(exercise.get("answer_kind", "text")).strip().lower()
    exercise_type = str(exercise.get("exercise_type", "")).strip()

    if not answer:
        if solution_steps:
            return (
                "Indice adaptatif. Avant de chercher le resultat, ecris d'abord la premiere etape utile. "
                f"Question-guide : comment peux-tu commencer par {solution_steps[0].lower()} ?"
            )
        return "Indice adaptatif. Ecris d'abord ce que l'enonce te donne et ce qu'il te demande vraiment de trouver."

    if exercise_type == "QCM":
        return (
            "Indice adaptatif. Ne regarde pas encore la bonne lettre. "
            "Reprends chaque proposition et demande-toi laquelle respecte la definition, la formule ou la propriete de l'enonce. "
            + _optional_step_tail(solution_steps)
        ).strip()

    if answer_kind == "numeric":
        return (
            "Indice adaptatif. Ta reponse ressemble a un resultat final. "
            "Quelle chaine de calcul justifie ce nombre ligne par ligne ? "
            "Verifie surtout les signes, les coefficients et la derniere operation. "
            + _optional_step_tail(solution_steps)
        ).strip()

    if answer_kind in {"expression", "set"}:
        return (
            "Indice adaptatif. Au lieu de sauter au resultat, justifie chaque transformation. "
            "Quelle regle utilises-tu pour passer de ta ligne actuelle a la suivante ? "
            + _optional_step_tail(solution_steps)
        ).strip()

    return (
        "Indice adaptatif. Reformule ton idee en une petite demarche : "
        "que sais-tu deja, quelle propriete comptes-tu utiliser, et qu'est-ce qu'il reste a montrer ? "
        + _optional_step_tail(solution_steps)
    ).strip()


def get_revealed_hints(exercise: dict[str, Any], level: int) -> list[str]:
    """Return only the currently revealed progressive hints."""
    safe_level = max(0, int(level))
    return build_progressive_hints(exercise)[:safe_level]


def _optional_step_tail(solution_steps: list[str]) -> str:
    """Append a small method reminder when a first step exists."""
    if not solution_steps:
        return ""
    return f" Piste utile : {solution_steps[0]}"


def _deduplicate(items: list[str]) -> list[str]:
    """Remove duplicates while preserving order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered
