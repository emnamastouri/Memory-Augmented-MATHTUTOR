"""Question-by-question solution coverage checks."""

from __future__ import annotations

import re
from typing import Any


def validate_question_answer_coverage(exercise: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    """Check that each explicit question has a visible answer in the solution."""
    questions = [str(item).strip() for item in (exercise.get("questions") or []) if str(item).strip()]
    solution = " ".join(
        str(exercise.get(field, "")) for field in ("hidden_solution", "display_answer")
    )
    if not questions:
        return True, [], {"unanswered_question_indices": [], "question_count": 0}
    issues: list[str] = []
    unanswered: list[int] = []
    normalized_solution = solution.lower()
    numbered_answers = len(re.findall(r"(?:^|\s)(?:question\s*)?\d+\s*[\).:-]", normalized_solution))
    if numbered_answers and numbered_answers < len(questions):
        issues.append(f"Question coverage failed: {numbered_answers} reponses numerotees pour {len(questions)} questions.")
        unanswered.extend(range(numbered_answers + 1, len(questions) + 1))
    vf_questions = sum(1 for question in questions if any(token in question.lower() for token in ("vrai", "faux")))
    if vf_questions:
        vf_answers = len(re.findall(r"\b(vrai|faux)\b", normalized_solution))
        if vf_answers < vf_questions:
            issues.append("Question coverage failed: le nombre de reponses Vrai/Faux est insuffisant.")
    for index, question in enumerate(questions, start=1):
        q_norm = question.lower()
        failed = False
        if any(token in q_norm for token in ("estimer", "calculer", "determiner", "déterminer")):
            if not re.search(r"[-+]?\d+(?:[.,]\d+)?", solution):
                failed = True
        if "equation" in q_norm and "=" not in solution:
            failed = True
        if "justifier" in q_norm and not any(token in normalized_solution for token in ("car", "donc", "parce", "puisque", "en effet")):
            failed = True
        if failed:
            unanswered.append(index)
    if unanswered:
        issues.append("Question coverage failed: questions sans reponse exploitable: " + ", ".join(map(str, unanswered)))
    return not issues, issues, {"unanswered_question_indices": unanswered, "question_count": len(questions)}
