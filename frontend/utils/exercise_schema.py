from __future__ import annotations

import re
from typing import Any
import unicodedata


QUESTION_CUES = [
    "calculer",
    "calculez",
    "determiner",
    "determinez",
    "donner",
    "donnez",
    "montrer",
    "montrez",
    "justifier",
    "justifiez",
    "prouver",
    "prouvez",
    "en deduire",
    "deduire",
    "resoudre",
    "resolvez",
    "etudier",
    "etudiez",
    "dresser",
    "construire",
    "tracer",
    "verifier",
    "verifiez",
    "completer",
    "comparer",
    "interpreter",
    "quelle est",
    "quelles sont",
    "est-ce que",
    "peut-on",
    "demontrer",
]

QUESTION_MARKER_PATTERN = re.compile(
    r"(?:^|\s)(?:question\s*\d+|[0-9]+\s*[\)\.]|[a-z]\)|[ivxlcdm]+\)|-\s*(?:calculer|determin|donner|montrer|justifier|prouver|resoudre|etudier|verifier|completer|comparer|interpreter|tracer|construire))",
    flags=re.IGNORECASE,
)
DIRECT_IMPERATIVE_PATTERN = re.compile(
    r"(?:^|[\.\?!;:]\s+)(?:calculer|calculez|determiner|determinez|donner|donnez|montrer|montrez|justifier|justifiez|prouver|prouvez|en deduire|deduire|resoudre|resolvez|etudier|etudiez|dresser|construire|tracer|verifier|verifiez|completer|comparer|interpreter|quelle est|quelles sont|est-ce que|peut-on|demontrer)\b",
    flags=re.IGNORECASE,
)

PLACEHOLDER_PATTERNS = (
    r"\btodo\b",
    r"\bvoir annexe\b",
    r"\breponse en attente\b",
    r"\bplaceholder\b",
    r"\bcompleter ici\b",
)


def normalize_exercise_schema(obj: dict[str, Any]) -> dict[str, Any]:
    """Normalize model outputs into the v7 structured exercise schema."""
    payload = dict(obj or {})

    title = str(payload.get("title", "") or "").strip()
    context = str(payload.get("context", "") or "").strip()
    instruction = str(payload.get("instruction") or payload.get("prompt") or "").strip()
    solution = str(payload.get("solution") or payload.get("full_solution") or "").strip()
    expected_answer = str(payload.get("expected_answer") or payload.get("display_answer") or "").strip()
    questions = _coerce_question_list(payload.get("questions"))

    if (not context or not questions) and instruction:
        split_context, split_questions = split_instruction_into_context_and_questions(instruction)
        context = context or split_context
        if not questions:
            questions = split_questions

    if not context and questions:
        context = _derive_context_from_instruction(instruction, questions)
    if not instruction and (context or questions):
        instruction = compose_instruction(context, questions)
    elif instruction and context and questions and not _instruction_contains_context_and_questions(instruction, context, questions):
        instruction = compose_instruction(context, questions)

    generation_metadata = payload.get("generation_metadata")
    if not isinstance(generation_metadata, dict):
        generation_metadata = {}

    normalized = {
        "title": title,
        "context": _compact_text(context),
        "questions": questions,
        "instruction": _compact_text(instruction),
        "prompt": _compact_text(instruction),
        "solution": _compact_text(solution),
        "full_solution": _compact_text(solution),
        "expected_answer": _compact_text(expected_answer),
        "answer_kind": str(payload.get("answer_kind", "") or "").strip().lower(),
        "solution_steps": _coerce_string_list(payload.get("solution_steps")),
        "learning_objective": _compact_text(payload.get("learning_objective", "")),
        "estimated_time": _compact_text(payload.get("estimated_time", "")),
        "table_data": _normalize_optional_object(payload.get("table_data")),
        "chart_data": _normalize_optional_object(payload.get("chart_data")),
        "graph_data": _normalize_optional_object(payload.get("graph_data")),
        "hint": _compact_text(payload.get("hint", "")),
        "options": _coerce_string_list(payload.get("options")),
        "memory_rationale": _compact_text(payload.get("memory_rationale", "")),
        "generation_metadata": {
            "target_section": str(generation_metadata.get("target_section") or payload.get("section") or "").strip(),
            "target_topic": str(generation_metadata.get("target_topic") or payload.get("topic") or "").strip(),
            "target_subtopic": str(generation_metadata.get("target_subtopic") or payload.get("subtopic") or "").strip(),
            "exercise_family": str(generation_metadata.get("exercise_family") or payload.get("exercise_type") or "").strip(),
            "requires_symbolic_check": bool(generation_metadata.get("requires_symbolic_check", False)),
            "requires_numeric_check": bool(generation_metadata.get("requires_numeric_check", False)),
        },
    }
    return normalized


def validate_exercise_schema_v7(obj: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate the strict v7 exercise schema used by the LLM path."""
    payload = normalize_exercise_schema(obj)
    issues: list[str] = []

    for field in ("title", "context", "instruction", "solution", "expected_answer", "learning_objective", "estimated_time"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"Champ requis manquant ou vide : {field}.")

    if not isinstance(payload.get("questions"), list) or len(payload["questions"]) < 2:
        issues.append("Le schema v7 exige au moins deux questions explicites.")
    else:
        for index, question in enumerate(payload["questions"], start=1):
            question_text = str(question or "").strip()
            if not question_text or question_text.rstrip(" ):.") in {"", str(index), chr(96 + index)}:
                issues.append(f"Question vide ou inexploitables a l'index {index}.")

    if not has_explicit_questions(payload.get("instruction", ""), payload.get("questions")):
        issues.append("L'enonce ne contient pas de questions explicites detectables.")

    if payload.get("context") and payload.get("questions") and not _instruction_contains_context_and_questions(
        payload["instruction"],
        payload["context"],
        payload["questions"],
    ):
        issues.append("L'instruction ne recompose pas correctement le contexte et les questions.")

    solution = payload.get("solution", "")
    if payload.get("questions"):
        if len(payload["questions"]) >= 2 and len(solution) < 60:
            issues.append("La solution est trop courte pour repondre a toutes les questions.")

    combined = " ".join(
        [
            payload.get("title", ""),
            payload.get("context", ""),
            payload.get("instruction", ""),
            payload.get("solution", ""),
            payload.get("expected_answer", ""),
        ]
    )
    normalized_combined = _normalize_lookup(combined)
    for pattern in PLACEHOLDER_PATTERNS:
        if re.search(pattern, normalized_combined, flags=re.IGNORECASE):
            issues.append("Le schema contient encore un placeholder interne.")
            break

    for field in ("table_data", "chart_data", "graph_data"):
        value = payload.get(field)
        if value is not None and not isinstance(value, dict):
            issues.append(f"Le champ {field} doit etre un objet ou null.")

    return not issues, issues


def has_explicit_questions(text: str, questions: list[str] | None = None) -> bool:
    """Detect whether a French mathematical statement contains explicit student tasks."""
    question_items = [str(question or "").strip() for question in (questions or []) if str(question or "").strip()]
    if len(question_items) >= 2:
        return True

    normalized = _normalize_lookup(text)
    if not normalized:
        return False

    if re.search(r"\?", text):
        return True

    imperative_count = sum(1 for cue in QUESTION_CUES if cue in normalized)
    has_numbering = bool(QUESTION_MARKER_PATTERN.search(normalized))
    if has_numbering and imperative_count >= 1:
        return True
    if DIRECT_IMPERATIVE_PATTERN.search(normalized):
        return True
    if imperative_count >= 2:
        return True
    if question_items and imperative_count >= 1:
        return True
    return False


def split_instruction_into_context_and_questions(instruction: str) -> tuple[str, list[str]]:
    """Split one legacy instruction into a pure context and a question list."""
    text = _compact_text(instruction)
    if not text:
        return "", []

    numbered = re.sub(
        r"\s+(?=(?:question\s*\d+|[0-9]+\s*[\)\.]|[a-z]\)|[ivxlcdm]+\)|-\s*(?:Calculer|Déterminer|Donner|Montrer|Justifier|Prouver|Résoudre|Étudier|Vérifier|Compléter|Comparer|Interpréter|Tracer|Construire)))",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    segments = [segment.strip(" -") for segment in numbered.splitlines() if segment.strip(" -")]
    context_parts: list[str] = []
    questions: list[str] = []

    for segment in segments:
        if _looks_like_question_segment(segment):
            questions.append(_strip_question_marker(segment))
        elif questions:
            questions[-1] = f"{questions[-1]} {segment}".strip()
        else:
            context_parts.append(segment)

    if questions:
        return _compact_text(" ".join(context_parts)), _clean_questions(questions)

    sentences = re.split(r"(?<=[\.\?!;:])\s+", text)
    for sentence in sentences:
        compact = sentence.strip()
        if not compact:
            continue
        if _looks_like_question_segment(compact):
            questions.append(_strip_question_marker(compact))
        elif questions:
            questions[-1] = f"{questions[-1]} {compact}".strip()
        else:
            context_parts.append(compact)

    return _compact_text(" ".join(context_parts)), _clean_questions(questions)


def compose_instruction(context: str, questions: list[str]) -> str:
    """Compose one student-facing instruction from context and question items."""
    context_text = _compact_text(context)
    clean_questions = _clean_questions(questions)
    if not clean_questions:
        return context_text
    numbered_questions = [f"{index}) {question}" for index, question in enumerate(clean_questions, start=1)]
    if context_text:
        return f"{context_text}\n" + "\n".join(numbered_questions)
    return "\n".join(numbered_questions)


def _instruction_contains_context_and_questions(instruction: str, context: str, questions: list[str]) -> bool:
    normalized_instruction = _normalize_lookup(instruction)
    normalized_context = _normalize_lookup(context)
    if normalized_context and normalized_context[:30] not in normalized_instruction:
        return False
    for question in _clean_questions(questions):
        snippet = _normalize_lookup(question)[:20]
        if snippet and snippet not in normalized_instruction:
            return False
    return True


def _derive_context_from_instruction(instruction: str, questions: list[str]) -> str:
    context, _ = split_instruction_into_context_and_questions(instruction)
    if context:
        return context
    cleaned_instruction = _compact_text(instruction)
    for question in _clean_questions(questions):
        cleaned_instruction = cleaned_instruction.replace(question, "").strip()
    return cleaned_instruction.strip(" :;-")


def _looks_like_question_segment(segment: str) -> bool:
    normalized = _normalize_lookup(segment)
    if not normalized:
        return False
    if QUESTION_MARKER_PATTERN.search(normalized):
        return True
    return any(cue in normalized for cue in QUESTION_CUES)


def _strip_question_marker(segment: str) -> str:
    cleaned = re.sub(r"^(?:question\s*\d+\s*[:\-]?\s*|[0-9]+\s*[\)\.]\s*|[a-z]\)\s*|[ivxlcdm]+\)\s*|-\s*)", "", segment, flags=re.IGNORECASE)
    return _compact_text(cleaned)


def _coerce_question_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return _clean_questions(value)
    if isinstance(value, str) and value.strip():
        _, questions = split_instruction_into_context_and_questions(value)
        return questions
    return []


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value in (None, ""):
        return []
    return [str(value).strip()]


def _normalize_optional_object(value: Any) -> dict[str, Any] | None:
    if value in ("", None, [], ()):
        return None
    return value if isinstance(value, dict) else None


def _clean_questions(questions: list[Any]) -> list[str]:
    result: list[str] = []
    for question in questions:
        cleaned = _compact_text(_strip_question_marker(str(question or "")))
        if cleaned:
            result.append(cleaned)
    return result


def _compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).strip()


def _normalize_lookup(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_value).strip().lower()
