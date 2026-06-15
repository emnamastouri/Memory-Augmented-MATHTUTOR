"""Math notation repair and guard utilities for generated exercises.

This module is intentionally conservative: it fixes only very common notation
corruptions observed in model outputs, then blocks anything that still looks
unsafe before the judge or the student-facing display gate.
"""

from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any

from frontend.utils.openrouter_client import (
    extract_openrouter_text,
    get_openrouter_client,
    get_openrouter_settings,
    has_openrouter_config,
)

STUDENT_FACING_FIELDS = (
    "title",
    "prompt",
    "hint",
    "learning_objective",
    "display_answer",
    "hidden_solution",
    "memory_adaptation_note",
)

MATH_TEXT_FIELDS = (
    "prompt",
    "display_answer",
    "hidden_solution",
    "hint",
    "learning_objective",
)

BAD_MATH_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bextbf\{", "La commande \\textbf a perdu son antislash."),
    (r"\bextit\{", "La commande \\textit a perdu son antislash."),
    (r"\bext\{", "La commande \\text a perdu son antislash."),
    (r"\brac\{", "La commande \\frac a perdu son antislash initial."),
    (r"\bhickapprox\b", "La commande \\approx est corrompue."),
    (r"\bextasciitilde\b", "La commande \\sim est corrompue."),
    (r"\\\\{2,}(?:ln|frac|dfrac|text|textit|sqrt|mathbb|int|sum|approx|sim)\b", "Une commande LaTeX contient trop d'antislashs."),
    (r"\begin\{", "Un environnement LaTeX begin a perdu son antislash."),
    (r"\bend\{", "Un environnement LaTeX end a perdu son antislash."),
    (r"\bfrace\b", "Le token 'frace' indique une fraction LaTeX corrompue."),
    (r"\bfracpi\b", "Le token 'fracpi' indique une fraction de pi corrompue."),
    (r"\bfrac(?:1|2|3|4|5|6|7|8|9)\b", "Une fraction compacte du type 'frac1'/'frac2' reste non corrigee."),
    (r"\bmathbb\s*[A-Z]\b", "Une notation d'ensemble du type 'mathbb R' reste non corrigee."),
    (r"\bvec\s*[ijk]\b", "Une notation vectorielle du type 'vec i' reste non corrigee."),
    (r"(?:\+|-|->|vers|tend(?:re)?\s+vers|lim(?:ite)?)[^.\n]{0,35}\bin\s+fty\b", "La borne infinie contient encore le token corrompu 'in fty'."),
    (r"\bin\s+t_\b", "Une integrale contient encore le token corrompu 'in t_'."),
    (r"\blim_\s*[a-z]\s*->", "Une limite du type 'lim_x ->' reste non convertie en LaTeX propre."),
    (r"e\^\{0\s*U\}_0", "La notation corrompue 'e^{0U}_0' reste presente."),
    (r"e\^\{-n\s*U\}_n", "La notation corrompue 'e^{-nU}_n' reste presente."),
    (r"\\\\int_\\alpha\^\{\{\{0 f\}\}\}", "Une notation d'integrale corrompue reste presente."),
)


def find_math_format_issues(exercise: dict[str, Any]) -> list[str]:
    """Return blocking formatting issues found in an exercise."""
    segments: list[str] = []
    for field in MATH_TEXT_FIELDS:
        segments.append(str(exercise.get(field, "")))
    segments.extend(str(step) for step in (exercise.get("solution_steps") or []))
    segments.extend(str(option) for option in (exercise.get("options") or []))
    text = "\n".join(segment for segment in segments if segment).strip()
    if not text:
        return []

    issues: list[str] = []
    lowered = text.lower()
    for pattern, message in BAD_MATH_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            issues.append(f"Formatting repair guard failed: {message}")
    return _deduplicate(issues)


def has_corrupted_math_exercise(exercise: dict[str, Any]) -> bool:
    """True when the exercise still contains known corrupted math tokens."""
    return bool(find_math_format_issues(exercise))


def repair_exercise_math_locally(exercise: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Apply deterministic repairs to common corrupted mathematical notation."""
    repaired = deepcopy(exercise)
    changed = False

    for field in STUDENT_FACING_FIELDS:
        if field in repaired:
            new_value = repair_math_text_locally(repaired.get(field, ""))
            if new_value != repaired.get(field, ""):
                repaired[field] = new_value
                changed = True

    if repaired.get("accepted_answers"):
        new_answers = [repair_math_text_locally(answer) for answer in repaired.get("accepted_answers", [])]
        if new_answers != repaired.get("accepted_answers"):
            repaired["accepted_answers"] = new_answers
            changed = True

    if repaired.get("solution_steps"):
        new_steps = [repair_math_text_locally(step) for step in repaired.get("solution_steps", [])]
        if new_steps != repaired.get("solution_steps"):
            repaired["solution_steps"] = new_steps
            changed = True

    if repaired.get("options"):
        new_options = [repair_math_text_locally(option) for option in repaired.get("options", [])]
        if new_options != repaired.get("options"):
            repaired["options"] = new_options
            changed = True

    if changed:
        repaired["math_format_repair_applied"] = True
        repaired.setdefault("math_format_repair_notes", []).append("Reparation locale des notations mathematiques courantes.")
    return repaired, changed


def repair_math_text_locally(value: Any) -> str:
    """Repair common model-output notation corruptions without changing meaning."""
    text = str(value or "").replace("\r", "").strip()
    if not text:
        return ""

    text = preserve_latex_backslashes(text)
    text = repair_corrupted_latex_commands(text)
    text = _normalize_latex_delimiters(text)

    # Canonical LaTeX commands accidentally output without backslashes/braces.
    text = re.sub(r"\bmathbb\s*R\b", r"\\mathbb{R}", text)
    text = re.sub(r"\bmathbb\s*N\b", r"\\mathbb{N}", text)
    text = re.sub(r"\bmathbb\s*Z\b", r"\\mathbb{Z}", text)
    text = re.sub(r"\bmathbb\s*C\b", r"\\mathbb{C}", text)
    text = re.sub(r"\bmathbb\s*Q\b", r"\\mathbb{Q}", text)
    text = re.sub(r"\bvec\s*([ijk])\b", r"\\vec{\1}", text)
    text = re.sub(r"\bcdot\b", r"\\cdot", text)
    text = re.sub(r"\bsqrt\s*\(?\s*([A-Za-z0-9_+\-*/^]+)\s*\)?", r"\\sqrt{\1}", text)

    # Infinity and limits.
    text = re.sub(r"\+\s*in\s+fty", r"+\\infty", text, flags=re.IGNORECASE)
    text = re.sub(r"-\s*in\s+fty", r"-\\infty", text, flags=re.IGNORECASE)
    text = re.sub(r"\bin\s+fty\b", r"\\infty", text, flags=re.IGNORECASE)
    text = re.sub(r"\+\s*(?<!\\)infty", r"+\\infty", text, flags=re.IGNORECASE)
    text = re.sub(r"-\s*(?<!\\)infty", r"-\\infty", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\\)\binfty\b", r"\\infty", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\blim_\s*([A-Za-z])\s*->\s*([^\s\)]+)",
        lambda m: r"\lim_{" + m.group(1) + r" \to " + _repair_limit_target(m.group(2)) + "}",
        text,
        flags=re.IGNORECASE,
    )

    # Integrals corrupted as "in t_0^1" or "in t_0^ln x".
    text = re.sub(r"\bin\s*t_\s*([^\s^]+)\s*\^\s*([^\s,;\)]+(?:\s+[A-Za-z]+)?)", _repair_integral_match, text)

    # Compact pi fractions and integer fractions.
    text = re.sub(r"\bfracpi\s*([0-9]+)\b", r"\\frac{\\pi}{\1}", text)
    text = re.sub(r"\bfrac\s*pi\s*([0-9]+)\b", r"\\frac{\\pi}{\1}", text)
    text = re.sub(r"\bfrac1(1\s*\+\s*[^\s,;\)]+)", r"\\frac{1}{\1}", text)
    text = re.sub(r"\bfrac1([A-Za-z])\b", r"\\frac{1}{\1}", text)
    text = re.sub(r"\bfrac\s*([0-9]+)\s*([A-Za-z]+)\b", r"\\frac{\1}{\2}", text)
    text = re.sub(r"\bfrac\s*([0-9])\s*([0-9])\b", r"\\frac{\1}{\2}", text)

    # Most common exponential fraction from the uploaded audit: frace^x1+e^2x.
    text = re.sub(
        r"\bfrace\^([A-Za-z])\s*1\s*\+\s*e\^2\1\b",
        r"\\frac{e^\1}{1+e^{2\1}}",
        text,
    )
    text = re.sub(
        r"\bfrace\^([A-Za-z])-1\s*\+\s*e\^2\1\b",
        r"\\frac{e^\1-1}{1+e^{2\1}}",
        text,
    )
    text = re.sub(
        r"\bfrace\^(-?[A-Za-z])\s*1\s*\+\s*e\^(-?2[A-Za-z])\b",
        r"\\frac{e^{\1}}{1+e^{\2}}",
        text,
    )
    text = re.sub(r"e\^\{0\s*U\}_0", r"e^0U_0", text)
    text = re.sub(r"e\^\{0U\}_0", r"e^0U_0", text)
    text = re.sub(r"e\^\{-n\s*U\}_n", r"e^{-n}U_n", text)
    text = re.sub(r"e\^\{-nU\}_n", r"e^{-n}U_n", text)
    text = re.sub(r"\\\\int_\\alpha\^\{\{\{0 f\}\}\}\(x\)\\,dx", r"\\int_\\alpha^0 f(x)\\,dx", text)
    text = re.sub(r"\\int_\\alpha\^\{\{\{0 f\}\}\}\(x\)\\,dx", r"\\int_\\alpha^0 f(x)\\,dx", text)
    text = re.sub(r"\bmathbb\s*N\b", r"\\mathbb{N}", text)
    text = re.sub(r"\bln\s*\(", r"\\ln(", text)

    # Convert explicit tokens that survived as 'frace' into '\frac{e' so they are no longer invisible corruption.
    text = re.sub(r"\bfrace\^([A-Za-z])", r"\\frac{e^\1}", text)
    text = re.sub(r"\bfrac\s*\(([^()]+)\)\s*([^\s,;]+)", r"\\frac{\1}{\2}", text)

    # Add braces to common exponent forms, but avoid touching already braced exponents.
    text = re.sub(r"e\^([0-9]+)([A-Za-z])", r"e^{\1\2}", text)
    text = re.sub(r"([A-Za-z])\^([0-9]+)([A-Za-z])", r"\1^{\2}\3", text)

    # Normalize repeated spaces while preserving LaTeX commands.
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" \\infty", r"\infty")
    return text.strip()


def preserve_latex_backslashes(text: str) -> str:
    """Keep valid LaTeX commands intact instead of stripping their backslashes."""
    if not text:
        return ""
    protected_commands = (
        "frac",
        "dfrac",
        "textbf",
        "textit",
        "text",
        "ln",
        "log",
        "exp",
        "sqrt",
        "mathbb",
        "overline",
        "bar",
        "hat",
        "vec",
        "approx",
        "sim",
        "le",
        "ge",
        "infty",
        "int",
        "sum",
    )
    # Collapse excessive escaping from JSON/string round-trips while keeping one
    # LaTeX command backslash. This targets commands only, not normal prose.
    command_pattern = "|".join(re.escape(command) for command in protected_commands)
    return re.sub(r"\\{2,}(?=(" + command_pattern + r")\b)", r"\\", str(text))


def repair_corrupted_latex_commands(text: str) -> str:
    """Repair known command corruptions without guessing arbitrary French words."""
    repaired = preserve_latex_backslashes(str(text or ""))
    repaired = re.sub(r"(?<!\\)\bextbf\{", r"\\textbf{", repaired)
    repaired = re.sub(r"(?<!\\)\bextit\{", r"\\textit{", repaired)
    repaired = re.sub(r"(?<!\\)\bext\{", r"\\text{", repaired)
    repaired = re.sub(r"(?<![A-Za-z\\])rac\{", r"\\frac{", repaired)
    repaired = re.sub(r"(?<!\\)\bdfrac\{", r"\\dfrac{", repaired)
    repaired = re.sub(r"(?<!\\)\bmathbb\{([A-Z])\}", r"\\mathbb{\1}", repaired)
    repaired = re.sub(r"\bhickapprox\b", r"\\approx", repaired)
    repaired = re.sub(r"\bextasciitilde\b", r"\\sim", repaired)
    repaired = re.sub(r"(?<!\\)\begin\{(cases|pmatrix|bmatrix)\}", r"\\begin{\1}", repaired)
    repaired = re.sub(r"(?<!\\)\bend\{(cases|pmatrix|bmatrix)\}", r"\\end{\1}", repaired)
    repaired = re.sub(r"\\{2,}(ln|frac|dfrac|text|textbf|textit|sqrt|mathbb|int|sum|approx|sim|begin|end)\b", r"\\\1", repaired)
    repaired = re.sub(r"\\{2,}\(", r"\\(", repaired)
    repaired = re.sub(r"\\{2,}\)", r"\\)", repaired)
    repaired = re.sub(r"\\{2,}\[", r"\\[", repaired)
    repaired = re.sub(r"\\{2,}\]", r"\\]", repaired)
    return repaired


def repair_exercise_math_with_openrouter(
    exercise: dict[str, Any],
    *,
    level: str = "",
    section: str = "",
    topic: str = "",
    subtopic: str = "",
    previous_issues: list[str] | None = None,
) -> tuple[dict[str, Any], bool, list[str]]:
    """Ask OpenRouter to repair notation only, then verify the result locally."""
    if not has_openrouter_config():
        return exercise, False, ["OpenRouter indisponible pour la reparation de notation."]

    settings = get_openrouter_settings()
    if settings is None:
        return exercise, False, ["Configuration OpenRouter absente."]

    client = get_openrouter_client()
    compact_exercise = {
        "title": exercise.get("title", ""),
        "prompt": exercise.get("prompt", ""),
        "hint": exercise.get("hint", ""),
        "learning_objective": exercise.get("learning_objective", ""),
        "expected_answer": exercise.get("display_answer", ""),
        "full_solution": exercise.get("hidden_solution", ""),
        "answer_kind": exercise.get("answer_kind", "text"),
        "solution_steps": exercise.get("solution_steps", []),
        "options": exercise.get("options", []),
    }
    issue_block = "\n".join(f"- {issue}" for issue in (previous_issues or [])) or "- Aucun historique."

    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un reparateur de notation mathematique LaTeX pour MathTutorAI. "
                "Tu ne dois pas resoudre l'exercice, ne pas changer les questions, ne pas changer les resultats. "
                "Tu corriges uniquement la notation mathematique corrompue. "
                "Retourne uniquement un objet JSON valide."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Contexte: niveau={level}, section={section}, theme={topic}, sous-theme={subtopic}.\n"
                "Problemes detectes avant reparation:\n"
                f"{issue_block}\n\n"
                "Repare uniquement les notations mathematiques corrompues. Exemples obligatoires:\n"
                "- frace^x1+e^2x -> \\frac{e^x}{1+e^{2x}}\n"
                "- fracpi4 -> \\frac{\\pi}{4}\n"
                "- mathbb R -> \\mathbb{R}\n"
                "- in fty ou infty -> \\infty\n"
                "- in t_0^1 -> \\int_0^1\n"
                "- lim_x -> + infty -> \\lim_{x \\to +\\infty}\n\n"
                "Ne modifie pas le contenu mathematique, ne simplifie pas, n'ajoute pas de nouvelle question.\n"
                "Retourne exactement ces champs JSON: title, prompt, hint, learning_objective, expected_answer, full_solution, answer_kind, solution_steps, options.\n"
                f"Exercice a reparer:\n{json.dumps(compact_exercise, ensure_ascii=False)}"
            ),
        },
    ]
    request_kwargs = {
        "model": settings.judge_model or settings.exercise_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 3200,
    }

    try:
        response = client.chat.completions.create(response_format={"type": "json_object"}, **request_kwargs)
    except Exception:
        response = client.chat.completions.create(**request_kwargs)

    content = extract_openrouter_text(response)
    payload = _extract_json_payload(content)
    if not payload:
        return exercise, False, ["Le reparateur de notation n'a pas renvoye un JSON exploitable."]

    repaired = deepcopy(exercise)
    field_map = {
        "title": "title",
        "prompt": "prompt",
        "hint": "hint",
        "learning_objective": "learning_objective",
        "expected_answer": "display_answer",
        "full_solution": "hidden_solution",
        "answer_kind": "answer_kind",
    }
    for source_field, target_field in field_map.items():
        value = payload.get(source_field)
        if isinstance(value, str) and value.strip():
            repaired[target_field] = value.strip()

    if isinstance(payload.get("solution_steps"), list):
        repaired["solution_steps"] = [str(step).strip() for step in payload["solution_steps"] if str(step).strip()]
    if isinstance(payload.get("options"), list):
        repaired["options"] = [str(option).strip() for option in payload["options"] if str(option).strip()]
    if repaired.get("display_answer"):
        repaired["accepted_answers"] = [repaired["display_answer"]]

    repaired, local_changed = repair_exercise_math_locally(repaired)
    remaining_issues = find_math_format_issues(repaired)
    if not remaining_issues:
        repaired["math_format_repair_applied"] = True
        repaired.setdefault("math_format_repair_notes", []).append("Reparation OpenRouter avant juge.")
        return repaired, True, []

    return repaired if local_changed else exercise, False, remaining_issues


def _normalize_latex_delimiters(text: str) -> str:
    text = re.sub(r"\$\$(.+?)\$\$", r"\\[\1\\]", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", r"\\(\1\\)", text)
    return text


def _repair_limit_target(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"\+\s*(?:in\s+fty|infty)", r"+\\infty", value, flags=re.IGNORECASE)
    value = re.sub(r"-\s*(?:in\s+fty|infty)", r"-\\infty", value, flags=re.IGNORECASE)
    value = value.replace("\\\\infty", "\\infty")
    return value


def _repair_integral_match(match: re.Match[str]) -> str:
    lower = match.group(1).strip()
    upper = match.group(2).strip()
    upper = upper.replace("ln x", r"\ln x")
    upper = re.sub(r"in\s+fty|infty", lambda _m: r"\infty", upper, flags=re.IGNORECASE)
    if " " in upper and not (upper.startswith("{") and upper.endswith("}")):
        upper = "{" + upper + "}"
    return rf"\int_{lower}^{upper}"


def _extract_json_payload(raw_content: str) -> dict[str, Any]:
    content = (raw_content or "").strip()
    if not content:
        return {}
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()
    for candidate in (content, _slice_json_object(content)):
        if not candidate:
            continue
        for payload in (candidate, _repair_invalid_json_backslashes(candidate)):
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def _slice_json_object(content: str) -> str:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return content[start : end + 1]


def _repair_invalid_json_backslashes(value: str) -> str:
    repaired = re.sub(r"\\(?=[A-Za-z]{2,})", r"\\\\", value)
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", repaired)


def _deduplicate(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result
