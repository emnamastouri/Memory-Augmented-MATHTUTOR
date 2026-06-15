from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache
import json
from math import log, sqrt
from pathlib import Path
import re
from typing import Any
import unicodedata

import streamlit as st

from frontend.utils.dataset_catalog import normalize_section_label
from frontend.utils.paths import DATA_DIR, DATASET_PATH

MAX_MEMORY_ITEMS = 32
OUTCOME_MEMORY_PATH = DATA_DIR / "generation_outcome_memory.jsonl"


@dataclass(frozen=True)
class DatasetExerciseCase:
    case_id: str
    section: str
    topic: str
    subtopic: str
    year: str
    instruction: str
    solution: str
    final_answer: str
    modality: str


def retrieve_dataset_cases(
    *,
    section: str,
    topic: str,
    subtopic: str,
    profile: dict[str, Any] | None = None,
    difficulty: str = "",
    exercise_type: str = "",
    top_k: int = 3,
) -> list[DatasetExerciseCase]:
    """Hybrid semantic+metadata retrieval over dataset cases."""
    return [case for _score, case in retrieve_dataset_case_matches(
        section=section,
        topic=topic,
        subtopic=subtopic,
        profile=profile,
        difficulty=difficulty,
        exercise_type=exercise_type,
        top_k=top_k,
    )]


def retrieve_dataset_case_matches(
    *,
    section: str,
    topic: str,
    subtopic: str,
    profile: dict[str, Any] | None = None,
    difficulty: str = "",
    exercise_type: str = "",
    top_k: int = 5,
) -> list[tuple[float, DatasetExerciseCase]]:
    """Return scored dataset cases with exact-subtopic/topic/section fallbacks."""
    normalized_section = normalize_section_label(section)
    normalized_topic = _normalize_text(topic)
    normalized_subtopic = _normalize_text(subtopic)
    query_text = " ".join(
        [
            section,
            topic,
            subtopic,
            str((profile or {}).get("current_focus", "")),
            " ".join((profile or {}).get("weak_topics", []) or []),
        ]
    )
    positive_scores = _case_success_priors()

    all_cases = load_dataset_exercise_cases()
    same_topic_cases = [
        case for case in all_cases
        if _same_topic(case.topic, topic) or _normalize_text(case.topic) == normalized_topic
    ]
    pools = [
        [
            case for case in all_cases
            if case.section == normalized_section
            and _same_topic(case.topic, topic)
            and _normalize_text(case.subtopic) == normalized_subtopic
        ],
        [
            case for case in all_cases
            if case.section == normalized_section
            and _same_topic(case.topic, topic)
        ],
        same_topic_cases,
        [case for case in all_cases if case.section == normalized_section] if not same_topic_cases else [],
        [
            case for case in all_cases
            if _same_topic(case.topic, topic)
            or _normalize_text(case.subtopic) == normalized_subtopic
        ],
    ]

    selected_pool: list[DatasetExerciseCase] = []
    for pool in pools:
        if pool:
            selected_pool = pool
            break
    selected_pool = _filter_topic_pollution(selected_pool, topic, subtopic)

    scored: list[tuple[float, DatasetExerciseCase]] = []
    for case in selected_pool:
        semantic_score = _semantic_similarity(query_text, f"{case.topic} {case.subtopic} {case.instruction} {case.solution}")
        metadata_score = 0.0
        if case.section == normalized_section:
            metadata_score += 0.20
        if _normalize_text(case.topic) == normalized_topic:
            metadata_score += 0.35
        metadata_score += SequenceMatcher(None, _normalize_text(case.subtopic), normalized_subtopic).ratio() * 0.45

        difficulty_score = 0.0
        modality_text = _normalize_text(case.modality)
        if exercise_type and _normalize_text(exercise_type) in modality_text:
            difficulty_score += 0.6
        if difficulty and _normalize_text(difficulty) in modality_text:
            difficulty_score += 0.4

        freshness_score = 0.2 if str(case.year).strip().endswith("principale") else 0.0
        freshness_score += min(positive_scores.get(case.case_id, 0.0), 1.0) * 0.8

        total = (
            semantic_score * 0.60
            + metadata_score * 0.25
            + difficulty_score * 0.10
            + freshness_score * 0.05
        )
        scored.append((max(total, 0.001), case))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def _same_topic(candidate: str, target: str) -> bool:
    candidate_norm = _normalize_text(candidate)
    target_norm = _normalize_text(target)
    return bool(candidate_norm and target_norm and (candidate_norm == target_norm or target_norm in candidate_norm or candidate_norm in target_norm))


def _filter_topic_pollution(cases: list[DatasetExerciseCase], topic: str, subtopic: str) -> list[DatasetExerciseCase]:
    normalized_topic = _normalize_text(topic)
    normalized_subtopic = _normalize_text(subtopic)
    if any(token in normalized_subtopic for token in ("bayes", "conditionnement", "probabilites", "probabilite")):
        allowed = ("probabil", "condition", "bayes", "arbre", "evenement", "p(", "sachant")
        banned = ("sphere", "plan", "repere", "droite dans l espace", "cercle", "vecteur")
        filtered = []
        for case in cases:
            text = _normalize_text(f"{case.topic} {case.subtopic} {case.instruction}")
            if any(token in text for token in allowed) and not any(token in text for token in banned):
                filtered.append(case)
        return filtered or cases[:0]
    if any(_same_topic(case.topic, topic) for case in cases):
        return [case for case in cases if _same_topic(case.topic, topic)]
    return cases


def retrieve_generation_memories(
    *,
    section: str,
    topic: str,
    subtopic: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Return recent positive generation memories close to the current request."""
    bank = _get_generation_memory_bank()
    normalized_section = normalize_section_label(section)
    normalized_topic = _normalize_text(topic)
    normalized_subtopic = _normalize_text(subtopic)

    scored: list[tuple[float, dict[str, Any]]] = []
    for item in bank:
        metadata_score = 0.0
        if item.get("section") == normalized_section:
            metadata_score += 0.5
        if _normalize_text(item.get("topic", "")) == normalized_topic:
            metadata_score += 0.2
        metadata_score += SequenceMatcher(None, _normalize_text(item.get("subtopic", "")), normalized_subtopic).ratio() * 0.3
        success_score = float(item.get("reward_proxy", 0.0))
        total = metadata_score * 0.65 + success_score * 0.35
        if total > 0:
            scored.append((total, item))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def retain_generation_memory(
    *,
    section: str,
    topic: str,
    subtopic: str,
    difficulty: str,
    exercise_type: str,
    generated_exercise: dict[str, Any],
    profile: dict[str, Any] | None = None,
    retrieved_cases: list[DatasetExerciseCase] | None = None,
) -> None:
    """Keep a compact positive in-session memory trace for prompt adaptation."""
    bank = _get_generation_memory_bank()
    reward_proxy = _estimate_reward_proxy(subtopic=subtopic, profile=profile)
    memory_entry = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "section": normalize_section_label(section),
        "topic": topic,
        "subtopic": subtopic,
        "difficulty": difficulty,
        "exercise_type": exercise_type,
        "title": generated_exercise.get("title", ""),
        "prompt_excerpt": _truncate(generated_exercise.get("prompt", ""), 320),
        "context_excerpt": _truncate(generated_exercise.get("context", ""), 240),
        "questions_excerpt": [str(item).strip() for item in (generated_exercise.get("questions") or [])[:3]],
        "learning_objective": generated_exercise.get("learning_objective", ""),
        "reward_proxy": reward_proxy,
        "source_case_ids": [case.case_id for case in (retrieved_cases or [])],
    }
    deduped = [item for item in bank if item.get("prompt_excerpt") != memory_entry["prompt_excerpt"]]
    st.session_state.generation_memory_bank = [memory_entry, *deduped][:MAX_MEMORY_ITEMS]


def register_generation_outcome(
    *,
    prompt_signature: str,
    section: str,
    topic: str,
    subtopic: str,
    retrieved_case_ids: list[str],
    generation_backend: str,
    is_true_llm_generation: bool,
    validation_result: str,
    judge_issues: list[str],
    local_issues: list[str],
    final_display_decision: str,
    student_facing_format_issues: list[str],
    failure_categories: list[str],
) -> None:
    """Persist one outcome memory entry for future prompt adaptation."""
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prompt_signature": prompt_signature,
        "section": normalize_section_label(section),
        "topic": topic,
        "subtopic": subtopic,
        "retrieved_case_ids": list(retrieved_case_ids),
        "generation_backend": generation_backend,
        "is_true_llm_generation": bool(is_true_llm_generation),
        "validation_result": validation_result,
        "judge_issues": list(judge_issues),
        "local_issues": list(local_issues),
        "final_display_decision": final_display_decision,
        "student_facing_format_issues": list(student_facing_format_issues),
        "failure_categories": list(failure_categories),
    }
    bank = st.session_state.get("generation_outcome_memory", [])
    st.session_state.generation_outcome_memory = [entry, *bank][:120]
    try:
        OUTCOME_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUTCOME_MEMORY_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def get_positive_memory_patterns(section: str, topic: str, subtopic: str, top_k: int = 3) -> list[dict[str, Any]]:
    patterns = [
        entry
        for entry in _load_generation_outcomes()
        if entry.get("is_true_llm_generation")
        and entry.get("final_display_decision") == "presented"
        and normalize_section_label(entry.get("section", "")) == normalize_section_label(section)
        and _normalize_text(entry.get("topic", "")) == _normalize_text(topic)
        and _normalize_text(entry.get("subtopic", "")) == _normalize_text(subtopic)
    ]
    return patterns[:top_k]


def get_negative_memory_patterns(section: str, topic: str, subtopic: str, top_k: int = 5) -> list[dict[str, Any]]:
    bucket: defaultdict[str, int] = defaultdict(int)
    for entry in _load_generation_outcomes():
        if normalize_section_label(entry.get("section", "")) != normalize_section_label(section):
            continue
        if _normalize_text(entry.get("topic", "")) != _normalize_text(topic):
            continue
        if _normalize_text(entry.get("subtopic", "")) != _normalize_text(subtopic):
            continue
        for category in entry.get("failure_categories", []) or []:
            bucket[str(category)] += 1
    ranked = sorted(bucket.items(), key=lambda item: item[1], reverse=True)
    return [{"pattern": category, "count": count} for category, count in ranked[:top_k]]


def build_memory_adapted_generation_prompt(
    section: str,
    topic: str,
    subtopic: str,
    difficulty: str,
    exercise_type: str,
    retrieved_cases: list[DatasetExerciseCase],
    positive_patterns: list[dict[str, Any]],
    negative_patterns: list[dict[str, Any]],
    previous_errors: list[str],
) -> list[dict[str, str]]:
    """Assemble one memory-aware user prompt block for exercise generation."""
    case_block = "\n".join(
        [
            f"- Cas {index}: theme={case.topic} | sous-theme={case.subtopic} | enonce={_truncate(case.instruction, 700)} | solution={_truncate(case.solution, 700)} | reponse={_truncate(case.final_answer, 160)}"
            for index, case in enumerate(retrieved_cases[:2], start=1)
        ]
    ) or "- Aucun cas dataset pertinent."
    positive_block = "\n".join(
        [
            f"- Pattern positif {index}: backend={entry.get('generation_backend')} | prompt={entry.get('prompt_signature', '')[:80]}"
            for index, entry in enumerate(positive_patterns[:3], start=1)
        ]
    ) or "- Aucun pattern positif memorise."
    negative_block = "\n".join(
        [
            f"- A eviter: {entry.get('pattern')} (vu {entry.get('count')} fois)"
            for entry in negative_patterns[:5]
        ]
    ) or "- Aucun pattern negatif memorise."
    error_block = "\n".join(f"- {error}" for error in previous_errors if str(error).strip()) or "- Aucun rejet precedent."

    content = (
        "Mémoire d'adaptation\n"
        f"- Section : {section}\n"
        f"- Thème : {topic}\n"
        f"- Sous-thème : {subtopic}\n"
        f"- Difficulté : {difficulty}\n"
        f"- Type : {exercise_type}\n\n"
        "Cas proches à imiter seulement pour la structure pédagogique\n"
        f"{case_block}\n\n"
        "Patterns positifs issus de vraies générations LLM validées\n"
        f"{positive_block}\n\n"
        "Patterns négatifs à éviter\n"
        f"{negative_block}\n\n"
        "Erreurs précédentes à corriger maintenant\n"
        f"{error_block}\n\n"
        "Ne recopie pas un cas source. Réutilise seulement la structure pédagogique utile."
    )
    return [{"role": "user", "content": content}]


def find_too_similar_source_case(
    generated_instruction: str,
    retrieved_cases: list[DatasetExerciseCase],
    threshold: float = 0.92,
) -> tuple[bool, str, float]:
    """Detect near-copy outputs against retrieved source cases."""
    candidate = _normalize_text(generated_instruction)
    best_case_id = ""
    best_score = 0.0
    for case in retrieved_cases:
        score = SequenceMatcher(None, candidate, _normalize_text(case.instruction)).ratio()
        if score > best_score:
            best_score = score
            best_case_id = case.case_id
    return best_score > threshold, best_case_id, best_score


@lru_cache(maxsize=1)
def load_dataset_exercise_cases() -> list[DatasetExerciseCase]:
    if not DATASET_PATH.exists():
        return []
    try:
        rows = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    cases: list[DatasetExerciseCase] = []
    for index, row in enumerate(rows, start=1):
        raw_section = str(row.get("section", "")).strip()
        raw_topic = str(row.get("topic", "")).strip()
        instruction = str(row.get("instruction", "")).strip()
        if not raw_section or not raw_topic or not instruction:
            continue
        topic_name, subtopic_name = _split_topic(raw_topic)
        cases.append(
            DatasetExerciseCase(
                case_id=f"dataset-{index:04d}",
                section=normalize_section_label(raw_section),
                topic=topic_name,
                subtopic=subtopic_name,
                year=str(row.get("year", "")).strip(),
                instruction=instruction,
                solution=str(row.get("solution", "")).strip(),
                final_answer=str(row.get("final_answer", "")).strip(),
                modality=str(row.get("modality", "") or "").strip(),
            )
        )
    return cases


def _get_generation_memory_bank() -> list[dict[str, Any]]:
    if "generation_memory_bank" not in st.session_state:
        st.session_state.generation_memory_bank = []
    return st.session_state.generation_memory_bank


def _load_generation_outcomes() -> list[dict[str, Any]]:
    session_bank = st.session_state.get("generation_outcome_memory", [])
    if session_bank:
        return session_bank
    entries: list[dict[str, Any]] = []
    if OUTCOME_MEMORY_PATH.exists():
        try:
            for line in OUTCOME_MEMORY_PATH.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entries.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            return []
    st.session_state.generation_outcome_memory = entries[:120]
    return st.session_state.generation_outcome_memory


def _case_success_priors() -> dict[str, float]:
    scores: defaultdict[str, float] = defaultdict(float)
    for entry in _load_generation_outcomes():
        if not entry.get("is_true_llm_generation") or entry.get("final_display_decision") != "presented":
            continue
        for case_id in entry.get("retrieved_case_ids", []) or []:
            scores[str(case_id)] += 0.2
    return dict(scores)


def _semantic_similarity(query: str, document: str) -> float:
    query_vector = _tfidf_vector(_normalize_text(query))
    document_vector = _tfidf_vector(_normalize_text(document))
    return _cosine_similarity(query_vector, document_vector)


@lru_cache(maxsize=4096)
def _tfidf_vector(text: str) -> dict[str, float]:
    tokens = _tokenize(text)
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = sum(counts.values())
    idf = _idf_scores()
    return {
        token: (count / total) * idf.get(token, 1.0)
        for token, count in counts.items()
    }


@lru_cache(maxsize=1)
def _idf_scores() -> dict[str, float]:
    documents = [
        _tokenize(f"{case.topic} {case.subtopic} {case.instruction} {case.solution}")
        for case in load_dataset_exercise_cases()
    ]
    doc_count = max(len(documents), 1)
    frequencies: defaultdict[str, int] = defaultdict(int)
    for document in documents:
        for token in set(document):
            frequencies[token] += 1
    return {
        token: log((1 + doc_count) / (1 + count)) + 1.0
        for token, count in frequencies.items()
    }


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    common = set(left).intersection(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = sqrt(sum(value * value for value in left.values()))
    right_norm = sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", _normalize_text(text))


def _split_topic(raw_topic: str) -> tuple[str, str]:
    cleaned = " ".join(raw_topic.split())
    for separator_pattern in (r"\s+â€”\s+", r"\s+â€“\s+", r"\s+-\s+"):
        parts = re.split(separator_pattern, cleaned, maxsplit=1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return parts[0].strip(), parts[1].strip()
    return cleaned, cleaned


def _estimate_reward_proxy(*, subtopic: str, profile: dict[str, Any] | None = None) -> float:
    normalized_subtopic = _normalize_text(subtopic)
    weak_topics = {_normalize_text(item) for item in (profile or {}).get("weak_topics", [])}
    current_focus = _normalize_text((profile or {}).get("current_focus", ""))
    if normalized_subtopic in weak_topics or normalized_subtopic == current_focus:
        return 1.0
    return 0.6


def _truncate(value: str, max_length: int) -> str:
    compact = " ".join(str(value or "").split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3].rstrip() + "..."


def _normalize_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_value).strip().lower()
