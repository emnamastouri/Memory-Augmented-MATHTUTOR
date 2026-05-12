"""Memory utilities for Memento-inspired exercise generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache
import json
import re
from typing import Any
import unicodedata

import streamlit as st

from frontend.utils.dataset_catalog import normalize_section_label
from frontend.utils.paths import DATASET_PATH

MAX_MEMORY_ITEMS = 24


@dataclass(frozen=True)
class DatasetExerciseCase:
    """Structured dataset case used as an external episodic memory entry."""

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
    top_k: int = 3,
) -> list[DatasetExerciseCase]:
    """Retrieve the most relevant dataset cases for the current generation request."""
    normalized_section = normalize_section_label(section)
    normalized_topic = _normalize_text(topic)
    normalized_subtopic = _normalize_text(subtopic)
    weak_topics = {_normalize_text(item) for item in (profile or {}).get("weak_topics", [])}
    current_focus = _normalize_text((profile or {}).get("current_focus", ""))

    scored_cases: list[tuple[float, DatasetExerciseCase]] = []
    for case in load_dataset_exercise_cases():
        if case.section != normalized_section:
            continue

        score = 0.0
        score += 6.0 if _normalize_text(case.topic) == normalized_topic else 0.0
        score += SequenceMatcher(None, _normalize_text(case.subtopic), normalized_subtopic).ratio() * 8.0
        if _normalize_text(case.subtopic) == normalized_subtopic:
            score += 8.0
        if _normalize_text(case.topic) in weak_topics or _normalize_text(case.subtopic) in weak_topics:
            score += 1.8
        if current_focus and (current_focus in _normalize_text(case.instruction) or current_focus == _normalize_text(case.subtopic)):
            score += 1.2
        if case.year.endswith("principale"):
            score += 0.3
        scored_cases.append((score, case))

    scored_cases.sort(key=lambda item: item[0], reverse=True)
    return [case for score, case in scored_cases[:top_k] if score > 0]


def retrieve_generation_memories(
    *,
    section: str,
    topic: str,
    subtopic: str,
    top_k: int = 2,
) -> list[dict[str, Any]]:
    """Retrieve recent generated exercises that are most similar to the current request."""
    bank = _get_generation_memory_bank()
    normalized_section = normalize_section_label(section)
    normalized_topic = _normalize_text(topic)
    normalized_subtopic = _normalize_text(subtopic)

    scored_memories: list[tuple[float, dict[str, Any]]] = []
    for item in bank:
        score = 0.0
        score += 5.0 if item.get("section") == normalized_section else 0.0
        score += 4.0 if _normalize_text(item.get("topic", "")) == normalized_topic else 0.0
        score += SequenceMatcher(None, _normalize_text(item.get("subtopic", "")), normalized_subtopic).ratio() * 6.0
        score += float(item.get("reward_proxy", 0.0))
        if score > 0:
            scored_memories.append((score, item))

    scored_memories.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in scored_memories[:top_k]]


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
    """Write a lightweight memory trace for future case-based retrieval."""
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
        "hint": generated_exercise.get("hint", ""),
        "learning_objective": generated_exercise.get("learning_objective", ""),
        "reward_proxy": reward_proxy,
        "source_case_ids": [case.case_id for case in (retrieved_cases or [])],
    }
    deduped = [item for item in bank if item.get("prompt_excerpt") != memory_entry["prompt_excerpt"]]
    st.session_state.generation_memory_bank = [memory_entry, *deduped][:MAX_MEMORY_ITEMS]


@lru_cache(maxsize=1)
def load_dataset_exercise_cases() -> list[DatasetExerciseCase]:
    """Load the exercise dataset into a structured memory bank."""
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

        topic, subtopic = _split_topic(raw_topic)
        cases.append(
            DatasetExerciseCase(
                case_id=f"dataset-{index:04d}",
                section=normalize_section_label(raw_section),
                topic=topic,
                subtopic=subtopic,
                year=str(row.get("year", "")).strip(),
                instruction=instruction,
                solution=str(row.get("solution", "")).strip(),
                final_answer=str(row.get("final_answer", "")).strip(),
                modality=str(row.get("modality", "") or "").strip(),
            )
        )
    return cases


def _get_generation_memory_bank() -> list[dict[str, Any]]:
    """Ensure the session memory bank exists before reading or writing."""
    if "generation_memory_bank" not in st.session_state:
        st.session_state.generation_memory_bank = []
    return st.session_state.generation_memory_bank


def _split_topic(raw_topic: str) -> tuple[str, str]:
    """Split a dataset topic into a family and a subtopic label."""
    cleaned = " ".join(raw_topic.split())
    for separator_pattern in (r"\s+—\s+", r"\s+â€”\s+", r"\s+–\s+"):
        parts = re.split(separator_pattern, cleaned, maxsplit=1)
        if len(parts) == 2:
            topic, subtopic = [part.strip() for part in parts]
            if topic and subtopic:
                return topic, subtopic
    return cleaned, cleaned


def _estimate_reward_proxy(*, subtopic: str, profile: dict[str, Any] | None = None) -> float:
    """Proxy feedback before automatic verification is available."""
    normalized_subtopic = _normalize_text(subtopic)
    weak_topics = {_normalize_text(item) for item in (profile or {}).get("weak_topics", [])}
    current_focus = _normalize_text((profile or {}).get("current_focus", ""))
    if normalized_subtopic in weak_topics or normalized_subtopic == current_focus:
        return 1.0
    return 0.55


def _truncate(value: str, max_length: int) -> str:
    """Shorten long memory fields to keep prompt assembly compact."""
    compact = " ".join(value.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3].rstrip() + "..."


def _normalize_text(value: str) -> str:
    """Normalize accents and spacing for matching and scoring."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_value.lower().split())
