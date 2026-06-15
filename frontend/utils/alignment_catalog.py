"""Official-program alignment metadata helpers for section/topic couples."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import re
from typing import Any
import unicodedata

from frontend.utils.dataset_catalog import get_sections as get_dataset_sections, normalize_section_label
from frontend.utils.paths import ALIGNMENT_METADATA_PATH


@dataclass(frozen=True)
class AlignmentRecord:
    """One official alignment reference for a section and detailed topic couple."""

    section_slug: str
    section_label: str
    topic_label: str
    topic_family: str
    subtopic_label: str
    official_program_scope: str
    topic_focus: str
    warnings: list[str]


def get_alignment_record(section: str, topic: str, subtopic: str) -> AlignmentRecord | None:
    """Return the exact alignment record for one section/topic/subtopic couple."""
    target_section = _simplify(normalize_section_label(section))
    target_topic = _simplify(topic)
    target_subtopic = _simplify(subtopic)

    for record in load_alignment_records():
        if _simplify(record.section_label) != target_section:
            continue
        if _simplify(record.topic_family) != target_topic:
            continue
        if _simplify(record.subtopic_label) != target_subtopic:
            continue
        return record
    return None


def get_supported_sections() -> list[str]:
    """Return sections covered by the official alignment reference."""
    mapping = _build_alignment_catalog()
    dataset_order = {label: index for index, label in enumerate(get_dataset_sections())}
    return sorted(
        mapping.keys(),
        key=lambda label: (dataset_order.get(label, len(dataset_order)), _simplify(label)),
    )


def get_supported_topics_for_section(section: str) -> list[str]:
    """Return supported topic families for one aligned section."""
    normalized_section = normalize_section_label(section)
    mapping = _build_alignment_catalog()
    return list(mapping.get(normalized_section, {}).keys())


def get_supported_subtopics_for_section_topic(section: str, topic: str) -> list[str]:
    """Return supported subtopics for one aligned section/topic pair."""
    normalized_section = normalize_section_label(section)
    section_topics = _build_alignment_catalog().get(normalized_section, {})
    return list(section_topics.get(topic, []))


@lru_cache(maxsize=1)
def load_alignment_records() -> list[AlignmentRecord]:
    """Load the official-program alignment metadata from the local project data folder."""
    if not ALIGNMENT_METADATA_PATH.exists():
        return []

    try:
        rows = json.loads(ALIGNMENT_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    records: list[AlignmentRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        raw_section = str(row.get("section", "")).strip()
        raw_topic = str(row.get("topic", "")).strip()
        if not raw_section or not raw_topic:
            continue

        topic_family, subtopic_label = _split_alignment_topic(raw_topic)
        warnings = row.get("warnings") or []
        if not isinstance(warnings, list):
            warnings = [str(warnings)]

        records.append(
            AlignmentRecord(
                section_slug=raw_section,
                section_label=normalize_section_label(raw_section),
                topic_label=raw_topic,
                topic_family=topic_family,
                subtopic_label=subtopic_label,
                official_program_scope=str(row.get("official_program_scope", "")).strip(),
                topic_focus=str(row.get("topic_focus", "")).strip(),
                warnings=[str(item).strip() for item in warnings if str(item).strip()],
            )
        )

    return records


@lru_cache(maxsize=1)
def _build_alignment_catalog() -> dict[str, dict[str, list[str]]]:
    """Index official records by section and topic for UI filtering."""
    catalog: dict[str, dict[str, set[str]]] = {}
    for record in load_alignment_records():
        catalog.setdefault(record.section_label, {}).setdefault(record.topic_family, set()).add(record.subtopic_label)

    return {
        section: {
            topic: sorted(subtopics, key=_simplify)
            for topic, subtopics in sorted(topic_map.items(), key=lambda item: _simplify(item[0]))
        }
        for section, topic_map in catalog.items()
    }


def _split_alignment_topic(raw_topic: str) -> tuple[str, str]:
    """Split one detailed topic label into theme family and subtopic."""
    cleaned = " ".join(raw_topic.split())
    for separator_pattern in (r"\s+—\s+", r"\s+–\s+"):
        parts = re.split(separator_pattern, cleaned, maxsplit=1)
        if len(parts) == 2:
            theme, subtopic = [part.strip() for part in parts]
            if theme and subtopic:
                return theme, subtopic
    return cleaned, cleaned


def _simplify(value: str) -> str:
    """Normalize accents, punctuation, and spacing for robust matching."""
    normalized = unicodedata.normalize("NFKD", value.replace("—", "-").replace("–", "-"))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    compact = re.sub(r"[^a-zA-Z0-9]+", " ", ascii_value)
    return " ".join(compact.lower().split())
