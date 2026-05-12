"""Dataset-backed catalog helpers for Bac sections and topics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from typing import Iterable
import unicodedata

from frontend.utils.paths import DATASET_PATH

TOPIC_SEPARATOR = "\u2014"

SECTION_LABELS = {
    "mathematiques": "Math\u00e9matiques",
    "sciences-experimentales": "Sciences exp\u00e9rimentales",
    "sciences-techniques": "Sciences techniques",
    "sciences-informatiques": "Sciences informatiques",
    "economie-et-gestion": "\u00c9conomie et gestion",
    "lettres": "Lettres",
    "sport": "Sport",
}

SECTION_ORDER = [
    "Math\u00e9matiques",
    "Sciences exp\u00e9rimentales",
    "Sciences techniques",
    "Sciences informatiques",
    "\u00c9conomie et gestion",
    "Lettres",
    "Sport",
]


@dataclass(frozen=True)
class DatasetCatalog:
    """Precomputed catalog of sections, topics, and subtopics."""

    sections: list[str]
    topics_by_section: dict[str, list[str]]
    subtopics_by_section_topic: dict[str, dict[str, list[str]]]


def get_sections() -> list[str]:
    """Return the available Bac sections."""
    return list(get_dataset_catalog().sections)


def get_topics_for_section(section: str) -> list[str]:
    """Return available topic families for a section."""
    normalized_section = normalize_section_label(section)
    return list(get_dataset_catalog().topics_by_section.get(normalized_section, []))


def get_subtopics_for_section_topic(section: str, topic: str) -> list[str]:
    """Return available subtopics for a section/topic pair."""
    normalized_section = normalize_section_label(section)
    topics = get_dataset_catalog().subtopics_by_section_topic.get(normalized_section, {})
    return list(topics.get(topic, []))


def get_default_section(preferred: str | None = None) -> str:
    """Return the best available default section."""
    sections = get_sections()
    if preferred:
        normalized = normalize_section_label(preferred)
        if normalized in sections:
            return normalized
    return sections[0] if sections else "Math\u00e9matiques"


def normalize_section_label(value: str) -> str:
    """Normalize a stored or raw section value to the display label."""
    cleaned = (value or "").strip()
    if not cleaned:
        return get_default_section()

    alias_map = _build_section_alias_map()
    return alias_map.get(_simplify(cleaned), cleaned)


@lru_cache(maxsize=1)
def get_dataset_catalog() -> DatasetCatalog:
    """Load the dataset and expose a reusable catalog."""
    section_topics: dict[str, dict[str, set[str]]] = {}

    if DATASET_PATH.exists():
        try:
            items = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            items = []
    else:
        items = []

    for item in items:
        raw_section = str(item.get("section", "")).strip()
        raw_topic = str(item.get("topic", "")).strip()
        if not raw_section or not raw_topic:
            continue

        section = _format_section(raw_section)
        topic, subtopic = _split_topic(raw_topic)
        section_topics.setdefault(section, {}).setdefault(topic, set()).add(subtopic)

    if not section_topics:
        section_topics = {
            "Math\u00e9matiques": {
                "Analyse": {"fonction exponentielle", "int\u00e9grales, primitives, aires et volumes"},
                "Probabilit\u00e9s": {"loi binomiale et sch\u00e9ma de Bernoulli"},
            }
        }

    ordered_sections = _order_labels(section_topics.keys())
    topics_by_section = {
        section: sorted(topic_map.keys(), key=_simplify)
        for section, topic_map in section_topics.items()
    }
    subtopics_by_section_topic = {
        section: {
            topic: sorted(subtopics, key=_simplify)
            for topic, subtopics in topic_map.items()
        }
        for section, topic_map in section_topics.items()
    }

    return DatasetCatalog(
        sections=ordered_sections,
        topics_by_section=topics_by_section,
        subtopics_by_section_topic=subtopics_by_section_topic,
    )


def _format_section(raw_section: str) -> str:
    """Convert a raw dataset section slug to a display label."""
    if raw_section in SECTION_LABELS:
        return SECTION_LABELS[raw_section]

    cleaned = raw_section.replace("-", " ").strip()
    return cleaned[:1].upper() + cleaned[1:] if cleaned else "Math\u00e9matiques"


def _split_topic(raw_topic: str) -> tuple[str, str]:
    """Split a raw dataset topic into a family and a detailed label."""
    cleaned = " ".join(raw_topic.split())
    if TOPIC_SEPARATOR in cleaned:
        topic, subtopic = [part.strip() for part in cleaned.split(TOPIC_SEPARATOR, 1)]
        if topic and subtopic:
            return topic, subtopic
    return cleaned, cleaned


@lru_cache(maxsize=1)
def _build_section_alias_map() -> dict[str, str]:
    """Build lookup aliases for both slugs and display labels."""
    aliases: dict[str, str] = {}
    catalog = get_dataset_catalog()

    for section in catalog.sections:
        aliases[_simplify(section)] = section

    for raw_value, display_label in SECTION_LABELS.items():
        aliases[_simplify(raw_value)] = display_label
        aliases[_simplify(display_label)] = display_label

    return aliases


def _order_labels(labels: Iterable[str]) -> list[str]:
    """Order known sections first, then any unexpected labels."""
    order_index = {label: index for index, label in enumerate(SECTION_ORDER)}
    return sorted(
        labels,
        key=lambda label: (order_index.get(label, len(SECTION_ORDER)), _simplify(label)),
    )


def _simplify(value: str) -> str:
    """Normalize accents and separators for deterministic comparisons."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_value.lower().replace("-", " ").split())
