"""Shared project paths used across the frontend."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "data-set.json"
ALIGNMENT_METADATA_PATH = DATA_DIR / "alignment_metadata_existing_couples_clean.json"
GENERATED_EXERCISES_LOG_PATH = DATA_DIR / "generated_exercises_audit.jsonl"
