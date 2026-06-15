"""Compatibility re-exports for the v7 memory adaptation layer."""

from frontend.utils.memory_adaptation import (
    DatasetExerciseCase,
    build_memory_adapted_generation_prompt,
    find_too_similar_source_case,
    get_negative_memory_patterns,
    get_positive_memory_patterns,
    load_dataset_exercise_cases,
    register_generation_outcome,
    retain_generation_memory,
    retrieve_dataset_case_matches,
    retrieve_dataset_cases,
    retrieve_generation_memories,
)

__all__ = [
    "DatasetExerciseCase",
    "build_memory_adapted_generation_prompt",
    "find_too_similar_source_case",
    "get_negative_memory_patterns",
    "get_positive_memory_patterns",
    "load_dataset_exercise_cases",
    "register_generation_outcome",
    "retain_generation_memory",
    "retrieve_dataset_case_matches",
    "retrieve_dataset_cases",
    "retrieve_generation_memories",
]
