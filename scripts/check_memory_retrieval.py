"""Debug dataset memory retrieval used before OpenRouter generation.

Example:
    python scripts/check_memory_retrieval.py --section "Sciences expérimentales" --topic "Statistiques" --subtopic "séries à deux caractères, régression et corrélation"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.utils.memory_adaptation import load_dataset_exercise_cases, retrieve_dataset_case_matches  # noqa: E402


def _short(value: str, limit: int = 300) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect MathTutorAI memory retrieval.")
    parser.add_argument("--section", default="")
    parser.add_argument("--topic", default="")
    parser.add_argument("--subtopic", default="")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cases = load_dataset_exercise_cases()
    print(f"dataset_items_loaded: {len(cases)}")
    matches = retrieve_dataset_case_matches(
        section=args.section,
        topic=args.topic,
        subtopic=args.subtopic,
        top_k=args.top_k,
    )
    if not matches:
        print("No cases found. Check dataset normalization or the provided labels.")
        return 0

    for rank, (score, case) in enumerate(matches[: args.top_k], start=1):
        print(f"\n#{rank}")
        print(f"case_id: {case.case_id}")
        print(f"score: {score:.4f}")
        print(f"section: {case.section}")
        print(f"topic: {case.topic}")
        print(f"subtopic: {case.subtopic}")
        print(f"instruction: {_short(case.instruction)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
