"""OpenRouter health check for MathTutorAI generation infrastructure.

Run with:
    python scripts/check_openrouter_generation.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.utils.openrouter_client import (  # noqa: E402
    extract_openrouter_text,
    get_openrouter_client,
    get_openrouter_settings,
    parse_json_object_detailed,
)


def _preview(value: Any, limit: int = 1000) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _response_to_preview(response: Any) -> str:
    try:
        model_dump_json = getattr(response, "model_dump_json", None)
        if callable(model_dump_json):
            return _preview(model_dump_json())
    except Exception:
        pass
    try:
        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            return _preview(json.dumps(model_dump(), ensure_ascii=False))
    except Exception:
        pass
    return _preview(response)


def _simple_schema() -> dict[str, Any]:
    return {
        "name": "mathtutorai_health_check",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["ok", "message"],
            "properties": {
                "ok": {"type": "boolean"},
                "message": {"type": "string"},
            },
        },
    }


def _run_mode(client: Any, model: str, mode_name: str, response_format: dict[str, Any] | None) -> None:
    print(f"\n=== mode: {mode_name} ===")
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": 'Return only valid JSON: {"ok": true, "message": "hello"}',
            }
        ],
        "temperature": 0,
        "top_p": 0.1,
        "max_tokens": 120,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 - diagnostic script must not crash.
        print("success: false")
        print(f"exception_type: {type(exc).__name__}")
        print(f"http_status: {getattr(exc, 'status_code', None)}")
        print(f"raw_response_preview: {_preview(exc)}")
        return

    content = extract_openrouter_text(response)
    parse_result = parse_json_object_detailed(content)
    print("success: true")
    print("http_status: None")
    print(f"raw_response_preview: {_response_to_preview(response)}")
    print(f"content_preview: {_preview(content)}")
    print(f"json_parse_ok: {parse_result.ok}")
    print(f"json_extraction_method: {parse_result.extraction_method}")
    if parse_result.ok:
        print(f"parsed_json: {json.dumps(parse_result.data, ensure_ascii=False)}")
    else:
        print(f"parse_error: {parse_result.error}")


def main() -> int:
    settings = get_openrouter_settings()
    print("MathTutorAI OpenRouter health check")
    print(f"api_key_present: {bool(settings and settings.api_key)}")
    print(f"base_url: {settings.base_url if settings else ''}")
    print(f"exercise_model: {settings.exercise_model_primary if settings else ''}")
    print(f"judge_model: {settings.judge_model if settings else ''}")
    print(f"response_format_config: {settings.response_mode if settings else ''}")
    if settings is None:
        print("OpenRouter is not configured. Add .streamlit/secrets.toml or environment variables.")
        return 0

    try:
        client = get_openrouter_client()
    except Exception as exc:  # noqa: BLE001
        print(f"client_error: {type(exc).__name__}: {_preview(exc)}")
        return 0

    _run_mode(client, settings.exercise_model_primary, "json_schema", {"type": "json_schema", "json_schema": _simple_schema()})
    _run_mode(client, settings.exercise_model_primary, "json_object", {"type": "json_object"})
    _run_mode(client, settings.exercise_model_primary, "prompt_only_json", None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
