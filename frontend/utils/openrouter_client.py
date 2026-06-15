"""Helpers for loading OpenRouter settings and extracting strict JSON outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import json
import os
import re
import time
import tomllib
from typing import Any

import streamlit as st
from openai import OpenAI

from frontend.utils.exercise_schema import normalize_exercise_schema, validate_exercise_schema_v7

REPO_ROOT = Path(__file__).resolve().parents[2]
SECRETS_PATH = REPO_ROOT / ".streamlit" / "secrets.toml"


@dataclass(frozen=True)
class OpenRouterSettings:
    """Runtime configuration for OpenRouter-based exercise generation."""

    api_key: str
    base_url: str
    exercise_model: str
    exercise_model_primary: str
    exercise_model_fallback: str
    judge_model: str
    validator_model: str
    tutor_model: str
    site_url: str
    app_name: str
    allow_dataset_demo: bool
    response_mode: str


@dataclass(frozen=True)
class JsonParseResult:
    """Detailed JSON extraction result for audit and debugging."""

    ok: bool
    data: dict[str, Any] | None
    error: str | None
    raw_preview: str
    extraction_method: str


@dataclass(frozen=True)
class OpenRouterCallResult:
    """Structured result for one OpenRouter completion request."""

    ok: bool
    content: str = ""
    raw_response_preview: str = ""
    http_status: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    model: str = ""
    response_format_mode: str = ""
    request_id: str | None = None
    provider: str | None = None
    usage: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)


def has_openrouter_config() -> bool:
    """Indicate whether OpenRouter credentials are available."""
    return get_openrouter_settings() is not None


def is_dataset_demo_allowed() -> bool:
    """Return whether dataset demo mode may be shown to the user."""
    settings = get_openrouter_settings()
    if settings is None:
        return True
    return bool(settings.allow_dataset_demo)


@lru_cache(maxsize=1)
def get_openrouter_settings() -> OpenRouterSettings | None:
    """Load OpenRouter settings from env vars, Streamlit secrets, or the local TOML file."""
    raw_settings = _load_secret_section("openrouter")

    api_key = _setting_value(
        raw_settings,
        key="api_key",
        env_name="OPENROUTER_API_KEY",
        aliases=("openrouter_api_key", "OPENROUTER_API_KEY"),
        default="",
    )
    if not api_key:
        return None

    exercise_model = _setting_value(
        raw_settings,
        key="exercise_model_primary",
        env_name="OPENROUTER_EXERCISE_MODEL",
        aliases=("exercise_model", "model", "OPENROUTER_MODEL"),
        default="qwen/qwen-2.5-7b-instruct",
    )
    exercise_model_fallback = _setting_value(
        raw_settings,
        key="exercise_model_fallback",
        env_name="OPENROUTER_EXERCISE_MODEL_FALLBACK",
        aliases=("OPENROUTER_FALLBACK_MODEL",),
        default=exercise_model,
    )

    return OpenRouterSettings(
        api_key=api_key,
        base_url=_setting_value(
            raw_settings,
            key="base_url",
            env_name="OPENROUTER_BASE_URL",
            aliases=("OPENROUTER_BASE_URL",),
            default="https://openrouter.ai/api/v1",
        ),
        exercise_model=exercise_model,
        exercise_model_primary=exercise_model,
        exercise_model_fallback=exercise_model_fallback,
        judge_model=_setting_value(
            raw_settings,
            key="judge_model",
            env_name="OPENROUTER_JUDGE_MODEL",
            aliases=("OPENROUTER_JUDGE_MODEL",),
            default=exercise_model,
        ),
        validator_model=_setting_value(
            raw_settings,
            key="validator_model",
            env_name="OPENROUTER_VALIDATOR_MODEL",
            aliases=("OPENROUTER_VALIDATOR_MODEL",),
            default=exercise_model,
        ),
        tutor_model=_setting_value(
            raw_settings,
            key="tutor_model",
            env_name="OPENROUTER_TUTOR_MODEL",
            aliases=("OPENROUTER_TUTOR_MODEL",),
            default=exercise_model,
        ),
        site_url=_setting_value(
            raw_settings,
            key="site_url",
            env_name="OPENROUTER_SITE_URL",
            aliases=("OPENROUTER_SITE_URL",),
            default="http://localhost:8501",
        ),
        app_name=_setting_value(
            raw_settings,
            key="app_name",
            env_name="OPENROUTER_APP_NAME",
            aliases=("OPENROUTER_APP_NAME",),
            default="MathTutorAI",
        ),
        allow_dataset_demo=_setting_bool(
            raw_settings,
            key="allow_dataset_demo",
            env_name="OPENROUTER_ALLOW_DATASET_DEMO",
            aliases=("ALLOW_DATASET_DEMO",),
            default=False,
        ),
        response_mode=_setting_value(
            raw_settings,
            key="openrouter_response_mode",
            env_name="OPENROUTER_RESPONSE_MODE",
            aliases=("response_mode",),
            default="auto",
        ).lower(),
    )


@lru_cache(maxsize=1)
def get_openrouter_client() -> OpenAI:
    """Build a reusable OpenAI-compatible client for OpenRouter."""
    settings = get_openrouter_settings()
    if settings is None:
        raise RuntimeError("OpenRouter n'est pas configure. Ajoutez les secrets dans .streamlit/secrets.toml.")

    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        default_headers={
            "HTTP-Referer": settings.site_url,
            "X-Title": settings.app_name,
        },
    )


def extract_openrouter_text(response: Any) -> str:
    """Extract the first textual answer from an OpenRouter chat completion response."""
    content = _extract_text_from_choices(getattr(response, "choices", None))
    if content:
        return content

    raw_payload = _to_response_dict(response)
    if raw_payload:
        content = _extract_text_from_choices(raw_payload.get("choices"))
        if content:
            return content
    return ""


def summarize_openrouter_response_issue(response: Any) -> str:
    """Build a short diagnostic when OpenRouter returns no usable text."""
    raw_payload = _to_response_dict(response)
    if not raw_payload:
        return "Reponse OpenRouter vide ou non structuree."

    error_payload = raw_payload.get("error")
    if isinstance(error_payload, dict):
        message = str(error_payload.get("message", "")).strip()
        if message:
            return f"OpenRouter a renvoye une erreur : {message}"

    if raw_payload.get("choices") is None:
        return "OpenRouter n'a renvoye aucun champ 'choices'."
    if raw_payload.get("choices") == []:
        return "OpenRouter a renvoye une liste 'choices' vide."
    return "OpenRouter n'a renvoye aucun contenu textuel exploitable."


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """Extract the first valid JSON object from a noisy model output."""
    return parse_json_object_detailed(raw).data


def parse_json_object_detailed(raw: str) -> JsonParseResult:
    """Extract one JSON object and explain how extraction behaved."""
    content = str(raw or "").strip()
    if not content:
        return JsonParseResult(False, None, "empty string", "", "failed")

    raw_preview = _preview(content)
    direct = _json_loads_dict(content)
    if direct is not None:
        return JsonParseResult(True, direct, None, raw_preview, "direct")

    fenced_chunks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.IGNORECASE | re.DOTALL)
    for chunk in fenced_chunks:
        parsed = _json_loads_dict(chunk.strip())
        if parsed is not None:
            return JsonParseResult(True, parsed, None, raw_preview, "fenced")

    for candidate in _iter_balanced_json_candidates(content):
        parsed = _json_loads_dict(candidate)
        if parsed is not None:
            return JsonParseResult(True, parsed, None, raw_preview, "balanced_object")

    repair_candidates = [content, _strip_markdown_fence(content), *_iter_balanced_json_candidates(content)]
    seen: set[str] = set()
    for candidate in repair_candidates:
        repaired = _repair_json_candidate(candidate)
        if not repaired or repaired in seen:
            continue
        seen.add(repaired)
        parsed = _json_loads_dict(repaired)
        if parsed is not None:
            return JsonParseResult(True, parsed, None, raw_preview, "trailing_comma_repair")

    return JsonParseResult(False, None, "No valid JSON object could be parsed.", raw_preview, "failed")


def call_openrouter_chat(
    *,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    purpose: str = "exercise",
    response_mode: str | None = None,
    json_schema: dict[str, Any] | None = None,
) -> OpenRouterCallResult:
    """Call OpenRouter with response-format fallback and structured diagnostics."""
    settings = get_openrouter_settings()
    if settings is None:
        return OpenRouterCallResult(
            ok=False,
            model=model,
            response_format_mode="not_configured",
            error_type="auth_error",
            error_message="OpenRouter is not configured.",
        )

    selected_mode = (response_mode or settings.response_mode or "auto").lower()
    modes = _response_format_modes(selected_mode)
    models = [model]
    if purpose == "exercise" and settings.exercise_model_fallback and settings.exercise_model_fallback not in models:
        models.append(settings.exercise_model_fallback)

    attempts: list[dict[str, Any]] = []
    last_result: OpenRouterCallResult | None = None
    for model_name in models:
        for mode in modes:
            transient_attempts = 3 if mode == modes[0] else 1
            for retry_index in range(transient_attempts):
                result = _call_openrouter_once(
                    messages=messages,
                    model=model_name,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    response_format_mode=mode,
                    purpose=purpose,
                    json_schema=json_schema,
                )
                attempts.append(_call_attempt_summary(result, retry_index + 1))
                if result.ok:
                    return OpenRouterCallResult(
                        ok=True,
                        content=result.content,
                        raw_response_preview=result.raw_response_preview,
                        http_status=result.http_status,
                        error_type=None,
                        error_message=None,
                        model=model_name,
                        response_format_mode=mode,
                        request_id=result.request_id,
                        provider=result.provider,
                        usage=result.usage,
                        attempts=attempts,
                    )
                last_result = result
                if result.error_type in {"connection_error", "timeout"} and retry_index < transient_attempts - 1:
                    time.sleep(0.5 * (2**retry_index))
                    continue
                break
            if last_result and last_result.error_type == "unsupported_response_format":
                continue
            if last_result and selected_mode != "auto":
                break

    if last_result is None:
        last_result = OpenRouterCallResult(ok=False, model=model, response_format_mode=selected_mode, error_type="unknown_error")
    return OpenRouterCallResult(
        ok=False,
        content=last_result.content,
        raw_response_preview=last_result.raw_response_preview,
        http_status=last_result.http_status,
        error_type=last_result.error_type,
        error_message=last_result.error_message,
        model=last_result.model or model,
        response_format_mode=last_result.response_format_mode or selected_mode,
        request_id=last_result.request_id,
        provider=last_result.provider,
        usage=last_result.usage,
        attempts=attempts,
    )


def get_exercise_json_schema() -> dict[str, Any]:
    """Return the strict exercise schema used for OpenRouter structured outputs."""
    return {
        "name": "mathtutorai_exercise",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "title",
                "context",
                "questions",
                "instruction",
                "solution",
                "expected_answer",
                "answer_kind",
                "solution_steps",
                "learning_objective",
                "estimated_time",
                "table_data",
                "chart_data",
                "graph_data",
                "generation_metadata",
            ],
            "properties": {
                "title": {"type": "string"},
                "context": {"type": "string"},
                "questions": {"type": "array", "items": {"type": "string"}},
                "instruction": {"type": "string"},
                "solution": {"type": "string"},
                "expected_answer": {"type": "string"},
                "answer_kind": {"type": "string"},
                "solution_steps": {"type": "array", "items": {"type": "string"}},
                "learning_objective": {"type": "string"},
                "estimated_time": {"type": "string"},
                "table_data": {"type": ["object", "null"], "additionalProperties": True},
                "chart_data": {"type": ["object", "null"], "additionalProperties": True},
                "graph_data": {"type": ["object", "null"], "additionalProperties": True},
                "generation_metadata": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "target_section",
                        "target_topic",
                        "target_subtopic",
                        "exercise_family",
                        "requires_symbolic_check",
                        "requires_numeric_check",
                    ],
                    "properties": {
                        "target_section": {"type": "string"},
                        "target_topic": {"type": "string"},
                        "target_subtopic": {"type": "string"},
                        "exercise_family": {"type": "string"},
                        "requires_symbolic_check": {"type": "boolean"},
                        "requires_numeric_check": {"type": "boolean"},
                    },
                },
            },
        },
    }


def _response_format_modes(mode: str) -> list[str]:
    if mode == "json_schema":
        return ["json_schema", "json_object", "prompt_only"]
    if mode == "json_object":
        return ["json_object", "prompt_only"]
    if mode == "prompt_only":
        return ["prompt_only"]
    return ["json_schema", "json_object", "prompt_only"]


def _call_openrouter_once(
    *,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    response_format_mode: str,
    purpose: str,
    json_schema: dict[str, Any] | None = None,
) -> OpenRouterCallResult:
    client = get_openrouter_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if response_format_mode == "json_schema":
        kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema or get_exercise_json_schema()}
    elif response_format_mode == "json_object":
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        return _exception_to_call_result(exc, model=model, response_format_mode=response_format_mode)

    content = extract_openrouter_text(response)
    response_dict = _to_response_dict(response)
    raw_preview = _preview(json.dumps(response_dict, ensure_ascii=False) if response_dict else str(response))
    if not content:
        return OpenRouterCallResult(
            ok=False,
            content="",
            raw_response_preview=raw_preview,
            error_type="empty_response",
            error_message=summarize_openrouter_response_issue(response),
            model=model,
            response_format_mode=response_format_mode,
            request_id=_extract_request_id(response, response_dict),
            provider=_extract_provider(response_dict),
            usage=_extract_usage(response_dict),
        )
    return OpenRouterCallResult(
        ok=True,
        content=content,
        raw_response_preview=_preview(content),
        model=model,
        response_format_mode=response_format_mode,
        request_id=_extract_request_id(response, response_dict),
        provider=_extract_provider(response_dict),
        usage=_extract_usage(response_dict),
    )


def _exception_to_call_result(exc: Exception, *, model: str, response_format_mode: str) -> OpenRouterCallResult:
    message = str(exc)
    status = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    error_type = _classify_openrouter_exception(exc, message, status)
    if status == 400 and response_format_mode in {"json_schema", "json_object"}:
        error_type = "unsupported_response_format"
    return OpenRouterCallResult(
        ok=False,
        content="",
        raw_response_preview=_preview(message),
        http_status=status if isinstance(status, int) else None,
        error_type=error_type,
        error_message=message,
        model=model,
        response_format_mode=response_format_mode,
        request_id=str(request_id) if request_id else None,
    )


def _classify_openrouter_exception(exc: Exception, message: str, status: Any) -> str:
    text = f"{type(exc).__name__} {message}".lower()
    if status in {401, 403} or "api key" in text or "auth" in text:
        return "auth_error"
    if status == 429 or "rate limit" in text:
        return "rate_limit"
    if "response_format" in text or "json_schema" in text or "json object" in text:
        return "unsupported_response_format"
    if "timeout" in text:
        return "timeout"
    if "connection" in text or "network" in text or "connect" in text:
        return "connection_error"
    if isinstance(status, int) and status >= 400:
        return "http_error"
    return "unknown_error"


def _call_attempt_summary(result: OpenRouterCallResult, retry_number: int) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "model": result.model,
        "response_format_mode": result.response_format_mode,
        "http_status": result.http_status,
        "error_type": result.error_type,
        "error_message": result.error_message,
        "raw_response_preview": result.raw_response_preview,
        "retry_number": retry_number,
    }


def _preview(value: str, limit: int = 1000) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _json_loads_dict(payload: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(str(payload or "").strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_request_id(response: Any, response_dict: dict[str, Any]) -> str | None:
    value = getattr(response, "id", None) or response_dict.get("id") or response_dict.get("request_id")
    return str(value) if value else None


def _extract_provider(response_dict: dict[str, Any]) -> str | None:
    provider = response_dict.get("provider") or response_dict.get("provider_name")
    return str(provider) if provider else None


def _extract_usage(response_dict: dict[str, Any]) -> dict[str, Any] | None:
    usage = response_dict.get("usage")
    return usage if isinstance(usage, dict) else None


def _legacy_extract_json_object(raw: str) -> dict[str, Any] | None:
    """Deprecated helper kept for compatibility during tests."""
    content = str(raw or "").strip()
    if not content:
        return None

    candidates: list[str] = []
    fenced_chunks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(chunk.strip() for chunk in fenced_chunks if chunk.strip())
    candidates.append(content)
    candidates.extend(_iter_balanced_json_candidates(content))

    seen: set[str] = set()
    for candidate in candidates:
        normalized_candidate = str(candidate or "").strip()
        if not normalized_candidate or normalized_candidate in seen:
            continue
        seen.add(normalized_candidate)
        parsed = _parse_json_candidate(normalized_candidate)
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_generated_exercise_payload(obj: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy and v7 keys into one consistent exercise payload."""
    return normalize_exercise_schema(obj)


def validate_generated_exercise_schema(obj: dict[str, Any]) -> tuple[bool, list[str]]:
    """Check whether one generated exercise payload respects the expected v7 schema."""
    return validate_exercise_schema_v7(obj)


def _load_secret_section(section_name: str) -> dict[str, Any]:
    """Load one secret section with a file fallback for local scripts and tests."""
    try:
        secret_section = st.secrets.get(section_name)
        if secret_section:
            return dict(secret_section)
    except Exception:
        pass

    if not SECRETS_PATH.exists():
        return {}

    try:
        content = tomllib.loads(SECRETS_PATH.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}

    section = content.get(section_name, {})
    return section if isinstance(section, dict) else {}


def _root_secret_value(key: str) -> str:
    """Read a root-level Streamlit secret safely."""
    try:
        value = st.secrets.get(key)
        if value:
            return str(value).strip()
    except Exception:
        pass
    return ""


def _setting_value(
    raw_settings: dict[str, Any],
    *,
    key: str,
    env_name: str,
    aliases: tuple[str, ...] = (),
    default: str = "",
) -> str:
    """Resolve a string setting from env vars, [openrouter] secrets, root secrets, then default."""
    env_value = os.getenv(env_name)
    if env_value:
        return env_value.strip()

    for candidate in (key, *aliases):
        value = raw_settings.get(candidate)
        if value:
            return str(value).strip()

    for candidate in (env_name, *aliases):
        value = _root_secret_value(candidate)
        if value:
            return value

    return default.strip()


def _setting_bool(
    raw_settings: dict[str, Any],
    *,
    key: str,
    env_name: str,
    aliases: tuple[str, ...] = (),
    default: bool,
) -> bool:
    """Resolve a boolean setting from env vars, [openrouter] secrets, root secrets, then default."""
    env_value = os.getenv(env_name)
    if env_value is not None:
        return _coerce_bool(env_value, default)

    for candidate in (key, *aliases):
        if candidate in raw_settings:
            return _coerce_bool(raw_settings.get(candidate), default)

    for candidate in (env_name, *aliases):
        value = _root_secret_value(candidate)
        if value:
            return _coerce_bool(value, default)

    return default


def _extract_text_from_choices(choices: Any) -> str:
    """Read text from SDK or dict-based choice objects."""
    if not isinstance(choices, list):
        return ""

    for choice in choices:
        content = _extract_text_from_choice(choice)
        if content:
            return content
    return ""


def _extract_text_from_choice(choice: Any) -> str:
    """Read one content string from a single choice."""
    if choice is None:
        return ""

    if isinstance(choice, dict):
        message = choice.get("message") or {}
        content = _extract_text_from_content(message.get("content"))
        if content:
            return content
        return _extract_text_from_content(choice.get("text"))

    message = getattr(choice, "message", None)
    content = _extract_text_from_content(getattr(message, "content", None))
    if content:
        return content
    return _extract_text_from_content(getattr(choice, "text", None))


def _extract_text_from_content(content: Any) -> str:
    """Normalize plain-string or part-list content into one text block."""
    if isinstance(content, str):
        return content.strip()

    if not isinstance(content, list):
        return ""

    chunks: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
        else:
            text = str(getattr(item, "text", "")).strip()
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()


def _to_response_dict(response: Any) -> dict[str, Any]:
    """Convert one SDK response object into a plain dict when possible."""
    if isinstance(response, dict):
        return response

    try:
        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            payload = model_dump()
            return payload if isinstance(payload, dict) else {}
    except Exception:
        pass

    try:
        model_dump_json = getattr(response, "model_dump_json", None)
        if callable(model_dump_json):
            payload = json.loads(model_dump_json())
            return payload if isinstance(payload, dict) else {}
    except Exception:
        pass

    return {}


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _iter_balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    start_index: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
            continue
        if character == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue
        if character == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start_index is not None:
                candidates.append(text[start_index : index + 1].strip())
                start_index = None
    return candidates


def _parse_json_candidate(candidate: str) -> dict[str, Any] | None:
    for payload in (
        candidate,
        _strip_markdown_fence(candidate),
        _repair_json_candidate(candidate),
        _repair_json_candidate(_strip_markdown_fence(candidate)),
    ):
        payload = str(payload or "").strip()
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _strip_markdown_fence(value: str) -> str:
    text = str(value or "").strip()
    if not text.startswith("```"):
        return text
    text = text.strip("`").strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()
    return text


def _repair_json_candidate(value: str) -> str:
    repaired = str(value or "").strip()
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    repaired = re.sub(r"\\(?=[A-Za-z]{2,})", r"\\\\", repaired)
    repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", repaired)
    return repaired
