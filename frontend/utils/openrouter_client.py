"""Helpers for loading OpenRouter settings and building the client."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import tomllib
from typing import Any

import streamlit as st
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
SECRETS_PATH = REPO_ROOT / ".streamlit" / "secrets.toml"


@dataclass(frozen=True)
class OpenRouterSettings:
    """Runtime configuration for OpenRouter-based exercise generation."""

    api_key: str
    base_url: str
    exercise_model: str
    judge_model: str
    tutor_model: str
    site_url: str
    app_name: str


def has_openrouter_config() -> bool:
    """Indicate whether OpenRouter credentials are available."""
    return get_openrouter_settings() is not None


@lru_cache(maxsize=1)
def get_openrouter_settings() -> OpenRouterSettings | None:
    """Load OpenRouter settings from Streamlit secrets or the local TOML file."""
    raw_settings = _load_secret_section("openrouter")
    api_key = str(raw_settings.get("api_key", "")).strip()
    if not api_key:
        return None

    return OpenRouterSettings(
        api_key=api_key,
        base_url=str(raw_settings.get("base_url", "https://openrouter.ai/api/v1")).strip(),
        exercise_model=str(raw_settings.get("exercise_model", "qwen/qwen-2.5-7b-instruct")).strip(),
        judge_model=str(raw_settings.get("judge_model", "qwen/qwen-2.5-7b-instruct")).strip(),
        tutor_model=str(raw_settings.get("tutor_model", raw_settings.get("exercise_model", "qwen/qwen-2.5-7b-instruct"))).strip(),
        site_url=str(raw_settings.get("site_url", "http://localhost:8501")).strip(),
        app_name=str(raw_settings.get("app_name", "MathTutorAI")).strip(),
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
        return "Réponse OpenRouter vide ou non structurée."

    error_payload = raw_payload.get("error")
    if isinstance(error_payload, dict):
        message = str(error_payload.get("message", "")).strip()
        if message:
            return f"OpenRouter a renvoyé une erreur : {message}"

    if raw_payload.get("choices") is None:
        return "OpenRouter n'a renvoyé aucun champ 'choices'."
    if raw_payload.get("choices") == []:
        return "OpenRouter a renvoyé une liste 'choices' vide."
    return "OpenRouter n'a renvoyé aucun contenu textuel exploitable."


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
