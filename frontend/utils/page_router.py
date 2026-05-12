"""Centralised routing helpers for Streamlit page switching."""

from __future__ import annotations

from typing import Any

import streamlit as st

_PAGE_REGISTRY: dict[str, Any] = {}


def register_pages(page_registry: dict[str, Any]) -> None:
    """Register Streamlit page objects for later switching."""
    _PAGE_REGISTRY.clear()
    _PAGE_REGISTRY.update(page_registry)


def switch_to_page(page_key: str) -> None:
    """Switch to a registered page by logical key."""
    page = _PAGE_REGISTRY.get(page_key)
    if page is None:
        raise RuntimeError(f"Page inconnue ou non enregistree : {page_key}")
    st.switch_page(page)
