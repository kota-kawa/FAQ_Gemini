"""Load shared model selection from Multi-Agent-Platform/model_settings.json."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

DEFAULT_SELECTION = {"provider": "openai", "model": "gpt-4.1-2025-04-14"}

PROVIDER_DEFAULTS: Dict[str, Dict[str, str | None]] = {
    "openai": {"api_key_env": "OPENAI_API_KEY", "base_url_env": "OPENAI_BASE_URL"},
    "claude": {"api_key_env": "CLAUDE_API_KEY", "langchain_api_key_env": "ANTHROPIC_API_KEY"},
    "gemini": {"api_key_env": "GOOGLE_API_KEY", "langchain_api_key_env": "GOOGLE_API_KEY"},
    "groq": {"api_key_env": "GROQ_API_KEY", "langchain_api_key_env": "GROQ_API_KEY"},
}

_OVERRIDE_SELECTION: Dict[str, str] | None = None


def _load_selection_file(agent_key: str) -> Dict[str, str]:
    """Return the model selection for the given agent key."""

    platform_path = Path(__file__).resolve().parent.parent / "Multi-Agent-Platform" / "model_settings.json"
    try:
        data = json.loads(platform_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(DEFAULT_SELECTION)

    selection = data.get("selection") or data
    if not isinstance(selection, dict):
        return dict(DEFAULT_SELECTION)

    chosen = selection.get(agent_key)
    if not isinstance(chosen, dict):
        return dict(DEFAULT_SELECTION)

    provider = (chosen.get("provider") or DEFAULT_SELECTION["provider"]).strip()
    model = (chosen.get("model") or DEFAULT_SELECTION["model"]).strip()
    return {"provider": provider, "model": model}


def apply_model_selection(agent_key: str = "lifestyle", override: Dict[str, str] | None = None) -> Tuple[str, str, str]:
    """Apply model selection to environment and return (provider, model, base_url)."""

    selection = override or _OVERRIDE_SELECTION or _load_selection_file(agent_key)
    provider = selection.get("provider") or DEFAULT_SELECTION["provider"]
    model = selection.get("model") or DEFAULT_SELECTION["model"]

    meta = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])
    api_key_env = meta["api_key_env"]
    langchain_api_key_env = meta.get("langchain_api_key_env", "OPENAI_API_KEY")
    base_url_env = meta.get("base_url_env")

    api_key = os.getenv(api_key_env) or os.getenv(api_key_env.lower())
    if api_key:
        os.environ[langchain_api_key_env] = api_key

    base_url = os.getenv(base_url_env, "") if base_url_env else ""
    if base_url:
        os.environ["OPENAI_API_BASE"] = base_url
        os.environ["OPENAI_BASE_URL"] = base_url

    return provider, model, base_url


def update_override(selection: Dict[str, str] | None) -> Tuple[str, str, str]:
    """Set an in-memory override and apply it immediately."""

    global _OVERRIDE_SELECTION
    _OVERRIDE_SELECTION = selection or None
    return apply_model_selection(override=_OVERRIDE_SELECTION or None)


def current_selection(agent_key: str = "lifestyle") -> Dict[str, str]:
    """Return the currently applied selection without requiring callers to know overrides."""

    provider, model, base_url = apply_model_selection(agent_key)
    return {"provider": provider, "model": model, "base_url": base_url}
