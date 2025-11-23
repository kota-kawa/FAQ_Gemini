from __future__ import annotations

from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

_DEFAULT_ENV_FILES = [Path(__file__).resolve().parent / "secrets.env"]


def load_secrets_env(additional_paths: Iterable[Path] | None = None) -> None:
    """Load secrets from the agent's env files, favoring secrets.env."""

    loaded = False
    for path in _DEFAULT_ENV_FILES + list(additional_paths or []):
        if load_dotenv(path, override=False):
            loaded = True
    if not loaded:
        # Fallback to python-dotenv's default resolution for legacy setups.
        load_dotenv()
