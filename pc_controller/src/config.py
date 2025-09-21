"""Central configuration loader for PC Controller.

Reads settings from pc_controller/config.json by default (NFR8).
Supports overriding the config file path via environment variable PC_CONFIG_PATH.
Provides cached accessors to avoid repeated disk I/O.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__CONFIG_CACHE: dict[str, Any] | None = None


def _default_config_path() -> Path:
    # This file lives at pc_controller/src/config.py
    # Default config.json resides two levels up: pc_controller/config.json
    here = Path(__file__).resolve()
    return here.parents[2] / "config.json"


def _load_from_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def reload_config() -> None:
    """Clear the in-memory cache so next get_config() will re-read from disk."""
    global __CONFIG_CACHE
    __CONFIG_CACHE = None


def get_config() -> dict[str, Any]:
    """Return the loaded configuration as a dictionary (cached).

    Resolution order:
    - If env PC_CONFIG_PATH is set, read that file
    - Else read the default pc_controller/config.json located at project root
    """
    global __CONFIG_CACHE
    if __CONFIG_CACHE is not None:
        return dict(__CONFIG_CACHE)

    env_path = os.environ.get("PC_CONFIG_PATH")
    path = Path(env_path) if env_path else _default_config_path()
    cfg = _load_from_file(path)
    __CONFIG_CACHE = cfg
    return dict(cfg)


def get(key: str, default: Any = None) -> Any:
    """Convenience accessor to fetch a single config value with default."""
    return get_config().get(key, default)
