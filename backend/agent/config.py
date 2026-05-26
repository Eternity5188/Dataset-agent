"""
config.py — Per-request API key configuration.

Uses a ContextVar so each request can carry its own API keys (sent via HTTP
headers from the frontend) without mutating process-level environment. All
tools read keys via get_api_key(); if a per-request value is missing, we fall
back to the corresponding environment variable, so deployments can still set
keys server-side.
"""

import contextvars
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Canonical key names → default env-var names
KEY_ENV_MAP: dict[str, str] = {
    "dashscope":         "DASHSCOPE_API_KEY",
    "github":            "GITHUB_TOKEN",
    "huggingface":       "HF_TOKEN",
    "semantic_scholar":  "SEMANTIC_SCHOLAR_API_KEY",
    "tavily":            "TAVILY_API_KEY",
}

_api_keys_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "api_keys", default={}
)


def set_api_keys(keys: dict) -> contextvars.Token:
    """Set the per-request API key bundle. Returns a token for reset()."""
    cleaned = {k: (v or "").strip() for k, v in (keys or {}).items() if v}
    return _api_keys_ctx.set(cleaned)


def reset_api_keys(token: contextvars.Token) -> None:
    try:
        _api_keys_ctx.reset(token)
    except Exception:
        pass


def get_api_key(name: str, default: str = "") -> str:
    """Get an API key by canonical name, falling back to env var, then default."""
    ctx_keys = _api_keys_ctx.get() or {}
    val = (ctx_keys.get(name) or "").strip()
    if val:
        return val
    env_name = KEY_ENV_MAP.get(name)
    if env_name:
        env_val = (os.getenv(env_name, "") or "").strip()
        if env_val:
            return env_val
    return default


def has_api_key(name: str) -> bool:
    return bool(get_api_key(name))


def configured_keys() -> dict:
    """Return a mask-friendly summary (only booleans) — never expose secrets."""
    return {name: has_api_key(name) for name in KEY_ENV_MAP}
