from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ProviderStatus:
    id: str
    name: str
    env_key: str
    configured: bool
    source: str  # "env" | "user" | "none"


def _read_user_keys() -> dict:
    """
    Reads `data/llm_user_keys.json` if present (same file used by core.ai_agent).
    """
    p = Path("data/llm_user_keys.json")
    if not p.exists():
        return {}
    try:
        import json

        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _has_nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def get_provider_statuses() -> list[ProviderStatus]:
    user = _read_user_keys()

    def user_key(pid: str) -> Optional[str]:
        v = (user.get(pid) or {}) if isinstance(user.get(pid), dict) else {}
        return v.get("api_key") if isinstance(v, dict) else None

    def env_key(name: str) -> Optional[str]:
        return os.environ.get(name)

    out: list[ProviderStatus] = []

    # OpenAI (ChatGPT, GPT-4o, etc.)
    openai_user = user_key("openai")
    openai_env = env_key("OPENAI_API_KEY")
    out.append(
        ProviderStatus(
            id="openai",
            name="OpenAI (ChatGPT)",
            env_key="OPENAI_API_KEY",
            configured=_has_nonempty(openai_user) or _has_nonempty(openai_env),
            source="user" if _has_nonempty(openai_user) else ("env" if _has_nonempty(openai_env) else "none"),
        )
    )

    # Anthropic (Claude)
    anth_user = user_key("anthropic")
    anth_env = env_key("ANTHROPIC_API_KEY")
    out.append(
        ProviderStatus(
            id="anthropic",
            name="Anthropic (Claude)",
            env_key="ANTHROPIC_API_KEY",
            configured=_has_nonempty(anth_user) or _has_nonempty(anth_env),
            source="user" if _has_nonempty(anth_user) else ("env" if _has_nonempty(anth_env) else "none"),
        )
    )

    # xAI (Grok) - OpenAI-compatible base URL in this codebase
    grok_user = user_key("grok")
    grok_env = env_key("GROK_API_KEY")
    out.append(
        ProviderStatus(
            id="grok",
            name="xAI (Grok)",
            env_key="GROK_API_KEY",
            configured=_has_nonempty(grok_user) or _has_nonempty(grok_env),
            source="user" if _has_nonempty(grok_user) else ("env" if _has_nonempty(grok_env) else "none"),
        )
    )

    return out


