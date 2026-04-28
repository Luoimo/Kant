"""
Backend prompt i18n module.

Central place for every LLM-facing prompt used by agents and retrievers.

Usage:
    from prompts import get_prompts
    p = get_prompts("en-US")     # or "zh-CN"
    system_prompt = p.router.system
"""
from __future__ import annotations

from . import zh_CN, en_US
from .types import PromptBundle

_BUNDLES: dict[str, PromptBundle] = {
    "zh-CN": zh_CN.BUNDLE,
    "en-US": en_US.BUNDLE,
}

DEFAULT_LOCALE = "zh-CN"


def normalize_locale(locale: str | None) -> str:
    """Normalize arbitrary locale strings to one of our supported ids."""
    if not locale:
        return DEFAULT_LOCALE
    low = locale.lower().replace("_", "-")
    if low.startswith("zh"):
        return "zh-CN"
    if low.startswith("en"):
        return "en-US"
    return DEFAULT_LOCALE


def get_prompts(locale: str | None = None) -> PromptBundle:
    """Return the prompt bundle for the requested locale (falls back to default)."""
    return _BUNDLES[normalize_locale(locale)]


__all__ = ["get_prompts", "normalize_locale", "PromptBundle", "DEFAULT_LOCALE"]
