# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Provider registry — map a provider name to a builder function.

Dependency-free providers (``mock``, ``callable``) are registered eagerly.
Providers that wrap an official vendor SDK (``openai``, ``bedrock``,
``google``, ``deepseek``, ``qwen``) are registered with lazy builders, so importing
``llm_bridge`` never pulls in a vendor SDK.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List

from llm_bridge.base import LLMClient

logger = logging.getLogger("llm_bridge.registry")

ProviderBuilder = Callable[[Dict[str, Any]], LLMClient]

_PROVIDERS: Dict[str, ProviderBuilder] = {}


def register_provider(name: str, builder: ProviderBuilder) -> None:
    """Register (or override) a provider builder under ``name``."""
    _PROVIDERS[name] = builder


def available_providers() -> List[str]:
    """Return the sorted list of registered provider names."""
    return sorted(_PROVIDERS)


def create_llm(config: Dict[str, Any], **overrides: Any) -> LLMClient:
    """Build a client from a config dict with at least a ``provider`` key.

    Extra keyword ``overrides`` replace config entries when not ``None``.
    """
    cfg = dict(config)
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    provider = cfg.get("provider")
    if not provider:
        raise ValueError("config must include a 'provider' key.")

    builder = _PROVIDERS.get(provider)
    if builder is None:
        raise ValueError(f"Unknown provider '{provider}'. Registered: {available_providers()}.")
    logger.debug("Building LLM client via provider '%s'", provider)
    return builder(cfg)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a config mapping from a JSON or YAML file.

    JSON uses the standard library. YAML requires PyYAML (imported lazily).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError("YAML config requires PyYAML: pip install pyyaml") from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _register_builtins() -> None:
    # Dependency-free providers (eager).
    from llm_bridge.providers import callable_provider, mock

    register_provider("mock", mock.build)
    register_provider("callable", callable_provider.build)

    # Optional vendor SDK providers (lazy — SDK imported only when used).
    def _openai(cfg: Dict[str, Any]) -> LLMClient:
        from llm_bridge.providers.openai_sdk import build

        return build(cfg)

    def _deepseek(cfg: Dict[str, Any]) -> LLMClient:
        from llm_bridge.providers.deepseek import build

        return build(cfg)

    def _qwen(cfg: Dict[str, Any]) -> LLMClient:
        from llm_bridge.providers.qwen import build

        return build(cfg)

    def _bedrock(cfg: Dict[str, Any]) -> LLMClient:
        from llm_bridge.providers.bedrock import build

        return build(cfg)

    def _google(cfg: Dict[str, Any]) -> LLMClient:
        from llm_bridge.providers.google_genai import build

        return build(cfg)

    register_provider("openai", _openai)
    register_provider("deepseek", _deepseek)
    register_provider("qwen", _qwen)
    register_provider("bedrock", _bedrock)
    register_provider("aws", _bedrock)  # alias
    register_provider("google", _google)
    register_provider("gemini", _google)  # alias


_register_builtins()
