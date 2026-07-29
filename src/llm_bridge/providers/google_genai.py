# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Google Gemini provider using the ``google-genai`` SDK.

Optional dependency — requires ``pip install llm-bridge[google]``.

Configuration (config keys, with environment fallbacks):
    model      (required)  e.g. "gemini-2.5-flash"
    api_key    GOOGLE_API_KEY or GEMINI_API_KEY
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Mapping, Optional

from llm_bridge.base import LLMClient, LLMResponse, Message, split_system_messages

_GENERATION_ARGUMENTS = ("top_p", "top_k", "stop_sequences", "seed")
_FINISH_REASON_TOKEN = re.compile(r"\A[A-Za-z][A-Za-z0-9_]*\Z")


def _generation_config_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and map Google-specific generation arguments."""
    unknown = sorted(set(kwargs) - set(_GENERATION_ARGUMENTS) - {"stop"})
    if unknown:
        raise TypeError(f"Unsupported Google generation arguments: {', '.join(unknown)}")
    if "stop" in kwargs and "stop_sequences" in kwargs:
        raise TypeError("'stop' and 'stop_sequences' cannot be used together")

    config = {name: kwargs[name] for name in _GENERATION_ARGUMENTS if name in kwargs}
    if "stop" in kwargs:
        config["stop_sequences"] = kwargs["stop"]
    return config


def _extract_finish_reason(response: Any) -> str:
    """Return the first lower-case Google finish-reason token."""
    try:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return "unknown"
        candidate = candidates[0]
        if candidate is None:
            return "unknown"
        reason = getattr(candidate, "finish_reason", None)
        if reason is None:
            return "unknown"
        try:
            value = reason.value
        except AttributeError:
            value = reason if isinstance(reason, str) else None
    except AttributeError:
        return "unknown"

    if not isinstance(value, str):
        return "unknown"
    value = value.strip()
    if value.upper() == "FINISH_REASON_UNSPECIFIED" or not _FINISH_REASON_TOKEN.fullmatch(value):
        return "unknown"
    return value.lower()


class GoogleClient(LLMClient):
    """Chat client backed by Google's Gen AI (Gemini) SDK."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        if not model:
            raise ValueError("google provider requires 'model'.")
        self._model = model

        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "The 'google' provider requires the google-genai SDK. "
                "Install it with: pip install llm-bridge[google]"
            ) from exc

        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self._genai = genai
        self._client = genai.Client(api_key=key)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "google"

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        generation_config = _generation_config_kwargs(kwargs)
        from google.genai import types

        system_text, turns = split_system_messages(messages)
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in turns
        ]
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_text,
            **generation_config,
        )

        start = time.perf_counter() * 1000
        resp = self._client.models.generate_content(
            model=self._model, contents=contents, config=config
        )
        latency = time.perf_counter() * 1000 - start

        usage = getattr(resp, "usage_metadata", None)
        return LLMResponse(
            content=getattr(resp, "text", "") or "",
            model=self._model,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            finish_reason=_extract_finish_reason(resp),
            latency_ms=latency,
            raw=resp,
        )


def build(config: Dict[str, Any]) -> GoogleClient:
    """Build a :class:`GoogleClient` from a config dict."""
    model = config.get("model")
    if not model:
        raise ValueError("google provider requires 'model'.")
    return GoogleClient(model=model, api_key=config.get("api_key"))
