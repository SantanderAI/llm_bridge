# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Alibaba Qwen provider using DashScope's OpenAI-compatible API.

Optional dependency — requires ``pip install llm-bridge[qwen]``.

Configuration (config keys, with environment fallbacks):
    model      (required)  e.g. "qwen-plus"
    api_key    DASHSCOPE_API_KEY
    base_url   DASHSCOPE_BASE_URL (optional; defaults to DashScope international)
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, cast

from llm_bridge.base import LLMClient, LLMResponse, Message

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class QwenClient(LLMClient):
    """Chat client backed by Alibaba Qwen through DashScope."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if not model:
            raise ValueError("qwen provider requires 'model'.")
        self._model = model

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'qwen' provider requires the openai SDK. "
                "Install it with: pip install llm-bridge[qwen]"
            ) from exc

        self._client = OpenAI(
            api_key=api_key or os.environ.get("DASHSCOPE_API_KEY"),
            base_url=base_url or os.environ.get("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "qwen"

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        start = time.perf_counter() * 1000
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=cast(Any, messages),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        latency = time.perf_counter() * 1000 - start

        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            content=choice.message.content or "",
            model=getattr(resp, "model", self._model),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            finish_reason=choice.finish_reason or "stop",
            latency_ms=latency,
            raw=resp,
        )


def build(config: Dict[str, Any]) -> QwenClient:
    """Build a :class:`QwenClient` from a config dict."""
    model = config.get("model")
    if not model:
        raise ValueError("qwen provider requires 'model'.")
    return QwenClient(
        model=model,
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
    )
