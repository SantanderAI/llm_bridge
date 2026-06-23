# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Bring-your-own-LLM provider.

Wrap ANY callable as a vendor-neutral client. This is the recommended way to
plug in a proprietary or internal inference backend: llm_bridge never needs to
know — or reveal — which backend you use.

Example:
    def my_backend(messages, temperature=0.7, max_tokens=1024, **kwargs):
        # call your own SDK / gateway / local model here
        return "the model output text"

    from llm_bridge.providers.callable_provider import CallableClient
    llm = CallableClient(my_backend)
"""

from __future__ import annotations

import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Union

from llm_bridge.base import LLMClient, LLMResponse, Message


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()) * 4 // 3)


def _get_signature(fn: Callable[..., Any]) -> Optional[inspect.Signature]:
    """Return the callable signature when Python can introspect it."""
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def _can_bind(
    signature: inspect.Signature,
    *args: Any,
    **kwargs: Any,
) -> bool:
    """Check argument compatibility without executing the callable."""
    try:
        signature.bind(*args, **kwargs)
    except TypeError:
        return False
    return True


class CallableClient(LLMClient):
    """Adapt a user-supplied function to the :class:`LLMClient` interface.

    The function may accept the keyword signature
    ``(messages, temperature, max_tokens, **kwargs)`` or just ``(messages)``.
    It must return either a plain string or a fully-formed
    :class:`LLMResponse`.
    """

    def __init__(
        self,
        fn: Callable[..., Union[str, LLMResponse]],
        model: str = "callable",
    ):
        if not callable(fn):
            raise TypeError("CallableClient requires a callable `fn`.")
        self._fn = fn
        self._signature = _get_signature(fn)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "callable"

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        start = time.perf_counter() * 1000
        signature = self._signature
        if signature is None or _can_bind(
            signature,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            out = self._fn(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        elif _can_bind(signature, messages):
            out = self._fn(messages)
        else:
            raise TypeError(
                "CallableClient backend must accept either "
                "(messages, temperature=..., max_tokens=..., **kwargs) "
                "or (messages)."
            )
        latency = time.perf_counter() * 1000 - start

        if isinstance(out, LLMResponse):
            return out

        content = str(out)
        prompt_text = " ".join(m.get("content", "") for m in messages)
        return LLMResponse(
            content=content,
            model=self._model,
            prompt_tokens=_estimate_tokens(prompt_text),
            completion_tokens=_estimate_tokens(content),
            finish_reason="stop",
            latency_ms=latency,
            raw=None,
        )


def build(config: Dict[str, Any]) -> CallableClient:
    """Build a :class:`CallableClient` from a config dict."""
    fn = config.get("callable")
    if fn is None:
        raise ValueError(
            "The 'callable' provider requires a 'callable' entry: a function "
            "(messages, temperature, max_tokens, **kwargs) -> str | LLMResponse."
        )
    return CallableClient(fn=fn, model=config.get("model", "callable"))
