# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Core abstractions for llm_bridge — a vendor-neutral LLM client interface.

A single small contract, :class:`LLMClient`, that every backend implements.
Application code depends only on this interface, never on a specific vendor
SDK or cloud provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# A chat message is a simple dict: {"role": "system|user|assistant", "content": "..."}
Message = Dict[str, str]


@dataclass
class LLMResponse:
    """A normalised response returned by every provider."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    raw: Any = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def split_system_messages(
    messages: List[Message],
) -> Tuple[Optional[str], List[Message]]:
    """Split a message list into ``(system_text, turns)``.

    ``system_text`` concatenates all ``system`` messages (joined by blank
    lines); ``turns`` keeps the ``user``/``assistant`` messages in order.
    Useful for providers (Bedrock, Gemini) that take the system prompt as a
    separate field rather than inline in the message list.
    """
    system_parts: List[str] = []
    turns: List[Message] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            if content:
                system_parts.append(content)
        else:
            turns.append({"role": role, "content": content})
    system_text = "\n\n".join(system_parts) if system_parts else None
    return system_text, turns


class LLMClient(ABC):
    """Vendor-neutral chat LLM client.

    Implementations only need to provide :meth:`chat` and the :attr:`model`
    property. :meth:`complete` is a convenience wrapper for single-turn calls.
    """

    @property
    @abstractmethod
    def model(self) -> str:
        """The model identifier this client talks to."""

    @property
    def provider(self) -> str:
        """Short provider name (e.g. ``"openai"``)."""
        return "unknown"

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a list of chat messages and return a normalised response."""

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Single-turn convenience: build messages from ``prompt``/``system``."""
        messages: List[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
