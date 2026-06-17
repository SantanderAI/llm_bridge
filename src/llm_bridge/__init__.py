# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""llm_bridge — a tiny, vendor-neutral wrapper for any LLM backend.

One small interface (:class:`LLMClient`) plus pluggable providers. Bring your
own backend with the ``callable`` provider, or use the bundled adapters for
OpenAI, AWS Bedrock, and Google Gemini. Nothing in the core install depends on
a vendor SDK — those are optional extras.

    from llm_bridge import create_llm

    llm = create_llm({"provider": "openai", "model": "gpt-4o-mini"})
    print(llm.complete("Say hello in one sentence.").content)
"""

from __future__ import annotations

from llm_bridge.base import LLMClient, LLMResponse, Message, split_system_messages
from llm_bridge.registry import (
    available_providers,
    create_llm,
    load_config,
    register_provider,
)

__version__ = "0.1.0"

__all__ = [
    "LLMClient",
    "LLMResponse",
    "Message",
    "split_system_messages",
    "create_llm",
    "register_provider",
    "available_providers",
    "load_config",
    "__version__",
]
