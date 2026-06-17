# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Provider implementations for llm_bridge.

    mock          in-process, deterministic (tests, demos, CI)
    callable      wrap ANY function -> bring your own backend
    openai        OpenAI SDK           (extra: pip install llm-bridge[openai])
    bedrock/aws   AWS Bedrock Converse (extra: pip install llm-bridge[aws])
    google/gemini Google Gemini SDK    (extra: pip install llm-bridge[google])

The ``openai`` adapter also targets any OpenAI-compatible endpoint (vLLM,
Ollama, Azure, internal gateway) via ``base_url``.

Use the registry (:mod:`llm_bridge.registry`) to build clients by name.
"""
