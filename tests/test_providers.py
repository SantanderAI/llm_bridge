# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Tests for the dependency-free providers (offline)."""

import pytest

from llm_bridge import create_llm
from llm_bridge.base import LLMResponse
from llm_bridge.providers.callable_provider import CallableClient
from llm_bridge.providers.mock import MockClient


def test_mock_echo():
    llm = create_llm({"provider": "mock"})
    assert "hello" in llm.complete("hello").content
    assert llm.provider == "mock"


def test_mock_fixed_response():
    llm = create_llm({"provider": "mock", "response": "fixed"})
    assert llm.complete("anything").content == "fixed"


def test_mock_responder():
    llm = MockClient(responder=lambda messages: f"n={len(messages)}")
    assert llm.chat([{"role": "user", "content": "a"}]).content == "n=1"


def test_callable_returns_str():
    llm = create_llm({"provider": "callable", "callable": lambda messages, **k: "hi"})
    resp = llm.complete("x")
    assert resp.content == "hi"
    assert llm.provider == "callable"


def test_callable_messages_only_signature():
    llm = CallableClient(lambda messages: "two")
    assert llm.complete("x").content == "two"


def test_callable_internal_type_error_is_not_retried():
    calls = 0

    def backend(messages, temperature=0.7, max_tokens=1024, **kwargs):
        nonlocal calls
        calls += 1
        raise TypeError("backend failure")

    llm = CallableClient(backend)

    with pytest.raises(TypeError, match="backend failure"):
        llm.complete("x")

    assert calls == 1


def test_callable_rejects_unsupported_signature_without_execution():
    calls = 0

    def backend(messages, required):
        nonlocal calls
        calls += 1
        return "unreachable"

    llm = CallableClient(backend)

    with pytest.raises(TypeError, match="must accept either"):
        llm.complete("x")

    assert calls == 0


def test_callable_llmresponse_passthrough():
    custom = LLMResponse(content="c", model="m", prompt_tokens=1, completion_tokens=2)
    llm = CallableClient(lambda **k: custom)
    assert llm.chat([{"role": "user", "content": "x"}]) is custom


def test_callable_requires_fn():
    with pytest.raises(ValueError):
        create_llm({"provider": "callable"})
