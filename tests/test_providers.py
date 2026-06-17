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


def test_callable_two_arg_signature():
    llm = CallableClient(lambda messages: "two")
    assert llm.complete("x").content == "two"


def test_callable_llmresponse_passthrough():
    custom = LLMResponse(content="c", model="m", prompt_tokens=1, completion_tokens=2)
    llm = CallableClient(lambda **k: custom)
    assert llm.chat([{"role": "user", "content": "x"}]) is custom


def test_callable_requires_fn():
    with pytest.raises(ValueError):
        create_llm({"provider": "callable"})
