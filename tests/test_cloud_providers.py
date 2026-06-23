# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the vendor SDK adapters.

The real vendor SDKs (``openai``, ``boto3``, ``google-genai``) are never
installed in CI's base job. These tests inject lightweight fakes into
``sys.modules`` so the adapter request/response mapping is exercised without
network access or credentials.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from llm_bridge import create_llm

MESSAGES = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "hello"},
]


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_openai(monkeypatch):
    captured: dict[str, Any] = {}

    class _Msg:
        content = "openai-reply"

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _Usage:
        prompt_tokens = 11
        completion_tokens = 7

    class _Resp:
        model = "gpt-test"
        choices = [_Choice()]
        usage = _Usage()

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _Chat:
        completions = _Completions()

    class OpenAI:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.chat = _Chat()

    mod = types.ModuleType("openai")
    mod.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)
    return captured


def test_openai_chat_maps_response(fake_openai):
    llm = create_llm({"provider": "openai", "model": "gpt-test", "api_key": "x"})
    assert llm.provider == "openai"
    resp = llm.chat(MESSAGES, temperature=0.1, max_tokens=64)
    assert resp.content == "openai-reply"
    assert resp.model == "gpt-test"
    assert resp.prompt_tokens == 11
    assert resp.completion_tokens == 7
    assert resp.total_tokens == 18
    assert fake_openai["model"] == "gpt-test"
    assert fake_openai["messages"] == MESSAGES


# --------------------------------------------------------------------------- #
# DeepSeek
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_deepseek_openai(monkeypatch):
    captured: dict[str, Any] = {}

    class _Msg:
        content = "deepseek-reply"

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _Usage:
        prompt_tokens = 13
        completion_tokens = 8

    class _Resp:
        model = "deepseek-v4-pro"
        choices = [_Choice()]
        usage = _Usage()

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _Chat:
        completions = _Completions()

    class OpenAI:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.chat = _Chat()

    mod = types.ModuleType("openai")
    mod.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)
    return captured


def test_deepseek_chat_maps_response(fake_deepseek_openai):
    llm = create_llm(
        {"provider": "deepseek", "model": "deepseek-v4-pro", "api_key": "deepseek-key"}
    )
    assert llm.provider == "deepseek"
    resp = llm.chat(MESSAGES, temperature=0.1, max_tokens=64, reasoning_effort="high")
    assert resp.content == "deepseek-reply"
    assert resp.model == "deepseek-v4-pro"
    assert resp.prompt_tokens == 13
    assert resp.completion_tokens == 8
    assert resp.total_tokens == 21
    assert fake_deepseek_openai["init"]["api_key"] == "deepseek-key"
    assert fake_deepseek_openai["init"]["base_url"] == "https://api.deepseek.com"
    assert fake_deepseek_openai["model"] == "deepseek-v4-pro"
    assert fake_deepseek_openai["messages"] == MESSAGES
    assert fake_deepseek_openai["reasoning_effort"] == "high"


def test_deepseek_base_url_override(fake_deepseek_openai):
    create_llm(
        {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "api_key": "x",
            "base_url": "https://example.test/deepseek",
        }
    )
    assert fake_deepseek_openai["init"]["base_url"] == "https://example.test/deepseek"


# --------------------------------------------------------------------------- #
# AWS Bedrock
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_boto3(monkeypatch):
    captured: dict[str, Any] = {}

    class _Client:
        def converse(self, **kwargs):
            captured.update(kwargs)
            return {
                "output": {"message": {"content": [{"text": "bedrock-reply"}]}},
                "usage": {"inputTokens": 5, "outputTokens": 3},
                "stopReason": "end_turn",
            }

    class _Session:
        def __init__(self, **kwargs):
            captured["session"] = kwargs

        def client(self, name, **kwargs):
            captured["client"] = name
            return _Client()

    mod = types.ModuleType("boto3")
    mod.Session = _Session
    monkeypatch.setitem(sys.modules, "boto3", mod)
    return captured


def test_bedrock_chat_maps_response(fake_boto3):
    llm = create_llm({"provider": "bedrock", "model": "anthropic.claude", "region": "eu-west-1"})
    assert llm.provider == "bedrock"
    resp = llm.chat(MESSAGES, temperature=0.2, max_tokens=128, top_p=0.9)
    assert resp.content == "bedrock-reply"
    assert resp.prompt_tokens == 5
    assert resp.completion_tokens == 3
    assert resp.finish_reason == "end_turn"
    assert fake_boto3["client"] == "bedrock-runtime"
    # system message is split out from the turns
    assert fake_boto3["system"] == [{"text": "be brief"}]
    assert fake_boto3["inferenceConfig"]["topP"] == 0.9


def test_bedrock_alias_aws(fake_boto3):
    llm = create_llm({"provider": "aws", "model": "meta.llama"})
    assert llm.chat(MESSAGES).content == "bedrock-reply"


# --------------------------------------------------------------------------- #
# Google Gemini
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_google(monkeypatch):
    captured: dict[str, Any] = {}

    class _Part:
        @staticmethod
        def from_text(text):
            return {"text": text}

    class _Content:
        def __init__(self, role, parts):
            self.role = role
            self.parts = parts

    class _GenerateContentConfig:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

    class _Usage:
        prompt_token_count = 4
        candidates_token_count = 6

    class _Resp:
        text = "gemini-reply"
        usage_metadata = _Usage()

    class _Models:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _Client:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.models = _Models()

    genai = types.ModuleType("google.genai")
    genai.Client = _Client
    types_mod = types.ModuleType("google.genai.types")
    types_mod.Part = _Part
    types_mod.Content = _Content
    types_mod.GenerateContentConfig = _GenerateContentConfig
    genai.types = types_mod

    google_pkg = types.ModuleType("google")
    google_pkg.genai = genai

    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)
    return captured


def test_google_chat_maps_response(fake_google):
    llm = create_llm({"provider": "google", "model": "gemini-2.5-flash", "api_key": "x"})
    assert llm.provider == "google"
    resp = llm.chat(MESSAGES, temperature=0.3, max_tokens=256)
    assert resp.content == "gemini-reply"
    assert resp.prompt_tokens == 4
    assert resp.completion_tokens == 6
    assert captured_model(fake_google) == "gemini-2.5-flash"


def test_google_alias_gemini(fake_google):
    llm = create_llm({"provider": "gemini", "model": "gemini-2.5-flash"})
    assert llm.chat(MESSAGES).content == "gemini-reply"


def captured_model(captured: dict) -> str:
    return captured["model"]
