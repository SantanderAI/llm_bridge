# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Focused request and response tests for the Google provider."""

from __future__ import annotations

import inspect
import sys
import types
from enum import Enum
from typing import Any

import pytest

from llm_bridge.providers.google_genai import (
    GoogleClient,
    _extract_finish_reason,
    _generation_config_kwargs,
    build,
)


class _FinishReason(Enum):
    STOP = "STOP"
    MAX_TOKENS = "MAX_TOKENS"
    SAFETY = "SAFETY"
    OTHER = "OTHER"
    UNSPECIFIED = "FINISH_REASON_UNSPECIFIED"


class _ConfigError(ValueError):
    pass


@pytest.fixture
def fake_google(monkeypatch):
    captured: dict[str, Any] = {
        "calls": 0,
        "control": {},
        "config_calls": 0,
        "init_calls": [],
        "part_calls": 0,
    }

    class _Part:
        @staticmethod
        def from_text(text):
            captured["part_calls"] += 1
            return {"text": text}

    class _Content:
        def __init__(self, role, parts):
            self.role = role
            self.parts = parts

    class _GenerateContentConfig:
        def __init__(self, **kwargs):
            captured["config_calls"] += 1
            if kwargs.get("top_p") == "invalid":
                raise _ConfigError("invalid top_p")
            captured["config_kwargs"] = kwargs

    class _Usage:
        prompt_token_count = 4
        candidates_token_count = 6

    class _Response:
        text = "gemini-reply"
        usage_metadata = _Usage()
        candidates = [types.SimpleNamespace(finish_reason=_FinishReason.STOP)]

    class _Models:
        def generate_content(self, **kwargs):
            captured["calls"] += 1
            control = captured["control"]
            if "transport_error" in control:
                raise control["transport_error"]
            captured.update(kwargs)
            return control.get("response", _Response())

    class _Client:
        def __init__(self, **kwargs):
            captured["init_calls"].append(kwargs)
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


def _response(reason: Any) -> Any:
    return types.SimpleNamespace(candidates=[types.SimpleNamespace(finish_reason=reason)])


def test_generation_config_maps_allowlist_falsey_values_and_alias():
    source = {"top_p": 0.0, "top_k": 0, "seed": 0, "stop": []}
    result = _generation_config_kwargs(source)

    assert result == {
        "top_p": 0.0,
        "top_k": 0,
        "seed": 0,
        "stop_sequences": [],
    }
    assert source == {"top_p": 0.0, "top_k": 0, "seed": 0, "stop": []}


def test_generation_config_preserves_stop_sequences():
    assert _generation_config_kwargs({"stop_sequences": []}) == {"stop_sequences": []}
    assert _generation_config_kwargs({"stop": None}) == {"stop_sequences": None}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"zeta": 1, "alpha": 2},
            "Unsupported Google generation arguments: alpha, zeta",
        ),
        (
            {"unknown": 1},
            "Unsupported Google generation arguments: unknown",
        ),
        (
            {"stop": ["A"], "stop_sequences": ["B"]},
            "'stop' and 'stop_sequences' cannot be used together",
        ),
        (
            {"unknown": 1, "stop": ["A"], "stop_sequences": ["B"]},
            "Unsupported Google generation arguments: unknown",
        ),
    ],
)
def test_generation_config_rejects_unknown_and_conflicting_arguments(kwargs, message):
    with pytest.raises(TypeError) as error:
        _generation_config_kwargs(kwargs)
    assert str(error.value) == message


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (_FinishReason.STOP, "stop"),
        (_FinishReason.MAX_TOKENS, "max_tokens"),
        (_FinishReason.SAFETY, "safety"),
        (_FinishReason.OTHER, "other"),
        (_FinishReason.UNSPECIFIED, "unknown"),
        ("finish_reason_unspecified", "unknown"),
        (" FUTURE_REASON ", "future_reason"),
        ("", "unknown"),
        ("   ", "unknown"),
        ("FinishReason.STOP", "unknown"),
        ("A-B", "unknown"),
        ("SÄFETY", "unknown"),
        (types.SimpleNamespace(value=1), "unknown"),
    ],
)
def test_finish_reason_mapping(reason, expected):
    assert _extract_finish_reason(_response(reason)) == expected


@pytest.mark.parametrize(
    "response",
    [
        object(),
        types.SimpleNamespace(candidates=None),
        types.SimpleNamespace(candidates=[]),
        types.SimpleNamespace(candidates=[None]),
        types.SimpleNamespace(candidates=[object()]),
        types.SimpleNamespace(candidates=[types.SimpleNamespace(finish_reason=None)]),
    ],
)
def test_finish_reason_missing_states_return_unknown(response):
    assert _extract_finish_reason(response) == "unknown"


def test_finish_reason_attribute_errors_return_unknown():
    class _Response:
        @property
        def candidates(self):
            raise AttributeError("not available")

    class _Candidate:
        @property
        def finish_reason(self):
            raise AttributeError("not available")

    class _Reason:
        @property
        def value(self):
            raise AttributeError("not available")

    assert _extract_finish_reason(_Response()) == "unknown"
    assert _extract_finish_reason(types.SimpleNamespace(candidates=[_Candidate()])) == "unknown"
    assert _extract_finish_reason(_response(_Reason())) == "unknown"


def test_finish_reason_candidate_lookup_attribute_error_returns_unknown():
    class _Candidates:
        def __bool__(self):
            return True

        def __getitem__(self, index):
            raise AttributeError("not available")

    assert _extract_finish_reason(types.SimpleNamespace(candidates=_Candidates())) == "unknown"


def test_finish_reason_non_attribute_error_propagates():
    class _Candidate:
        @property
        def finish_reason(self):
            raise RuntimeError("provider failure")

    response = types.SimpleNamespace(candidates=[_Candidate()])
    with pytest.raises(RuntimeError, match="provider failure"):
        _extract_finish_reason(response)


def test_constructor_properties_and_api_key_precedence(fake_google, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-env")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-env")

    explicit = GoogleClient("gemini-explicit", api_key="explicit")
    google_env = GoogleClient("gemini-google")
    monkeypatch.delenv("GOOGLE_API_KEY")
    gemini_env = GoogleClient("gemini-fallback")
    monkeypatch.delenv("GEMINI_API_KEY")
    no_key = GoogleClient("gemini-none")

    assert explicit.model == "gemini-explicit"
    assert explicit.provider == "google"
    assert google_env.model == "gemini-google"
    assert gemini_env.model == "gemini-fallback"
    assert no_key.model == "gemini-none"
    assert fake_google["init_calls"] == [
        {"api_key": "explicit"},
        {"api_key": "google-env"},
        {"api_key": "gemini-env"},
        {"api_key": None},
    ]


def test_constructor_rejects_missing_model():
    with pytest.raises(ValueError, match="google provider requires 'model'"):
        GoogleClient("")


def test_constructor_reports_missing_optional_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "google", None)
    with pytest.raises(ImportError, match=r"pip install llm-bridge\[google\]"):
        GoogleClient("gemini-test", api_key="fake")


def test_chat_maps_request_response_and_native_finish_reason(fake_google):
    client = GoogleClient("gemini-test", api_key="fake")
    raw = types.SimpleNamespace(
        text="partial",
        usage_metadata=types.SimpleNamespace(prompt_token_count=9, candidates_token_count=3),
        candidates=[types.SimpleNamespace(finish_reason=_FinishReason.MAX_TOKENS)],
    )
    fake_google["control"]["response"] = raw

    response = client.chat(
        [
            {"role": "system", "content": "be precise"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "draft"},
        ],
        temperature=0.2,
        max_tokens=64,
        top_p=0.3,
        top_k=5,
        stop_sequences=["END"],
        seed=7,
    )

    assert fake_google["config_kwargs"] == {
        "temperature": 0.2,
        "max_output_tokens": 64,
        "system_instruction": "be precise",
        "top_p": 0.3,
        "top_k": 5,
        "stop_sequences": ["END"],
        "seed": 7,
    }
    assert [item.role for item in fake_google["contents"]] == ["user", "model"]
    assert [item.parts for item in fake_google["contents"]] == [
        [{"text": "question"}],
        [{"text": "draft"}],
    ]
    assert fake_google["model"] == "gemini-test"
    assert fake_google["calls"] == 1
    assert response.content == "partial"
    assert response.model == "gemini-test"
    assert response.prompt_tokens == 9
    assert response.completion_tokens == 3
    assert response.finish_reason == "max_tokens"
    assert response.latency_ms >= 0
    assert response.raw is raw


def test_chat_handles_absent_text_usage_and_candidate(fake_google):
    client = GoogleClient("gemini-test")
    raw = types.SimpleNamespace(text=None, usage_metadata=None)
    fake_google["control"]["response"] = raw

    response = client.chat([{"role": "user", "content": "question"}])

    assert response.content == ""
    assert response.prompt_tokens == 0
    assert response.completion_tokens == 0
    assert response.finish_reason == "unknown"
    assert response.raw is raw


def test_complete_forwards_google_arguments_and_system_message(fake_google):
    client = GoogleClient("gemini-test")
    response = client.complete(
        "question",
        system="be brief",
        temperature=0.0,
        max_tokens=0,
        stop=["END"],
        seed=0,
    )

    assert response.content == "gemini-reply"
    assert fake_google["config_kwargs"] == {
        "temperature": 0.0,
        "max_output_tokens": 0,
        "system_instruction": "be brief",
        "stop_sequences": ["END"],
        "seed": 0,
    }
    assert [item.role for item in fake_google["contents"]] == ["user"]
    assert [item.parts for item in fake_google["contents"]] == [[{"text": "question"}]]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"unknown": True},
            "Unsupported Google generation arguments: unknown",
        ),
        (
            {"stop": ["A"], "stop_sequences": ["B"]},
            "'stop' and 'stop_sequences' cannot be used together",
        ),
    ],
)
def test_chat_rejects_arguments_before_sdk_construction_or_transport(fake_google, kwargs, message):
    client = GoogleClient("gemini-test")
    with pytest.raises(TypeError) as error:
        client.chat([{"role": "user", "content": "question"}], **kwargs)
    assert str(error.value) == message

    assert fake_google["part_calls"] == 0
    assert fake_google["config_calls"] == 0
    assert fake_google["calls"] == 0


def test_complete_uses_the_same_pre_call_validation(fake_google):
    client = GoogleClient("gemini-test")
    with pytest.raises(TypeError, match="Unsupported Google generation arguments: unknown"):
        client.complete("question", unknown=True)
    assert fake_google["calls"] == 0


def test_sdk_config_validation_and_transport_errors_propagate(fake_google):
    client = GoogleClient("gemini-test")
    with pytest.raises(_ConfigError, match="invalid top_p"):
        client.chat([{"role": "user", "content": "question"}], top_p="invalid")
    assert fake_google["calls"] == 0

    fake_google["control"]["transport_error"] = RuntimeError("transport failed")
    with pytest.raises(RuntimeError, match="transport failed"):
        client.chat([{"role": "user", "content": "question"}])


def test_public_signature_is_unchanged():
    assert str(inspect.signature(GoogleClient.chat)) == (
        "(self, messages: 'List[Message]', *, temperature: 'float' = 0.7, "
        "max_tokens: 'int' = 1024, **kwargs: 'Any') -> 'LLMResponse'"
    )


def test_build_validates_and_constructs_client(fake_google):
    with pytest.raises(ValueError, match="google provider requires 'model'"):
        build({})

    client = build({"model": "gemini-test", "api_key": "fake"})
    assert isinstance(client, GoogleClient)
    assert client.model == "gemini-test"
