# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Tests for the core abstractions."""

from llm_bridge.base import LLMClient, LLMResponse, split_system_messages


def test_total_tokens():
    r = LLMResponse(content="x", model="m", prompt_tokens=3, completion_tokens=4)
    assert r.total_tokens == 7


def test_split_system_messages():
    messages = [
        {"role": "system", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "system", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    system, turns = split_system_messages(messages)
    assert system == "a\n\nc"
    assert [t["role"] for t in turns] == ["user", "assistant"]


def test_split_system_messages_none():
    system, turns = split_system_messages([{"role": "user", "content": "hi"}])
    assert system is None
    assert len(turns) == 1


def test_complete_builds_messages():
    captured = {}

    class Dummy(LLMClient):
        @property
        def model(self):
            return "m"

        def chat(self, messages, *, temperature=0.7, max_tokens=1024, **kwargs):
            captured["messages"] = messages
            return LLMResponse(content="ok", model="m")

    Dummy().complete("hi", system="sys")
    assert captured["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
