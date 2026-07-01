# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Tests for the provider registry (offline)."""

import pytest

from llm_bridge import available_providers, create_llm


def test_builtins_registered():
    names = available_providers()
    for expected in (
        "mock",
        "callable",
        "openai",
        "deepseek",
        "qwen",
        "bedrock",
        "aws",
        "google",
        "gemini",
    ):
        assert expected in names


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        create_llm({"provider": "does-not-exist"})


def test_missing_provider_key_raises():
    with pytest.raises(ValueError):
        create_llm({"model": "x"})


def test_overrides_apply():
    llm = create_llm({"provider": "mock"}, model="custom")
    assert llm.model == "custom"


@pytest.mark.parametrize(
    "provider", ["openai", "deepseek", "qwen", "bedrock", "aws", "google", "gemini"]
)
def test_cloud_provider_validates_model_before_sdk(provider):
    # build() validates required fields before importing any optional SDK,
    # so this raises ValueError regardless of whether the SDK is installed.
    with pytest.raises(ValueError):
        create_llm({"provider": provider})
