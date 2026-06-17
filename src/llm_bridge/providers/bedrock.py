# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""AWS Bedrock provider using the unified Converse API.

Optional dependency — requires ``pip install llm-bridge[aws]``.
The Converse API is model-agnostic on Bedrock (Anthropic Claude, Meta Llama,
Mistral, Amazon Titan, Cohere, ...), so a single client covers all families.

Configuration (config keys):
    model          (required)  Bedrock model id or inference-profile id
    region         default: us-east-1
    profile_name   optional AWS profile
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from llm_bridge.base import LLMClient, LLMResponse, Message, split_system_messages


class BedrockClient(LLMClient):
    """Chat client backed by the AWS Bedrock Converse API."""

    def __init__(
        self,
        model: str,
        region: str = "us-east-1",
        profile_name: Optional[str] = None,
    ):
        if not model:
            raise ValueError("bedrock provider requires 'model'.")
        self._model = model
        self._region = region

        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'bedrock' provider requires boto3. "
                "Install it with: pip install llm-bridge[aws]"
            ) from exc

        session_kwargs: Dict[str, Any] = {"region_name": region}
        if profile_name:
            session_kwargs["profile_name"] = profile_name
        session = boto3.Session(**session_kwargs)
        self._client = session.client("bedrock-runtime", region_name=region)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "bedrock"

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        system_text, turns = split_system_messages(messages)
        system = [{"text": system_text}] if system_text else []
        bedrock_messages = [
            {
                "role": "assistant" if m["role"] == "assistant" else "user",
                "content": [{"text": m["content"]}],
            }
            for m in turns
        ]

        inference_config: Dict[str, Any] = {
            "maxTokens": max_tokens,
            "temperature": temperature,
        }
        if "top_p" in kwargs:
            inference_config["topP"] = kwargs["top_p"]

        start = time.perf_counter() * 1000
        resp = self._client.converse(
            modelId=self._model,
            system=system,
            messages=bedrock_messages,
            inferenceConfig=inference_config,
        )
        latency = time.perf_counter() * 1000 - start

        blocks = resp.get("output", {}).get("message", {}).get("content", [])
        content = blocks[0].get("text", "") if blocks else ""
        usage = resp.get("usage", {})
        return LLMResponse(
            content=content,
            model=self._model,
            prompt_tokens=usage.get("inputTokens", 0),
            completion_tokens=usage.get("outputTokens", 0),
            finish_reason=resp.get("stopReason", "stop"),
            latency_ms=latency,
            raw=resp,
        )


def build(config: Dict[str, Any]) -> BedrockClient:
    """Build a :class:`BedrockClient` from a config dict."""
    model = config.get("model")
    if not model:
        raise ValueError("bedrock provider requires 'model'.")
    return BedrockClient(
        model=model,
        region=config.get("region", "us-east-1"),
        profile_name=config.get("profile_name"),
    )
