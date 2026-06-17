# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""AWS Bedrock provider (Converse API).

Requires:
    pip install "llm-bridge[aws]"
    AWS credentials configured (env vars, shared config, or an IAM role)

Run:
    python examples/bedrock_example.py
"""

import os

from llm_bridge import create_llm


def main() -> None:
    model = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
    region = os.environ.get("AWS_REGION", "us-east-1")

    try:
        llm = create_llm({"provider": "bedrock", "model": model, "region": region})
    except ImportError as exc:
        print(exc)
        return

    resp = llm.complete("Summarize the benefit of a vendor-neutral LLM wrapper in one sentence.")
    print(resp.content)
    print("tokens:", resp.total_tokens)


if __name__ == "__main__":
    main()
