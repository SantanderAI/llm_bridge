# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Alibaba Qwen provider.

Requires:
    pip install "llm-bridge[qwen]"
    export DASHSCOPE_API_KEY=...

Run:
    python examples/qwen_example.py
"""

import os

from llm_bridge import create_llm


def main() -> None:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("Set DASHSCOPE_API_KEY to run this example.")
        return

    try:
        llm = create_llm(
            {
                "provider": "qwen",
                "model": os.environ.get("QWEN_MODEL", "qwen-plus"),
            }
        )
    except ImportError as exc:
        print(exc)
        return

    resp = llm.complete("Say hello in one short sentence.", system="You are friendly and concise.")
    print(resp.content)
    print("tokens:", resp.total_tokens)


if __name__ == "__main__":
    main()
