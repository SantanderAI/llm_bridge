# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""OpenAI provider.

Requires:
    pip install "llm-bridge[openai]"
    export OPENAI_API_KEY=...

Run:
    python examples/openai_example.py
"""

import os

from llm_bridge import create_llm


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        return

    try:
        llm = create_llm(
            {"provider": "openai", "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini")}
        )
    except ImportError as exc:
        print(exc)
        return

    resp = llm.complete("Say hello in one short sentence.", system="You are friendly and concise.")
    print(resp.content)
    print("tokens:", resp.total_tokens)


if __name__ == "__main__":
    main()
