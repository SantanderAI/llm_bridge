# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Mock provider — fully offline, no credentials.

Run:
    python examples/mock_example.py
"""

from llm_bridge import create_llm


def main() -> None:
    llm = create_llm({"provider": "mock"})

    print(llm.complete("Hello, who are you?").content)

    resp = llm.chat(
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Name three primary colors."},
        ]
    )
    print(resp.content, "| tokens:", resp.total_tokens)


if __name__ == "__main__":
    main()
