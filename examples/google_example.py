# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Google Gemini provider.

Requires:
    pip install "llm-bridge[google]"
    export GOOGLE_API_KEY=...        # or GEMINI_API_KEY

Run:
    python examples/google_example.py
"""

import os

from llm_bridge import create_llm


def main() -> None:
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        print("Set GOOGLE_API_KEY (or GEMINI_API_KEY) to run this example.")
        return

    try:
        llm = create_llm(
            {"provider": "google", "model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")}
        )
    except ImportError as exc:
        print(exc)
        return

    resp = llm.complete("Say hello in one short sentence.", system="Be concise.")
    print(resp.content)
    print("tokens:", resp.total_tokens)


if __name__ == "__main__":
    main()
