# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Bring your own backend with the `callable` provider.

Replace the body of `my_backend` with a call to your own SDK, gateway, or local
model. llm_bridge never needs to know which backend you use.

Run:
    python examples/callable_example.py
"""

from llm_bridge import create_llm


def my_backend(messages, temperature=0.7, max_tokens=1024, **kwargs):
    # TODO: call your real backend here and return its text output.
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return f"(echo) {last_user}"


def main() -> None:
    llm = create_llm({"provider": "callable", "callable": my_backend, "model": "my-backend"})
    print(llm.complete("Ping?").content)


if __name__ == "__main__":
    main()
