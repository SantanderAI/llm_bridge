# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Print the contents of `[project].dependencies` from pyproject.toml.

Used by the `license-check` workflow to assert the core stays dependency-free:
an empty print means no required runtime dependencies. Vendor SDKs must live
under `[project.optional-dependencies]`.
"""

from __future__ import annotations

import pathlib
import re


def required_dependencies(pyproject: str) -> str:
    match = re.search(r"(?ms)^\s*dependencies\s*=\s*\[(.*?)\]", pyproject)
    body = match.group(1).strip() if match else ""
    return body.replace("\n", " ").strip()


if __name__ == "__main__":
    text = pathlib.Path("pyproject.toml").read_text(encoding="utf-8")
    print(required_dependencies(text))
