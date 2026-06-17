# Contributing to llm_bridge

Thanks for your interest in contributing! `llm_bridge` is a small, vendor-neutral
LLM client library. Contributions of all kinds are welcome: bug reports,
documentation, new provider adapters, and tests.

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug** — open a [bug report issue](.github/ISSUE_TEMPLATE/bug_report.yml).
- **Request a feature** — open a [feature request issue](.github/ISSUE_TEMPLATE/feature_request.yml).
- **Submit a change** — follow the fork-based pull request flow below.
- **Report a vulnerability** — see [SECURITY.md](SECURITY.md) (do **not** open a public issue).

## Development setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -e ".[dev]"
```

## Running the tests

```bash
pytest
```

Tests run fully offline using the `mock` and `callable` providers — no network
access, no credentials, and no vendor SDKs required.

## Adding a new provider

1. Create a module under `src/llm_bridge/providers/` implementing
   `llm_bridge.base.LLMClient` (you only need `model` and `chat`).
2. Expose a `build(config: dict) -> LLMClient` entry point. Validate required
   fields **before** importing any heavy/optional SDK.
3. Register it in `src/llm_bridge/registry.py` — eagerly if it is
   dependency-free, lazily if it needs an optional dependency.
4. If it needs a third-party SDK, add it as an optional extra in
   `pyproject.toml` and import it lazily inside the client (with a clear
   error message pointing at the extra).
5. Add an offline test (stub the transport/SDK where needed).

## Guidelines

- **Keep the core dependency-free.** Required `dependencies` in
  `pyproject.toml` must stay empty; vendor SDKs are always optional extras.
- **No secrets in the repo.** Read credentials from environment variables.

## Pull Request Process

### For external contributors

1. **Fork** the repository to your GitHub account.
2. **Create a branch** from `main` with a descriptive name (e.g. `feat/add-cohere-provider`).
3. **Make your changes** following the code style below.
4. **Add offline tests** for any new functionality (stub the SDK/transport — no network).
5. **Update documentation** if your changes affect the public interface or providers.
6. **Commit** using [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `ci:`, `chore:`).
7. **Push** your branch and open a Pull Request against `main`.
8. **Sign the CLA** when prompted by the CLA Assistant bot.

### For internal contributors (Santander)

1. **Create a branch** from `main` (no fork needed if you are a member of the org).
2. Follow steps 3–7 above and request review from the team in [CODEOWNERS](CODEOWNERS).

### PR requirements

All pull requests must pass the automated checks before merge:

- [ ] **CI** (`ci`) — ruff + black + mypy + pytest matrix (3.9–3.12) with coverage
- [ ] **Security scan** (`codeql`, `dep-scan`) — SAST and dependency audit
- [ ] **License check** (`license-check`) — SPDX headers + core stays dependency-free
- [ ] **Pattern check** (`pattern-check`) — no internal URLs, IPs, or corporate email addresses
- [ ] **CLA signed** (for external contributors)

At least **1 maintainer approval** is required, all conversations resolved, and
the branch up to date with `main`.

## Code style

- [PEP 8](https://peps.python.org/pep-0008/), formatted with [Black](https://black.readthedocs.io/) (line length 100).
- Linted with [Ruff](https://docs.astral.sh/ruff/) and type-checked with [mypy](https://mypy-lang.org/).
- Every source file must start with the SPDX header:
  ```python
  # Copyright (c) 2026 Santander Group
  # SPDX-License-Identifier: Apache-2.0
  ```
- Run the full toolchain before submitting:
  ```bash
  ruff check . && black --check . && mypy src/llm_bridge && pytest
  ```

## Contributor License Agreement (CLA)

By submitting a pull request, you agree to the terms of our Contributor License
Agreement. The [CLA Assistant](https://cla-assistant.io/) bot will check your PR
and ask you to sign if you have not already. The CLA ensures contributions can
be distributed under the project's Apache 2.0 license.
