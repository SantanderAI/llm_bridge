# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Prevent the `callable` provider from retrying a backend when the backend
  itself raises `TypeError`, preserving the original exception and avoiding
  duplicate side effects.

### Added
- Open-source readiness scaffolding:
  - `NOTICE`, `CONTRIBUTING.md` (CLA), `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CODEOWNERS`
  - Issue templates (bug, feature) and PR template
  - `pyproject.toml` tooling config (ruff, black, mypy, pytest, coverage)
  - SPDX headers on Python sources and tests
  - GitHub Actions workflows (third-party actions pinned to SHA digests):
    - `ci.yml` — ruff + black + mypy + pytest matrix (3.9–3.12) with Codecov
    - `codeql.yml` — CodeQL SAST (push, PR, weekly cron)
    - `dep-scan.yml` — `pip-audit` (push, PR, daily cron)
    - `license-check.yml` — SPDX header verification + no-required-dependency guard
    - `pattern-check.yml` — internal-pattern scan with allowlist
    - `scorecard.yml` — OpenSSF Scorecard supply-chain analysis
    - `cla.yml` — CLA Assistant Lite
    - `stale.yml` — stale issues/PRs automation
    - `release.yml` — build sdist + wheel and attach to GitHub Releases
  - `.github/dependabot.yml` — monthly Python and GitHub Actions updates
  - README badges, attribution tagline, and Requirements/Contributing/Security/License/Citation sections
  - Offline tests for the cloud provider adapters (OpenAI, Bedrock, Gemini) via stubbed SDKs

## [0.1.0] - 2026-06-12

### Added
- `llm_bridge`: a tiny, vendor-neutral wrapper for any LLM backend
- Single contract `LLMClient` with `chat()` / `complete()` returning a normalised `LLMResponse`
- Dependency-free providers: `mock` (deterministic, offline) and `callable` (bring your own backend)
- Optional vendor adapters behind extras: `openai`, `bedrock`/`aws`, `google`/`gemini`
- Provider registry with eager (stdlib) and lazy (vendor SDK) builders
- `load_config` helper for JSON/YAML configuration

[Unreleased]: https://github.com/SantanderAI/llm_bridge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SantanderAI/llm_bridge/releases/tag/v0.1.0
