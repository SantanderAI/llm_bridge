# llm_bridge

> **Open source by Santander AI Lab.** A tiny, vendor-neutral **LLM client library** — one interface for **OpenAI, DeepSeek, Alibaba Qwen, AWS Bedrock and Google Gemini** (or bring your own AI backend).

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/SantanderAI/llm_bridge/actions/workflows/ci.yml/badge.svg)](https://github.com/SantanderAI/llm_bridge/actions/workflows/ci.yml)
[![CodeQL](https://github.com/SantanderAI/llm_bridge/actions/workflows/codeql.yml/badge.svg)](https://github.com/SantanderAI/llm_bridge/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/SantanderAI/llm_bridge/badge)](https://scorecard.dev/viewer/?uri=github.com/SantanderAI/llm_bridge)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196.svg)](https://conventionalcommits.org)

Part of [**Santander AI Open Source**](https://github.com/SantanderAI) — open source AI projects from Banco Santander ([santander.com](https://santander.com)).

A tiny, **vendor-neutral wrapper for any LLM backend**. One small interface,
pluggable providers. Write your application against `LLMClient` once and switch
between OpenAI, DeepSeek, Alibaba Qwen, AWS Bedrock, Google Gemini, a local
server, or your own
internal backend — without touching your code.

- **Canonical interface + thin SDK adapters.** One contract (`LLMClient`) with
  a small, modular adapter per official vendor SDK. The SDKs do the heavy
  lifting (retries, streaming, types); llm_bridge just normalises the calls.
- **Zero required dependencies.** The core (`mock`, `callable`) is pure standard
  library. Each vendor SDK is an optional extra you install on demand.
- **Bring your own backend.** Wrap any function with the `callable` provider —
  the library never needs to know which backend you use.
- **One interface.** `chat(messages)` and `complete(prompt)` returning a
  normalised `LLMResponse` (content + token usage + latency).

## Installation

```bash
pip install llm-bridge                 # core only (no vendor SDKs)
pip install "llm-bridge[openai]"       # + OpenAI SDK
pip install "llm-bridge[deepseek]"     # + OpenAI SDK for DeepSeek
pip install "llm-bridge[qwen]"         # + OpenAI SDK for Qwen/DashScope
pip install "llm-bridge[aws]"          # + AWS Bedrock (boto3)
pip install "llm-bridge[google]"       # + Google Gemini (google-genai)
pip install "llm-bridge[all]"          # everything
```

## Quickstart

```python
from llm_bridge import create_llm

# Offline, no credentials — great for tests and demos.
llm = create_llm({"provider": "mock"})
print(llm.complete("Hello!").content)

# Switch provider by changing one dict — your code stays the same.
llm = create_llm({"provider": "openai", "model": "gpt-4o-mini"})       # needs [openai] + OPENAI_API_KEY
llm = create_llm({"provider": "deepseek", "model": "deepseek-v4-pro"})  # needs [deepseek] + DEEPSEEK_API_KEY
llm = create_llm({"provider": "qwen", "model": "qwen-plus"})            # needs [qwen] + DASHSCOPE_API_KEY
llm = create_llm({"provider": "bedrock", "model": "<bedrock-model-id>"}) # needs [aws] + AWS creds
llm = create_llm({"provider": "google", "model": "gemini-2.5-flash"})   # needs [google] + GOOGLE_API_KEY

resp = llm.chat([
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Name three primary colors."},
])
print(resp.content, resp.total_tokens)
```

## Bring your own LLM

The `callable` provider wraps any function — the recommended way to plug in a
proprietary or internal backend:

```python
from llm_bridge import create_llm

def my_backend(messages, temperature=0.7, max_tokens=1024, **kwargs):
    # call your own SDK / gateway / local model here, return the text
    return "the model output"

llm = create_llm({"provider": "callable", "callable": my_backend})
print(llm.complete("Hi").content)
```

## Providers

| Provider | Name(s) | Dependency |
| --- | --- | --- |
| Mock (offline) | `mock` | none |
| Bring your own | `callable` | none |
| OpenAI (and OpenAI-compatible) | `openai` | `[openai]` |
| DeepSeek | `deepseek` | `[deepseek]` |
| Alibaba Qwen | `qwen` | `[qwen]` |
| AWS Bedrock (Converse) | `bedrock`, `aws` | `[aws]` |
| Google Gemini | `google`, `gemini` | `[google]` |

Credentials are read from environment variables (`OPENAI_API_KEY`,
`DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`, `GOOGLE_API_KEY`/`GEMINI_API_KEY`,
standard AWS credential chain). Never
hardcode secrets.

The `openai` provider also targets any **OpenAI-compatible** endpoint (vLLM,
Ollama, Azure OpenAI, or an internal gateway): pass a `base_url` (or set
`OPENAI_BASE_URL`). For local servers without auth, set a dummy `OPENAI_API_KEY`.

The `deepseek` provider uses DeepSeek's OpenAI-compatible API via the OpenAI
SDK, defaults to `https://api.deepseek.com`, and accepts `base_url` or
`DEEPSEEK_BASE_URL` for compatible endpoints.

The `qwen` provider uses Alibaba Model Studio/DashScope's OpenAI-compatible API
via the OpenAI SDK. It defaults to
`https://dashscope-intl.aliyuncs.com/compatible-mode/v1` and accepts `base_url`
or `DASHSCOPE_BASE_URL` for other regions or workspaces.

## The interface

```python
class LLMClient:
    @property
    def model(self) -> str: ...
    @property
    def provider(self) -> str: ...
    def chat(self, messages, *, temperature=0.7, max_tokens=1024, **kwargs) -> LLMResponse: ...
    def complete(self, prompt, *, system=None, temperature=0.7, max_tokens=1024, **kwargs) -> LLMResponse: ...

@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    latency_ms: float
    raw: Any            # the provider's raw response
    # .total_tokens
```

## Adding a provider

Implement `LLMClient`, expose `build(config) -> LLMClient`, and register it in
`llm_bridge.registry`. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Examples

See [`examples/`](examples): `mock_example.py`, `callable_example.py`,
`openai_example.py`, `deepseek_example.py`, `bedrock_example.py`,
`qwen_example.py`, `bedrock_example.py`, `google_example.py`.

## Requirements

- Python 3.9+
- No required runtime dependencies for the core (`mock`, `callable`).
- Optional vendor SDKs are installed on demand via extras (`[openai]`, `[deepseek]`, `[qwen]`, `[aws]`, `[google]`, `[all]`).

## Contributing

Contributions are welcome! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and
our [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). External contributors will be
asked to sign the CLA (handled automatically by the CLA Assistant bot).

## Security

Please report vulnerabilities privately — see [`SECURITY.md`](SECURITY.md). Do
**not** open a public issue for security reports.

## Citation

If you use this software, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
