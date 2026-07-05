# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Flask microservice — multi-provider LLM API powered by llm_bridge.

Three providers, two concrete use-case endpoints (+ generic chat):

  POST /chat       — multi-provider chat (mock / callable / openai)
  POST /summarize  — extractive summarisation with a structured prompt
  GET  /health     — readiness check

The ``callable`` provider wraps a local text analyser (counts words, detects
language, estimates reading time) — no network or API key needed.

Requires:
    pip install llm-bridge flask
    # Optional remote providers:
    pip install "llm-bridge[openai]"
    export OPENAI_API_KEY=...

Run:
    python examples/flask_microservice.py
"""

import os
import re
import time

from flask import Flask, jsonify, request

from llm_bridge import create_llm

app = Flask(__name__)
_PROVIDERS_SUPPORTED = ("mock", "callable", "openai")


# ---------------------------------------------------------------------------
# Callable — text analyser (zero dependencies, no network)
# ---------------------------------------------------------------------------


def _word_count(text: str) -> int:
    return len(text.split())


def _char_count(text: str) -> int:
    return len(text)


def _sentence_count(text: str) -> int:
    return max(1, len(re.split(r"[.!?]+", text)) - 1)


def _reading_time_sec(word_count: int) -> float:
    return round(word_count / 3.0, 1)  # ~180 wpm → 3 words/sec


def _detect_common_language(text: str) -> str:
    """Heuristic language detection based on stop-word frequency."""
    lang_map = {
        "en": {"the", "is", "and", "in", "of", "to", "it", "that", "this", "with"},
        "es": {"el", "la", "los", "las", "de", "en", "y", "que", "es", "por"},
        "pt": {"o", "a", "os", "as", "de", "em", "e", "que", "do", "da"},
        "fr": {"le", "la", "les", "de", "et", "est", "dans", "que", "pour", "pas"},
        "de": {"der", "die", "das", "ist", "und", "in", "den", "von", "zu", "mit"},
    }
    words = {w.lower() for w in re.findall(r"\w+", text)}
    scores = {lang: len(words & stops) for lang, stops in lang_map.items()}
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "unknown"


def text_analyser_backend(messages, temperature=0.7, max_tokens=1024, **kwargs):
    """Analyse the last user message: counts, reading time, language.

    No LLM needed — useful for quick validation and testing without
    any cloud dependency.
    """
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        "",
    )
    wc = _word_count(last_user)
    cc = _char_count(last_user)
    sc = _sentence_count(last_user)
    rt = _reading_time_sec(wc)
    lang = _detect_common_language(last_user)

    return (
        f"[text-analyser]\n"
        f"  words:       {wc}\n"
        f"  characters:  {cc}\n"
        f"  sentences:   {sc}\n"
        f"  reading:     {rt}s\n"
        f"  language:    {lang}"
    )


# ---------------------------------------------------------------------------
# LLM builder
# ---------------------------------------------------------------------------


def _build_llm(data: dict):
    """Parse JSON body and return an LLMClient for the requested provider."""
    provider = data.get("provider", "mock")
    config: dict = {"provider": provider}

    if provider == "callable":
        config["callable"] = text_analyser_backend

    elif provider == "openai":
        config["model"] = data.get(
            "model",
            os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        )

    elif provider == "mock":
        pass

    else:
        raise ValueError(
            f"Unsupported provider '{provider}'. " f"Supported: {', '.join(_PROVIDERS_SUPPORTED)}."
        )

    return create_llm(config)


def _error(msg: str, status: int = 400):
    return jsonify({"error": msg}), status


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    """Generic multi-provider chat. Provider selected per-request."""
    data = request.get_json(force=True)
    if not data:
        return _error("Request body is required")

    message = data.get("message", "")
    system = data.get("system", "You are helpful.")
    provider = data.get("provider", "mock")

    if not message:
        return _error("'message' field is required")

    try:
        llm = _build_llm(data)
    except (ImportError, ValueError) as exc:
        return _error(str(exc), 400)

    start = time.perf_counter()
    try:
        resp = llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": message},
            ]
        )
        elapsed = round((time.perf_counter() - start) * 1000, 1)

        return jsonify(
            {
                "provider": provider,
                "model": resp.model,
                "content": resp.content,
                "tokens": resp.total_tokens,
                "latency_ms": elapsed,
            }
        )
    except Exception as exc:
        return _error(str(exc), 502)


@app.route("/summarize", methods=["POST"])
def summarize():
    """Concrete use case: summarise a longer text.

    Uses a structured prompt that asks for a JSON-like summary with
    key points, sentiment, and reading level.  Demonstrates prompt
    engineering with the ``system`` field.
    """
    data = request.get_json(force=True)
    if not data:
        return _error("Request body is required")

    text = data.get("text", "")
    provider = data.get("provider", "openai")
    max_words = data.get("max_words", 60)

    if not text or len(text) < 20:
        return _error("'text' must be at least 20 characters")

    try:
        llm = _build_llm({"provider": provider})
    except (ImportError, ValueError) as exc:
        return _error(str(exc), 400)

    prompt = (
        f"Summarise the following text in at most {max_words} words.\n"
        "Return your answer as plain JSON with keys: "
        "summary, keywords (list of 3-5), sentiment (positive/neutral/negative), "
        "and reading_level (beginner/intermediate/advanced).\n"
        "Do not wrap in markdown code blocks.\n\n"
        f"---\n{text}\n---"
    )

    start = time.perf_counter()
    try:
        resp = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a precise summarisation assistant. "
                        "Always respond with valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )
        elapsed = round((time.perf_counter() - start) * 1000, 1)

        return jsonify(
            {
                "provider": provider,
                "model": resp.model,
                "summary_raw": resp.content,
                "tokens": resp.total_tokens,
                "latency_ms": elapsed,
            }
        )
    except Exception as exc:
        return _error(str(exc), 502)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_usage():
    print("llm_bridge Flask microservice")
    print("=" * 35)
    print("  GET  /health")
    print("  POST /chat       {'provider': 'mock', 'message': '...'}")
    print("  POST /summarize  {'text': '...', 'provider': 'openai'}")
    print()
    print("Providers: mock, callable, openai")
    print()
    print("Examples:")
    print("  curl http://localhost:5000/chat -X POST \\")
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"provider": "mock", "message": "Tell me a joke"}\'')
    print()
    print("  curl http://localhost:5000/summarize -X POST \\")
    print('    -H "Content-Type: application/json" \\')
    print("    -d" ' \'{"text": "Long article text here...", "provider": "mock"}\'')


def main() -> None:
    _print_usage()
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
