"""Optional local LLM summarization using Ollama."""

from __future__ import annotations

import json
import os
from urllib import request


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral:latest"


def summarize_with_ollama(
    text: str,
    model: str | None = None,
    url: str | None = None,
    timeout: float = 60.0,
) -> str:
    """Summarize text using a locally running Ollama model."""
    if not text.strip():
        return ""

    model = model or os.getenv("AUDIO_BRIEF_LLM_MODEL", DEFAULT_MODEL)
    url = url or os.getenv("AUDIO_BRIEF_OLLAMA_URL", DEFAULT_OLLAMA_URL)

    prompt = (
        "Summarize the following transcript clearly and concisely. "
        "Keep the most important facts, decisions, and topics. "
        "Do not invent information.\n\n"
        f"Transcript:\n{text}"
    )

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(req, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result.get("response", "").strip()