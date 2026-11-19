"""Small URL helpers shared by the embedded memory middlewares.

Mem0 and MemOS talk to Ollama through two different client stacks:

* native ``ollama.Client`` — expects ``http://host:11434`` (no ``/v1``)
* OpenAI-compatible clients — expect ``http://host:11434/v1``

Profiles such as ``ollama_memory`` carry a single ``llm_base_url`` in the
OpenAI-compatible form because Graphiti / EverMemOS / Hindsight hand it to
the OpenAI SDK.  The helpers below let the Mem0 / MemOS middlewares derive
the native endpoint without requiring a second profile key.
"""

from __future__ import annotations


def native_ollama_url(url: str | None) -> str | None:
    """Return the native Ollama endpoint for an OpenAI-compatible URL.

    ``http://localhost:11434/v1`` -> ``http://localhost:11434``.
    URLs that do not end in ``/v1`` are returned unchanged (minus any
    trailing slash), so an explicitly native URL keeps working.
    """
    if not url:
        return None
    return url.rstrip("/").removesuffix("/v1").rstrip("/")
