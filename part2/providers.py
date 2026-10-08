"""Every model choice lives here, so the rest of Part 2 never names a vendor.

Search and evals need NO key: embeddings default to a small local model (all-MiniLM-L6-v2,
run through ONNX by chromadb, ~80 MB, downloaded once by `python part2/shelf.py --warm`).

The agent needs a chat model with tool calling. Any OpenAI-compatible endpoint works:

    LLM_PROVIDER=ollama      # local, no key  (ollama pull llama3.1)
    LLM_PROVIDER=openai      # OPENAI_API_KEY
    LLM_PROVIDER=gateway     # Vercel AI Gateway, AI_GATEWAY_API_KEY
    LLM_PROVIDER=openrouter  # OPENROUTER_API_KEY
    LLM_PROVIDER=groq        # GROQ_API_KEY
    LLM_PROVIDER=custom      # LLM_BASE_URL + LLM_API_KEY, e.g. LM Studio, vLLM, Together

LLM_MODEL overrides the preset's default model. Embeddings follow the same idea:
EMBED_PROVIDER=local (default) or EMBED_PROVIDER=openai with EMBED_BASE_URL / EMBED_API_KEY / EMBED_MODEL.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from llama_index.core.base.embeddings.base import BaseEmbedding


@dataclass(frozen=True)
class Preset:
    base_url: str | None
    key_env: str | None
    model: str


PRESETS: dict[str, Preset] = {
    "openai": Preset(None, "OPENAI_API_KEY", "gpt-4o-mini"),
    "gateway": Preset("https://ai-gateway.vercel.sh/v1", "AI_GATEWAY_API_KEY", "openai/gpt-4o-mini"),
    "openrouter": Preset("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai/gpt-4o-mini"),
    "groq": Preset("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "ollama": Preset("http://localhost:11434/v1", None, "llama3.1"),
    "custom": Preset(None, "LLM_API_KEY", ""),
}


class ConfigError(RuntimeError):
    """Raised with a message a student can act on, instead of a stack trace from deep in a SDK."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str | None
    api_key: str
    model: str


def llm_config() -> LLMConfig:
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower() or _guess_provider()
    if provider not in PRESETS:
        raise ConfigError(f"LLM_PROVIDER={provider!r} is not one of {sorted(PRESETS)}")
    preset = PRESETS[provider]
    base_url = os.environ.get("LLM_BASE_URL") or preset.base_url
    model = os.environ.get("LLM_MODEL") or preset.model
    if provider == "custom" and not (base_url and model):
        raise ConfigError("LLM_PROVIDER=custom needs LLM_BASE_URL and LLM_MODEL")
    if preset.key_env is None:
        # Local servers ignore the key, but the OpenAI SDK refuses an empty one.
        api_key = os.environ.get("LLM_API_KEY", "local")
    else:
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get(preset.key_env, "")
        if not api_key:
            raise ConfigError(f"LLM_PROVIDER={provider} needs {preset.key_env} (or LLM_API_KEY). See part2/README.md.")
    return LLMConfig(provider, base_url, api_key, model)


def _guess_provider() -> str:
    """Pick whichever key the student already has, so `python part2/agent.py` just works."""
    for name in ("openai", "gateway", "openrouter", "groq"):
        key_env = PRESETS[name].key_env
        if key_env and os.environ.get(key_env):
            return name
    if os.environ.get("LLM_BASE_URL"):
        return "custom"
    return "ollama"


def chat_client():
    from openai import OpenAI

    cfg = llm_config()
    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=60, max_retries=3)
    return client, cfg


# ---------- embeddings ----------

LOCAL_EMBED_MODEL = "local:all-MiniLM-L6-v2"


def embed_model_name() -> str:
    """Goes into the index fingerprint, so switching embedders always rebuilds the index."""
    if _embed_provider() == "local":
        return LOCAL_EMBED_MODEL
    return f"openai-compatible:{os.environ.get('EMBED_BASE_URL', 'default')}:{_remote_embed_model()}"


def make_embed_model() -> BaseEmbedding:
    if _embed_provider() == "local":
        return LocalMiniLM()
    from llama_index.embeddings.openai import OpenAIEmbedding

    key = os.environ.get("EMBED_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ConfigError("EMBED_PROVIDER=openai needs EMBED_API_KEY (or OPENAI_API_KEY)")
    # model_name (not model) skips LlamaIndex's enum check, so any OpenAI-compatible server works.
    return OpenAIEmbedding(
        model_name=_remote_embed_model(),
        api_key=key,
        api_base=os.environ.get("EMBED_BASE_URL"),
        max_retries=3,
        timeout=30,
    )


def _embed_provider() -> str:
    provider = os.environ.get("EMBED_PROVIDER", "local").strip().lower()
    if provider not in ("local", "openai"):
        raise ConfigError(f"EMBED_PROVIDER={provider!r} must be 'local' or 'openai'")
    return provider


def _remote_embed_model() -> str:
    return os.environ.get("EMBED_MODEL", "text-embedding-3-small")


class LocalMiniLM(BaseEmbedding):
    """all-MiniLM-L6-v2 through chromadb's bundled ONNX runtime: no key, no torch, runs on a laptop CPU."""

    def __init__(self, **kwargs) -> None:
        super().__init__(model_name=LOCAL_EMBED_MODEL, embed_batch_size=64, **kwargs)

    @property
    def _fn(self):
        # Built lazily and cached on the class: pydantic models reject ad-hoc instance attributes.
        if not hasattr(LocalMiniLM, "_onnx"):
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            LocalMiniLM._onnx = DefaultEmbeddingFunction()
        return LocalMiniLM._onnx

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._fn(texts)]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed([query])[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)
