import os
import unittest
from unittest.mock import patch

from providers import ConfigError, embed_model_name, llm_config

KEYS = ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "OPENAI_API_KEY", "AI_GATEWAY_API_KEY",
        "OPENROUTER_API_KEY", "GROQ_API_KEY", "EMBED_PROVIDER", "EMBED_MODEL", "EMBED_BASE_URL")


def env(**values):
    clean = {k: v for k, v in os.environ.items() if k not in KEYS}
    clean.update(values)
    return patch.dict(os.environ, clean, clear=True)


class LLMConfig(unittest.TestCase):
    def test_no_keys_falls_back_to_local_ollama(self):
        with env():
            cfg = llm_config()
        self.assertEqual(cfg.provider, "ollama")
        self.assertIn("11434", cfg.base_url)

    def test_guesses_provider_from_whichever_key_exists(self):
        with env(GROQ_API_KEY="g"):
            cfg = llm_config()
        self.assertEqual((cfg.provider, cfg.api_key), ("groq", "g"))

    def test_explicit_provider_without_key_is_a_clear_error(self):
        with env(LLM_PROVIDER="openrouter"), self.assertRaisesRegex(ConfigError, "OPENROUTER_API_KEY"):
            llm_config()

    def test_custom_endpoint_and_model_override(self):
        with env(LLM_PROVIDER="custom", LLM_BASE_URL="http://localhost:1234/v1", LLM_API_KEY="x", LLM_MODEL="qwen"):
            cfg = llm_config()
        self.assertEqual((cfg.base_url, cfg.model), ("http://localhost:1234/v1", "qwen"))

    def test_unknown_provider(self):
        with env(LLM_PROVIDER="skynet"), self.assertRaises(ConfigError):
            llm_config()


class EmbedConfig(unittest.TestCase):
    def test_default_is_local_and_switching_changes_the_fingerprint_input(self):
        with env():
            local = embed_model_name()
        with env(EMBED_PROVIDER="openai", EMBED_MODEL="text-embedding-3-large"):
            remote = embed_model_name()
        self.assertTrue(local.startswith("local:"))
        self.assertNotEqual(local, remote)


if __name__ == "__main__":
    unittest.main()
