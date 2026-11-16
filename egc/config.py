"""All tunable settings in one place. Every value can be overridden from the
environment (or a .env file), so experiments are reproducible from a config dump."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:  # .env is optional
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _env(name: str, default, cast=str):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return cast(raw)


@dataclass
class Settings:
    # storage
    store_dir: str = field(default_factory=lambda: _env("EGC_STORE_DIR", "store"))

    # chunking (words)
    chunk_words: int = field(default_factory=lambda: _env("EGC_CHUNK_WORDS", 180, int))
    chunk_overlap: int = field(default_factory=lambda: _env("EGC_CHUNK_OVERLAP", 40, int))

    # embeddings: a sentence-transformers model name, or "hashing" (offline, tests only)
    embedder: str = field(
        default_factory=lambda: _env("EGC_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # retrieval
    n_candidates: int = field(default_factory=lambda: _env("EGC_N_CANDIDATES", 20, int))
    rrf_k: int = field(default_factory=lambda: _env("EGC_RRF_K", 60, int))
    top_k_ask: int = field(default_factory=lambda: _env("EGC_TOP_K_ASK", 6, int))
    top_k_explain: int = field(default_factory=lambda: _env("EGC_TOP_K_EXPLAIN", 8, int))
    top_k_summarize: int = field(default_factory=lambda: _env("EGC_TOP_K_SUMMARIZE", 10, int))
    max_per_paper: int = field(default_factory=lambda: _env("EGC_MAX_PER_PAPER", 4, int))

    # generator / verifier LLM (any OpenAI-compatible chat-completions endpoint)
    llm_provider: str = field(default_factory=lambda: _env("EGC_LLM_PROVIDER", "openai_compat"))
    llm_base_url: str = field(
        default_factory=lambda: _env("EGC_LLM_BASE_URL", "http://localhost:11434/v1")
    )
    llm_api_key: str = field(default_factory=lambda: _env("EGC_LLM_API_KEY", "not-needed"))
    llm_model: str = field(default_factory=lambda: _env("EGC_LLM_MODEL", ""))
    llm_json_mode: bool = field(default_factory=lambda: _env("EGC_LLM_JSON_MODE", True, bool))
    llm_min_interval: float = field(default_factory=lambda: _env("EGC_LLM_MIN_INTERVAL", 0.0, float))
    llm_max_tokens: int = field(default_factory=lambda: _env("EGC_LLM_MAX_TOKENS", 2000, int))
    llm_timeout: int = field(default_factory=lambda: _env("EGC_LLM_TIMEOUT", 120, int))

    # verification
    verifier: str = field(default_factory=lambda: _env("EGC_VERIFIER", "llm"))  # llm | nli | lexical
    verify_batch: int = field(default_factory=lambda: _env("EGC_VERIFY_BATCH", 10, int))
    strict_quote: bool = field(default_factory=lambda: _env("EGC_STRICT_QUOTE", True, bool))
    quote_overlap: float = field(default_factory=lambda: _env("EGC_QUOTE_OVERLAP", 0.85, float))
    nli_model: str = field(
        default_factory=lambda: _env("EGC_NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
    )
    nli_threshold: float = field(default_factory=lambda: _env("EGC_NLI_THRESHOLD", 0.5, float))

    def public(self) -> dict:
        d = asdict(self)
        d.pop("llm_api_key", None)
        return d

    @property
    def store_path(self) -> Path:
        return Path(self.store_dir)
