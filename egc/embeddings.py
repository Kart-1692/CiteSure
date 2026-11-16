"""Dense embedders. The default is a sentence-transformers model; `hashing` is a
dependency-free stand-in so the test-suite runs offline (do not report results with it)."""
from __future__ import annotations

import hashlib
import math

from .textutil import content_tokens


class Embedder:
    name = "base"

    def encode(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, kept lazy

        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
        return [v.tolist() for v in vecs]


class HashingEmbedder(Embedder):
    """Signed feature hashing of unigrams and bigrams into a fixed-size unit vector."""
    name = "hashing"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            toks = content_tokens(text)
            grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            vec = [0.0] * self.dim
            for g in grams:
                h = int.from_bytes(hashlib.md5(g.encode()).digest()[:8], "little")
                vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def get_embedder(name: str) -> Embedder:
    return HashingEmbedder() if name == "hashing" else SentenceTransformerEmbedder(name)
