"""Persistent index: ChromaDB for dense vectors + metadata, BM25 for keywords.

BM25 documents are the chunk text *plus* the paper's header fields, so a surname, a year or a
DOI typed by the user hits the right paper even when the chunk itself never repeats them.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import chromadb

from .embeddings import Embedder
from .schema import Chunk, Paper
from .textutil import tokenize


class BM25:
    """Okapi BM25 with the always-positive IDF  log(1 + (N - n + 0.5) / (n + 0.5)).

    The classic IDF turns zero or negative for a term that occurs in half of the chunks or more,
    which in a small session-sized corpus is exactly what a paper's own title words, author
    names and DOI do. The positive form keeps those identifiers searchable.
    """
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        self.lengths = [len(d) for d in docs]
        self.avg = (sum(self.lengths) / self.n) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.n
        for term in query:
            for i, tf in self.postings.get(term, ()):
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                scores[i] += self.idf[term] * tf * (self.k1 + 1) / norm
        return scores


class Store:
    def __init__(self, path: str | Path, embedder: Embedder):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(self.path / "chroma"))
        self.col = self.client.get_or_create_collection(
            "chunks", metadata={"hnsw:space": "cosine", "embedder": embedder.name}
        )
        built_with = (self.col.metadata or {}).get("embedder", embedder.name)
        if built_with != embedder.name and self.col.count() > 0:
            raise RuntimeError(
                f"Store at {self.path} was built with embedder '{built_with}', not '{embedder.name}'. "
                "Use the same EGC_EMBEDDER or point EGC_STORE_DIR at a new folder."
            )
        self._registry = self.path / "papers.json"
        self.papers: dict[str, Paper] = {}
        if self._registry.exists():
            self.papers = {k: Paper(**v) for k, v in json.loads(self._registry.read_text()).items()}
        self._bm25: Optional[tuple[BM25, list[Chunk]]] = None

    # ------------------------------------------------------------ write
    def _save(self):
        self._registry.write_text(json.dumps({k: asdict(v) for k, v in self.papers.items()}, indent=2))
        self._bm25 = None

    def add_paper(self, paper: Paper, chunks: list[Chunk]) -> None:
        if paper.paper_id in self.papers:
            self.remove_paper(paper.paper_id)
        for i in range(0, len(chunks), 64):
            batch = chunks[i:i + 64]
            self.col.add(
                ids=[c.chunk_id for c in batch],
                embeddings=self.embedder.encode([c.text for c in batch]),
                documents=[c.text for c in batch],
                metadatas=[{
                    "paper_id": c.paper_id, "idx": c.idx, "page_start": c.page_start,
                    "page_end": c.page_end, "section": c.section,
                } for c in batch],
            )
        self.papers[paper.paper_id] = paper
        self._save()

    def remove_paper(self, paper_id: str) -> None:
        self.col.delete(where={"paper_id": paper_id})
        self.papers.pop(paper_id, None)
        self._save()

    # ------------------------------------------------------------ read
    @staticmethod
    def _chunk(cid: str, doc: str, m: dict) -> Chunk:
        return Chunk(cid, m["paper_id"], int(m["idx"]), doc, int(m["page_start"]), int(m["page_end"]), m.get("section", ""))

    def all_chunks(self, paper_id: Optional[str] = None) -> list[Chunk]:
        got = self.col.get(where={"paper_id": paper_id} if paper_id else None, include=["documents", "metadatas"])
        chunks = [self._chunk(i, d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])]
        return sorted(chunks, key=lambda c: c.chunk_id)

    def get_chunks(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        got = self.col.get(ids=ids, include=["documents", "metadatas"])
        by_id = {i: self._chunk(i, d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}
        return [by_id[i] for i in ids if i in by_id]

    def resolve_paper(self, key: str) -> Paper:
        """Accept a paper id, a file name, or a unique fragment of the title."""
        if key in self.papers:
            return self.papers[key]
        hits = [p for p in self.papers.values()
                if key.lower() in p.title.lower() or key.lower() == p.source_file.lower()
                or key.lower() == Path(p.source_file).stem.lower()]
        if len(hits) == 1:
            return hits[0]
        raise KeyError(f"'{key}' matches {len(hits)} papers; use a paper id from `list`.")

    # ------------------------------------------------------------ search
    def dense(self, query: str, n: int, paper_id: Optional[str] = None) -> list[tuple[str, float]]:
        total = self.col.count()
        if total == 0:
            return []
        res = self.col.query(
            query_embeddings=self.embedder.encode([query]), n_results=min(n, total),
            where={"paper_id": paper_id} if paper_id else None, include=["distances"],
        )
        return [(cid, 1.0 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]

    def keyword(self, query: str, n: int, paper_id: Optional[str] = None) -> list[tuple[str, float]]:
        if self._bm25 is None:
            chunks = self.all_chunks()
            if not chunks:
                return []
            docs = []
            for c in chunks:
                p = self.papers.get(c.paper_id)
                header = f"{p.title} {' '.join(p.authors)} {p.year or ''} {p.doi} {p.arxiv_id}" if p else ""
                docs.append(tokenize(f"{c.text} {header}"))
            self._bm25 = (BM25(docs), chunks)
        bm25, chunks = self._bm25
        scores = bm25.get_scores(tokenize(query))
        ranked = sorted(
            ((c.chunk_id, float(s)) for c, s in zip(chunks, scores)
             if s > 0 and (paper_id is None or c.paper_id == paper_id)),
            key=lambda x: -x[1],
        )
        return ranked[:n]
