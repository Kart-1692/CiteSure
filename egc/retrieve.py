"""Hybrid retrieval: dense and BM25 rankings fused with reciprocal rank fusion (RRF).

One retrieval stage per request. A request may carry several queries (the fixed facet
queries of Explain / Summarize); every (query, retriever) ranking enters the same fusion.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .schema import Passage
from .store import Store


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """score(d) = sum over rankings of 1 / (k + rank_r(d)), ranks starting at 1."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return dict(scores)


def retrieve(store: Store, queries: list[str] | str, top_k: int, paper_id: Optional[str] = None,
             mode: str = "hybrid", n_candidates: int = 20, rrf_k: int = 60,
             max_per_paper: Optional[int] = None) -> list[Passage]:
    """mode: hybrid | dense | bm25 (the last two exist for the ablation study)."""
    if isinstance(queries, str):
        queries = [queries]
    rankings: list[list[str]] = []
    for q in queries:
        if mode in ("hybrid", "dense"):
            rankings.append([cid for cid, _ in store.dense(q, n_candidates, paper_id)])
        if mode in ("hybrid", "bm25"):
            rankings.append([cid for cid, _ in store.keyword(q, n_candidates, paper_id)])
    fused = sorted(rrf(rankings, rrf_k).items(), key=lambda kv: (-kv[1], kv[0]))

    picked: list[tuple[str, float]] = []
    per_paper: dict[str, int] = defaultdict(int)
    for cid, score in fused:
        pid = cid.split(":")[0]
        if max_per_paper and paper_id is None and per_paper[pid] >= max_per_paper:
            continue
        per_paper[pid] += 1
        picked.append((cid, score))
        if len(picked) == top_k:
            break

    chunks = {c.chunk_id: c for c in store.get_chunks([cid for cid, _ in picked])}
    passages = []
    for cid, score in picked:
        if cid in chunks and chunks[cid].paper_id in store.papers:
            passages.append(Passage(f"S{len(passages) + 1}", chunks[cid], store.papers[chunks[cid].paper_id], score))
    return passages
