"""The three modes, end to end. Every mode is the same four steps with a different
retrieval scope and prompt:  retrieve -> generate -> verify -> (render)."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Optional

from . import prompts
from .config import Settings
from .embeddings import Embedder, get_embedder
from .generate import baseline as run_baseline
from .generate import generate
from .ingest import ingest_pdf
from .llm import LLM, get_llm
from .retrieve import retrieve
from .schema import Answer, Paper
from .store import Store
from .verify import Verifier, get_verifier, verify_statements


class Engine:
    def __init__(self, settings: Optional[Settings] = None, llm: Optional[LLM] = None,
                 embedder: Optional[Embedder] = None, verifier: Optional[Verifier] = None):
        self.settings = settings or Settings()
        self.embedder = embedder or get_embedder(self.settings.embedder)
        self.store = Store(self.settings.store_path, self.embedder)
        self._llm, self._verifier = llm, verifier

    # lazy, so `ingest` and `list` work without any LLM configured
    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = get_llm(self.settings)
        return self._llm

    @property
    def verifier(self) -> Verifier:
        if self._verifier is None:
            self._verifier = get_verifier(self.settings, self.llm)
        return self._verifier

    # ------------------------------------------------------------ corpus
    def ingest(self, path: str | Path, enrich: bool = False) -> Paper:
        paper, chunks = ingest_pdf(path, self.settings.chunk_words, self.settings.chunk_overlap, enrich)
        self.store.add_paper(paper, chunks)
        return paper

    def papers(self) -> list[Paper]:
        return sorted(self.store.papers.values(), key=lambda p: p.title.lower())

    # ------------------------------------------------------------ modes
    def ask(self, question: str, verify: bool = True) -> Answer:
        return self._run("ask", question, [question], None, self.settings.top_k_ask, verify, question=question)

    def explain(self, paper_key: str, focus: str = "", verify: bool = True) -> Answer:
        paper = self.store.resolve_paper(paper_key)
        queries = [focus] if focus else prompts.EXPLAIN_FACETS
        return self._run("explain", focus or paper.title, queries, paper, self.settings.top_k_explain, verify, focus=focus)

    def summarize(self, paper_key: str, verify: bool = True) -> Answer:
        paper = self.store.resolve_paper(paper_key)
        return self._run("summarize", paper.title, prompts.SUMMARY_FACETS, paper, self.settings.top_k_summarize, verify)

    def baseline(self, mode: str, question: str = "", paper_key: str = "") -> dict:
        paper = self.store.resolve_paper(paper_key) if paper_key else None
        t0 = time.time()
        out = run_baseline(self.llm, mode, question, paper)
        out["seconds"] = round(time.time() - t0, 3)
        return out

    # ------------------------------------------------------------ shared path
    def _run(self, mode, query, queries, paper, top_k, verify, question="", focus="") -> Answer:
        s = self.settings
        ans = Answer(mode=mode, query=query, paper_id=paper.paper_id if paper else "")
        calls0 = self.llm.calls

        t = time.time()
        passages = retrieve(self.store, queries, top_k, paper.paper_id if paper else None, "hybrid",
                            s.n_candidates, s.rrf_k, s.max_per_paper if paper is None else None)
        if paper is not None:                          # single paper: present passages in reading order
            passages.sort(key=lambda p: p.chunk.idx)
            for i, p in enumerate(passages, start=1):
                p.tag = f"S{i}"
        ans.passages = passages
        ans.timings["retrieve"] = round(time.time() - t, 3)
        if not passages:
            ans.insufficient = True
            return ans

        t = time.time()
        draft, ans.insufficient = generate(self.llm, mode, passages, question, focus)
        ans.timings["generate"] = round(time.time() - t, 3)
        ans.draft = copy.deepcopy(draft)

        t = time.time()
        if verify and draft:
            ans.statements, ans.removed = verify_statements(draft, passages, self.verifier, s.strict_quote, s.quote_overlap)
        else:
            ans.statements = draft
        ans.timings["verify"] = round(time.time() - t, 3)
        ans.timings["total"] = round(sum(ans.timings.values()), 3)
        ans.llm_calls = self.llm.calls - calls0
        if not ans.statements:
            ans.insufficient = True                    # nothing survived: say so instead of guessing
        return ans
