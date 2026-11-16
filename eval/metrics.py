"""Metric definitions (statement-level, micro-averaged).

Citation recall / precision follow ALCE (Gao et al., 2023):
  recall    = share of factual statements entailed by the passages they cite
  precision = share of citations that support their statement: the passage entails the statement
              on its own, or the statement loses its support when that passage is taken away
Faithfulness is a support rate against the corpus, computed the same way for every system:
  a statement is faithful if one of the top-k corpus passages retrieved *for that statement*
  entails it (FActScore-style retrieve-then-check; RAGAS-style ratio of supported claims).
The judge is an NLI model that is not used anywhere inside the system under test.
"""
from __future__ import annotations

from egc.retrieve import retrieve
from egc.textutil import normalize, similarity


class Judge:
    """Memoised entailment judge on top of an egc Verifier (NLI by default)."""
    def __init__(self, verifier):
        self.verifier, self.cache = verifier, {}

    def prime(self, pairs):
        todo = list({p for p in pairs if p not in self.cache})
        for p, v in zip(todo, self.verifier.check(todo)):
            self.cache[p] = v.verdict == "full"

    def __call__(self, claim: str, evidence: str) -> bool:
        if (claim, evidence) not in self.cache:
            self.prime([(claim, evidence)])
        return self.cache[(claim, evidence)]


def alce(statements: list[dict], text_of: dict[str, str], judge: Judge) -> dict:
    """statements: [{"text", "cites": [tag, ...]}], factual statements only."""
    pairs = []
    for st in statements:
        for c in st["cites"]:
            pairs.append((st["text"], text_of[c]))
        if len(st["cites"]) > 1:
            pairs.append((st["text"], " ".join(text_of[c] for c in st["cites"])))
    judge.prime(pairs)
    rec_hit = prec_hit = n_cit = 0
    for st in statements:
        cites = st["cites"]
        single = [judge(st["text"], text_of[c]) for c in cites]
        supported = any(single) or (len(cites) > 1 and judge(st["text"], " ".join(text_of[c] for c in cites)))
        rec_hit += bool(supported)
        for i, c in enumerate(cites):
            n_cit += 1
            if not supported:
                continue
            if single[i]:
                prec_hit += 1
            elif len(cites) > 1:
                rest = " ".join(text_of[x] for j, x in enumerate(cites) if j != i)
                prec_hit += not judge(st["text"], rest)   # needed for the joint support
    return {"n_statements": len(statements), "recall_hits": rec_hit, "n_citations": n_cit, "precision_hits": prec_hit}


def corpus_support(texts: list[str], store, judge: Judge, paper_id=None, top_k: int = 5, settings=None) -> list[bool]:
    out = []
    for text in texts:
        kw = dict(n_candidates=settings.n_candidates, rrf_k=settings.rrf_k) if settings else {}
        passages = retrieve(store, text, top_k, paper_id, "hybrid", **kw)
        judge.prime([(text, p.chunk.text) for p in passages])
        out.append(any(judge(text, p.chunk.text) for p in passages))
    return out


def match_corpus(title: str, papers: dict) -> str:
    """paper_id of the stored paper with this title, or ''."""
    best = max(papers.values(), key=lambda p: similarity(title, p.title), default=None)
    return best.paper_id if best and similarity(title, best.title) >= 0.85 else ""


def metadata_error(ref: dict, record: dict) -> bool:
    """A real reference whose year (beyond +-1, preprint vs. proceedings) or first author is wrong."""
    wrong_year = bool(ref.get("year") and record.get("year") and abs(int(ref["year"]) - int(record["year"])) > 1)
    first = normalize(ref.get("authors", "")).replace(" et al.", "").split(",")[0].split(" and ")[0].split()
    surname = first[-1].lower().strip(".") if first else ""
    names = " ".join(record.get("authors") or []).lower()
    wrong_author = bool(surname and len(surname) > 1 and names and surname not in names)
    return wrong_year or wrong_author


def pct(num: float, den: float):
    return round(100.0 * num / den, 1) if den else None
