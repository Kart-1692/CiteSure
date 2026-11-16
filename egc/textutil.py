"""Small text helpers used across stages (no heavy dependencies)."""
from __future__ import annotations

import re

_ABBREV = {
    "al.", "fig.", "figs.", "eq.", "eqs.", "sec.", "no.", "vs.", "e.g.", "i.e.", "cf.", "etc.",
    "dr.", "prof.", "mr.", "ms.", "vol.", "pp.", "ref.", "refs.", "approx.", "resp.", "tab.",
}
_TOKEN = re.compile(r"[a-z0-9]+(?:[./:_-][a-z0-9]+)*")
STOP = set(
    "a an the of in on at to for from by with and or is are was were be been being this that these "
    "those it its as into than then which who whom whose what when where how why can could may might "
    "will would shall should do does did has have had not no nor but if so such we our their they he "
    "she you i also both each more most other some any all between over under about".split()
)


def normalize(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Lower-cased tokens. Compound tokens such as a DOI (10.1234/abc-1) are kept whole
    *and* split, so exact identifiers and their parts are both searchable."""
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        out.append(tok)
        if re.search(r"[./:_-]", tok):
            out.extend(p for p in re.split(r"[./:_-]", tok) if p)
    return out


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOP and len(t) > 1]


def containment(needle: str, haystack: str) -> float:
    """Share of the content tokens of `needle` that also occur in `haystack`."""
    n = content_tokens(needle)
    if not n:
        return 0.0
    h = set(content_tokens(haystack))
    return sum(t in h for t in n) / len(n)


def split_sentences(text: str) -> list[str]:
    words = text.split()
    sents, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        if not re.search(r"[.?!][\"')\]]*$", w):
            continue
        if w.lower() in _ABBREV or re.fullmatch(r"[A-Z]\.", w):
            continue
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if nxt and not re.match(r"[\"'(\[]?[A-Z0-9]", nxt):
            continue
        sents.append(" ".join(cur))
        cur = []
    if cur:
        sents.append(" ".join(cur))
    return sents


def similarity(a: str, b: str) -> float:
    """Symmetric token-set similarity in [0, 1] (used to match titles)."""
    ta, tb = set(content_tokens(a)), set(content_tokens(b))
    if not ta or not tb:
        return 0.0
    return 2 * len(ta & tb) / (len(ta) + len(tb))
