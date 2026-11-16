"""PDF -> (Paper, [Chunk]).

Three things matter for citation quality and all happen here:
  1. the reference list is cut off, so a bibliography entry can never be retrieved as "evidence";
  2. bibliographic fields are read once and stored, so a citation is copied, never generated;
  3. chunks are short, sentence-aligned and overlapping, and remember their pages and section.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pymupdf

from . import metadata as md
from .schema import Chunk, Paper
from .textutil import _ABBREV, normalize

_KNOWN = (
    r"abstract|introduction|related works?|background|preliminaries|literature (?:survey|review)|"
    r"methods?|methodology|approach|proposed (?:method|system|approach|framework)|"
    r"system (?:design|architecture|overview)|architecture|implementation(?: details)?|"
    r"experiments?|experimental (?:setup|results|settings?)|evaluation|analysis|"
    r"results?(?: and (?:discussion|analysis))?|discussions?|ablation(?: stud(?:y|ies))?|"
    r"limitations?(?: and future work)?|threats to validity|ethic(?:s|al) (?:statement|considerations)|"
    r"conclusions?(?: and future work)?|future work|acknowledge?ments?|references|bibliography"
)
_NUM = r"(?:[IVX]{1,5}\.?|\d{1,2}(?:\.\d{1,2}){0,2}\.?|[A-Z]\.)"
_HEAD_KNOWN = re.compile(rf"^(?:{_NUM}\s+)?({_KNOWN})\s*[:.]?$", re.I)
_HEAD_NUMBERED = re.compile(rf"^{_NUM}\s+([A-Z][A-Za-z0-9 ,:&/()'\-]{{2,70}})$")
_NOISE = re.compile(r"^(?:\d{1,4}|page \d+(?: of \d+)?|arXiv:\S+.*)$", re.I)
_AFFIL = re.compile(
    r"universit|institut|department|dept\.?|laborator|\blab\b|college|school|research|"
    r"inc\.|google|microsoft|\bmeta\b|openai|academy|centre|center|@|https?:", re.I
)


def heading_name(line: str) -> str:
    m = _HEAD_KNOWN.match(line)
    if m:
        return m.group(1).strip().title()
    m = _HEAD_NUMBERED.match(line)
    if m and len(line.split()) <= 10 and not line.endswith((".", ",", ";")):
        return m.group(1).strip()
    return ""


def paper_id_for(path: Path) -> str:
    return "p" + hashlib.sha1(path.read_bytes()).hexdigest()[:10]


# ------------------------------------------------------------------ metadata
def _title_by_font(page) -> str:
    rows = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if len(text) > 3 and not text.lower().startswith("arxiv"):
                rows.append((max(s["size"] for s in line["spans"]), line["bbox"][1], text))
    rows = [r for r in rows if r[1] < page.rect.height * 0.5]
    if not rows:
        return ""
    top = max(r[0] for r in rows)
    lines = sorted((r for r in rows if r[0] >= top - 0.6), key=lambda r: r[1])
    return normalize(" ".join(r[2] for r in lines))[:300]


def _authors_heuristic(first_page: str, title: str) -> list[str]:
    head = re.split(r"\babstract\b", first_page, maxsplit=1, flags=re.I)[0]
    out: list[str] = []
    title_tokens = set(title.lower().split())
    for line in head.splitlines():
        line = re.sub(r"[\d*†‡§¶∗]+", " ", line)
        line = re.sub(r"\b(?:Dr|Prof|Mr|Mrs|Ms)\.\s*", "", line).strip(" ,")
        if not line or _AFFIL.search(line) or set(line.lower().split()) <= title_tokens:
            continue
        for name in re.split(r",|;|\band\b|\s{2,}", line):
            name = name.strip()
            words = name.split()
            if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words) and name not in out:
                out.append(name)
    return out[:12]


def guess_metadata(doc, first_pages: list[str]) -> dict:
    info = doc.metadata or {}
    head = "\n".join(first_pages[:2])
    title = normalize(info.get("title") or "")
    source = "pdf-metadata"
    if not (10 <= len(title) <= 300) or re.search(r"\.(pdf|dvi|tex|docx?)$|^untitled|^microsoft word", title, re.I):
        title, source = _title_by_font(doc[0]), "heuristic"
    raw_auth = normalize(info.get("author") or "")
    authors = [a.strip() for a in re.split(r";|,|\band\b", raw_auth) if len(a.strip()) > 2]
    if not authors:
        authors, source = _authors_heuristic(first_pages[0] if first_pages else "", title), "heuristic"
    doi = md.DOI_RE.search(head)
    arx = md.ARXIV_RE.search(head)
    year = None
    stamp = re.search(r"arXiv:\S+\s+\[[\w.\-]+\]\s+\d{1,2}\s+\w{3}\s+((?:19|20)\d{2})", head)
    if stamp:
        year = int(stamp.group(1))
    else:
        years = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", first_pages[0] if first_pages else "")]
        years = [y for y in years if 1950 <= y <= 2100]
        if years:
            year = max(years)
        elif re.search(r"(19|20)\d{2}", info.get("creationDate") or ""):
            year = int(re.search(r"((?:19|20)\d{2})", info["creationDate"]).group(1))
    return {
        "title": title or "Untitled document", "authors": authors, "year": year,
        "doi": doi.group(1).rstrip(".,;)") if doi else "", "arxiv_id": arx.group(1) if arx else "",
        "venue": f"arXiv:{arx.group(1)}" if arx else "", "meta_source": source,
    }


# ------------------------------------------------------------------ text units
def text_units(doc) -> list[tuple[str, int, str]]:
    """Flatten the paper into (word, page, section) triples, stopping at the reference list."""
    lines = [(pno, ln.strip()) for pno, page in enumerate(doc, start=1) for ln in page.get_text("text").splitlines()]
    units: list[tuple[str, int, str]] = []
    section, hyphen_open = "Front matter", False
    for i, (pno, line) in enumerate(lines):
        if not line or _NOISE.match(line):
            continue
        name = heading_name(line)
        if name:
            if name.lower() in {"references", "bibliography"} and i > 0.4 * len(lines):
                break
            section, hyphen_open = name, False
            continue
        words = line.replace("\u00ad", "").split()
        if hyphen_open and units and words and words[0][:1].islower():
            w, p, s = units[-1]
            units[-1] = (w[:-1] + words[0], p, s)
            words = words[1:]
        units.extend((w, pno, section) for w in words)
        hyphen_open = bool(words) and words[-1].endswith("-") and len(words[-1]) > 2
    return units


def _sentence_ends(words: list[str]) -> list[int]:
    ends = []
    for i, w in enumerate(words):
        if not re.search(r"[.?!][\"')\]]*$", w) or w.lower() in _ABBREV or re.fullmatch(r"[A-Z]\.", w):
            continue
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if nxt and not re.match(r"[\"'(\[]?[A-Z0-9]", nxt):
            continue
        ends.append(i + 1)
    if not ends or ends[-1] != len(words):
        ends.append(len(words))
    return ends


def chunk_units(units, paper_id: str, size: int = 180, overlap: int = 40) -> list[Chunk]:
    """Pack whole sentences into ~`size`-word chunks; consecutive chunks of a section share
    their trailing sentences (up to `overlap` words). Chunks never cross a section heading."""
    spans: list[tuple[int, int]] = []  # [start, end) into units
    run_start = 0
    for i in range(1, len(units) + 1):
        if i < len(units) and units[i][2] == units[run_start][2]:
            continue
        words = [u[0] for u in units[run_start:i]]
        sents, prev = [], 0
        for e in _sentence_ends(words):
            for s in range(prev, e, size):                       # hard-split run-on "sentences"
                sents.append((run_start + s, run_start + min(s + size, e)))
            prev = e
        cur: list[tuple[int, int]] = []
        for sent in sents:
            cur_len = sum(b - a for a, b in cur)
            if cur and cur_len + (sent[1] - sent[0]) > size:
                spans.append((cur[0][0], cur[-1][1]))
                keep, kept = [], 0
                for a, b in reversed(cur):                       # sentence-level overlap
                    if kept + (b - a) > overlap:
                        break
                    keep.insert(0, (a, b))
                    kept += b - a
                cur = keep
            cur.append(sent)
        if cur:
            last = (cur[0][0], cur[-1][1])
            if spans and last[1] - last[0] < 25 and units[spans[-1][0]][2] == units[last[0]][2]:
                spans[-1] = (spans[-1][0], last[1])               # fold a tiny tail into its neighbour
            else:
                spans.append(last)
        run_start = i
    chunks = []
    for a, b in spans:
        if b - a < 8:
            continue
        idx = len(chunks)
        chunks.append(Chunk(
            chunk_id=f"{paper_id}:{idx:04d}", paper_id=paper_id, idx=idx,
            text=" ".join(u[0] for u in units[a:b]),
            page_start=units[a][1], page_end=units[b - 1][1], section=units[a][2],
        ))
    return chunks


# ------------------------------------------------------------------ entry point
def ingest_pdf(path: str | Path, chunk_words: int = 180, chunk_overlap: int = 40,
               enrich: bool = False) -> tuple[Paper, list[Chunk]]:
    path = Path(path)
    pid = paper_id_for(path)
    with pymupdf.open(path) as doc:
        first_pages = [doc[i].get_text("text") for i in range(min(2, len(doc)))]
        meta = guess_metadata(doc, first_pages)
        units = text_units(doc)
        n_pages = len(doc)
    if enrich:
        meta = md.enrich(meta)
    meta = md.sidecar(path, meta)
    chunks = chunk_units(units, pid, chunk_words, chunk_overlap)
    if not chunks:
        raise ValueError(f"{path.name}: no extractable text (scanned PDF? run OCR first)")
    paper = Paper(
        paper_id=pid, title=meta["title"], authors=list(meta.get("authors") or []), year=meta.get("year"),
        doi=meta.get("doi", ""), arxiv_id=meta.get("arxiv_id", ""), venue=meta.get("venue", ""),
        source_file=path.name, n_pages=n_pages, n_chunks=len(chunks), meta_source=meta.get("meta_source", "heuristic"),
    )
    return paper, chunks
