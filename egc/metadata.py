"""Bibliographic metadata: enrichment of ingested papers and lookup of free-text
references (used by the evaluation to audit references a plain LLM writes from memory).

Every network call is optional and fails soft: no connection -> None, never an exception.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import requests

from .textutil import normalize, similarity

UA = {"User-Agent": "evidence-grounded-citation/0.1 (academic project)"}
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>{}|\\^`]+)", re.I)
ARXIV_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_ATOM = {"a": "http://www.w3.org/2005/Atom"}


def _get(url: str, params: dict | None = None, timeout: int = 20):
    try:
        r = requests.get(url, params=params, headers=UA, timeout=timeout)
        if r.status_code == 200:
            return r
    except requests.RequestException:
        pass
    return None


# ---------------------------------------------------------------- by identifier
def crossref_by_doi(doi: str) -> Optional[dict]:
    r = _get(f"https://api.crossref.org/works/{doi}")
    if not r:
        return None
    return _from_crossref(r.json().get("message", {}))


def arxiv_by_id(arxiv_id: str) -> Optional[dict]:
    r = _get("https://export.arxiv.org/api/query", {"id_list": arxiv_id})
    if not r:
        return None
    recs = _from_arxiv_feed(r.text)
    return recs[0] if recs else None


def _from_crossref(m: dict) -> Optional[dict]:
    title = normalize(" ".join(m.get("title") or []))
    if not title:
        return None
    authors = [
        normalize(f"{a.get('given', '')} {a.get('family', '')}") for a in m.get("author", []) if a.get("family")
    ]
    year = None
    for key in ("published-print", "published-online", "issued", "created"):
        parts = (m.get(key) or {}).get("date-parts") or [[None]]
        if parts and parts[0] and parts[0][0]:
            year = int(parts[0][0])
            break
    return {
        "title": title, "authors": authors, "year": year, "doi": m.get("DOI", ""),
        "venue": normalize(" ".join(m.get("container-title") or [])), "source": "crossref",
    }


def _from_arxiv_feed(xml_text: str) -> list[dict]:
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for e in root.findall("a:entry", _ATOM):
        title = normalize(e.findtext("a:title", "", _ATOM))
        if not title or title.lower() == "error":
            continue
        pub = e.findtext("a:published", "", _ATOM)
        ident = (e.findtext("a:id", "", _ATOM).rsplit("/abs/", 1) + [""])[1] if "/abs/" in e.findtext("a:id", "", _ATOM) else ""
        out.append({
            "title": title,
            "authors": [normalize(a.findtext("a:name", "", _ATOM)) for a in e.findall("a:author", _ATOM)],
            "year": int(pub[:4]) if pub[:4].isdigit() else None,
            "doi": "", "arxiv_id": re.sub(r"v\d+$", "", ident),
            "venue": f"arXiv:{re.sub(r'v[0-9]+$', '', ident)}" if ident else "arXiv", "source": "arxiv",
        })
    return out


def enrich(meta: dict) -> dict:
    """Replace heuristic fields with registry fields when a DOI or arXiv id is known."""
    rec = None
    if meta.get("doi"):
        rec = crossref_by_doi(meta["doi"])
    if rec is None and meta.get("arxiv_id"):
        rec = arxiv_by_id(meta["arxiv_id"])
    if rec:
        for k in ("title", "authors", "year", "venue"):
            if rec.get(k):
                meta[k] = rec[k]
        meta["doi"] = meta.get("doi") or rec.get("doi", "")
        meta["meta_source"] = rec["source"]
    return meta


def sidecar(pdf_path: Path, meta: dict) -> dict:
    """`paper.pdf` + `paper.meta.json` -> hand-checked fields win over everything else."""
    side = pdf_path.with_suffix(".meta.json")
    if side.exists():
        meta.update({k: v for k, v in json.loads(side.read_text()).items() if v})
        meta["meta_source"] = "sidecar"
    return meta


# ---------------------------------------------------------------- by free text
def search_title(title: str) -> dict:
    """Look a title up in Crossref, OpenAlex and arXiv.

    Returns {"status": "found" | "not_found" | "unchecked", "record": dict | None, "tried": [...]}.
    `unchecked` means no registry could be reached, so nothing can be said about the title.
    """
    title = normalize(title)
    tried, reached = [], False
    for name, fn in (("crossref", _search_crossref), ("openalex", _search_openalex), ("arxiv", _search_arxiv)):
        cands = fn(title)
        tried.append(name)
        if cands is None:
            continue
        reached = True
        best = max(cands, key=lambda c: similarity(title, c["title"]), default=None)
        if best and similarity(title, best["title"]) >= 0.85:
            return {"status": "found", "record": best, "tried": tried}
    return {"status": "not_found" if reached else "unchecked", "record": None, "tried": tried}


def _search_crossref(title: str):
    r = _get("https://api.crossref.org/works", {"query.bibliographic": title, "rows": 5})
    if not r:
        return None
    return [c for c in (_from_crossref(m) for m in r.json().get("message", {}).get("items", [])) if c]


def _search_openalex(title: str):
    r = _get("https://api.openalex.org/works", {"search": title, "per-page": 5})
    if not r:
        return None
    out = []
    for w in r.json().get("results", []):
        if w.get("display_name"):
            out.append({
                "title": normalize(w["display_name"]),
                "authors": [a.get("author", {}).get("display_name", "") for a in w.get("authorships", [])],
                "year": w.get("publication_year"), "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                "venue": "", "source": "openalex",
            })
    return out


def _search_arxiv(title: str):
    q = re.sub(r"[^\w\s]", " ", title)
    r = _get("https://export.arxiv.org/api/query", {"search_query": f'ti:"{q}"', "max_results": 5})
    if not r:
        return None
    return _from_arxiv_feed(r.text)
