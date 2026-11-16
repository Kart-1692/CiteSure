"""Passage-first generation: the prompt holds only retrieved passages, and the model must
return labelled statements that each name the passage tag(s) they rest on."""
from __future__ import annotations

import re

from . import prompts
from .llm import LLM
from .schema import LABELS, Paper, Passage, Statement


def format_passages(passages: list[Passage]) -> str:
    blocks = []
    for p in passages:
        head = f'[{p.tag}] paper: "{p.paper.title}" ({p.paper.short()}) | {p.chunk.pages} | section: {p.chunk.section or "n/a"}'
        blocks.append(f"{head}\n{p.chunk.text}")
    return "\n\n".join(blocks)


def _clean_statements(raw, valid_tags: set[str]) -> list[Statement]:
    out = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        label = str(item.get("label", "finding")).strip().lower()
        label = label if label in LABELS else "finding"
        cites_raw = item.get("cites") or []
        if isinstance(cites_raw, (str, int)):
            cites_raw = [cites_raw]
        cites = []
        for c in cites_raw:
            m = re.search(r"(\d+)", str(c))
            tag = f"S{m.group(1)}" if m else ""
            if tag in valid_tags and tag not in cites:
                cites.append(tag)                      # a tag that was never in the prompt is dropped
        out.append(Statement(text=text, label=label, cites=cites[:3], heading=str(item.get("heading", "")).strip()))
    return out


def generate(llm: LLM, mode: str, passages: list[Passage], question: str = "", focus: str = "") -> tuple[list[Statement], bool]:
    block = format_passages(passages)
    if mode == "ask":
        user = prompts.ASK_USER.format(passages=block, question=question)
    elif mode == "explain":
        user = prompts.EXPLAIN_USER.format(passages=block, focus=f", focusing on: {focus}" if focus else "")
    elif mode == "summarize":
        user = prompts.SUMMARIZE_USER.format(passages=block)
    else:
        raise ValueError(f"unknown mode: {mode}")
    data = llm.chat_json(prompts.GENERATE_SYSTEM, user, task="generate")
    statements = _clean_statements(data.get("statements"), {p.tag for p in passages})
    return statements, bool(data.get("insufficient")) or not statements


def baseline(llm: LLM, mode: str, question: str = "", paper: Paper | None = None) -> dict:
    """Plain LLM: same request, no documents. Returns its statements and the references it wrote."""
    if mode == "ask":
        user = prompts.BASELINE_ASK.format(question=question)
    else:
        who = f" ({paper.short()})" if paper and paper.authors else ""
        template = prompts.BASELINE_EXPLAIN if mode == "explain" else prompts.BASELINE_SUMMARIZE
        user = template.format(title=paper.title, who=who)
    data = llm.chat_json(prompts.BASELINE_SYSTEM, user, task="baseline")
    refs = {}
    for r in data.get("references") or []:
        if isinstance(r, dict) and str(r.get("title", "")).strip():
            m = re.search(r"\d+", str(r.get("n", "")))
            n = int(m.group()) if m else len(refs) + 1
            year = re.search(r"(?:19|20)\d{2}", str(r.get("year", "")))
            refs[n] = {"n": n, "authors": str(r.get("authors", "")), "title": str(r["title"]).strip(),
                       "venue": str(r.get("venue", "")), "year": int(year.group()) if year else None,
                       "doi": str(r.get("doi", "") or "")}
    statements = []
    for s in data.get("statements") or []:
        if isinstance(s, dict) and str(s.get("text", "")).strip():
            nums = s.get("refs") or []
            nums = [nums] if isinstance(nums, (int, str)) else nums
            cited = [int(m.group()) for m in (re.search(r"\d+", str(n)) for n in nums) if m]
            statements.append({"text": str(s["text"]).strip(), "refs": [n for n in cited if n in refs]})
    return {"statements": statements, "references": list(refs.values())}
