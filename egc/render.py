"""Turn an Answer into text. Bibliographic details are copied from the stored paper
record; nothing in a citation or reference is written by the language model."""
from __future__ import annotations

from .schema import Answer, Paper

LABEL_NOTE = {
    "finding": "Finding",
    "limitation": "Limitation (stated by the paper)",
    "interpretation": "Interpretation (the system's reading, not stated by the paper)",
}
NO_ANSWER = ("The uploaded papers do not contain enough evidence to answer this. "
             "Nothing is shown rather than an unsupported answer.")


def _initials(name: str) -> str:
    parts = name.replace(".", " ").split()
    if len(parts) < 2:
        return name
    return " ".join(f"{p[0]}." for p in parts[:-1]) + " " + parts[-1]


def ieee_reference(p: Paper) -> str:
    names = [_initials(a) for a in p.authors]
    if len(names) > 6:
        who = f"{names[0]} et al."
    elif len(names) > 2:
        who = ", ".join(names[:-1]) + ", and " + names[-1]
    else:
        who = " and ".join(names)
    parts = [f"{who}, " if who else "", f"\u201c{p.title},\u201d "]
    if p.venue:
        parts.append(f"{p.venue}, ")
    parts.append(f"{p.year}." if p.year else "n.d.")
    if p.doi:
        parts.append(f" doi: {p.doi}.")
    return "".join(parts)


def numbering(answer: Answer) -> tuple[dict[str, int], dict[str, int]]:
    """Evidence numbers per passage tag and reference numbers per paper, in order of first use."""
    by_tag = {p.tag: p for p in answer.passages}
    ev, ref = {}, {}
    for st in answer.statements:
        for tag in st.cites:
            if tag in by_tag and tag not in ev:
                ev[tag] = len(ev) + 1
                ref.setdefault(by_tag[tag].paper.paper_id, len(ref) + 1)
    return ev, ref


def to_markdown(answer: Answer, show_removed: bool = True) -> str:
    if answer.insufficient and not answer.statements:
        body = [NO_ANSWER]
    else:
        ev, ref = numbering(answer)
        by_tag = {p.tag: p for p in answer.passages}
        body, heading = [], None
        for st in answer.statements:
            if st.heading and st.heading != heading:
                heading = st.heading
                body.append(f"\n### {heading}")
            marks = "".join(f"[{ev[t]}]" for t in st.cites if t in ev)
            body.append(f"- **{LABEL_NOTE[st.label].split(' (')[0]}.** {st.text} {marks}")
        body.append("\n**Evidence**")
        quotes = {(s.tag): s.quote for st in answer.statements for s in st.support if s.quote}
        for tag, n in ev.items():
            p = by_tag[tag]
            line = f"[{n}] {p.paper.short()}, {p.chunk.pages}, section \u201c{p.chunk.section or 'n/a'}\u201d (ref. {ref[p.paper.paper_id]})"
            if quotes.get(tag):
                line += f" \u2014 \u201c{quotes[tag]}\u201d"
            body.append(line + "  ")
        body.append("\n**References**")
        papers = {p.paper.paper_id: p.paper for p in answer.passages}
        for pid, n in ref.items():
            body.append(f"{n}. {ieee_reference(papers[pid])}")
    if show_removed and answer.removed:
        body.append(f"\n**Removed by verification ({len(answer.removed)})**")
        body.extend(f"- ~~{st.text}~~ ({st.reason})" for st in answer.removed)
    return "\n".join(body).strip()
