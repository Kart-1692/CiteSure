"""Plain data containers shared by every stage."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

FACTUAL_LABELS = ("finding", "limitation")
LABELS = ("finding", "interpretation", "limitation")


@dataclass
class Paper:
    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: str = ""
    arxiv_id: str = ""
    venue: str = ""
    source_file: str = ""
    n_pages: int = 0
    n_chunks: int = 0
    meta_source: str = "heuristic"  # heuristic | pdf-metadata | crossref | arxiv | sidecar

    def short(self) -> str:
        """`Surname et al., 2024` style label built only from stored fields."""
        if not self.authors:
            who = self.title[:40]
        else:
            last = self.authors[0].split()[-1]
            who = last if len(self.authors) == 1 else (
                f"{last} and {self.authors[1].split()[-1]}" if len(self.authors) == 2 else f"{last} et al."
            )
        return f"{who}, {self.year}" if self.year else who


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    idx: int
    text: str
    page_start: int
    page_end: int
    section: str = ""

    @property
    def pages(self) -> str:
        return f"p. {self.page_start}" if self.page_start == self.page_end else (
            f"pp. {self.page_start}-{self.page_end}"
        )


@dataclass
class Passage:
    """A retrieved chunk, given a short per-request tag (S1, S2, ...) for the prompt."""
    tag: str
    chunk: Chunk
    paper: Paper
    score: float = 0.0


@dataclass
class Support:
    tag: str
    verdict: str  # full | partial | none | contradicted
    quote: str = ""
    anchored: bool = False


@dataclass
class Statement:
    text: str
    label: str = "finding"
    cites: list[str] = field(default_factory=list)
    heading: str = ""
    status: str = "draft"  # draft | kept | removed
    reason: str = ""
    support: list[Support] = field(default_factory=list)


@dataclass
class Answer:
    mode: str
    query: str
    paper_id: str = ""
    statements: list[Statement] = field(default_factory=list)  # kept after verification
    removed: list[Statement] = field(default_factory=list)
    draft: list[Statement] = field(default_factory=list)       # generator output, before verification
    passages: list[Passage] = field(default_factory=list)
    insufficient: bool = False
    timings: dict = field(default_factory=dict)
    llm_calls: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
