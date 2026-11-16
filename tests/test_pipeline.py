"""Offline tests: hashing embedder + mock LLM + synthetic PDFs. No network, no model download.
They check the plumbing and the rules (what gets cited, what gets removed), not answer quality."""
import pymupdf
import pytest

from egc.config import Settings
from egc.embeddings import HashingEmbedder
from egc.ingest import chunk_units, heading_name, ingest_pdf
from egc.llm import MockLLM, parse_json
from egc.pipeline import Engine
from egc.render import ieee_reference, to_markdown
from egc.retrieve import retrieve, rrf
from egc.schema import Chunk, Paper, Passage, Statement
from egc.verify import LexicalVerifier, Verdict, Verifier, anchored, verify_statements

BODY_A = (
    "Calibration of low-cost strain gauges is slow when every sensor is tested by hand. "
    "We propose a batch calibration rig that tests forty gauges in one thermal cycle. "
    "The rig holds the gauges on an aluminium beam and applies a known load with a stepper motor. ") * 6
BODY_B = (
    "Vibration from the spindle shortens the life of ceramic cutting tools. "
    "This study measures tool wear at three spindle speeds on hardened steel. "
    "Tool life fell by thirty percent when the spindle speed doubled. ") * 6


def make_pdf(path, title, author, body, doi=""):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(60, 50, 540, 90), title, fontsize=17)
    page.insert_textbox(pymupdf.Rect(60, 95, 540, 115), author, fontsize=11)
    text = f"Abstract\n{body[:300]}\n" + (f"DOI: {doi}\n" if doi else "") + f"1 Introduction\n{body}\n2 Method\n{body}"
    page.insert_textbox(pymupdf.Rect(60, 125, 540, 760), text, fontsize=8)
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(60, 50, 540, 760),
                        f"3 Results\n{body}\nReferences\n[1] X. Zebra, Quokka handling manual, 1999.", fontsize=8)
    doc.set_metadata({})
    doc.save(path)
    doc.close()


@pytest.fixture()
def engine(tmp_path):
    make_pdf(tmp_path / "a.pdf", "Batch Calibration of Strain Gauges", "Asha Rao, Vikram Nair", BODY_A, "10.1234/gauge.2024.17")
    make_pdf(tmp_path / "b.pdf", "Spindle Vibration and Ceramic Tool Wear", "Meera Iyer", BODY_B)
    s = Settings()
    s.store_dir, s.embedder, s.llm_provider, s.verifier = str(tmp_path / "store"), "hashing", "mock", "llm"
    eng = Engine(s, llm=MockLLM(), embedder=HashingEmbedder())
    eng.ingest(tmp_path / "a.pdf")
    eng.ingest(tmp_path / "b.pdf")
    return eng


def test_ingest_metadata_sections_and_reference_cut(engine):
    papers = {p.source_file: p for p in engine.papers()}
    a = papers["a.pdf"]
    assert a.title == "Batch Calibration of Strain Gauges"
    assert a.doi == "10.1234/gauge.2024.17"
    assert "Asha Rao" in a.authors
    chunks = engine.store.all_chunks(a.paper_id)
    assert {"Introduction", "Method", "Results"} <= {c.section for c in chunks}
    assert not any("Quokka" in c.text for c in chunks), "reference list must not be indexed"
    assert all(len(c.text.split()) <= 230 for c in chunks)


def test_chunks_overlap_and_headings():
    words = ("Alpha beta gamma delta epsilon zeta eta theta iota kappa. " * 60).split()
    chunks = chunk_units([(w, 1, "Method") for w in words], "p1", size=100, overlap=30)
    assert len(chunks) > 3
    assert chunks[1].text.split()[:10] == chunks[0].text.split()[-30:][:10]
    assert heading_name("III. METHODOLOGY") == "Methodology"
    assert heading_name("4.2 Ablation Study") == "Ablation Study"
    assert heading_name("We ran 4 trials.") == ""


def test_rrf_and_keyword_finds_identifiers(engine):
    assert rrf([["a", "b"], ["b", "c"]], k=60)["b"] > rrf([["a", "b"], ["b", "c"]], k=60)["a"]
    by_doi = retrieve(engine.store, "10.1234/gauge.2024.17", 3, mode="bm25")
    assert by_doi and by_doi[0].paper.source_file == "a.pdf"
    by_name = retrieve(engine.store, "Iyer", 3, mode="bm25")
    assert by_name and by_name[0].paper.source_file == "b.pdf"
    hybrid = retrieve(engine.store, "tool life at higher spindle speed", 4)
    assert hybrid[0].paper.source_file == "b.pdf" and [p.tag for p in hybrid] == ["S1", "S2", "S3", "S4"]


def test_ask_removes_unsupported_claim_and_cites_from_stored_fields(engine):
    ans = engine.ask("How many strain gauges does the calibration rig test in one thermal cycle?")
    assert ans.statements and not ans.insufficient
    assert any(MockLLM.FAKE == s.text for s in ans.removed)
    assert all(MockLLM.FAKE != s.text for s in ans.statements)
    assert all(s.cites for s in ans.statements)
    assert all(sup.anchored for s in ans.statements if s.label != "interpretation" for sup in s.support if sup.verdict == "full")
    assert len(ans.draft) == len(ans.statements) + len(ans.removed)
    text = to_markdown(ans)
    assert "A. Rao and V. Nair" in text and "10.1234/gauge.2024.17" in text and "Removed by verification" in text


def test_explain_and_summarize_stay_inside_one_paper(engine):
    b = next(p for p in engine.papers() if p.source_file == "b.pdf")
    for ans in (engine.explain(b.paper_id), engine.summarize("Spindle Vibration")):
        assert ans.passages and {p.paper.paper_id for p in ans.passages} == {b.paper_id}
        assert [p.chunk.idx for p in ans.passages] == sorted(p.chunk.idx for p in ans.passages)
        assert ans.statements


def test_out_of_corpus_question_is_declined(engine):
    ans = engine.ask("Which enzyme unwinds DNA during replication in yeast?")
    assert ans.insufficient and not ans.statements
    assert "do not contain enough evidence" in to_markdown(ans)


class Scripted(Verifier):
    def __init__(self, verdicts):
        self.verdicts = verdicts

    def check(self, pairs):
        return [self.verdicts.pop(0) for _ in pairs]


def _passages():
    paper = Paper("p1", "T", ["Ann Lee"], 2024)
    texts = ["The motor draws two amperes at full load.", "Efficiency was ninety percent in the bench test."]
    return [Passage(f"S{i + 1}", Chunk(f"p1:{i}", "p1", i, t, 1, 1, "Results"), paper) for i, t in enumerate(texts)]


def test_verification_rules():
    q1, q2 = "The motor draws two amperes at full load.", "Efficiency was ninety percent"
    statements = [
        Statement("Full with real quote.", "finding", ["S1", "S2"]),      # keeps S1 only
        Statement("Full but invented quote.", "finding", ["S1"]),         # downgraded -> removed
        Statement("Two partials, jointly full.", "finding", ["S1", "S2"]),
        Statement("Contradicted.", "limitation", ["S2"]),
        Statement("Interpretation with partial basis.", "interpretation", ["S2"]),
        Statement("Interpretation with no basis.", "interpretation", ["S1"]),
        Statement("No citation at all.", "finding", []),
    ]
    verdicts = [
        Verdict("full", q1), Verdict("none"),
        Verdict("full", "this quote is nowhere in the passage text"),
        Verdict("partial", q1), Verdict("partial", q2),
        Verdict("contradicted"),
        Verdict("partial", q2),
        Verdict("none"),
        Verdict("full", q1),                                               # the joint re-check
    ]
    kept, removed = verify_statements(statements, _passages(), Scripted(verdicts))
    assert [s.text for s in kept] == ["Full with real quote.", "Two partials, jointly full.", "Interpretation with partial basis."]
    assert kept[0].cites == ["S1"] and kept[1].reason == "supported jointly"
    reasons = {s.text: s.reason for s in removed}
    assert reasons["Full but invented quote."] == "not supported by the cited passage"
    assert reasons["Contradicted."] == "contradicted by the cited passage"
    assert reasons["Interpretation with no basis."] == "no basis in the cited passage"
    assert reasons["No citation at all."] == "no citation"


def test_small_helpers():
    assert anchored("draws two amperes at full", "The motor draws two amperes at full load.")
    assert not anchored("draws nine volts at idle speed", "The motor draws two amperes at full load.")
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    ref = ieee_reference(Paper("p", "A Title", ["Ann B. Lee", "Raj Kumar", "Li Wei"], 2023, doi="10.1/x"))
    assert ref == "A. B. Lee, R. Kumar, and L. Wei, \u201cA Title,\u201d 2023. doi: 10.1/x."
    assert LexicalVerifier().check([("motor draws two amperes", "The motor draws two amperes.")])[0].verdict == "full"
