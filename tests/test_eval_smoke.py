"""Runs the whole evaluation path offline on two synthetic PDFs (mock LLM, hashing embedder,
lexical judge). Checks that every number the paper's tables need is produced; the values
themselves mean nothing."""
import json
import sys

from tests.test_pipeline import BODY_A, BODY_B, make_pdf


def test_eval_and_report(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    make_pdf(corpus / "gauge.pdf", "Batch Calibration of Strain Gauges", "Asha Rao, Vikram Nair", BODY_A)
    make_pdf(corpus / "tool.pdf", "Spindle Vibration and Ceramic Tool Wear", "Meera Iyer", BODY_B)
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps({
        "ask": [
            {"id": "a1", "question": "How many strain gauges does the calibration rig test in one thermal cycle?", "gold": ["gauge"]},
            {"id": "a2", "question": "How did tool life change when the spindle speed doubled?", "gold": ["tool"]},
            {"id": "a3", "question": "Skipped because its paper is missing.", "gold": ["absent"]},
        ],
        "unanswerable": [{"id": "u1", "question": "Which enzyme unwinds DNA during replication in yeast?"}],
        "paper_tasks": ["gauge"],
    }))
    for k, v in {"EGC_EMBEDDER": "hashing", "EGC_LLM_PROVIDER": "mock", "EGC_VERIFIER": "llm"}.items():
        monkeypatch.setenv(k, v)
    out = tmp_path / "results"
    from eval import make_report, run_eval

    monkeypatch.setattr(sys, "argv", ["run_eval", "--corpus", str(corpus), "--questions", str(questions),
                                      "--out", str(out), "--judge", "lexical", "--no-refcheck"])
    assert run_eval.main() == 0
    m = json.loads((out / "metrics.json").read_text())
    assert m["meta"]["smoke"] is True and m["meta"]["n_ask"] == 2 and m["meta"]["n_papers"] == 2
    for name in ("draft", "verified"):
        assert m["main"][name]["citation_recall"] is not None and m["main"][name]["citation_precision"] is not None
    assert m["main"]["verified"]["citation_precision"] >= m["main"]["draft"]["citation_precision"]
    assert m["verification"]["removed"] >= 1 and m["references"]["unchecked"] == m["references"]["total"] > 0
    assert m["abstention"]["system_abstained_pct"] == 100.0
    assert set(m["retrieval"]) == {"dense", "bm25", "hybrid"}
    assert (out / "removed_claims.jsonl").read_text().strip()

    gen = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", ["make_report", "--metrics", str(out / "metrics.json"), "--out", str(gen)])
    assert make_report.main() == 0
    tex = (gen / "results.tex").read_text()
    assert "SMOKE-TEST NUMBERS" in tex and "\\newcommand{\\ResTableMain}" in tex
    assert (gen / "fig_main.pdf").exists() and (out / "summary.md").exists()
