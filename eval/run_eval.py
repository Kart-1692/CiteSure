"""Run the evaluation and write results/metrics.json.

    python -m eval.download_corpus
    python -m eval.run_eval                 # resumable: finished requests are cached in results/raw
    python -m eval.make_report              # tables + figures for the paper

Three systems answer the same requests:
  baseline : the plain LLM, no documents (answers and references from memory)
  draft    : this system before verification (retrieval + passage-first generation)
  verified : this system (draft + per-claim verification)
"draft" and "verified" share one generation, so their difference is the verification step alone.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from egc import metadata as md
from egc.config import Settings
from egc.pipeline import Engine
from egc.retrieve import retrieve
from egc.schema import FACTUAL_LABELS
from egc.textutil import normalize
from egc.verify import LexicalVerifier, NLIVerifier

from .metrics import Judge, alce, corpus_support, match_corpus, metadata_error, pct

HERE = Path(__file__).parent


def load_or_run(path: Path, fn):
    if path.exists():
        return json.loads(path.read_text())
    data = fn()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(HERE / "corpus"))
    ap.add_argument("--questions", default=str(HERE / "questions.json"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--judge", choices=["nli", "lexical"], default="nli",
                    help="lexical = token overlap, for offline smoke tests only")
    ap.add_argument("--limit", type=int, default=0, help="only the first N ask questions (quick trial)")
    ap.add_argument("--no-refcheck", action="store_true", help="do not look baseline references up online")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    settings = Settings()
    settings.store_dir = str(out / "store")
    engine = Engine(settings)
    store = engine.store

    # ---------------------------------------------------------------- corpus
    have = {p.source_file for p in store.papers.values()}
    for pdf in sorted(Path(args.corpus).glob("*.pdf")):
        if pdf.name not in have:
            paper = engine.ingest(pdf)
            print(f"ingested {pdf.name}: {paper.n_chunks} chunks")
    key_to_id = {Path(p.source_file).stem: p.paper_id for p in store.papers.values()}
    if not key_to_id:
        raise SystemExit(f"No PDFs in {args.corpus}. Run `python -m eval.download_corpus` first.")
    spec = json.loads(Path(args.questions).read_text())
    ask = [q for q in spec["ask"] if all(g in key_to_id for g in q["gold"])]
    if args.limit:
        ask = ask[:args.limit]
    unans = [] if args.limit else spec.get("unanswerable", [])
    paper_keys = [] if args.limit else [k for k in spec.get("paper_tasks", []) if k in key_to_id]
    skipped = len(spec["ask"]) - len(ask) if not args.limit else 0
    if skipped:
        print(f"note: {skipped} questions skipped because their gold paper is not in the corpus folder")

    # ---------------------------------------------------------------- retrieval ablation (no LLM)
    retrieval = {}
    for mode in ("dense", "bm25", "hybrid"):
        hit = cov = mrr = 0.0
        for q in ask:
            gold = {key_to_id[g] for g in q["gold"]}
            got = [p.paper.paper_id for p in retrieve(store, q["question"], settings.top_k_ask, None, mode,
                                                      settings.n_candidates, settings.rrf_k, settings.max_per_paper)]
            hit += bool(gold & set(got))
            cov += len(gold & set(got)) / len(gold)
            mrr += next((1.0 / r for r, pid in enumerate(got, start=1) if pid in gold), 0.0)
        n = max(len(ask), 1)
        retrieval[mode] = {"hit": pct(hit, n), "coverage": pct(cov, n), "mrr": round(mrr / n, 3)}
    print("retrieval:", retrieval)

    # ---------------------------------------------------------------- run the three systems
    tasks = [{"id": q["id"], "mode": "ask", "question": q["question"], "answerable": True} for q in ask]
    tasks += [{"id": q["id"], "mode": "ask", "question": q["question"], "answerable": False} for q in unans]
    for k in paper_keys:
        tasks.append({"id": f"e-{k}", "mode": "explain", "paper": key_to_id[k], "answerable": True})
        tasks.append({"id": f"s-{k}", "mode": "summarize", "paper": key_to_id[k], "answerable": True})

    runs = []
    for i, t in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] {t['id']} ({t['mode']})", flush=True)

        def run_system():
            if t["mode"] == "ask":
                return engine.ask(t["question"]).to_dict()
            return (engine.explain(t["paper"]) if t["mode"] == "explain" else engine.summarize(t["paper"])).to_dict()

        def run_base():
            return engine.baseline(t["mode"], question=t.get("question", ""), paper_key=t.get("paper", ""))

        runs.append({"task": t,
                     "system": load_or_run(out / "raw" / f"{t['id']}.system.json", run_system),
                     "baseline": load_or_run(out / "raw" / f"{t['id']}.baseline.json", run_base)})

    # ---------------------------------------------------------------- judge
    judge_model = LexicalVerifier() if args.judge == "lexical" else NLIVerifier(settings.nli_model, settings.nli_threshold)
    judge = Judge(judge_model)
    cache_path = out / "refcache.json"
    refcache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def new_bucket():
        return Counter()

    groups = ["all", "ask", "explain", "summarize"]
    agg = {g: {s: new_bucket() for s in ("baseline", "draft", "verified")} for g in groups}
    refs = Counter()
    reasons = Counter()
    removed_log, abst = [], Counter()

    for run in runs:
        t, sysa, base = run["task"], run["system"], run["baseline"]
        pid = t.get("paper")
        text_of = {p["tag"]: p["chunk"]["text"] for p in sysa["passages"]}
        if not t["answerable"]:
            abst["unans_total"] += 1
            abst["unans_system_abstained"] += not sysa["statements"]
            abst["unans_baseline_abstained"] += not base["statements"]
            continue
        abst["ans_total"] += 1
        abst["ans_system_abstained"] += not sysa["statements"]

        for name, stmts in (("draft", sysa["draft"]), ("verified", sysa["statements"])):
            factual = [s for s in stmts if s["label"] in FACTUAL_LABELS]
            a = alce(factual, text_of, judge)
            faithful = corpus_support([s["text"] for s in factual], store, judge, pid, settings=settings)
            for g in ("all", t["mode"]):
                b = agg[g][name]
                b.update(a)
                b["faithful"] += sum(faithful)
                b["answers"] += 1
                b["statements_all"] += len(stmts)
                b["interpretations"] += sum(s["label"] == "interpretation" for s in stmts)
                if name == "verified":
                    b["seconds"] += sysa["timings"].get("total", 0.0)
                    b["llm_calls"] += sysa["llm_calls"]
                else:
                    b["seconds"] += sysa["timings"].get("retrieve", 0.0) + sysa["timings"].get("generate", 0.0)
                    b["llm_calls"] += 1
        for s in sysa["removed"]:
            reasons[s["reason"]] += 1
            removed_log.append({"task": t["id"], "mode": t["mode"], "label": s["label"], "text": s["text"],
                                "reason": s["reason"], "cited": [text_of.get(c, "")[:400] for c in s["cites"]]})

        # ---- baseline: audit its references, then score its statements
        status = {}
        for r in base["references"]:
            key = normalize(r["title"]).lower()
            in_corpus = match_corpus(r["title"], store.papers)
            if in_corpus:
                rec = asdict(store.papers[in_corpus])
                status[r["n"]] = ("in_corpus", in_corpus)
                refs["metadata_errors"] += metadata_error(r, rec)
            elif args.no_refcheck:
                status[r["n"]] = ("unchecked", "")
            else:
                if key not in refcache:
                    refcache[key] = md.search_title(r["title"])
                    cache_path.write_text(json.dumps(refcache, indent=2))
                res = refcache[key]
                kind = {"found": "real_outside", "not_found": "fabricated", "unchecked": "unchecked"}[res["status"]]
                status[r["n"]] = (kind, "")
                if res["record"]:
                    refs["metadata_errors"] += metadata_error(r, res["record"])
            refs[status[r["n"]][0]] += 1
            refs["total"] += 1
        texts = [s["text"] for s in base["statements"]]
        faithful = corpus_support(texts, store, judge, pid, settings=settings)
        b_stats = Counter(n_statements=len(texts), faithful=sum(faithful), answers=1, statements_all=len(texts),
                          seconds=base.get("seconds", 0.0), llm_calls=1)
        for s in base["statements"]:
            b_stats["uncited"] += not s["refs"]
            supported_any = False
            for n in s["refs"]:
                kind, ref_pid = status.get(n, ("unchecked", ""))
                if kind in ("real_outside", "unchecked"):
                    b_stats["citations_unscored"] += 1     # real or unknown, but no source text to check against
                    continue
                b_stats["n_citations"] += 1
                if kind == "in_corpus":
                    ok = corpus_support([s["text"]], store, judge, ref_pid, settings=settings)[0]
                    b_stats["precision_hits"] += ok
                    supported_any |= ok
            b_stats["recall_hits"] += supported_any
        for g in ("all", t["mode"]):
            agg[g]["baseline"].update(b_stats)

    # ---------------------------------------------------------------- aggregate
    def row(b: Counter) -> dict:
        n = max(b["answers"], 1)
        return {
            "citation_recall": pct(b["recall_hits"], b["n_statements"]),
            "citation_precision": pct(b["precision_hits"], b["n_citations"]),
            "faithfulness": pct(b["faithful"], b["n_statements"]),
            "statements_per_answer": round(b["statements_all"] / n, 1),
            "factual_statements": b["n_statements"], "citations": b["n_citations"],
            "interpretations": b["interpretations"], "uncited_pct": pct(b["uncited"], b["n_statements"]),
            "citations_unscored": b["citations_unscored"],
            "seconds_per_answer": round(b["seconds"] / n, 1), "llm_calls_per_answer": round(b["llm_calls"] / n, 1),
            "answers": b["answers"],
        }

    checked = refs["total"] - refs["unchecked"]
    draft_n = agg["all"]["draft"]["statements_all"]
    smoke = args.judge == "lexical" or settings.llm_provider == "mock" or settings.embedder == "hashing"
    metrics = {
        "meta": {
            "date": dt.date.today().isoformat(), "smoke": smoke,
            "llm": engine.llm.name, "embedder": settings.embedder, "verifier": engine.verifier.name,
            "judge": judge_model.name, "n_papers": len(store.papers), "n_chunks": store.col.count(),
            "n_ask": len(ask), "n_cross": sum(len(q["gold"]) > 1 for q in ask), "n_unanswerable": len(unans),
            "n_paper_tasks": len(paper_keys), "settings": settings.public(),
            "verifier_failures": engine.verifier.failures,
        },
        "main": {name: row(agg["all"][name]) for name in ("baseline", "draft", "verified")},
        "by_mode": {g: {name: row(agg[g][name]) for name in ("baseline", "draft", "verified")} for g in groups[1:]},
        "references": {
            "total": refs["total"], "checked": checked, "fabricated": refs["fabricated"],
            "real_in_corpus": refs["in_corpus"], "real_outside": refs["real_outside"], "unchecked": refs["unchecked"],
            "fabricated_pct": pct(refs["fabricated"], checked),
            "metadata_errors": refs["metadata_errors"],
            "metadata_error_pct": pct(refs["metadata_errors"], refs["in_corpus"] + refs["real_outside"]),
        },
        "retrieval": retrieval,
        "verification": {
            "draft_statements": draft_n, "removed": sum(reasons.values()),
            "removed_pct": pct(sum(reasons.values()), draft_n), "reasons": dict(reasons),
        },
        "abstention": {
            "unanswerable": abst["unans_total"],
            "system_abstained_pct": pct(abst["unans_system_abstained"], abst["unans_total"]),
            "baseline_abstained_pct": pct(abst["unans_baseline_abstained"], abst["unans_total"]),
            "answerable": abst["ans_total"],
            "system_false_abstention_pct": pct(abst["ans_system_abstained"], abst["ans_total"]),
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out / "removed_claims.jsonl").open("w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in removed_log)
    print(json.dumps({k: metrics[k] for k in ("main", "references", "verification", "abstention")}, indent=2))
    if smoke:
        print("\nSMOKE RUN (mock LLM, hashing embedder or lexical judge): these numbers test the code path only.")
    print(f"\nwrote {out / 'metrics.json'} and {out / 'removed_claims.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
