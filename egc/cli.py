"""Command line:  python -m egc.cli <command>

  ingest PATH [PATH ...] [--enrich]   add PDFs (files or folders) to the store
  list                                show stored papers and their metadata
  remove PAPER                        drop a paper
  ask "QUESTION"                      answer across all stored papers
  explain PAPER [--focus TEXT]        explain one paper
  summarize PAPER                     structured summary of one paper
  baseline ask "QUESTION"             the same request to the plain LLM, no documents
  models                              list model ids served by the configured endpoint
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .pipeline import Engine
from .render import to_markdown


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="egc", description="Evidence-grounded answers over research papers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest"); p.add_argument("paths", nargs="+"); p.add_argument("--enrich", action="store_true",
        help="fill title/authors/year from Crossref or arXiv when a DOI or arXiv id is found (needs internet)")
    sub.add_parser("list")
    p = sub.add_parser("remove"); p.add_argument("paper")
    for name in ("ask", "explain", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("target", help="question (ask) or paper id / file name / title fragment")
        p.add_argument("--json", action="store_true")
        p.add_argument("--no-verify", action="store_true", help="skip verification (ablation)")
        if name == "explain":
            p.add_argument("--focus", default="")
    p = sub.add_parser("baseline"); p.add_argument("mode", choices=["ask", "explain", "summarize"]); p.add_argument("target")
    sub.add_parser("models")
    args = ap.parse_args(argv)

    engine = Engine(Settings())
    if args.cmd == "ingest":
        files = []
        for raw in args.paths:
            path = Path(raw)
            files.extend(sorted(path.glob("*.pdf")) if path.is_dir() else [path])
        for f in files:
            try:
                paper = engine.ingest(f, enrich=args.enrich)
                print(f"+ {paper.paper_id}  {paper.n_chunks:4d} chunks  [{paper.meta_source}]  {paper.title[:80]}")
            except Exception as exc:
                print(f"! {f.name}: {exc}", file=sys.stderr)
    elif args.cmd == "list":
        for paper in engine.papers():
            print(f"{paper.paper_id}  {paper.year or '----'}  {paper.short():28.28s}  {paper.title[:70]}")
            print(f"{'':12s}doi={paper.doi or '-'}  file={paper.source_file}  metadata={paper.meta_source}")
    elif args.cmd == "remove":
        engine.store.remove_paper(engine.store.resolve_paper(args.paper).paper_id)
    elif args.cmd in ("ask", "explain", "summarize"):
        if args.cmd == "ask":
            ans = engine.ask(args.target, verify=not args.no_verify)
        elif args.cmd == "explain":
            ans = engine.explain(args.target, focus=args.focus, verify=not args.no_verify)
        else:
            ans = engine.summarize(args.target, verify=not args.no_verify)
        print(json.dumps(ans.to_dict(), indent=2, ensure_ascii=False) if args.json else to_markdown(ans))
    elif args.cmd == "baseline":
        out = engine.baseline(args.mode, question=args.target if args.mode == "ask" else "",
                              paper_key="" if args.mode == "ask" else args.target)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    elif args.cmd == "models":
        print("\n".join(engine.llm.list_models()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
