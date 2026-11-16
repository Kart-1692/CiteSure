"""Download the evaluation corpus from arXiv.

    python -m eval.download_corpus            # -> eval/corpus/<key>.pdf + <key>.meta.json

Each arXiv id in eval/corpus.json is checked against the expected title before anything is
saved; an id whose title does not match is skipped with a warning (so a wrong id can never
put the wrong paper under a key). Metadata is written from the arXiv record into a sidecar
file, which ingestion then treats as authoritative.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

from egc.metadata import UA, arxiv_by_id
from egc.textutil import similarity

HERE = Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "corpus"))
    ap.add_argument("--spec", default=str(HERE / "corpus.json"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for item in json.loads(Path(args.spec).read_text()):
        pdf, side = out / f"{item['key']}.pdf", out / f"{item['key']}.meta.json"
        if pdf.exists() and side.exists():
            print(f"= {item['key']}: already there")
            ok += 1
            continue
        rec = arxiv_by_id(item["arxiv_id"])
        time.sleep(3)                                  # arXiv asks for at most one request every 3 s
        if rec is None:
            print(f"! {item['key']}: arXiv API not reachable, skipped", file=sys.stderr)
            continue
        if similarity(rec["title"], item["title"]) < 0.8:
            print(f"! {item['key']}: arXiv:{item['arxiv_id']} is \"{rec['title']}\", expected \"{item['title']}\" - skipped."
                  " Fix the id in eval/corpus.json or drop the PDF in by hand.", file=sys.stderr)
            continue
        r = requests.get(f"https://arxiv.org/pdf/{item['arxiv_id']}", headers=UA, timeout=120)
        time.sleep(3)
        if r.status_code != 200 or not r.content.startswith(b"%PDF"):
            print(f"! {item['key']}: PDF download failed (HTTP {r.status_code})", file=sys.stderr)
            continue
        pdf.write_bytes(r.content)
        side.write_text(json.dumps({
            "title": rec["title"], "authors": rec["authors"], "year": rec["year"],
            "arxiv_id": item["arxiv_id"], "venue": f"arXiv:{item['arxiv_id']}",
        }, indent=2))
        print(f"+ {item['key']}: {rec['title']}")
        ok += 1
    print(f"{ok} papers ready in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
