# Evidence-Grounded Generative AI for Engineering Research Document Citation

A research-paper assistant that only says what it can point to. It reads PDFs, answers in three
modes, ties every factual sentence to the exact passage that supports it, checks each claim
against that passage, and removes what fails the check. No model is trained or fine-tuned.

| Mode | What it does |
|---|---|
| **Ask Research** | Answers a question across all uploaded papers |
| **Explain Paper** | Explains one paper's problem, method, results and key concepts in plain language |
| **Summarize Paper** | Structured summary: objective, method, data, results, limitations, takeaway |

Every statement is labelled **finding** (stated by the paper), **limitation** (a limit the paper
itself states) or **interpretation** (the system's own reading).

## How it works

```
PDF ─► text (reference list cut) ─► metadata (title, authors, year, DOI) ─► 180-word chunks
                                                         │
                         ┌───────────────────────────────┴──────────────┐
                 vector DB (ChromaDB)                          BM25 keyword index
                         └──────────────┬───────────────────────────────┘
request ─► hybrid retrieval (reciprocal rank fusion) ─► passages S1..Sk
        ─► passage-first generation (LLM call 1: statements + label + tags)
        ─► claim verification (LLM call 2: one check per claim–passage pair, quote anchoring)
        ─► supported claims only ─► citations and references copied from stored metadata
```

Design rules, all enforced in code and covered by tests:

1. The model never writes a title, author, year or DOI. References are printed from the stored record.
2. The model can only cite passage tags that were in its prompt; any other tag is dropped.
3. A finding or limitation survives only if a cited passage fully supports it **and** the verifier's
   supporting quote is really in that passage. Citations that do not support are dropped.
4. Unsupported claims are removed, not rewritten, and are shown in a "removed" list.
5. If nothing survives, the system says the papers do not contain the evidence.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # then put your LLM endpoint, key and model in .env
python -m egc.cli models                                # lists the model ids your endpoint serves
```

Any OpenAI-compatible chat endpoint works (Groq, Gemini, OpenRouter, a local Ollama). The first run
downloads the embedding model (`all-MiniLM-L6-v2`, about 90 MB).

## Use

```bash
python -m egc.cli ingest papers/ --enrich     # index every PDF in papers/ (--enrich: Crossref/arXiv metadata)
python -m egc.cli list
python -m egc.cli ask "How does the proposed controller handle sensor noise?"
python -m egc.cli explain <paper-id or title fragment> --focus "the loss function"
python -m egc.cli summarize <paper-id or title fragment>
python -m egc.cli baseline ask "same question"          # the plain LLM, no documents

streamlit run app.py                                    # web app with the three modes
```

Metadata check: `list` shows where each paper's metadata came from. `heuristic` means it was guessed
from the PDF layout; fix it by adding `paper.meta.json` next to `paper.pdf`
(`{"title": "...", "authors": ["..."], "year": 2024, "doi": "..."}`) and ingesting again.

## Evaluation (numbers for the paper)

```bash
python -m eval.download_corpus     # 12 open-access papers from arXiv -> eval/corpus/
python -m eval.run_eval            # runs plain LLM vs grounded vs grounded+verification -> results/
python -m eval.make_report         # -> paper/generated/results.tex + fig_main.pdf, results/summary.md
cd paper && pdflatex sections_III_to_V.tex && pdflatex sections_III_to_V.tex
```

* `run_eval` is resumable: every finished request is cached in `results/raw/`, so a rate-limit stop
  costs nothing. Try `--limit 3` first. About 46 requests per system, roughly 150 to 200 LLM calls.
* The judge is an NLI cross-encoder (`cross-encoder/nli-deberta-v3-base`, downloaded on first use)
  that is **not** part of the system, so the system is not marking its own work.
  Set `EGC_NLI_MODEL` to use a different one.
* Baseline references are looked up by title in Crossref, OpenAlex and arXiv (needs internet).
* To evaluate on your own engineering papers, put the PDFs in `eval/corpus/` and edit
  `eval/questions.json` (`gold` = file names without `.pdf`).

Metrics (defined in `eval/metrics.py`): citation recall and precision (ALCE), faithfulness (share of
statements entailed by retrieved corpus passages), fabricated-reference rate of the plain LLM,
retrieval Hit@k / MRR for dense vs BM25 vs hybrid, abstention on out-of-corpus questions, calls and
seconds per answer.

## Tests

```bash
pytest -q
```

The tests run offline with a scripted model, a hashing embedder and synthetic PDFs. They check the
rules above and the evaluation plumbing. They say nothing about answer quality; numbers from the
mock model or the `lexical` judge are stamped "not for reporting".

## Layout

```
egc/        ingest.py  metadata.py  embeddings.py  store.py  retrieve.py
            prompts.py generate.py  verify.py      pipeline.py render.py  llm.py  cli.py
app.py      Streamlit interface
eval/       corpus.json  questions.json  download_corpus.py  run_eval.py  metrics.py  make_report.py
paper/      sections_III_to_V.tex (Methodology, Results and Discussion, Future Work, Conclusion)
            architecture.pdf/.png (Fig. 1 on its own)   generated/ (written by make_report.py)
tests/
```

## Known limits

Single retrieval pass (no query rewriting or second hop); tables, equations and figures are read as
plain text; scanned PDFs need OCR first; the verifier is the same LLM as the generator (quote
anchoring and an independent evaluation judge limit, but do not remove, shared blind spots).
