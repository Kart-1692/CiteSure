"""Post-hoc verification. Every (claim, cited passage) pair is checked on its own; a factual
claim survives only if at least one cited passage fully supports it. Unsupported claims are
removed, not rewritten: the verifier never adds text, so it cannot add a hallucination.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from . import prompts
from .llm import LLM
from .schema import FACTUAL_LABELS, Passage, Statement, Support
from .textutil import containment, normalize, split_sentences

VERDICTS = ("full", "partial", "none", "contradicted")


@dataclass
class Verdict:
    verdict: str = "none"
    quote: str = ""


def anchored(quote: str, passage: str, min_overlap: float = 0.85) -> bool:
    """True when the supporting quote really is in the passage (exact after normalisation,
    or nearly all of its content words are). Guards against a verifier that invents a quote."""
    q, p = normalize(quote).lower(), normalize(passage).lower()
    if len(q.split()) < 3:
        return False
    return q in p or containment(q, p) >= min_overlap


class Verifier:
    name = "base"
    calls = 0
    failures = 0

    def check(self, pairs: list[tuple[str, str]]) -> list[Verdict]:
        raise NotImplementedError


class LLMVerifier(Verifier):
    """A separate call to the frozen LLM, in batches, with a prompt that only sees claim + passage."""
    def __init__(self, llm: LLM, batch: int = 10):
        self.llm, self.batch, self.name = llm, batch, f"llm:{llm.name}"

    def check(self, pairs):
        out: list[Verdict] = []
        for i in range(0, len(pairs), self.batch):
            chunk = pairs[i:i + self.batch]
            items = [{"id": j + 1, "claim": c, "evidence": e} for j, (c, e) in enumerate(chunk)]
            verdicts = {}
            try:
                data = self.llm.chat_json(prompts.VERIFY_SYSTEM, json.dumps({"items": items}, ensure_ascii=False), task="verify")
                self.calls += 1
                for v in data.get("verdicts") or []:
                    if isinstance(v, dict) and str(v.get("verdict", "")).lower() in VERDICTS:
                        verdicts[int(v.get("id", 0))] = Verdict(str(v["verdict"]).lower(), str(v.get("quote", "") or ""))
            except Exception:
                self.failures += 1                    # fail closed: an unverified claim is not kept
            out.extend(verdicts.get(j + 1, Verdict()) for j in range(len(chunk)))
        return out


class NLIVerifier(Verifier):
    """Cross-encoder NLI model (premise = passage, hypothesis = claim). Also the evaluation judge."""
    def __init__(self, model_name: str, threshold: float = 0.5):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.name, self.threshold = f"nli:{model_name}", threshold
        config = getattr(self.model, "config", None) or self.model.model.config
        labels = {int(k): str(v).lower() for k, v in config.id2label.items()}
        self.i_ent = next(k for k, v in labels.items() if v.startswith("entail"))
        self.i_con = next(k for k, v in labels.items() if v.startswith("contra"))

    def check(self, pairs):
        if not pairs:
            return []
        import numpy as np

        logits = np.asarray(self.model.predict([(e, c) for c, e in pairs], show_progress_bar=False))
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        self.calls += 1
        out = []
        for (claim, evidence), p in zip(pairs, probs):
            if p[self.i_ent] >= self.threshold:
                out.append(Verdict("full", _best_sentence(claim, evidence)))
            elif p[self.i_con] >= self.threshold:
                out.append(Verdict("contradicted"))
            else:
                out.append(Verdict("none"))
        return out


class LexicalVerifier(Verifier):
    """Token-overlap check. Offline smoke tests only; not a measure of support."""
    name = "lexical"

    def check(self, pairs):
        self.calls += 1
        out = []
        for claim, evidence in pairs:
            c = containment(claim, evidence)
            out.append(Verdict("full" if c >= 0.85 else "partial" if c >= 0.4 else "none",
                               _best_sentence(claim, evidence) if c >= 0.4 else ""))
        return out


def _best_sentence(claim: str, evidence: str) -> str:
    sents = split_sentences(evidence) or [evidence]
    return max(sents, key=lambda s: containment(claim, s))


def get_verifier(settings, llm: LLM) -> Verifier:
    if settings.verifier == "nli":
        return NLIVerifier(settings.nli_model, settings.nli_threshold)
    if settings.verifier == "lexical":
        return LexicalVerifier()
    return LLMVerifier(llm, settings.verify_batch)


def verify_statements(statements: list[Statement], passages: list[Passage], verifier: Verifier,
                      strict_quote: bool = True, quote_overlap: float = 0.85) -> tuple[list[Statement], list[Statement]]:
    """Returns (kept, removed). Statements are updated in place with status, reason and support."""
    text_of = {p.tag: p.chunk.text for p in passages}
    pairs, owners = [], []
    for si, st in enumerate(statements):
        st.support = []
        for tag in st.cites:
            pairs.append((st.text, text_of[tag]))
            owners.append((si, tag))
    for (si, tag), v in zip(owners, verifier.check(pairs)):
        ok = anchored(v.quote, text_of[tag], quote_overlap) if v.verdict in ("full", "partial") else False
        verdict = v.verdict
        if verdict == "full" and strict_quote and not ok:
            verdict = "partial"                       # a "full" verdict needs a quote that is really there
        statements[si].support.append(Support(tag, verdict, v.quote if ok else "", ok))

    # second look: a factual claim that two or more passages each support in part may be
    # supported by them jointly (multi-passage synthesis)
    joint_idx, joint_pairs = [], []
    for si, st in enumerate(statements):
        if st.label in FACTUAL_LABELS and not any(s.verdict == "full" for s in st.support):
            partial = [s.tag for s in st.support if s.verdict == "partial"]
            if len(partial) >= 2:
                joint_idx.append(si)
                joint_pairs.append((st.text, " ".join(text_of[t] for t in partial)))
    joint_ok = set()
    if joint_pairs:
        for si, v in zip(joint_idx, verifier.check(joint_pairs)):
            if v.verdict == "full":
                joint_ok.add(si)

    kept, removed = [], []
    for si, st in enumerate(statements):
        verdicts = {s.tag: s.verdict for s in st.support}
        full = [t for t, v in verdicts.items() if v == "full"]
        partial = [t for t, v in verdicts.items() if v == "partial"]
        contra = [t for t, v in verdicts.items() if v == "contradicted"]
        if not st.cites:
            st.status, st.reason = "removed", "no citation"
        elif st.label in FACTUAL_LABELS:
            if full:
                st.status, st.cites = "kept", full     # citations that do not support are dropped
            elif si in joint_ok:
                st.status, st.cites, st.reason = "kept", partial, "supported jointly"
            else:
                st.status = "removed"
                st.reason = "contradicted by the cited passage" if contra else "not supported by the cited passage"
        else:  # interpretation: must have a real basis and must not conflict with it
            if contra:
                st.status, st.reason = "removed", "contradicted by the cited passage"
            elif full or partial:
                st.status, st.cites = "kept", full + partial
            else:
                st.status, st.reason = "removed", "no basis in the cited passage"
        (kept if st.status == "kept" else removed).append(st)
    return kept, removed
