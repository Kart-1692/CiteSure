"""LLM access. One client speaks the OpenAI-compatible /chat/completions protocol, which
covers Ollama (local), Groq, Gemini's OpenAI endpoint, OpenRouter and OpenAI itself.
The model is frozen: nothing here trains or fine-tunes.
"""
from __future__ import annotations

import json
import re
import time

import requests

from .textutil import containment, content_tokens, split_sentences


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model reply")
    return json.loads(text[start:end + 1])


class LLM:
    name = "base"
    calls = 0
    prompt_tokens = 0
    completion_tokens = 0

    def chat(self, system: str, user: str, task: str = "", json_mode: bool = True) -> str:
        raise NotImplementedError

    def chat_json(self, system: str, user: str, task: str = "") -> dict:
        reply = self.chat(system, user, task=task, json_mode=True)
        try:
            return parse_json(reply)
        except (ValueError, json.JSONDecodeError):
            reply = self.chat(
                system, user + "\n\nYour previous reply was not valid JSON. Return only the JSON object.",
                task=task, json_mode=True,
            )
            return parse_json(reply)


class OpenAICompatLLM(LLM):
    def __init__(self, base_url: str, api_key: str, model: str, json_mode: bool = True,
                 min_interval: float = 0.0, max_tokens: int = 2000, timeout: int = 120):
        if not model:
            raise ValueError("Set EGC_LLM_MODEL (see .env.example; `python -m egc.cli models` lists what the endpoint serves).")
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model
        self.name = model
        self.json_mode, self.min_interval = json_mode, min_interval
        self.max_tokens, self.timeout = max_tokens, timeout
        self._last = 0.0

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def list_models(self) -> list[str]:
        r = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=30)
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []))

    def chat(self, system: str, user: str, task: str = "", json_mode: bool = True) -> str:
        payload = {
            "model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode and self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        last_error = ""
        for attempt in range(7):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = requests.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                                  json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = str(exc)
                time.sleep(min(2 ** attempt, 30))
                continue
            if r.status_code == 200:
                body = r.json()
                self.calls += 1
                usage = body.get("usage") or {}
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
                return body["choices"][0]["message"]["content"] or ""
            last_error = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code == 400 and "response_format" in payload:
                payload.pop("response_format")       # endpoint has no JSON mode; rely on the prompt
                self.json_mode = False
                continue
            if r.status_code in (408, 429, 500, 502, 503, 504):
                retry_after = r.headers.get("retry-after", "")
                delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 2 ** attempt * 2
                time.sleep(min(delay, 90))
                continue
            break
        raise RuntimeError(f"LLM request failed ({last_error})")


class MockLLM(LLM):
    """Deterministic stand-in so the whole pipeline can be tested without a model or network.

    It drafts statements by copying sentences from the passages, adds one statement that the
    passages do not support (so verification has something to remove), and 'verifies' by
    token overlap. It is a test double, not a baseline: never report numbers produced with it.
    """
    name = "mock"
    FAKE = "The reported accuracy is 99.9 percent on every public benchmark."

    def chat(self, system: str, user: str, task: str = "", json_mode: bool = True) -> str:
        self.calls += 1
        if task == "verify":
            out = []
            for item in json.loads(user[user.find("{"):])["items"]:
                c = containment(item["claim"], item["evidence"])
                sents = split_sentences(item["evidence"]) or [item["evidence"]]
                quote = max(sents, key=lambda s: containment(item["claim"], s))
                verdict = "full" if c >= 0.85 else "partial" if c >= 0.4 else "none"
                out.append({"id": item["id"], "verdict": verdict, "quote": quote if verdict != "none" else ""})
            return json.dumps({"verdicts": out})
        if task == "baseline":
            return json.dumps({
                "statements": [
                    {"text": "Retrieval-augmented generation reduces hallucination in language models.", "refs": [1]},
                    {"text": "Citation accuracy improves when every claim is checked against its source.", "refs": [2]},
                ],
                "references": [
                    {"n": 1, "authors": "A. Author and B. Writer", "title": "A Wholly Invented Study of Imaginary Retrieval",
                     "venue": "Journal of Nonexistent Results", "year": 2024, "doi": ""},
                    {"n": 2, "authors": "C. Nobody", "title": "Verification Methods That Were Never Published",
                     "venue": "Proc. Imaginary Conf.", "year": 2023, "doi": ""},
                ],
            })
        blocks = re.findall(r"^\[(S\d+)\][^\n]*\n(.+?)(?=\n\n\[S\d+\]|\n\n---|\Z)", user, flags=re.S | re.M)
        question = re.search(r"^QUESTION: (.+)$", user, flags=re.M)
        if question and blocks and containment(question.group(1), " ".join(b for _, b in blocks)) < 0.5:
            return json.dumps({"insufficient": True, "statements": []})
        statements = []
        for tag, body in blocks[:3]:
            sents = [s for s in split_sentences(body) if len(s.split()) >= 8]
            if sents:
                statements.append({"heading": "", "label": "finding", "text": sents[0], "cites": [tag]})
        if blocks:
            tag, body = blocks[0]
            statements.append({"heading": "", "label": "finding", "text": self.FAKE, "cites": [tag]})
            words = " ".join(content_tokens(body)[:8])
            statements.append({"heading": "", "label": "interpretation",
                               "text": f"In simple terms, this passage is about {words}.", "cites": [tag]})
        return json.dumps({"insufficient": not statements, "statements": statements})


def get_llm(settings) -> LLM:
    if settings.llm_provider == "mock":
        return MockLLM()
    return OpenAICompatLLM(
        settings.llm_base_url, settings.llm_api_key, settings.llm_model, settings.llm_json_mode,
        settings.llm_min_interval, settings.llm_max_tokens, settings.llm_timeout,
    )
