"""Phase 1 · Lesson 02 — structure: nested work and the RAG shape.

Lesson 01 recorded one flat generation. Real pipelines have *structure*: steps
nest inside each other, and the trace should mirror that. This lesson runs a
tiny in-file RAG pipeline over a six-note corpus and records every stage with
the SDK's structural primitives:

  * ``with bir.trace(...)`` — the explicit trace root (the alternative to
    ``@observe`` from Lesson 01), with trace-level metadata.
  * ``span(...)`` — a nested unit of plain work (query preparation), with
    ``set_metadata``.
  * ``retrieval(...)`` — Bir's documented RAG event shape, shown both ways:
    recall records hits one by one with ``add_document(...)``, rerank records
    an already-built list with ``set_documents(...)``.
  * ``tool_call(...)`` — a non-LLM step (building the context block), with
    ``set_output``.
  * the Ollama generation from Lesson 01, now nested inside the pipeline.
  * ``score(...)`` — two numeric judgments attached to the trace.

Afterward the script reloads the trace and prints it as an indented tree, so
you can *see* the nesting that the context managers produced.

Run it:
  * offline (no Ollama, no network, deterministic — what CI runs):
      uv run python main.py --smoke
  * real (needs a local Ollama server + a pulled model):
      ollama pull llama3.2:1b
      uv run python main.py --prompt "How do scores relate to a trace?"

Ollama is local and keyless, so there is no API key to set.
"""

from __future__ import annotations

import argparse
import os
import string
from pathlib import Path

from bir import TraceEvent, configure, load_traces, retrieval, score, span, tool_call, trace
from bir.integrations.ollama import trace_chat as trace_ollama_chat

DEFAULT_TRACE_PATH = Path(__file__).resolve().parent / ".bir" / "traces.jsonl"


def _default_model() -> str:
    # The whole tour runs on one local model, named once in cookbook.env at
    # the repo root — edit that single line (and `ollama pull` the new model)
    # to rerun every lesson on it. Precedence: --model flag > OLLAMA_MODEL
    # env var > cookbook.env > this built-in fallback.
    from_env = os.environ.get("OLLAMA_MODEL")
    if from_env:
        return from_env
    for parent in Path(__file__).resolve().parents:
        config = parent / "cookbook.env"
        if config.is_file():
            for line in config.read_text().splitlines():
                key, sep, value = line.partition("=")
                if sep and key.strip() == "OLLAMA_MODEL" and value.strip():
                    return value.strip()
            break
    return "llama3.2:1b"


DEFAULT_MODEL = _default_model()
DEFAULT_PROMPT = "How do spans, generations and scores relate to a trace?"

RECALL_TOP_K = 4  # documents the keyword search returns
RERANK_TOP_N = 2  # documents the rerank keeps for the context

# The corpus the pipeline retrieves from: six short notes about Bir's own event
# types, so the recipe documents the SDK while demonstrating it.
_CORPUS = [
    {
        "id": "note-trace",
        "source": "notes/trace.md",
        "text": "A trace is one recorded run of your program. Every span, generation, "
        "tool call, and score created during that run shares its trace id.",
    },
    {
        "id": "note-span",
        "source": "notes/span.md",
        "text": "Spans mark nested units of work inside a trace. Each span records its "
        "own timing and status, and new events attach to the innermost open span.",
    },
    {
        "id": "note-generation",
        "source": "notes/generation.md",
        "text": "A generation is the event for a model call: it stores the model name, "
        "the prompt input, the output text, and the token usage.",
    },
    {
        "id": "note-tool-call",
        "source": "notes/tool-call.md",
        "text": "A tool call records one call to an external function or API, with its "
        "input and output captured for debugging.",
    },
    {
        "id": "note-retrieval",
        "source": "notes/retrieval.md",
        "text": "Retrieval events record a search: the query goes in the input, and the "
        "returned documents, each with rank, score, and source, go in the output.",
    },
    {
        "id": "note-score",
        "source": "notes/score.md",
        "text": "Scores attach numeric judgments, such as groundedness or relevance, to "
        "the current trace so runs can be filtered and compared later.",
    },
]

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "how",
    "in", "is", "it", "its", "of", "on", "or", "the", "to", "was", "what", "with",
}

# Keep the client module-level so it is never captured as an input (see CLAUDE.md).
_CLIENT = None


# --------------------------------------------------------------------------- #
# The pipeline's plain-Python steps. Deliberately trivial — the lesson is the
# trace structure around them, not the algorithms.
# --------------------------------------------------------------------------- #
def _words(text: str) -> list[str]:
    """Lowercased words with punctuation stripped and a crude plural fold."""

    cleaned = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [w.rstrip("s") if len(w) > 3 else w for w in cleaned.split()]


def _extract_keywords(question: str) -> list[str]:
    seen: dict[str, None] = {}
    for word in _words(question):
        if word not in _STOPWORDS:
            seen.setdefault(word)
    return list(seen)


def _keyword_search(keywords: list[str], top_k: int) -> list[dict]:
    """Score each corpus note by how many distinct keywords it contains."""

    hits = []
    for doc in _CORPUS:
        doc_words = set(_words(doc["text"]))
        matched = sum(1 for kw in keywords if kw in doc_words)
        if matched:
            hits.append({**doc, "score": float(matched)})
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_k]


def _rerank(hits: list[dict], keywords: list[str], top_n: int) -> list[dict]:
    """Re-score hits by keyword *density* so short, focused notes win."""

    reranked = []
    for hit in hits:
        doc_words = _words(hit["text"])
        density = sum(1 for w in doc_words if w in keywords) / max(len(doc_words), 1)
        reranked.append({**hit, "score": round(density, 4)})
    reranked.sort(key=lambda h: h["score"], reverse=True)
    return reranked[:top_n]


def _grounded_share(answer: str, context: str) -> float:
    """Share of the answer's content words that appear in the retrieved context."""

    answer_words = [w for w in _words(answer) if w not in _STOPWORDS]
    if not answer_words:
        return 0.0
    context_words = set(_words(context))
    return round(sum(1 for w in answer_words if w in context_words) / len(answer_words), 4)


# --------------------------------------------------------------------------- #
# Offline fake — only used with --smoke. Same response shape as Lesson 01's
# fake (``model``, ``message.content``, ``prompt_eval_count`` / ``eval_count``,
# a ``model_dump``). Its answer quotes the first context note it is given, so
# the groundedness score is deterministically high.
# --------------------------------------------------------------------------- #
class _FakeMessage:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def __getitem__(self, key: str):
        return getattr(self, key)

    def model_dump(self) -> dict:
        return {"role": self.role, "content": self.content}


class _FakeChatResponse:
    def __init__(self, *, model: str, content: str, prompt_eval_count: int, eval_count: int) -> None:
        self.model = model
        self.message = _FakeMessage("assistant", content)
        self.prompt_eval_count = prompt_eval_count
        self.eval_count = eval_count
        self.done = True

    def __getitem__(self, key: str):
        return getattr(self, key)

    def model_dump(self) -> dict:
        return {
            "model": self.model,
            "message": self.message.model_dump(),
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
            "done": self.done,
        }


class _FakeOllamaClient:
    """Deterministic stand-in for ``ollama.Client`` used only in --smoke mode."""

    def chat(self, *, model: str, messages: list[dict], **_kwargs) -> _FakeChatResponse:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        first_note = next(
            (line[2:] for line in system.splitlines() if line.startswith("- ")),
            "the context was empty",
        )
        content = f"(smoke) According to the notes: {first_note}"
        return _FakeChatResponse(
            model=model,
            content=content,
            prompt_eval_count=sum(len(m["content"].split()) for m in messages),
            eval_count=len(content.split()),
        )


def _build_client(smoke: bool):
    if smoke:
        return _FakeOllamaClient()

    # Real mode: talk to a local Ollama server. Import the provider library
    # lazily so --smoke needs neither the package nor a running server. Ollama
    # needs no API key — the client honors the OLLAMA_HOST env var if set.
    try:
        from ollama import Client
    except ModuleNotFoundError as exc:  # pragma: no cover - real-path only
        raise SystemExit(
            "The 'ollama' package is required for a real run "
            "(`uv add ollama`), or run offline with --smoke."
        ) from exc

    client = Client()
    try:
        client.list()  # cheap reachability check for a clear error up front
    except Exception as exc:  # pragma: no cover - real-path only
        raise SystemExit(
            "Could not reach a local Ollama server at http://localhost:11434.\n"
            "Start Ollama and pull a model (`ollama pull llama3.2:1b`), "
            "or run offline with --smoke."
        ) from exc
    return client


def answer_question(question: str, model: str) -> str:
    """Run the RAG pipeline, recording each stage under one explicit trace root."""

    # Lesson 01 rooted the trace with @observe; ``with bir.trace(...)`` is the
    # explicit equivalent, handy when the unit of work isn't one function.
    with trace("ollama_structure", metadata={"recipe": "ollama-02-structure"}):
        # A span is plain nested work — anything worth timing that isn't an
        # LLM call, a tool, or a retrieval.
        with span("prepare_query") as prep:
            keywords = _extract_keywords(question)
            prep.set_metadata({"keywords": keywords})

        # Recall stage: hits are discovered one by one, so record each with
        # add_document — id, text, rank, score, and source all land in the
        # event's documents list.
        with retrieval("keyword_search", query=question) as recall:
            hits = _keyword_search(keywords, top_k=RECALL_TOP_K)
            for rank, hit in enumerate(hits):
                recall.add_document(
                    id=hit["id"],
                    text=hit["text"],
                    rank=rank,
                    score=hit["score"],
                    source=hit["source"],
                )
            recall.set_metadata({"corpus_size": len(_CORPUS), "returned": len(hits)})

        # Rerank stage: the final list already exists, so set_documents replaces
        # the documents wholesale — the bulk alternative to add_document.
        with retrieval("rerank", query=question, metadata={"strategy": "keyword-density"}) as rerank:
            top = _rerank(hits, keywords, top_n=RERANK_TOP_N)
            rerank.set_documents(
                {"id": h["id"], "text": h["text"], "rank": i, "score": h["score"], "source": h["source"]}
                for i, h in enumerate(top)
            )

        # A tool_call records any non-LLM step with an input and an output.
        with tool_call("build_context", input={"document_ids": [h["id"] for h in top]}) as build:
            context = "\n".join(f"- {h['text']}" for h in top)
            build.set_output({"characters": len(context), "documents": len(top)})

        response = trace_ollama_chat(
            _CLIENT.chat,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer in one or two sentences using ONLY these notes:\n" + context,
                },
                {"role": "user", "content": question},
            ],
            bir_metadata={"recipe": "ollama-02-structure"},
        )
        answer = response.message.content

        # Scores attach numeric judgments to the trace; metadata explains them.
        score(
            "retrieval_top_score",
            hits[0]["score"] if hits else 0.0,
            metadata={"stage": "keyword_search", "keywords": keywords},
        )
        score(
            "groundedness",
            _grounded_share(answer, context),
            metadata={"method": "share of answer content words present in the retrieved context"},
        )
        return answer


# --------------------------------------------------------------------------- #
# Self-verification: reload the trace from disk and print it as a tree, so the
# nesting produced by the context managers above is visible in the terminal.
# --------------------------------------------------------------------------- #
def _event_details(event: TraceEvent) -> str:
    if event.type == "generation":
        total = (event.usage or {}).get("total_tokens")
        return f"model={event.model} tokens={total}"
    if event.type == "score":
        return f"value={event.value}"
    if isinstance(event.output, dict) and isinstance(event.output.get("documents"), list):
        return f"documents={len(event.output['documents'])}"
    return ""


def _print_tree(events: list[TraceEvent]) -> None:
    children: dict[str | None, list[TraceEvent]] = {}
    by_id = {event.id: event for event in events}
    for event in events:
        # The trace root's parent_id is None; treat any missing parent as root-level.
        parent = event.parent_id if event.parent_id in by_id else None
        children.setdefault(parent, []).append(event)
    for siblings in children.values():
        siblings.sort(key=lambda e: e.start_time)

    def walk(event: TraceEvent, depth: int) -> None:
        label = "retrieval" if event.metadata.get("kind") == "retrieval" else event.type
        details = _event_details(event)
        print(f"[bir] {'  ' * depth}{label:<10} {event.name}" + (f"  ({details})" if details else ""))
        for child in children.get(event.id, []):
            walk(child, depth + 1)

    for root in children.get(None, []):
        walk(root, 0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lesson 02 — trace a nested RAG pipeline (spans, tools, retrieval, scores)."
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--trace-path", default=str(DEFAULT_TRACE_PATH))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Offline mode: in-file fake Ollama client (no server, no network). "
        "Also enabled by BIR_COOKBOOK_SMOKE=1.",
    )
    args = parser.parse_args()

    smoke = args.smoke or os.environ.get("BIR_COOKBOOK_SMOKE") == "1"
    trace_path = Path(args.trace_path)
    configure(trace_path=trace_path, capture_inputs=True, capture_outputs=True)

    global _CLIENT
    _CLIENT = _build_client(smoke)

    try:
        text = answer_question(args.prompt, args.model)
    except Exception as exc:  # pragma: no cover - real-path only
        if smoke:
            raise
        raise SystemExit(
            f"Ollama call failed: {exc}\n"
            f"Ensure Ollama is running and the model is pulled "
            f"(`ollama pull {args.model}`), or run with --smoke."
        ) from exc

    print(text)

    latest = load_traces(trace_path)[-1]
    print(f"\n[bir] trace_id={latest.id}")
    print(f"[bir] events={len(latest.events)}")
    _print_tree(latest.events)
    print(f"[bir] wrote {trace_path}")


if __name__ == "__main__":
    main()
