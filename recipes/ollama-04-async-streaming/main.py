"""Phase 1 · Lesson 04 — async, streaming, generators: tracing code that doesn't block.

Lessons 01–03 traced synchronous, one-shot calls. Real services await, stream,
and yield — and the SDK traces all three without changing how the code behaves:

  * **async** — ``@observe`` decorates ``async def`` functions too, and
    ``trace_chat_async`` awaits an ``ollama.AsyncClient`` chat inside one
    generation event. Two calls run concurrently under ``asyncio.gather``;
    contextvars are task-local, so each task opens its OWN trace root and the
    concurrent traces never cross-contaminate. The summary proves it by
    matching each task's ``get_current_trace_id()`` against the reloaded
    traces and checking neither captured request leaked the other's question.
  * **streaming** — ``trace_chat(..., stream=True)`` returns a lazy iterable
    that yields Ollama's chunks unchanged, so tokens hit the terminal as they
    arrive; meanwhile the wrapper assembles the output from each chunk's
    ``message.content`` delta and reads token usage from the terminal ``done``
    chunk. The reloaded generation event holds the full text AND the usage.
  * **generators** — ``@observe`` also traces generator functions, for their
    full iteration lifetime: creation stays lazy, the trace stays open across
    every ``next``, and it finalizes on exhaustion
    (``metadata.generator.outcome == "completed"``) or early ``close()``
    (``"closed"``). Yielded values are never buffered — only a bounded
    ``metadata.generator.items`` count is recorded.

Run it:
  * offline (no Ollama, no network, deterministic — what CI runs):
      uv run python main.py --smoke
  * real (needs a local Ollama server + a pulled model):
      ollama pull llama3.2:1b
      uv run python main.py --prompt "Why show an answer word by word?"

Ollama is local and keyless, so there is no API key to set.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import os
from pathlib import Path

from bir import configure, get_current_trace_id, load_traces, observe
from bir.integrations.ollama import (
    trace_chat as trace_ollama_chat,
    trace_chat_async as trace_ollama_chat_async,
)

DEFAULT_TRACE_PATH = Path(__file__).resolve().parent / ".bir" / "traces.jsonl"
DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_PROMPT = "In one short sentence: why show a chatbot's answer word by word instead of all at once?"

RECIPE = "ollama-04-async-streaming"

# Part A asks two DIFFERENT questions concurrently, so the isolation check can
# look for one task's question leaking into the other task's captured request.
CONCURRENT_QUESTIONS = (
    "In one short sentence: what is a race condition?",
    "In one short sentence: what is backpressure?",
)

# Part C consumes this many items before close() — enough to prove the
# generator ran, few enough that the stream is clearly cut short.
EARLY_CLOSE_AFTER = 3

# One run appends exactly these five traces: two async tasks, one streamed
# call, and the generator consumed twice (fully, then closed early).
TRACES_PER_RUN = 5

# Keep the clients module-level so they are never passed as @observe arguments
# and therefore never captured as inputs (see CLAUDE.md).
_CLIENT = None
_ASYNC_CLIENT = None


# --------------------------------------------------------------------------- #
# Offline fakes — only used with --smoke. The sync fake mirrors both shapes the
# streaming wrapper reads: with ``stream=True`` it returns an iterator of
# chunks carrying ``message.content`` deltas, ending in a ``done`` chunk that
# holds the token counts at the top level (``prompt_eval_count`` /
# ``eval_count``) exactly like Ollama's terminal chunk. The async fake mirrors
# ``ollama.AsyncClient``: its ``chat`` is a coroutine the async wrapper awaits.
# --------------------------------------------------------------------------- #
_SMOKE_STREAM_TOKENS = (
    "(smoke) ", "Streaming ", "keeps ", "the ", "tokens ", "flowing ",
    "while ", "the ", "trace ", "records ", "the ", "whole ", "answer.",
)


class _FakeMessage:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def model_dump(self) -> dict:
        return {"role": self.role, "content": self.content}


class _FakeChatResponse:
    def __init__(self, *, model: str, content: str, prompt_eval_count: int, eval_count: int) -> None:
        self.model = model
        self.message = _FakeMessage("assistant", content)
        self.prompt_eval_count = prompt_eval_count
        self.eval_count = eval_count
        self.done = True

    def model_dump(self) -> dict:
        return {
            "model": self.model,
            "message": self.message.model_dump(),
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
            "done": self.done,
        }


class _FakeChatChunk:
    """One streamed chunk: a ``message.content`` delta, counts only when done."""

    def __init__(self, *, model: str, content: str, done: bool,
                 prompt_eval_count: int | None = None, eval_count: int | None = None) -> None:
        self.model = model
        self.message = _FakeMessage("assistant", content)
        self.done = done
        self.prompt_eval_count = prompt_eval_count
        self.eval_count = eval_count


class _FakeOllamaClient:
    """Deterministic stand-in for ``ollama.Client`` used only in --smoke mode."""

    def chat(self, *, model: str, messages: list[dict], stream: bool = False, **_kwargs):
        question = messages[-1]["content"] if messages else ""
        if stream:
            return self._stream(model, messages)
        content = f"(smoke) '{question}' answered in its own isolated trace."
        return _FakeChatResponse(
            model=model,
            content=content,
            prompt_eval_count=sum(len(m["content"].split()) for m in messages),
            eval_count=len(content.split()),
        )

    def _stream(self, model: str, messages: list[dict]):
        for token in _SMOKE_STREAM_TOKENS:
            yield _FakeChatChunk(model=model, content=token, done=False)
        # The terminal chunk repeats an empty delta and carries the counts —
        # the shape the wrapper reads the final usage from.
        yield _FakeChatChunk(
            model=model,
            content="",
            done=True,
            prompt_eval_count=sum(len(m["content"].split()) for m in messages),
            eval_count=len(_SMOKE_STREAM_TOKENS),
        )


class _FakeAsyncOllamaClient:
    """Deterministic stand-in for ``ollama.AsyncClient`` used only in --smoke mode."""

    def __init__(self) -> None:
        self._sync = _FakeOllamaClient()

    async def chat(self, *, model: str, messages: list[dict], **_kwargs) -> _FakeChatResponse:
        # Yield the event loop once so gather genuinely interleaves the tasks.
        await asyncio.sleep(0)
        return self._sync.chat(model=model, messages=messages)


def _build_clients(smoke: bool):
    if smoke:
        return _FakeOllamaClient(), _FakeAsyncOllamaClient()

    # Real mode: talk to a local Ollama server. Import the provider library
    # lazily so --smoke needs neither the package nor a running server. Ollama
    # needs no API key — the client honors the OLLAMA_HOST env var if set.
    try:
        from ollama import AsyncClient, Client
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
    return client, AsyncClient()


# --------------------------------------------------------------------------- #
# Part A — async: @observe on an async def, trace_chat_async, asyncio.gather.
# --------------------------------------------------------------------------- #
@observe(name="async_answer", capture_inputs=True, capture_outputs=True, metadata={"recipe": RECIPE})
async def async_answer(question: str, model: str) -> dict[str, str]:
    """One traced async chat call — run concurrently with its twin under gather."""

    # asyncio.gather wraps each coroutine in a Task, and every Task runs in a
    # COPY of the current context. No trace is active when the tasks start, so
    # each @observe call here opens its own trace root, and contextvars keep
    # the two concurrent traces from cross-contaminating.
    trace_id = get_current_trace_id()

    # trace_chat_async awaits the AsyncClient's chat coroutine inside a single
    # generation event; arguments are forwarded to Ollama unchanged.
    response = await trace_ollama_chat_async(
        _ASYNC_CLIENT.chat,
        model=model,
        messages=[{"role": "user", "content": question}],
        bir_metadata={"recipe": RECIPE},
    )
    return {"trace_id": trace_id, "question": question, "answer": response.message.content}


async def _gather_answers(model: str) -> list[dict[str, str]]:
    return await asyncio.gather(*(async_answer(q, model) for q in CONCURRENT_QUESTIONS))


# --------------------------------------------------------------------------- #
# Part B — streaming: trace_chat(stream=True) yields chunks live while the
# wrapper assembles the full output and final usage behind the scenes.
# --------------------------------------------------------------------------- #
@observe(name="stream_answer", capture_inputs=True, capture_outputs=True, metadata={"recipe": RECIPE})
def stream_answer(question: str, model: str) -> str:
    """Stream one chat call to the terminal; return the text we saw arrive."""

    stream = trace_ollama_chat(
        _CLIENT.chat,
        model=model,
        messages=[{"role": "user", "content": question}],
        stream=True,
        bir_metadata={"recipe": RECIPE},
    )
    parts: list[str] = []
    for chunk in stream:
        # Chunks pass through unchanged; the terminal done chunk repeats an
        # empty delta, so only non-empty text is printed.
        text = chunk.message.content
        if text:
            print(text, end="", flush=True)
            parts.append(text)
    print()
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Part C — generators: @observe traces a generator function for its whole
# iteration lifetime, not just its creation.
# --------------------------------------------------------------------------- #
@observe(name="stream_tokens", capture_inputs=True, capture_outputs=True, metadata={"recipe": RECIPE})
def stream_tokens(question: str, model: str):
    """Yield each text delta from a traced streaming call — chunk objects in,
    plain strings out.

    Nothing runs until the first ``next``; the nested generation attaches to
    this generator's trace; exhaustion vs ``close()`` is recorded under the
    root's ``metadata.generator``.
    """

    stream = trace_ollama_chat(
        _CLIENT.chat,
        model=model,
        messages=[{"role": "user", "content": question}],
        stream=True,
        bir_metadata={"recipe": RECIPE},
    )
    try:
        for chunk in stream:
            text = chunk.message.content
            if text:
                yield text
    finally:
        # On early close() the GeneratorExit lands here; closing the wrapper's
        # stream deterministically finalizes the generation event with the
        # partial output accumulated so far (the done chunk never arrived, so
        # that generation records no usage).
        stream.close()


def _consume_fully(question: str, model: str) -> int:
    print("[gen] ", end="")
    items = 0
    for token in stream_tokens(question, model):
        print(token, end="", flush=True)
        items += 1
    print()
    return items


def _close_early(question: str, model: str) -> None:
    gen = stream_tokens(question, model)
    taken = list(itertools.islice(gen, EARLY_CLOSE_AFTER))
    gen.close()
    print(f"[gen] took {len(taken)} items ({''.join(taken).strip()!r}…) then close() — the body never ran again")


# --------------------------------------------------------------------------- #
# Self-verification: reload this run's traces and check each part's claim —
# isolated async traces, the assembled streamed output + usage, and the
# generator outcomes.
# --------------------------------------------------------------------------- #
def _root(trace):
    return next(e for e in trace.events if e.type == "trace")


def _generation(trace):
    return next(e for e in trace.events if e.type == "generation")


def _summarize(trace_path: Path, async_results: list[dict[str, str]], streamed_text: str) -> None:
    traces = load_traces(trace_path)[-TRACES_PER_RUN:]

    print(f"\n[bir] traces this run ({len(traces)}):")
    for trace in traces:
        root = _root(trace)
        gens = [e for e in trace.events if e.type == "generation"]
        total = sum((g.usage or {}).get("total_tokens", 0) for g in gens)
        model = gens[-1].model if gens else "?"
        print(f"[bir]   {trace.id}  root={root.name:<13}  events={len(trace.events)}  model={model}  total_tokens={total}")

    # A — each task's get_current_trace_id() names a real, distinct trace whose
    # captured request holds only that task's question.
    by_id = {t.id: t for t in traces}
    print("[bir] async isolation (contextvars):")
    for result in async_results:
        gen = _generation(by_id[result["trace_id"]])
        request = str(gen.input)
        own = result["question"] in request
        leaked = any(r["question"] in request for r in async_results if r is not result)
        print(
            f"[bir]   trace {result['trace_id'][:8]}…  own question captured: {own}  "
            f"other task's question leaked: {leaked}"
        )

    # B — the generation persisted the FULL assembled text plus the usage from
    # the terminal done chunk, even though the caller only saw chunks.
    gen = _generation(next(t for t in traces if _root(t).name == "stream_answer"))
    usage = gen.usage or {}
    print(
        f"[bir] streaming: reloaded output == streamed text: {gen.output == streamed_text}  "
        f"chars={len(str(gen.output))}  usage: in={usage.get('input_tokens')} "
        f"out={usage.get('output_tokens')} total={usage.get('total_tokens')}"
    )

    # C — the generator roots carry outcome + bounded item count, never the
    # yielded values themselves.
    print("[bir] generator lifetimes (metadata.generator):")
    for trace in traces:
        root = _root(trace)
        if root.name != "stream_tokens":
            continue
        meta = root.metadata.get("generator", {})
        print(f"[bir]   stream_tokens  outcome={meta.get('outcome'):<9}  items={meta.get('items')}")

    print(f"[bir] wrote {trace_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lesson 04 — async tracing, token streaming, and observed generators."
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--trace-path", default=str(DEFAULT_TRACE_PATH))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Offline mode: in-file fake Ollama clients (no server, no network). "
        "Also enabled by BIR_COOKBOOK_SMOKE=1.",
    )
    args = parser.parse_args()

    smoke = args.smoke or os.environ.get("BIR_COOKBOOK_SMOKE") == "1"
    trace_path = Path(args.trace_path)
    configure(trace_path=trace_path, capture_inputs=True, capture_outputs=True)

    global _CLIENT, _ASYNC_CLIENT
    _CLIENT, _ASYNC_CLIENT = _build_clients(smoke)

    try:
        print("== A · async: two concurrent traced calls (asyncio.gather) ==")
        async_results = asyncio.run(_gather_answers(args.model))
        for i, result in enumerate(async_results, start=1):
            print(f"[task {i}] trace={result['trace_id']}")
            print(f"[task {i}] Q: {result['question']}")
            print(f"[task {i}] A: {result['answer']}")

        print("\n== B · streaming: trace_chat(stream=True) ==")
        streamed_text = stream_answer(args.prompt, args.model)

        print("\n== C · generators: @observe on a generator function ==")
        _consume_fully(args.prompt, args.model)
        _close_early(args.prompt, args.model)
    except Exception as exc:  # pragma: no cover - real-path only
        if smoke:
            raise
        raise SystemExit(
            f"Ollama call failed: {exc}\n"
            f"Ensure Ollama is running and the model is pulled "
            f"(`ollama pull {args.model}`), or run with --smoke."
        ) from exc

    _summarize(trace_path, async_results, streamed_text)


if __name__ == "__main__":
    main()
