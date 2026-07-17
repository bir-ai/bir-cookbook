# Ollama · Lesson 04 — async, streaming, generators

**Phase 1, Lesson 04 of the Ollama feature tour.** Tracing code that doesn't
block: two async chat calls run **concurrently** without mixing up their traces,
a **streamed** call prints tokens live while the trace still captures the whole
answer, an **async streamed generate** call combines the two and completes the
SDK's Ollama wrapper matrix, and an `@observe`-decorated **generator** is traced
across its entire iteration lifetime — including what happens when you
`close()` it early.

## What it shows

- **async** — `@observe` on an `async def` function, with
  `bir.integrations.ollama.trace_chat_async` awaiting an `ollama.AsyncClient`
  chat inside one generation event. Two calls run under `asyncio.gather`;
  contextvars are task-local, so each task opens its own trace root and the
  concurrent traces never cross-contaminate. The summary proves it: each
  task's `get_current_trace_id()` matches a distinct reloaded trace, and
  neither captured request contains the other task's question.
- **streaming** — `trace_chat(..., stream=True)` returns a lazy iterable that
  yields Ollama's chunks unchanged (tokens print as they arrive), while the
  wrapper assembles the output from each chunk's `message.content` delta and
  reads token usage from the terminal `done` chunk. The reloaded generation
  event holds the full text **and** the final usage.
- **async + streaming** — `trace_generate_async(..., stream=True)` covers the
  fourth Ollama wrapper (chat / generate × sync / async). Awaiting it resolves
  to an **async iterator**; `generate` chunks carry the text delta at
  `response` (not `message.content`), the terminal `done` chunk carries the
  top-level `prompt_eval_count` / `eval_count`, and the generation's output
  and usage finalize only once the stream is fully consumed.
- **generators** — `@observe` also traces generator functions, for their full
  iteration lifetime: creation stays lazy, the trace stays open across every
  `next`, and the root records `metadata.generator.outcome` (`"completed"` on
  exhaustion, `"closed"` on early `close()`) plus a bounded
  `metadata.generator.items` count — yielded values are never buffered.

## Key

**None.** Ollama runs locally and is keyless — there is no API key and no `.env`
to fill in.

## Run it

```bash
# Offline smoke — no Ollama, no network, deterministic (what CI runs):
uv run python main.py --smoke

# Real run — needs a local Ollama server and a pulled model:
ollama pull llama3.2:1b
uv run python main.py --prompt "Why show an answer word by word?"
```

Flags: `--prompt` (feeds the streaming parts; part A asks two fixed questions
concurrently), `--model` (default `llama3.2:1b`), `--trace-path`, `--smoke`
(also `BIR_COOKBOOK_SMOKE=1`).

## What you'll see

```
== A · async: two concurrent traced calls (asyncio.gather) ==
[task 1] trace=…
[task 1] Q: In one short sentence: what is a race condition?
[task 1] A: …
[task 2] trace=…
…

== B · streaming: trace_chat(stream=True) ==
…tokens printing as they arrive…

== C · async + streaming: trace_generate_async(stream=True) ==
…tokens printing as they arrive, now from the generate surface…

== D · generators: @observe on a generator function ==
[gen] …tokens again, now yielded by an observed generator…
[gen] took 3 items (…) then close() — the body never ran again

[bir] traces this run (6):
[bir]   …  root=async_answer        events=2  model=llama3.2:1b  total_tokens=…
[bir]   …  root=async_answer        events=2  model=llama3.2:1b  total_tokens=…
[bir]   …  root=stream_answer       events=2  model=llama3.2:1b  total_tokens=…
[bir]   …  root=astream_completion  events=2  model=llama3.2:1b  total_tokens=…
[bir]   …  root=stream_tokens       events=2  model=llama3.2:1b  total_tokens=…
[bir]   …  root=stream_tokens       events=2  model=llama3.2:1b  total_tokens=0
[bir] async isolation (contextvars):
[bir]   trace …  own question captured: True  other task's question leaked: False
[bir]   trace …  own question captured: True  other task's question leaked: False
[bir] streaming: reloaded output == streamed text: True  chars=…  usage: in=… out=… total=…
[bir] async streaming generate: reloaded output == streamed text: True  chars=…  usage: in=… out=… total=…
[bir] generator lifetimes (metadata.generator):
[bir]   stream_tokens  outcome=completed  items=…
[bir]   stream_tokens  outcome=closed     items=3
[bir] wrote ./.bir/traces.jsonl
```

One run appends **six** traces: one per async task (that's the isolation
lesson), one for the streamed chat call, one for the async streamed generate
call, and two for the generator (consumed fully, then closed early). The
closed generator's nested generation records the partial output but **no
usage** — Ollama's token counts ride on the terminal `done` chunk, which never
arrived. Inspect the raw records with:

```bash
cat .bir/traces.jsonl
```
