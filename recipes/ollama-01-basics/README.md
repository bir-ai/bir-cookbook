# Ollama · Lesson 01 — basics

**Phase 1, Lesson 01 of the Ollama feature tour.** Your first traced Ollama
calls: make one real, local call on each sync surface — `chat` and `generate` —
and record both in a single Bir trace.

This is the smallest possible slice of the SDK — just enough to see a trace on
disk. Later lessons add spans, tool calls, retrieval, prompts, governance, cost,
persistence, and evals.

## What it shows

- `configure(trace_path=..., capture_inputs=True, capture_outputs=True)` writing
  to the recipe-local `./.bir/traces.jsonl`.
- An `@observe`-decorated function that makes one call on each sync Ollama
  surface, each recorded as a Bir **generation** with the model and token usage:
  a chat call wrapped with `bir.integrations.ollama.trace_chat` (imported here
  as `trace_ollama_chat`) and a generate call wrapped with `trace_generate`
  (imported as `trace_ollama_generate`).
- The two surfaces' different response shapes: chat answers at
  `message.content`, generate at `response`; both report token usage at the
  top-level `prompt_eval_count` / `eval_count`.
- `load_traces(...)` afterward — the script prints the `trace_id`, event count,
  and each generation's model and token usage so a run is self-verifying.

## Key

**None.** Ollama runs locally and is keyless — there is no API key and no `.env`
to fill in.

## Run it

```bash
# Offline smoke — no Ollama, no network, deterministic (what CI runs):
uv run python main.py --smoke

# Real run — needs a local Ollama server and a pulled model:
ollama pull llama3.2:1b
uv run python main.py --prompt "In one sentence, what is LLM observability?"
```

Flags: `--prompt`, `--model` (default: the `cookbook.env` model), `--trace-path`, `--smoke`
(also `BIR_COOKBOOK_SMOKE=1`).

If Ollama isn't reachable, the real run exits with a message pointing you to
`ollama pull llama3.2:1b` and to `--smoke`.

## What you'll see

```
[bir] trace_id=...
[bir] events=3
[bir] generation ollama.chat      model=llama3.2:1b  usage: in=… out=… total=…
[bir] generation ollama.generate  model=llama3.2:1b  usage: in=… out=… total=…
[bir] wrote ./.bir/traces.jsonl
```

Three events: the `@observe` trace root (`ollama_basics`) and the two
`generation`s (`ollama.chat` and `ollama.generate`) nested inside it. Inspect
the raw records with:

```bash
cat .bir/traces.jsonl
```

The same file can also be browsed without any Python — the SDK ships a `bir`
console script, so `uv run bir traces` lists it as a table (the full CLI tour
is in [Lesson 07](../ollama-07-persistence/)).
