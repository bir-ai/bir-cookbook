# Ollama · Lesson 01 — basics

**Phase 1, Lesson 01 of the Ollama feature tour.** Your first traced Ollama
call: make ONE real, local chat call and record it as a Bir trace.

This is the smallest possible slice of the SDK — just enough to see a trace on
disk. Later lessons add spans, tool calls, retrieval, prompts, governance, cost,
persistence, and evals.

## What it shows

- `configure(trace_path=..., capture_inputs=True, capture_outputs=True)` writing
  to the recipe-local `./.bir/traces.jsonl`.
- An `@observe`-decorated function that makes one Ollama chat call wrapped with
  `bir.integrations.ollama.trace_chat` (imported here as `trace_ollama_chat`),
  recorded as a Bir **generation** with the model and token usage.
- `load_traces(...)` afterward — the script prints the `trace_id`, event count,
  model, and token usage so a run is self-verifying.

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

Flags: `--prompt`, `--model` (default `llama3.2:1b`), `--trace-path`, `--smoke`
(also `BIR_COOKBOOK_SMOKE=1`).

If Ollama isn't reachable, the real run exits with a message pointing you to
`ollama pull llama3.2:1b` and to `--smoke`.

## What you'll see

```
[bir] trace_id=...
[bir] events=2  model=llama3.2:1b  usage: in=… out=… total=…
[bir] wrote ./.bir/traces.jsonl
```

Two events: the `@observe` trace root (`ollama_basics`) and the `generation`
(`ollama.chat`) nested inside it. Inspect the raw records with:

```bash
cat .bir/traces.jsonl
```
