# Ollama · Lesson 02 — structure

**Phase 1, Lesson 02 of the Ollama feature tour.** Lesson 01 recorded one flat
generation; this lesson records a *nested pipeline*. A tiny in-file RAG flow —
keyword recall → rerank → context building → one local Ollama call → scoring —
where every stage becomes its own event in the trace tree.

## What it shows

- `with bir.trace(name, metadata=...)` — the explicit trace root, the
  alternative to Lesson 01's `@observe`.
- `span(...)` — nested plain work (query preparation), plus `set_metadata`.
- `retrieval(...)` — Bir's RAG event shape, **both ways**:
  - recall records hits one by one with `add_document(id=, text=, rank=,
    score=, source=)`;
  - rerank replaces the list wholesale with `set_documents([...])`.
- `tool_call(...)` — a non-LLM step (building the context block), plus
  `set_output`.
- `score(name, value, metadata=...)` — two numeric judgments (top retrieval
  score, answer groundedness) attached to the trace.
- After the run, the script reloads the trace with `load_traces(...)` and
  prints it as an **indented tree**, so the nesting is visible.

## Key

**None.** Ollama runs locally and is keyless — there is no API key and no `.env`
to fill in.

## Run it

```bash
# Offline smoke — no Ollama, no network, deterministic (what CI runs):
uv run python main.py --smoke

# Real run — needs a local Ollama server and a pulled model:
ollama pull llama3.2:1b
uv run python main.py --prompt "How do scores relate to a trace?"
```

Flags: `--prompt`, `--model` (default: the `cookbook.env` model), `--trace-path`, `--smoke`
(also `BIR_COOKBOOK_SMOKE=1`).

## What you'll see

```
[bir] trace_id=...
[bir] events=8
[bir] trace      ollama_structure
[bir]   span       prepare_query
[bir]   retrieval  keyword_search  (documents=4)
[bir]   retrieval  rerank  (documents=2)
[bir]   tool_call  build_context
[bir]   generation ollama.chat  (model=llama3.2:1b tokens=…)
[bir]   score      retrieval_top_score  (value=…)
[bir]   score      groundedness  (value=…)
[bir] wrote ./.bir/traces.jsonl
```

The pipeline retrieves from six in-file notes describing Bir's own event types,
so the recipe documents the SDK while demonstrating it. Inspect the raw records
with:

```bash
cat .bir/traces.jsonl
```
