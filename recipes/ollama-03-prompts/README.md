# Ollama · Lesson 03 — prompts & correlation

**Phase 1, Lesson 03 of the Ollama feature tour.** Two production questions,
one lesson: *which prompt version produced this output?* and *which trace does
this log line belong to?* The same question is asked through **two versions of
the same named prompt** (one Ollama generation each), while ordinary stdlib log
lines are stamped with the active trace and span ids.

## What it shows

- `bir.prompt(name, version=, template=, variables=)` — a versioned
  `PromptRecord` attached via `generation(..., prompt=record)`, so every
  generation event records exactly which prompt produced it. A
  `template_sha256` is always stored; the template / variables / rendered text
  appear only because the `capture_template` / `capture_variables` /
  `capture_rendered` flags are set (all default to `False` — prompt content
  stays private by default).
- the **manual `generation(...)` context manager** — the primitive the
  integration wrappers are built on (the Ollama wrapper from Lessons 01–02
  doesn't take a prompt record), with `set_model`, `set_output`, `set_usage`.
- `get_current_trace_id()` / `get_current_span_id()` — the ids every event of
  the active trace carries, readable from anywhere inside it.
- `bir.logging.install_trace_id_filter` — stamps those ids onto every stdlib
  `LogRecord` (`%(bir_trace_id)s` / `%(bir_span_id)s`), so log ↔ trace
  correlation needs zero per-call plumbing.
- After the run: a **version → output** table from the reloaded trace, and a
  **correlation demo** that takes one stamped log line and uses only its ids to
  find the matching trace *and* exact event in `./.bir/traces.jsonl`.

## Key

**None.** Ollama runs locally and is keyless — there is no API key and no `.env`
to fill in.

## Run it

```bash
# Offline smoke — no Ollama, no network, deterministic (what CI runs):
uv run python main.py --smoke

# Real run — needs a local Ollama server and a pulled model:
ollama pull llama3.2:1b
uv run python main.py --prompt "Why version prompts at all?"
```

Flags: `--prompt`, `--model` (default `llama3.2:1b`), `--trace-path`, `--smoke`
(also `BIR_COOKBOOK_SMOKE=1`).

## What you'll see

Stamped log lines on stderr while it runs — `trace=None` outside the trace,
real ids inside (inside a generation, `span=` is that generation's own id):

```
INFO app [trace=None span=None] lesson 03 starting (smoke=True)
INFO app [trace=6636e01a… span=6636e01a…] get_current_trace_id()=… get_current_span_id()=…
INFO app [trace=6636e01a… span=e56c1a2b…] calling ollama with prompt cookbook-qa@v1
INFO app [trace=6636e01a… span=91d24c3f…] calling ollama with prompt cookbook-qa@v2
```

In a real run even the ollama client's own `httpx` log lines come out stamped —
the filter sits on the *handler*, so third-party loggers that propagate to root
are correlated too, with zero changes to their code.

then the self-verifying summary:

```
cookbook-qa@v1: …
cookbook-qa@v2: …

[bir] trace_id=…
[bir] events=3  model=llama3.2:1b  total_tokens=…
[bir] prompt versions in this trace:
[bir]   cookbook-qa@v1  template_sha256=…  -> '…'
[bir]   cookbook-qa@v2  template_sha256=…  -> '…'
[bir] log ↔ trace correlation:
[bir]   log line:   INFO app [trace=… span=…] calling ollama with prompt cookbook-qa@v2
[bir]   trace_id -> trace … (3 events) in traces.jsonl
[bir]   span_id  -> generation 'ask_v2' (model=llama3.2:1b)
[bir] wrote ./.bir/traces.jsonl
```

Inspect the raw records — each generation's `metadata.prompt` holds the name,
version, hash, template, variables, and rendered text:

```bash
cat .bir/traces.jsonl
```
