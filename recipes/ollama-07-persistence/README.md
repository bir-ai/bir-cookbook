# Ollama · Lesson 07 — persistence

**Phase 1, Lesson 07 of the Ollama feature tour.** Everything Bir records lands
in one local JSONL file — which grows forever unless you cap it, and helps
nobody else until you ship it. This lesson covers both halves: opt-in size
rotation (`configure(max_bytes=…, backup_count=…)`) and uploading with
`send_events()` to a Bir ingestion server.

There is no free hosted Bir server, so the recipe ships its own: an in-file,
**in-memory fake** (stdlib `http.server` on `127.0.0.1`, ephemeral port) that
speaks the exact wire protocol `send_events()` uses — `POST /v1/events/batch`
answering `{"accepted": <int>, "event_ids": […]}` — and is idempotent on event
ids like a real one. The server is the *subject* of the lesson, so it runs in
**both** modes: `--smoke` still means no *external* services, and only the
Ollama client gets faked. Every part self-verifies by reloading the local
files and querying the fake server's in-memory store; all asserted checks are
deterministic in both modes.

## What it shows

- **A · rotation** — `configure(max_bytes=4096, backup_count=3)`, then two
  real wrapped Ollama calls plus thirty cheap traced journal entries for write
  volume. A write that would push the active file past the cap rotates it
  first: `traces.jsonl` → `.1`, the old `.1` → `.2`, … keeping at most
  `backup_count` backups. Rotation only cuts on whole-line boundaries, so
  every file stays valid JSONL. Default `load_events()` reads **only** the
  active file; `include_rotated=True` adds the backups oldest-first,
  preserving write order.
- **B · the retention trade-off** — the run writes more than `backup_count`
  files can hold, so the oldest backups get **dropped**: the retained
  `include_rotated=True` set is strictly smaller than what was written (the
  run's two *real* Ollama traces are among the casualties — oldest first,
  however precious). Also shown (print-only): a trace split across a rotation
  boundary appears in `load_events()` but is silently *skipped* by
  `load_traces()` for any file that lacks its root — the root line is written
  last, so it lives in the newest of the trace's files.
- **C · `send_events` to a Bir server** — one send with defaults uploads the
  active file: `accepted == attempted ==` the active file's event count, the
  server's stored ids equal `load_events()` ids exactly, and whole traces
  upload **root-first** (the first event the server receives for each trace is
  its root).
- **D · safe re-sends** — a plain re-send re-attempts everything and the
  idempotent server accepts 0 new (`.skipped == attempted`); with
  `mark_sent=True` the acknowledged ids land in a `traces.jsonl.sent` sidecar,
  so the next send attempts 0 events without even POSTing.
- **E · rotation × sending, and retries** — a default send strands rotated
  events on disk; `send_events(include_rotated=True)` sweeps them up,
  deduplicated by id, so the server ends up with exactly the retained set.
  Finally the server is scripted to fail one batch request with a 503: the
  send still succeeds via the built-in retry (`retries=2` by default, sleeping
  `backoff · 2^attempt` between tries), and the server counts exactly two
  batch attempts. A non-404 4xx would raise immediately instead.

## Key

**None.** Ollama runs locally and is keyless, and the Bir "server" is a
loopback fake the script starts and stops itself — nothing leaves your
machine.

## Run it

```bash
# Offline smoke — no Ollama, no external network, deterministic (what CI runs;
# the loopback fake server still runs — it is the lesson):
uv run python main.py --smoke

# Real run — needs a local Ollama server and the pulled model:
ollama pull llama3.2:1b
uv run python main.py --prompt "Why rotate log files?"
```

Flags: `--prompt` (feeds the first real chat call; the rest is fixed),
`--model` (default: the `cookbook.env` model), `--trace-path` (**cleared at startup**,
including `.N` backups and the `.sent` sidecar — the lesson's accounting needs
a clean slate), `--smoke` (also `BIR_COOKBOOK_SMOKE=1`).

## What you'll see

```
[bir] fake Bir ingestion server listening on http://127.0.0.1:xxxxx (in-process, loopback only)

== A · rotation: cap the active file, keep numbered backups ==
[bir] configure(max_bytes=4096, backup_count=3)  # opt-in; the default is ONE ever-growing file
…the model's answers…
[bir] trace_id=…  events=2  model=llama3.2:1b  usage={…}
[bir]   traces.jsonl.3       … bytes
[bir]   traces.jsonl.2       … bytes
[bir]   traces.jsonl.1       … bytes
[bir]   traces.jsonl         … bytes  (active)
[bir] ✓ A: exactly backup_count=3 rotated files on disk (.1 newest … .3 oldest)
[bir] ✓ A: every rotated file is valid, non-empty JSONL — rotation only cuts on whole-line boundaries
[bir] ✓ A: default load_events() reads ONLY the active file — include_rotated=True adds the backups, oldest first

== B · retention: backups are a bounded cache, not an archive ==
[bir] written this run:  32 traces / 94 events
[bir] retained on disk:  … traces / … events (include_rotated=True)
[bir] ✓ B: the retained traces are a strict subset of what was written — the oldest backups were dropped
[bir] ✓ B: past backup_count, even include_rotated=True cannot see everything — rotation deletes, send first
[bir] split trace … spans 2 files; traces.jsonl.… holds its first … event(s)

== C · send_events: upload the active file to a Bir server ==
[bir] ✓ C: the first send accepted everything the active file holds (… events)
[bir] ✓ C: the server's stored ids equal load_events() ids exactly
[bir] ✓ C: for every trace, the FIRST event the server received was its root — whole traces upload root-first

== D · re-sends: idempotent by default, cheap with mark_sent ==
[bir] ✓ D: a plain re-send re-attempts every event; the idempotent server accepts 0 new
[bir] ✓ D: mark_sent=True recorded the acknowledged ids in the traces.jsonl.sent sidecar
[bir] ✓ D: next mark_sent send: every id already recorded — 0 attempted

== E · rotation × sending, and transient failures ==
[bir] ✓ E: rotated events are stranded — the server has none of them
[bir] ✓ E: server ids == the include_rotated local ids — deduplicated by id, nothing double-stored
[bir] ✓ E: the scripted 503 cost one attempt, the retry succeeded — exactly two batch POSTs on the wire

[bir] all persistence checks passed — the server holds … events from … traces
[bir] wrote ./.bir/traces.jsonl (+ .1–.3 backups and the traces.jsonl.sent sidecar)
```

Exact byte sizes and event counts differ between smoke and real runs (model
answers differ in length) — every `✓` compares *relative* quantities reloaded
from disk and from the server's store, so all checks are deterministic either
way. Inspect the raw files with:

```bash
ls -l .bir/            # active file, .1–.3 backups, .sent sidecar
cat .bir/traces.jsonl* # each one valid JSONL on its own
```

## Inspect with the CLI

The SDK installs a `bir` console script (also `python -m bir`), so every
recipe's environment already has it — and its default path is exactly this
recipe's `./.bir/traces.jsonl` when run from this directory (`--path` points it
anywhere else). `bir traces` lists what a load would see — and honors the same
rotation boundary as `load_events()`: by default only the active file, with
`--include-rotated` adding the backups:

```bash
uv run bir traces
# START                             STATUS   DURATION  EVENTS  NAME
# 2026-07-17T20:42:02.433076+00:00  success  1.0ms     3       journal_entry
# 2026-07-17T20:42:02.431023+00:00  success  1.6ms     3       journal_entry
# 2026-07-17T20:42:02.420405+00:00  success  1.5ms     1       journal_entry

uv run bir traces --include-rotated   # same table, 13 traces — the backups too
```

`bir show <trace_id>` renders one trace as the indented event tree (grab an id
from `bir traces --json`):

```bash
uv run bir show 0ab46860-fbaf-4158-91a5-f2b1a03b6f7c
# trace journal_entry [success] 1.0ms
#   span entry.compose [success] 0.0ms
#   span entry.store [success] 0.0ms
```

`bir stats` aggregates counts, tokens, cost, and latency over the same
selection:

```bash
uv run bir stats --include-rotated
# METRIC         VALUE
# traces         13
# success        13
# error          0
# input_tokens   0
# output_tokens  0
# total_tokens   0
# cost           -
# latency_count  13
# latency_mean   4.4ms
# latency_p95    17.1ms
```

Zero tokens even with `--include-rotated` is part B rendered as a table: the
run's generation traces were the oldest, so retention dropped them — rotation
deletes, send first.
