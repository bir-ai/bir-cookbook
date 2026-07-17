# Ollama · Lesson 05 — governance

**Phase 1, Lesson 05 of the Ollama feature tour.** Production controls: the
same traced Ollama calls as earlier lessons, now governed — tagged by
deployment, switchable off during an incident, sampled under volume, redacted
beyond the built-in rules, capped in how much of any one payload reaches disk,
and — for one sensitive call — recorded without its payloads entirely.

All but the last are `configure()` arguments. `configure()` mutates one
process-global config, so the lesson calls it several times on purpose — once
per part, printing each call — and self-verifies every part by reloading
`./.bir/traces.jsonl` and counting what did (or deliberately did not) get
recorded. The last control (part F) rides on the individual call instead of
the global config.

## What it shows

- **A · deployment tags** — `configure(service_name=, environment=, source=)`.
  The reloaded trace ROOT carries `metadata.service` (keys `name` /
  `environment`) and `metadata.source`; child events never do.
- **B · the `enabled` kill switch** — with `configure(enabled=False)` the
  traced call still runs and still returns its answer, and
  `get_current_trace_id()` still returns a live in-process id (log correlation
  keeps working), but the on-disk trace count does not move — a trace already
  in flight stops writing too. `configure(enabled=True)` restores recording for
  traces started afterward. Env counterpart: a truthy `BIR_DISABLED` (an
  explicit `enabled=True` overrides it).
- **C · sampling** — `sample_rules` maps EXACT trace-root names to per-root
  rates that override the global `sample_rate`: rules at `1.0` / `0.0` show
  always-kept vs never-kept deterministically (the never-kept call still
  answers — the work runs, the write doesn't). Then a fractional global
  `sample_rate=0.25` is shown statistically over 200 cheap traced calls — the
  keep/drop decision is made once per trace root and inherited by every
  descendant event. Passing `sample_rules` replaces the prior rule table (`{}`
  clears it).
- **D · redaction** — `additional_secret_keys` / `additional_redaction_patterns`
  only ever WIDEN the built-in redaction; they can never weaken it. A request
  payload with an obviously fake `api_key` field (caught by the always-on
  built-ins), a `session-fingerprint` field (caught by the additional key —
  whole-name, case-insensitive, `-` == `_`), and a token matching a custom
  regex (redacted inside the captured Ollama generation input) goes in;
  `[redacted]` is all that reaches disk.
- **E · capture limits** — `max_value_length` truncates a captured string with
  a visible `…[truncated]` marker; `max_collection_items` keeps a collection's
  first N items plus one marker. Truncation always runs AFTER redaction, so a
  cut can never expose part of a secret. Both default to `None` (unlimited).
- **F · per-call capture override** — no `configure()` change at all. One call
  handles sensitive data (a customer name and account id no redaction pattern
  would flag), so it passes `bir_capture_input=False` /
  `bir_capture_output=False` to the wrapper — the `bir_`-prefixed params map
  straight to `bir.generation(capture_input=..., capture_output=...)` and
  override the global capture config for that single call. The reloaded trace
  shows that generation with no input/output payloads but its event, model,
  and token usage intact — while a baseline call in the SAME trace captured
  everything. Keep the trace, not the payload.

## Key

**None.** Ollama runs locally and is keyless — there is no API key and no `.env`
to fill in. The "secrets" in Part D and the "customer PII" in Part F are fake
by construction and exist only to be kept off disk.

## Run it

```bash
# Offline smoke — no Ollama, no network, deterministic (what CI runs):
uv run python main.py --smoke

# Real run — needs a local Ollama server and a pulled model:
ollama pull llama3.2:1b
uv run python main.py --prompt "Why tag traces with an environment?"
```

Flags: `--prompt` (feeds parts A and B; C–F use fixed inputs), `--model`
(default `llama3.2:1b`), `--trace-path`, `--smoke` (also `BIR_COOKBOOK_SMOKE=1`).
The statistical kept-count in part C varies run to run; every `✓` check is
deterministic.

## What you'll see

```
== A · deployment tags: service_name / environment / source ==
[bir] configure(service_name="support-copilot", environment="staging", source="ollama-05-governance")
…the model's answer…
[bir] ✓ A: one new trace on disk
[bir] ✓ A: root metadata.service == {'name': 'support-copilot', 'environment': 'staging'}
[bir] ✓ A: root metadata.source == 'ollama-05-governance'
[bir] ✓ A: tags land on trace ROOTS only, never on child events

== B · the enabled kill switch: code runs, nothing is written ==
[bir] configure(enabled=False)   # incident toggle; env twin: BIR_DISABLED=1
…the model still answers…
[bir] ✓ B: trace count on disk did NOT change while disabled
[bir] ✓ B: get_current_trace_id() still returned a live id while disabled

== C · sampling: exact-name rules, then a fractional global rate ==
[bir] ✓ C: two traced calls ran, exactly one trace reached disk
[bir] global sample_rate=0.25: kept 47/200 traced calls (expected ≈50; …)

== D · redaction: widen the built-in rules, never weaken them ==
[bir] ✓ D: built-in KEY rule redacted "api_key" — always on, no configuration needed
[bir] ✓ D: custom pattern redacted the token inside the captured Ollama generation input
[bir] ✓ D: no fake secret appears anywhere in the raw on-disk trace

== E · capture limits: bound one huge payload, redact before the cut ==
[bir] ✓ E: string capped at 80 chars of redacted text + '…[truncated]'
[bir] ✓ E: redaction ran BEFORE the cut — no fragment of the token survived

== F · per-call capture override: keep the trace, drop the payload ==
[bir] no configure() call — global capture stays ON; the override rides on one call
[bir] sensitive answer (terminal only, never captured): …
[bir] ✓ F: the baseline generation in the SAME trace captured input and output
[bir] ✓ F: the sensitive generation captured NEITHER input nor output
[bir] ✓ F: …but its model was still recorded (llama3.2:1b)
[bir] ✓ F: …and its token usage too ({'input_tokens': 19, 'output_tokens': 31, 'total_tokens': 50})
[bir] ✓ F: the fake customer PII appears nowhere in the raw on-disk trace

[bir] all governance checks passed — traces on disk: …
[bir] wrote ./.bir/traces.jsonl
```

Note the count in part C's rules check: two Ollama calls ran, one trace was
written. Sampled-out (and kill-switched) code is never skipped — only the
recording is. Inspect the raw records with:

```bash
cat .bir/traces.jsonl
```
