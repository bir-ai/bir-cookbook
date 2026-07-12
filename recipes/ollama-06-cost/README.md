# Ollama · Lesson 06 — cost

**Phase 1, Lesson 06 of the Ollama feature tour.** Spend tracking: Bir bundles
**no prices** — cost is opt-in and local-only. You hand `configure()` a
`model_prices` table of per-**token** `input`/`output` rates (plus an optional
`currency`, default `"USD"`), and any generation that ends with token usage and
a model exactly matching a table entry — but no explicitly set cost — gets its
`input_cost`/`output_cost`/`total_cost` derived automatically.

Ollama is free and local, so **every rate in this lesson is fictional** — the
numbers exist only to make the arithmetic visible. Every part self-verifies by
reloading `./.bir/traces.jsonl` and recomputing the expected cost from the
reloaded event's *own* usage × the configured rate, so the checks hold in both
smoke and real mode even though token counts differ.

## What it shows

- **A · auto-cost** — `configure(model_prices={model: {"input": …, "output":
  …}})`, then one call through the `trace_chat` wrapper. The wrapper already
  records the model and usage (`prompt_eval_count` / `eval_count`), so the
  reloaded generation carries a derived cost with zero extra code. Lookup is by
  EXACT model name against the generation's *final* recorded model — the
  wrapper refines it from the response, so key the table by the response's
  model name (for Ollama that equals the requested one, e.g. `llama3.2:1b`).
- **B · manual `set_usage`** — a raw `client.chat` call inside
  `with bir.generation(...)`, no wrapper; usage recorded by hand from the
  response, omitting `total_tokens` to show `set_usage()` derives it from the
  two halves. The table prices this generation too: auto-cost applies to ANY
  generation with a model and usage, not just integration wrappers.
- **C · explicit `set_cost` wins** — same call shape plus
  `set_cost(total_cost=0.0042, currency="EUR")` (a fictional flat contracted
  price). The reloaded cost is exactly the explicit numbers and currency; a
  matching table entry and recorded usage are both present, and neither
  overwrites it.
- **D · when no cost appears** — two deterministic misses, both reloading with
  `cost=None`: a call on `--alt-model` (no table entry; exact-name lookup, no
  fuzzy matching, never a guess), and a generation whose usage was set with
  `total_tokens` only — with no input/output split, neither side can be priced.
- **E · the spend rollup** — reload every trace the run wrote and print a
  per-model summary: calls, tokens, and summed `total_cost` kept per currency
  (USD from the table, EUR from the contract — never mixed).

## Key

**None.** Ollama runs locally and is keyless — there is no API key and no `.env`
to fill in. Nothing in this lesson bills anything; the rates are teaching
numbers.

## Run it

```bash
# Offline smoke — no Ollama, no network, deterministic (what CI runs):
uv run python main.py --smoke

# Real run — needs a local Ollama server and BOTH pulled models
# (the default one is priced in the table; the alt one is deliberately not):
ollama pull llama3.2:1b
ollama pull qwen2.5:0.5b
uv run python main.py --prompt "Why track the cost of LLM calls?"
```

Flags: `--prompt` (feeds part A; B–E use fixed inputs), `--model` (default
`llama3.2:1b` — the exact name the price table keys), `--alt-model` (default
`qwen2.5:0.5b` — deliberately left out of the table), `--trace-path`, `--smoke`
(also `BIR_COOKBOOK_SMOKE=1`).

## What you'll see

```
== A · auto-cost: a price table + the wrapper, zero extra code ==
[bir] configure(model_prices={"llama3.2:1b": {"input": 2e-07, "output": 8e-07}})  # per-TOKEN rates, fictional — Ollama is free
…the model's answer…
[bir] ✓ A: the wrapper recorded token usage (both halves + the derived total)
[bir] ✓ A: reloaded cost == this generation's own usage × the configured rates
[bir] ✓ A: currency defaulted to "USD"

== B · manual set_usage: the table prices ANY generation, not just wrappers ==
[bir] ✓ B: total_tokens was auto-derived from the two halves (it was never passed)
[bir] ✓ B: the table auto-priced this hand-recorded generation too

== C · explicit set_cost wins: a flat contracted price, non-USD ==
[bir] ✓ C: reloaded cost is exactly the explicit {'total_cost': 0.0042} — no table-derived halves
[bir] ✓ C: currency is the explicit "EUR", not the table's "USD" — the table never overwrote it

== D · when no cost appears: unknown model, or usage without a split ==
[bir] ✓ D: …but reloads with cost=None — "qwen2.5:0.5b" has no table entry (lookup is exact-name)
[bir] ✓ D: …so neither side can be priced and it reloads with cost=None (never a guess)

== E · the spend rollup: reload everything this run wrote ==
[bir] spend rollup for this run (4 traces, 5 generations):
[bir]   llama3.2:1b    calls=4  tokens=…   EUR 0.00420000  USD 0.0000…
[bir]   qwen2.5:0.5b   calls=1  tokens=…   (no priced calls)
[bir]   uncosted generations: 2 (no price entry / no token split)

[bir] all cost checks passed — traces on disk: …
[bir] wrote ./.bir/traces.jsonl
```

Token counts (and therefore the USD amounts) differ between smoke and real
runs — every `✓` recomputes its expectation from the reloaded usage, so all
checks are deterministic either way. Inspect the raw records with:

```bash
cat .bir/traces.jsonl
```
