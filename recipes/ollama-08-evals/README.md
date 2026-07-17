# Ollama · Lesson 08 — evals

**Phase 1, Lesson 08 of the Ollama feature tour — the final lesson.** Every
lesson so far recorded what your LLM code *did*; this one measures how *well*
it did. The whole eval loop is offline and file-based: a `Dataset` of examples,
deterministic evaluator factories, `run_experiment` persisting one JSONL row
per example plus a `.summary.json` sidecar, `render_experiment_report` for a
shareable report, and `compare_experiments` as the regression gate a CI job can
key on.

The model's prose differs between smoke and real runs, so every asserted check
keys on structure the *task's code* controls — the `[doc-id]` citation the code
appends, the `{"answer", "contexts"}` mapping it builds — never on live model
text. That is also the lesson's design advice: let code, not the model, own the
fields your evaluators judge.

## What it shows

- **A · datasets** — a `Dataset` built in code; an example's `input` *mapping*
  becomes the task's kwargs (`task(question=…, doc_id=…)`), so dataset inputs
  and task signatures are designed together. Duplicate example ids raise a
  `ValueError` at construction; `to_jsonl` / `from_jsonl` round-trip the
  examples exactly (redacted by default, like every Bir artifact).
- **B · evaluators standalone** — `evaluator.evaluate()` needs no experiment.
  One factory of each flavor: expected-from-example fallback (`exact_match`
  raises without any expected), configured expected (`similarity_above` with
  the difflib ratio as inspectable metadata and an inclusive threshold
  boundary), context-based (`latency_under` / `cost_under` judge the run via
  an `EvaluationContext`, and refuse to run without one), `custom_evaluator`
  (a returned bool is coerced to exactly 1.0/0.0), and a RAG check
  (`answer_context_overlap` with the overlap ratio and unsupported words as
  evidence). Every score is exactly 1.0 or 0.0 — the evidence lives in
  `EvalResult.metadata`, not the score.
- **C · `run_experiment`** — the grounded-QA task over all six examples with
  `record_traces=True` and `raise_on_error=False`. One example fails
  deterministically (its `doc_id` matches no document, before any model call)
  and becomes a `status="error"` row while the run continues; aggregates are
  the per-evaluator *means*, recomputed in-script from the reloaded rows and
  matched against both the result and the summary sidecar. `record_traces=True`
  is the bridge to Lessons 01–07: each example runs in its own
  `experiment.<name>.<example_id>` trace in the same `traces.jsonl` every other
  lesson wrote to, every score doubles as a score event on that trace, the
  wrapped Ollama generation nests inside, and each row's `trace_id` links the
  two stores. A second tiny run shows the default `raise_on_error=True`: rows
  and summary are persisted through the failing example, *then* the exception
  re-raises — fail-fast, not fail-silent.
- **D · reports** — `render_experiment_report` in both formats: one
  self-contained string each (stdlib only, everything escaped), written next to
  the experiment file, byte-identical when re-rendered, with every evaluator
  and every example (the errored one included) present in both.
- **E · the regression gate** — the *same* dataset through the baseline task
  and a deliberately degraded candidate (a plain string instead of the RAG
  mapping; one designated example loses its citation), so the diff is exact:
  three evaluators regress, one improves, one is unchanged, one is
  baseline-only (the candidate dropped it) and one candidate-only (never a
  regression). `per_example=True` pins the small 0.2 citation drop to the one
  example that caused it; `score_tolerances` absorbs that drop (the boundary is
  inclusive) without loosening the other checks, and an override naming a
  non-shared evaluator raises; `tolerance=1.0` opens the gate and
  `missing_score="regress"` closes it again, because dropping an evaluator is a
  coverage regression. `compare_experiments` accepts result objects *or* paths,
  so CI can diff two persisted runs without re-running anything — gate on
  `diff.has_regressions`. Finally, `list_experiments` reads every
  `.summary.json` newest-first.

Not covered here: `run_experiment_async` (Lesson 04 taught the async story) and
`send_experiment` (Lesson 07 taught the server story).

## Key

**None.** Ollama runs locally and is keyless, and the eval loop is entirely
file-based — nothing leaves your machine.

## Run it

```bash
# Offline smoke — no Ollama, no network, deterministic (what CI runs):
uv run python main.py --smoke

# Real run — needs a local Ollama server and the pulled model:
ollama pull llama3.2:1b
uv run python main.py --prompt "In one sentence, what does a trace record?"
```

Flags: `--prompt` (seeds the question of the first dataset example; the rest is
fixed), `--model` (default `llama3.2:1b`, the task's model), `--trace-path`
(**cleared at startup**, along with the sibling `experiments/` and `datasets/`
output — the lesson asserts exact counts), `--smoke` (also
`BIR_COOKBOOK_SMOKE=1`).

## What you'll see

```
== A · datasets: uniquely identified examples, JSONL round-trip ==
[bir] ✓ A: datasets are sized and iterable, in example order
[bir] Dataset with a duplicate id -> ValueError: dataset contains duplicate example IDs: e1-tracing
[bir] ✓ A: duplicate example ids raise a ValueError naming the offending id
[bir] ✓ A: to_jsonl -> from_jsonl round-trips ids, inputs, and expected values exactly

== B · evaluators standalone: exact 1.0/0.0, evidence in metadata ==
[bir] similarity ratios: near=0.863 far=0.308 (threshold 0.8)
[bir] ✓ B: similarity_above is binary — the ratio is evidence, not the score
…
== C · run_experiment: score the task over the dataset, traced ==
[bir] ✓ C: raise_on_error=False turned the poisoned example into a status=error row, not a crash
[bir] e6-missing-doc -> error: retrieval failed: no document with id 'docs-nonexistent'
[bir] ✓ C: aggregate_scores recomputed from the reloaded rows match result and summary
[bir] ✓ C: every row's trace_id resolves to a trace named experiment.qa-quality.<example_id>
…
== E · the regression gate: baseline vs degraded candidate ==
[bir]   Δ answer_contains_citation   -0.20
[bir]   Δ has_rag_shape              -1.00
[bir]   Δ is_plain_string            +1.00
[bir]   Δ json_valid                 -1.00
[bir]   Δ latency_under              +0.00
[bir] ✓ E: regressed == exactly the three checks the degraded task breaks
[bir] ✓ E: has_regressions — the boolean a CI gate keys on — is True
[bir] ✓ E: missing_score='regress' closes it again — dropping an evaluator is a coverage regression
[bir] list_experiments -> ['qa-candidate', 'qa-baseline', 'qa-failfast', 'qa-quality']

[bir] all eval checks passed — 4 experiments, 16 experiment traces
[bir] trace_id=…  events=8  model=llama3.2:1b  usage={…}
[bir] wrote ./.bir/traces.jsonl and ./.bir/experiments/ (results, summaries, reports)
```

Every `✓` is deterministic in both modes because the evaluators only judge
structure the task's code controls. Inspect the artifacts with:

```bash
ls .bir/experiments/           # 4 × .jsonl + .summary.json, plus the reports
cat .bir/experiments/qa-quality.summary.json
open .bir/experiments/qa-quality.report.html
cat .bir/datasets/qa.jsonl     # the exported dataset, one example per line
```
