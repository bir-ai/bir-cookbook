# Bir Cookbook

Hands-on recipes for the [**bir-sdk**](https://github.com/bir-ai/bir-python)
tracing / observability SDK (import name `bir`), built to teach **how you
actually use the SDK and every one of its features** — against **real, free,
local LLM calls** via [Ollama](https://ollama.com).

The SDK's own repo ships deliberately minimal, offline regression examples. This
cookbook is the teaching layer: runnable, real, and progressive.

---

## Two phases, in order

### Phase 1 — Feature tour on Ollama  ← the focus

A **progressive, numbered series** that exercises the *full* SDK surface against
real local Ollama calls. Ollama is free, local, keyless, and unlimited — ideal
for exhaustive demos (many calls, sampling, whole eval experiments) with no cost
or rate limits. Each lesson is one runnable script that also has an offline
`--smoke` path for CI.

### Phase 2 — Free integration recipes  (later)

Thin per-provider recipes showing how to wire each SDK **integration** (Gemini,
Mistral, Groq, CrewAI, LlamaIndex, …) using **free** providers. These come
*after* the tour on purpose: their only job is the integration wiring — the SDK
features themselves are already taught in Phase 1.

> **Status:** repo skeleton + CI in place. Phase 1, Lessons 01–06
> ([`ollama-01-basics`](recipes/ollama-01-basics/),
> [`ollama-02-structure`](recipes/ollama-02-structure/),
> [`ollama-03-prompts`](recipes/ollama-03-prompts/),
> [`ollama-04-async-streaming`](recipes/ollama-04-async-streaming/),
> [`ollama-05-governance`](recipes/ollama-05-governance/),
> [`ollama-06-cost`](recipes/ollama-06-cost/)) are
> implemented. Lessons 07–08 are next.

---

## Roadmap

### Phase 1 — Ollama feature tour

Each lesson is one self-contained, runnable recipe (with `--smoke`). Together
they cover essentially the whole SDK surface.

| # | Lesson | What it teaches | SDK surface exercised |
| --- | --- | --- | --- |
| 01 | [basics](recipes/ollama-01-basics/) ✅ | your first traced Ollama call | `configure`, `@observe`, `generation` (via `trace_ollama_chat`), `load_traces` |
| 02 | [structure](recipes/ollama-02-structure/) ✅ | nested work + the RAG shape | `trace` / `span`, `tool_call`, `retrieval` (`add_document`/`set_documents`), `score` |
| 03 | [prompts & correlation](recipes/ollama-03-prompts/) ✅ | prompt versioning + log linking | `prompt()` (templates/versions), `generation(..., prompt=)`, `get_current_trace_id` / `get_current_span_id`, `bir.logging.install_trace_id_filter` |
| 04 | [async, streaming, generators](recipes/ollama-04-async-streaming/) ✅ | non-blocking + token streaming | async `@observe`, streaming generation, generator tracing |
| 05 | [governance](recipes/ollama-05-governance/) ✅ | production controls | `sample_rate` / `sample_rules`, `enabled` kill-switch, redaction (`additional_secret_keys` / `additional_redaction_patterns`), capture limits, `service` / `environment` / `source` tags |
| 06 | [cost](recipes/ollama-06-cost/) ✅ | spend tracking | `model_prices` auto-cost, `set_cost`, `set_usage` |
| 07 | persistence | files + server | file rotation (`max_bytes` / `backup_count`), `send_events` to a Bir server |
| 08 | evals | the offline eval loop | `Dataset`, evaluators, `run_experiment`, `render_experiment_report`, `compare_experiments` |

### Phase 2 — Free integration recipes (after the tour)

- **Provider wrappers:** `mistral`, `google` / Gemini, `cohere`, Groq / OpenRouter
  via `openai` + `base_url` or `litellm`, `ollama`.
- **Frameworks on a free model:** `llamaindex`, `crewai`, `haystack`, `autogen`,
  `openai_agents`, `pydantic_ai`, `dspy`, `instructor`.
- **Exporter / other:** `otel` (`export_traces_to_otlp`), a real-dependency
  `evals` recipe.
- **Out of scope (paid-only):** Anthropic, AWS Bedrock, Vertex AI — excluded so
  every recipe stays free to run.

---

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) — each recipe is its own uv project.
- For **real** Ollama runs: install [Ollama](https://ollama.com) and pull a small
  model, e.g. `ollama pull llama3.2:1b`. The offline `--smoke` path needs neither
  Ollama nor a model.

## The `--smoke` contract

**Every recipe's entry script supports `--smoke` (or `BIR_COOKBOOK_SMOKE=1`).**
In smoke mode the recipe uses a tiny, deterministic **in-file fake** of the
provider client — no server, no network — so the *Bir tracing wiring* is
regression-tested in CI without a running Ollama. Ollama's live path is
**local-only** (CI has no Ollama server, so CI runs `--smoke`).

## Layout

```
recipes/
├── _template/         # copy-to-start scaffold for a new lesson/recipe
└── <lesson-or-recipe>/
    ├── main.py        # runnable; supports --smoke
    ├── pyproject.toml # uv project; deps = ["bir-sdk==0.3.0", …]
    ├── README.md
    └── .env.example
```

`scripts/smoke.py` discovers recipes and runs each one's `--smoke` path (it also
feeds the CI matrix via `--list`). Folders starting with `_` are scaffolding and
are skipped.

## Versioning & drift

Recipes pin **`bir-sdk==0.3.0`**. CI additionally runs a **nightly** leg that
installs the SDK from `main` (`git+https://github.com/bir-ai/bir-python@main`)
and re-runs every recipe's smoke path, so upstream API drift is caught here
before it reaches users. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## License

[Apache-2.0](LICENSE), matching `bir-sdk`.
