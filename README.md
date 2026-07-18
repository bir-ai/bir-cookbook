# Bir Cookbook

Runnable recipes for [bir-sdk](https://github.com/bir-ai/bir-python), a tracing
and observability SDK for LLM applications (import name `bir`). Every recipe
runs against a free provider, starting with local [Ollama](https://ollama.com),
so you can work through the entire SDK without an API key or a usage bill.

The SDK repo keeps its own examples minimal and offline. This cookbook is the
teaching layer: each recipe is a small script you can run, read, and copy from.

## Quick start

You need [uv](https://docs.astral.sh/uv/); each recipe is its own uv project.

```bash
cd recipes/ollama-01-basics

# Offline mode: a deterministic in-file fake stands in for Ollama.
# No model, no network, no key.
uv run python main.py --smoke

# Real mode: needs a local Ollama server and a small model.
ollama pull llama3.2:1b
uv run python main.py
```

Every recipe writes traces to a local `./.bir/traces.jsonl` and prints the
`trace_id`, event count, model, and token usage, so a run verifies itself.

## The Ollama feature tour

The core of the cookbook is a numbered series of eight lessons that covers the
full SDK surface in order. Ollama is local, keyless, and unlimited, which makes
it a good fit for the more exhaustive demos: many calls, sampling, cost
tracking, whole eval experiments. All eight lessons are implemented.

| # | Lesson | Covers |
| --- | --- | --- |
| 01 | [basics](recipes/ollama-01-basics/) | your first traced calls: `configure`, `@observe`, a `generation` via each of `trace_ollama_chat` and `trace_generate`, `load_traces` |
| 02 | [structure](recipes/ollama-02-structure/) | nested work and the RAG shape: `trace` / `span`, `tool_call`, `retrieval`, `score` |
| 03 | [prompts](recipes/ollama-03-prompts/) | prompt versioning and log correlation: `prompt()`, `generation(..., prompt=)`, `get_current_trace_id` / `get_current_span_id`, the logging filter |
| 04 | [async-streaming](recipes/ollama-04-async-streaming/) | async `@observe`, streaming generations, `trace_generate_async` (streaming generate), generator tracing |
| 05 | [governance](recipes/ollama-05-governance/) | production controls: sampling, the `enabled` kill switch, redaction, capture limits, per-call capture override (`bir_capture_input` / `bir_capture_output`), `service` / `environment` / `source` tags |
| 06 | [cost](recipes/ollama-06-cost/) | spend tracking: `model_prices` auto-cost, `set_cost`, `set_usage` |
| 07 | [persistence](recipes/ollama-07-persistence/) | trace files and servers: rotation (`max_bytes` / `backup_count`), `send_events`, the `bir traces` / `show` / `stats` CLI |
| 08 | [evals](recipes/ollama-08-evals/) | the offline eval loop: `Dataset`, evaluators, `run_experiment` + `run_experiment_async`, `render_experiment_report`, `compare_experiments`, `send_experiment`, the `bir experiments` / `experiment-show` / `experiment-report` CLI |

## Integration recipes (phase 2, planned)

With the tour as the reference for SDK features, a second set of recipes will
show only the wiring for each SDK integration, still on free providers:

- Providers: Gemini, Mistral, Cohere, and Groq / OpenRouter through the
  OpenAI-compatible client or litellm.
- Frameworks on a free model: LlamaIndex, CrewAI, Haystack, AutoGen,
  OpenAI Agents, Pydantic AI, DSPy, Instructor.
- Exporters: OTLP via `export_traces_to_otlp`.

Paid-only providers (Anthropic, AWS Bedrock, Vertex AI) are out of scope, so
that everything in this repo stays free to run.

## Smoke mode

Every entry script supports `--smoke` (or `BIR_COOKBOOK_SMOKE=1`). In smoke
mode the provider client is replaced by a small deterministic fake defined in
the script itself, so the tracing wiring is tested with no network, no key, and
no Ollama install. CI runs every recipe this way; the live Ollama path is
local-only.

## Repository layout

```
recipes/
├── _template/         # scaffold to copy when adding a recipe
└── <recipe>/
    ├── main.py        # runnable entry point, supports --smoke
    ├── pyproject.toml # uv project, pins bir-sdk==0.3.0
    ├── README.md
    └── .env.example
```

`scripts/smoke.py` discovers recipes and runs each one's smoke path; CI builds
its matrix from `scripts/smoke.py --list`. Folders starting with `_` are
scaffolding and are skipped.

## Versioning

Recipes pin `bir-sdk==0.3.0`. A nightly CI leg reinstalls the SDK from its
`main` branch and re-runs every smoke path, so upstream API drift is caught
here before it reaches users. See
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## License

[Apache-2.0](LICENSE), same as bir-sdk.
