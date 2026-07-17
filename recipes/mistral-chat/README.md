# Mistral · chat

**Phase 2 integration recipe.** One traced chat call against Mistral's hosted
[La Plateforme](https://console.mistral.ai/) API. Phase 1 (the
[Ollama feature tour](../../README.md)) teaches the SDK's features; this recipe
shows only the wiring for the Mistral integration.

## What it shows

- `bir.integrations.mistral.trace_chat` wrapping `client.chat.complete` from
  the `mistralai` v1 SDK, inside an `@observe` trace, recorded as one Bir
  **generation**. Import it from the module explicitly — it is a different
  function from the Ollama `trace_chat` in Lesson 01.
- The OpenAI-shaped response: assistant text at `choices[0].message.content`,
  usage at `usage.prompt_tokens` / `completion_tokens` / `total_tokens`, and
  the generation's model refined from `response.model`.
- `load_traces(...)` afterward — the script prints the `trace_id`, event count,
  model, and token usage so a run is self-verifying.

Streaming (`stream=True` around `client.chat.stream`) and the async wrapper
(`trace_chat_async` around `client.chat.complete_async`) work the same way as
Lesson 04's Ollama streaming — not repeated here.

## Key

A **free** Mistral API key: [console.mistral.ai](https://console.mistral.ai/) →
API Keys. The free tier is enough. Export it as `MISTRAL_API_KEY` (see
`.env.example`) — the script reads it from the environment only.

## Run it

```bash
# Offline smoke — no key, no network, deterministic (what CI runs):
uv run python main.py --smoke

# Real run — needs MISTRAL_API_KEY exported:
export MISTRAL_API_KEY=...   # https://console.mistral.ai/
uv run python main.py --prompt "In one sentence, what is LLM observability?"
```

Flags: `--prompt`, `--model` (default `mistral-small-latest`), `--trace-path`,
`--smoke` (also `BIR_COOKBOOK_SMOKE=1`).

If the key is missing, the real run exits with a message pointing to the free
key page and to `--smoke`.

## What you'll see

```
[bir] trace_id=...
[bir] events=2
[bir] generation mistral.chat  model=mistral-small-latest  usage: in=… out=… total=…
[bir] wrote ./.bir/traces.jsonl
```

Two events: the `@observe` trace root (`mistral_chat`) and the `mistral.chat`
generation nested inside it. Inspect the raw records with:

```bash
cat .bir/traces.jsonl
```
