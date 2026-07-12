"""Phase 1 · Lesson 06 — cost: spend tracking.

Bir bundles NO prices. Cost is opt-in and local-only: you hand ``configure()``
a ``model_prices`` table (per-TOKEN ``input``/``output`` rates, optional
``currency``), and any generation that ends with a model matching a table entry
and token usage — but no explicitly set cost — gets its
``input_cost``/``output_cost``/``total_cost`` derived automatically at exit.
Ollama is free and local, so every rate in this file is a FICTIONAL teaching
number; nothing here bills anything, ever.

Every part self-verifies by reloading ``./.bir/traces.jsonl`` and recomputing
the expected cost from the reloaded event's OWN usage × the configured rate,
so the checks hold in both smoke and real mode even though token counts differ:

  A. auto-cost — ``configure(model_prices={model: {"input": ..., "output":
     ...}})``, then one wrapped Ollama call. The wrapper already records usage
     from ``prompt_eval_count`` / ``eval_count``, so the reloaded generation
     carries a derived cost with ZERO extra code. Lookup is by EXACT model
     name — the wrapper refines the model from the response, so the table must
     key the response's model name (for Ollama that equals the requested one).
  B. manual ``set_usage`` — one RAW client call inside ``with
     bir.generation(...)``, no wrapper; usage recorded by hand from the
     response, omitting ``total_tokens`` to show it is auto-derived. The table
     prices this generation too: auto-cost applies to ANY generation with a
     model and usage, not just integration wrappers.
  C. explicit ``set_cost`` wins — same call shape plus a flat contracted price
     in EUR. The reloaded cost is exactly the explicit numbers and currency;
     the table never overwrites an explicit cost.
  D. when NO cost appears — two deterministic misses: a model with no table
     entry (the ``--alt-model``), and a usage recorded with ``total_tokens``
     only (no input/output split, so neither side can be priced). Both reload
     with ``cost=None``.
  E. the spend rollup — reload every trace this run wrote and print a
     per-model summary (calls, tokens, summed total_cost by currency).

Run it:
  * offline (no Ollama, no network, deterministic — what CI runs):
      uv run python main.py --smoke
  * real (needs a local Ollama server + the two pulled models):
      ollama pull llama3.2:1b
      ollama pull qwen2.5:0.5b
      uv run python main.py --prompt "Why track the cost of LLM calls?"

Ollama is local and keyless, so there is no API key to set.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from bir import configure, generation, load_traces, observe
from bir.integrations.ollama import trace_chat as trace_ollama_chat

DEFAULT_TRACE_PATH = Path(__file__).resolve().parent / ".bir" / "traces.jsonl"
DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_ALT_MODEL = "qwen2.5:0.5b"
DEFAULT_PROMPT = "In one sentence, why track the cost of LLM calls?"

# FICTIONAL per-TOKEN rates (model_prices is per-token, NOT per-million).
# Providers quote per-million, so divide by 1e6 when copying a price sheet.
# Ollama is free — these numbers exist only to make the math visible.
INPUT_RATE = 2e-07  # "$0.20 per million input tokens", as a per-token rate
OUTPUT_RATE = 8e-07  # "$0.80 per million output tokens", as a per-token rate

# Part C: a flat, contracted per-call price in a non-USD currency — also
# fictional. Explicit set_cost() always beats the table.
FLAT_PRICE = 0.0042
FLAT_CURRENCY = "EUR"

# Parts B-D use fixed, distinct prompts (--prompt feeds part A). Distinct
# prompts keep real-mode token counts honest: a repeated identical prompt can
# hit Ollama's prompt cache, which may skip prompt evaluation entirely.
PROMPT_B = "In one sentence, what does the cost of an LLM call depend on?"
PROMPT_C = "In one sentence, what is a flat contracted price?"
PROMPT_D = "In one sentence, why might a cost be unknown?"

# Keep the client module-level so it is never passed as an @observe argument and
# therefore never captured as an input (see CLAUDE.md).
_CLIENT = None


def _check(ok: bool, label: str) -> None:
    """Print a visible verification line; any failed check fails the run."""

    print(f"[bir] {'✓' if ok else '✗'} {label}")
    if not ok:
        raise SystemExit(f"self-check failed: {label}")


def _expected_cost(usage: dict | None) -> dict | None:
    """Recompute what the price table should derive from this usage.

    Mirrors the SDK's arithmetic exactly (input rate × input tokens, output
    rate × output tokens, total summed when both sides are priced), so the
    equality checks below are deterministic in both modes — the expectation is
    computed from the reloaded event's own token counts, never hardcoded.
    """

    input_tokens = (usage or {}).get("input_tokens")
    output_tokens = (usage or {}).get("output_tokens")
    cost: dict[str, float] = {}
    if input_tokens is not None:
        cost["input_cost"] = INPUT_RATE * input_tokens
    if output_tokens is not None:
        cost["output_cost"] = OUTPUT_RATE * output_tokens
    if "input_cost" in cost and "output_cost" in cost:
        cost["total_cost"] = cost["input_cost"] + cost["output_cost"]
    return cost or None


# --------------------------------------------------------------------------- #
# Offline fake — only used with --smoke. Same response shape as Lesson 01's
# fake (``model``, ``message.content``, ``prompt_eval_count`` / ``eval_count``,
# a ``model_dump``). It happily serves ANY model name, which Part D relies on.
# --------------------------------------------------------------------------- #
class _FakeMessage:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def __getitem__(self, key: str):
        return getattr(self, key)

    def model_dump(self) -> dict:
        return {"role": self.role, "content": self.content}


class _FakeChatResponse:
    def __init__(self, *, model: str, content: str, prompt_eval_count: int, eval_count: int) -> None:
        self.model = model
        self.message = _FakeMessage("assistant", content)
        self.prompt_eval_count = prompt_eval_count
        self.eval_count = eval_count
        self.done = True

    def __getitem__(self, key: str):
        return getattr(self, key)

    def model_dump(self) -> dict:
        return {
            "model": self.model,
            "message": self.message.model_dump(),
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
            "done": self.done,
        }


class _FakeOllamaClient:
    """Deterministic stand-in for ``ollama.Client`` used only in --smoke mode."""

    def chat(self, *, model: str, messages: list[dict], **_kwargs) -> _FakeChatResponse:
        prompt = messages[-1]["content"] if messages else ""
        content = f"(smoke) Cost is usage times a rate you supply. You asked: {prompt}"
        return _FakeChatResponse(
            model=model,
            content=content,
            prompt_eval_count=sum(len(m["content"].split()) for m in messages),
            eval_count=len(content.split()),
        )


def _build_client(smoke: bool):
    if smoke:
        return _FakeOllamaClient()

    # Real mode: talk to a local Ollama server. Import the provider library
    # lazily so --smoke needs neither the package nor a running server. Ollama
    # needs no API key — the client honors the OLLAMA_HOST env var if set.
    try:
        from ollama import Client
    except ModuleNotFoundError as exc:  # pragma: no cover - real-path only
        raise SystemExit(
            "The 'ollama' package is required for a real run "
            "(`uv add ollama`), or run offline with --smoke."
        ) from exc

    client = Client()
    try:
        client.list()  # cheap reachability check for a clear error up front
    except Exception as exc:  # pragma: no cover - real-path only
        raise SystemExit(
            "Could not reach a local Ollama server at http://localhost:11434.\n"
            "Start Ollama and pull the models (`ollama pull llama3.2:1b` and "
            "`ollama pull qwen2.5:0.5b`), or run offline with --smoke."
        ) from exc
    return client


# --------------------------------------------------------------------------- #
# The traced entry points — one @observe root per part, so each part reloads
# exactly the trace it just wrote.
# --------------------------------------------------------------------------- #
@observe(name="auto_cost_answer", capture_inputs=True, capture_outputs=True)
def auto_cost_answer(prompt: str, model: str) -> str:
    # The wrapper records model + usage; the configured table does the rest.
    response = trace_ollama_chat(
        _CLIENT.chat,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        bir_name="chat.auto_cost",
        bir_metadata={"recipe": "ollama-06-cost"},
    )
    return response.message.content


@observe(name="manual_usage_answer", capture_inputs=True, capture_outputs=True)
def manual_usage_answer(prompt: str, model: str) -> str:
    # No wrapper: a raw client call inside a hand-opened generation. Usage is
    # recorded by hand from Ollama's top-level token counts — total_tokens is
    # deliberately omitted to show set_usage() derives it. (`or 0` guards the
    # rare real-mode case where Ollama omits a count, e.g. a cached prompt.)
    messages = [{"role": "user", "content": prompt}]
    with generation(
        "chat.manual_usage",
        model=model,
        input={"model": model, "messages": messages},
        metadata={"recipe": "ollama-06-cost"},
    ) as gen:
        response = _CLIENT.chat(model=model, messages=messages)
        gen.set_output(response.message.content)
        gen.set_usage(
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
        )
    return response.message.content


@observe(name="contracted_answer", capture_inputs=True, capture_outputs=True)
def contracted_answer(prompt: str, model: str) -> str:
    # Usage AND a matching table entry are both present, yet the explicit
    # set_cost() below is what reaches disk — auto-cost never overwrites it.
    messages = [{"role": "user", "content": prompt}]
    with generation(
        "chat.contracted",
        model=model,
        input={"model": model, "messages": messages},
        metadata={"recipe": "ollama-06-cost"},
    ) as gen:
        response = _CLIENT.chat(model=model, messages=messages)
        gen.set_output(response.message.content)
        gen.set_usage(
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
        )
        gen.set_cost(total_cost=FLAT_PRICE, currency=FLAT_CURRENCY)
    return response.message.content


@observe(name="no_cost_answer", capture_inputs=True, capture_outputs=True)
def no_cost_answer(prompt: str, model: str, alt_model: str) -> str:
    # Miss 1: a wrapped call on a model with no table entry — usage and model
    # are recorded, but exact-name lookup finds nothing.
    response = trace_ollama_chat(
        _CLIENT.chat,
        model=alt_model,
        messages=[{"role": "user", "content": prompt}],
        bir_name="chat.unpriced_model",
        bir_metadata={"recipe": "ollama-06-cost"},
    )

    # Miss 2: the priced model, but usage carries only total_tokens — with no
    # input/output split, neither side can be priced, so no cost is derived.
    messages = [{"role": "user", "content": prompt}]
    with generation(
        "chat.total_only",
        model=model,
        input={"model": model, "messages": messages},
        metadata={"recipe": "ollama-06-cost"},
    ) as gen:
        total_only = _CLIENT.chat(model=model, messages=messages)
        gen.set_output(total_only.message.content)
        gen.set_usage(total_tokens=(total_only.prompt_eval_count or 0) + (total_only.eval_count or 0))
    return response.message.content


# --------------------------------------------------------------------------- #
# The five parts. Each prints what it configures/records, then reloads
# ./.bir/traces.jsonl and checks the cost that did (or did not) reach disk.
# --------------------------------------------------------------------------- #
def _generation_named(trace, name: str):
    return next(event for event in trace.events if event.type == "generation" and event.name == name)


def part_a_auto_cost(prompt: str, model: str, trace_path: Path) -> tuple[str, float]:
    print("\n== A · auto-cost: a price table + the wrapper, zero extra code ==")
    print(
        f'[bir] configure(model_prices={{"{model}": {{"input": {INPUT_RATE}, "output": {OUTPUT_RATE}}}}})'
        "  # per-TOKEN rates, fictional — Ollama is free"
    )
    configure(model_prices={model: {"input": INPUT_RATE, "output": OUTPUT_RATE}})

    print(auto_cost_answer(prompt, model))

    latest = load_traces(trace_path)[-1]
    gen = _generation_named(latest, "chat.auto_cost")
    expected = _expected_cost(gen.usage)
    _check(
        gen.usage is not None and {"input_tokens", "output_tokens", "total_tokens"} <= set(gen.usage),
        "A: the wrapper recorded token usage (both halves + the derived total)",
    )
    _check(
        gen.cost == expected,
        "A: reloaded cost == this generation's own usage × the configured rates",
    )
    _check(gen.currency == "USD", 'A: currency defaulted to "USD"')
    print(f"[bir] usage={gen.usage}")
    print(f"[bir] cost={gen.cost} {gen.currency}")
    return latest.id, gen.cost["total_cost"]


def part_b_manual_usage(model: str, trace_path: Path) -> tuple[str, float]:
    print("\n== B · manual set_usage: the table prices ANY generation, not just wrappers ==")
    print("[bir] raw client.chat inside `with bir.generation(...)`; set_usage(input_tokens=…, output_tokens=…)")

    print(manual_usage_answer(PROMPT_B, model))

    latest = load_traces(trace_path)[-1]
    gen = _generation_named(latest, "chat.manual_usage")
    _check(
        gen.usage is not None
        and gen.usage.get("total_tokens") == gen.usage.get("input_tokens") + gen.usage.get("output_tokens"),
        "B: total_tokens was auto-derived from the two halves (it was never passed)",
    )
    _check(
        gen.cost == _expected_cost(gen.usage) and gen.currency == "USD",
        "B: the table auto-priced this hand-recorded generation too",
    )
    print(f"[bir] usage={gen.usage}")
    print(f"[bir] cost={gen.cost} {gen.currency}")
    return latest.id, gen.cost["total_cost"]


def part_c_explicit_cost(model: str, trace_path: Path) -> str:
    print("\n== C · explicit set_cost wins: a flat contracted price, non-USD ==")
    print(f'[bir] set_cost(total_cost={FLAT_PRICE}, currency="{FLAT_CURRENCY}")  # fictional flat rate')

    print(contracted_answer(PROMPT_C, model))

    latest = load_traces(trace_path)[-1]
    gen = _generation_named(latest, "chat.contracted")
    _check(
        gen.cost == {"total_cost": FLAT_PRICE},
        f"C: reloaded cost is exactly the explicit {{'total_cost': {FLAT_PRICE}}} — no table-derived halves",
    )
    _check(
        gen.currency == FLAT_CURRENCY,
        f'C: currency is the explicit "{FLAT_CURRENCY}", not the table\'s "USD" — the table never overwrote it',
    )
    print(f"[bir] usage={gen.usage}  (present, and ignored for pricing)")
    print(f"[bir] cost={gen.cost} {gen.currency}")
    return latest.id


def part_d_no_cost(model: str, alt_model: str, trace_path: Path) -> str:
    print("\n== D · when no cost appears: unknown model, or usage without a split ==")

    print(no_cost_answer(PROMPT_D, model, alt_model))

    latest = load_traces(trace_path)[-1]
    unpriced = _generation_named(latest, "chat.unpriced_model")
    _check(
        unpriced.usage is not None and unpriced.model == alt_model,
        f'D: the "{alt_model}" call recorded usage and its model as usual',
    )
    _check(
        unpriced.cost is None and unpriced.currency is None,
        f'D: …but reloads with cost=None — "{alt_model}" has no table entry (lookup is exact-name)',
    )
    total_only = _generation_named(latest, "chat.total_only")
    _check(
        total_only.usage is not None and set(total_only.usage) == {"total_tokens"},
        "D: the second generation's usage carries total_tokens only — no input/output split",
    )
    _check(
        total_only.cost is None,
        "D: …so neither side can be priced and it reloads with cost=None (never a guess)",
    )
    return latest.id


def part_e_spend_rollup(
    run_trace_ids: list[str],
    expected_usd_total: float,
    model: str,
    alt_model: str,
    trace_path: Path,
) -> None:
    print("\n== E · the spend rollup: reload everything this run wrote ==")

    run_traces = [trace for trace in load_traces(trace_path) if trace.id in run_trace_ids]
    generations = [event for trace in run_traces for event in trace.events if event.type == "generation"]

    # Aggregate per model: call count, summed tokens, and spend by currency.
    # total_cost can be absent on a cost dict (e.g. an explicit input-only
    # set_cost), so the rollup counts only what it can sum.
    rollup: dict[str, dict] = {}
    uncosted = 0
    for gen in generations:
        entry = rollup.setdefault(gen.model, {"calls": 0, "tokens": 0, "spend": {}})
        entry["calls"] += 1
        entry["tokens"] += (gen.usage or {}).get("total_tokens", 0)
        if gen.cost is None:
            uncosted += 1
        elif "total_cost" in gen.cost:
            entry["spend"][gen.currency] = entry["spend"].get(gen.currency, 0.0) + gen.cost["total_cost"]

    print(f"[bir] spend rollup for this run ({len(run_traces)} traces, {len(generations)} generations):")
    for name, entry in sorted(rollup.items()):
        spend = "  ".join(f"{currency} {amount:.8f}" for currency, amount in sorted(entry["spend"].items()))
        print(f"[bir]   {name:<14} calls={entry['calls']}  tokens={entry['tokens']}  {spend or '(no priced calls)'}")
    print(f"[bir]   uncosted generations: {uncosted} (no price entry / no token split)")

    _check(len(run_traces) == 4 and len(generations) == 5, "E: the run wrote 4 traces holding 5 generations")
    _check(
        rollup[model]["spend"] == {"USD": expected_usd_total, FLAT_CURRENCY: FLAT_PRICE},
        f'E: "{model}" spend == parts A+B in USD plus part C\'s flat {FLAT_CURRENCY} price, kept per currency',
    )
    _check(
        rollup[alt_model]["calls"] == 1 and rollup[alt_model]["spend"] == {},
        f'E: "{alt_model}" shows its call and tokens but no spend',
    )
    _check(uncosted == 2, "E: exactly the two part-D generations carry no cost")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lesson 06 — cost: model_prices auto-cost, set_usage, set_cost, and a spend rollup."
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Feeds part A; B-E use fixed inputs.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="The model the price table keys (exact name).")
    parser.add_argument(
        "--alt-model",
        default=DEFAULT_ALT_MODEL,
        help="A second pulled model deliberately LEFT OUT of the price table (part D).",
    )
    parser.add_argument("--trace-path", default=str(DEFAULT_TRACE_PATH))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Offline mode: in-file fake Ollama client (no server, no network). "
        "Also enabled by BIR_COOKBOOK_SMOKE=1.",
    )
    args = parser.parse_args()

    smoke = args.smoke or os.environ.get("BIR_COOKBOOK_SMOKE") == "1"
    trace_path = Path(args.trace_path)
    configure(trace_path=trace_path, capture_inputs=True, capture_outputs=True)

    global _CLIENT
    _CLIENT = _build_client(smoke)

    try:
        trace_a, total_a = part_a_auto_cost(args.prompt, args.model, trace_path)
        trace_b, total_b = part_b_manual_usage(args.model, trace_path)
        trace_c = part_c_explicit_cost(args.model, trace_path)
        trace_d = part_d_no_cost(args.model, args.alt_model, trace_path)
        part_e_spend_rollup(
            [trace_a, trace_b, trace_c, trace_d],
            total_a + total_b,
            args.model,
            args.alt_model,
            trace_path,
        )
    except Exception as exc:  # pragma: no cover - real-path only
        if smoke:
            raise
        raise SystemExit(
            f"Ollama call failed: {exc}\n"
            f"Ensure Ollama is running and both models are pulled "
            f"(`ollama pull {args.model}` and `ollama pull {args.alt_model}`), or run with --smoke."
        ) from exc

    print(f"\n[bir] all cost checks passed — traces on disk: {len(load_traces(trace_path))}")
    print(f"[bir] wrote {trace_path}")


if __name__ == "__main__":
    main()
