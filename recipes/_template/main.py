"""TEMPLATE recipe — copy this folder to recipes/<name> and fill in the TODOs.

Pattern: wrap ONE real provider call with a Bir integration so it is recorded as
a Bir event inside an @observe trace. Read the SDK integration you are wrapping
first (../../../bir-python/src/bir/integrations/<name>.py) to match its real call
shape — which kwarg carries the model, where token usage lives, streaming.

Contract (see ../../CLAUDE.md):
  * `--smoke` / BIR_COOKBOOK_SMOKE=1 -> offline: use the in-file fake client,
    no key, no network. Import the real provider lib lazily (non-smoke branch).
  * real mode -> read credentials from env vars only; exit clearly if missing.
  * wrap the provider call inside an active trace; write ./.bir/traces.jsonl.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from bir import configure, load_traces, observe

# TODO: import the integration entry point you are demonstrating, e.g.
#   from bir.integrations.ollama import trace_chat as trace_ollama_chat  # Phase 1
#   from bir.integrations.mistral import trace_chat                      # Phase 2

DEFAULT_TRACE_PATH = Path(__file__).resolve().parent / ".bir" / "traces.jsonl"
DEFAULT_MODEL = "TODO-model-id"
DEFAULT_PROMPT = "In one sentence, what is LLM observability?"

# Keep the client module-level so it is never captured as an @observe input.
_CLIENT = None


# --------------------------------------------------------------------------- #
# Offline fake — only used with --smoke. Match the shape the SDK wrapper reads
# from the real response (output, model, token usage).
# --------------------------------------------------------------------------- #
class _FakeClient:
    """TODO: mirror the real client's call surface used below in answer()."""

    def call(self, *, model: str, prompt: str):  # rename to the real method
        raise NotImplementedError("TODO: return a fake response of the right shape")


def _build_client(smoke: bool):
    if smoke:
        return _FakeClient()

    # TODO: build the real client. Phase 1 uses local Ollama, which needs NO key
    # (it talks to a local server), e.g.:
    #   from ollama import Client
    #   return Client()
    # A keyed free provider (Phase 2) reads its key from an env var, e.g.:
    #   from mistralai import Mistral
    #   key = os.environ.get("MISTRAL_API_KEY")
    #   if not key:
    #       raise SystemExit("Set MISTRAL_API_KEY (free tier), or run with --smoke.")
    #   return Mistral(api_key=key)
    raise SystemExit("TODO: build the real client; or run with --smoke.")


@observe(name="TODO_recipe", capture_inputs=True, capture_outputs=True)
def answer(prompt: str, model: str) -> str:
    """TODO: wrap the real provider call with the Bir integration wrapper."""

    # Example (wrap-a-callable) — match the real response shape from the SDK
    # integration source before finalizing. For Ollama (Phase 1):
    #   response = trace_ollama_chat(
    #       _CLIENT.chat,
    #       model=model,
    #       messages=[{"role": "user", "content": prompt}],
    #       bir_metadata={"recipe": "TODO"},
    #   )
    #   return response["message"]["content"]
    raise NotImplementedError("TODO: call the provider through the Bir wrapper")


def main() -> None:
    parser = argparse.ArgumentParser(description="TODO: one-line recipe description.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--trace-path", default=str(DEFAULT_TRACE_PATH))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Offline mode: in-file fake client (no key, no network). Also BIR_COOKBOOK_SMOKE=1.",
    )
    args = parser.parse_args()

    smoke = args.smoke or os.environ.get("BIR_COOKBOOK_SMOKE") == "1"
    trace_path = Path(args.trace_path)
    configure(trace_path=trace_path, capture_inputs=True, capture_outputs=True)

    global _CLIENT
    _CLIENT = _build_client(smoke)

    text = answer(args.prompt, args.model)
    print(text)

    latest = load_traces(trace_path)[-1]
    print(f"\n[bir] wrote {len(latest.events)} events to {trace_path}  trace_id={latest.id}")


if __name__ == "__main__":
    main()
