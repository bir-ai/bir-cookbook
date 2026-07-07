"""Phase 1 · Lesson 01 — basics: your first traced Ollama call.

Make ONE real, local Ollama chat call and record it as a Bir trace. This is the
simplest lesson in the tour; later lessons add spans, tools, retrieval, evals,
and governance. Here we exercise exactly this slice of the SDK:

  * ``configure(trace_path=..., capture_inputs=True, capture_outputs=True)`` ->
    write the recipe-local ``./.bir/traces.jsonl``.
  * an ``@observe``-decorated function that makes one Ollama chat call wrapped
    with ``trace_ollama_chat(...)`` so it is recorded as a Bir ``generation``.
  * ``load_traces(...)`` afterward -> print the trace_id, event count, model,
    and token usage so the run is self-verifying.

Run it:
  * offline (no Ollama, no network, deterministic — what CI runs):
      uv run python main.py --smoke
  * real (needs a local Ollama server + a pulled model):
      ollama pull llama3.2
      uv run python main.py --prompt "In one sentence, what is tracing?"

Ollama is local and keyless, so there is no API key to set.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from bir import configure, load_traces, observe
from bir.integrations.ollama import trace_chat as trace_ollama_chat

DEFAULT_TRACE_PATH = Path(__file__).resolve().parent / ".bir" / "traces.jsonl"
DEFAULT_MODEL = "llama3.2"
DEFAULT_PROMPT = "In one sentence, what is LLM observability?"

# Keep the client module-level so it is never passed as an @observe argument and
# therefore never captured as an input (see CLAUDE.md).
_CLIENT = None


# --------------------------------------------------------------------------- #
# Offline fake — only used with --smoke. It mirrors the shape the Ollama wrapper
# reads from a real response: ``model``, the assistant text at
# ``message.content``, and token usage at ``prompt_eval_count`` / ``eval_count``.
# The real ``ollama.ChatResponse`` is a pydantic model exposing ``model_dump``,
# so the fake provides one too — that is the path ``_response_output`` takes.
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
        content = f"(smoke) A trace is a recorded run of your program. You asked: {prompt}"
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
            "Start Ollama and pull a model (`ollama pull llama3.2`), "
            "or run offline with --smoke."
        ) from exc
    return client


@observe(name="ollama_basics", capture_inputs=True, capture_outputs=True)
def answer(prompt: str, model: str) -> str:
    """Make one Ollama chat call, recorded as a Bir generation via the wrapper."""

    # trace_ollama_chat takes the chat callable first, then forwards everything
    # else to Ollama unchanged; ``model`` selects the model, ``messages`` carries
    # the prompt. The client is read from module scope, never a captured input.
    response = trace_ollama_chat(
        _CLIENT.chat,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        bir_metadata={"recipe": "ollama-01-basics"},
    )
    return response.message.content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lesson 01 — record a Bir trace for one local Ollama chat call."
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
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
        text = answer(args.prompt, args.model)
    except Exception as exc:  # pragma: no cover - real-path only
        if smoke:
            raise
        raise SystemExit(
            f"Ollama call failed: {exc}\n"
            f"Ensure Ollama is running and the model is pulled "
            f"(`ollama pull {args.model}`), or run with --smoke."
        ) from exc

    print(text)

    # Reload from disk and summarize so the run is self-verifying.
    latest = load_traces(trace_path)[-1]
    gen = next((event for event in latest.events if event.type == "generation"), None)
    model_name = gen.model if gen is not None else args.model
    usage = (gen.usage if gen is not None else None) or {}
    print(f"\n[bir] trace_id={latest.id}")
    print(
        f"[bir] events={len(latest.events)}  model={model_name}  "
        f"usage: in={usage.get('input_tokens')} "
        f"out={usage.get('output_tokens')} total={usage.get('total_tokens')}"
    )
    print(f"[bir] wrote {trace_path}")


if __name__ == "__main__":
    main()
