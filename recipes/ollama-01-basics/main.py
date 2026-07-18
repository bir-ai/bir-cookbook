"""Phase 1 · Lesson 01 — basics: your first traced Ollama calls.

Make one real, local Ollama call on each sync surface — ``chat`` and
``generate`` — and record both in a single Bir trace. This is the simplest
lesson in the tour; later lessons add spans, tools, retrieval, evals, and
governance. Here we exercise exactly this slice of the SDK:

  * ``configure(trace_path=..., capture_inputs=True, capture_outputs=True)`` ->
    write the recipe-local ``./.bir/traces.jsonl``.
  * an ``@observe``-decorated function that makes one ``chat`` call wrapped with
    ``trace_ollama_chat(...)`` and one ``generate`` call wrapped with
    ``trace_ollama_generate(...)``, each recorded as a Bir ``generation``. The
    two surfaces answer differently: chat puts the text at ``message.content``,
    generate at ``response``; both report usage at the top-level
    ``prompt_eval_count`` / ``eval_count``.
  * ``load_traces(...)`` afterward -> print the trace_id, event count, and each
    generation's model and token usage so the run is self-verifying.

Run it:
  * offline (no Ollama, no network, deterministic — what CI runs):
      uv run python main.py --smoke
  * real (needs a local Ollama server + a pulled model):
      ollama pull llama3.2:1b
      uv run python main.py --prompt "In one sentence, what is tracing?"

Ollama is local and keyless, so there is no API key to set.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from bir import configure, load_traces, observe
from bir.integrations.ollama import trace_chat as trace_ollama_chat
from bir.integrations.ollama import trace_generate as trace_ollama_generate

DEFAULT_TRACE_PATH = Path(__file__).resolve().parent / ".bir" / "traces.jsonl"


def _default_model() -> str:
    # The whole tour runs on one local model, named once in cookbook.env at
    # the repo root — edit that single line (and `ollama pull` the new model)
    # to rerun every lesson on it. Precedence: --model flag > OLLAMA_MODEL
    # env var > cookbook.env > this built-in fallback.
    from_env = os.environ.get("OLLAMA_MODEL")
    if from_env:
        return from_env
    for parent in Path(__file__).resolve().parents:
        config = parent / "cookbook.env"
        if config.is_file():
            for line in config.read_text().splitlines():
                key, sep, value = line.partition("=")
                if sep and key.strip() == "OLLAMA_MODEL" and value.strip():
                    return value.strip()
            break
    return "llama3.2:1b"


DEFAULT_MODEL = _default_model()
DEFAULT_PROMPT = "In one sentence, what is LLM observability?"

# Keep the client module-level so it is never passed as an @observe argument and
# therefore never captured as an input (see CLAUDE.md).
_CLIENT = None


# --------------------------------------------------------------------------- #
# Offline fake — only used with --smoke. It mirrors the shapes the Ollama
# wrappers read from real responses: ``model`` plus the text at
# ``message.content`` (chat) or ``response`` (generate), and token usage at the
# top-level ``prompt_eval_count`` / ``eval_count`` for both. The real
# ``ollama.ChatResponse``/``GenerateResponse`` are pydantic models exposing
# ``model_dump``, so the fakes provide one too — that is the path
# ``_response_output`` takes.
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


class _FakeGenerateResponse:
    def __init__(self, *, model: str, response: str, prompt_eval_count: int, eval_count: int) -> None:
        self.model = model
        self.response = response
        self.prompt_eval_count = prompt_eval_count
        self.eval_count = eval_count
        self.done = True

    def __getitem__(self, key: str):
        return getattr(self, key)

    def model_dump(self) -> dict:
        return {
            "model": self.model,
            "response": self.response,
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

    def generate(self, *, model: str, prompt: str, **_kwargs) -> _FakeGenerateResponse:
        response = f"(smoke) Generate completes raw prompts. You asked: {prompt}"
        return _FakeGenerateResponse(
            model=model,
            response=response,
            prompt_eval_count=len(prompt.split()),
            eval_count=len(response.split()),
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
            "Start Ollama and pull a model (`ollama pull llama3.2:1b`), "
            "or run offline with --smoke."
        ) from exc
    return client


@observe(name="ollama_basics", capture_inputs=True, capture_outputs=True)
def answer(prompt: str, model: str) -> dict[str, str]:
    """Make one chat and one generate call, each recorded as a Bir generation."""

    # Each wrapper takes the Ollama callable first, then forwards everything
    # else to Ollama unchanged; ``model`` selects the model. The client is read
    # from module scope, never a captured input.
    chat_response = trace_ollama_chat(
        _CLIENT.chat,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        bir_metadata={"recipe": "ollama-01-basics"},
    )

    # Ollama's second sync surface: ``generate`` takes a raw ``prompt`` instead
    # of ``messages`` and answers with the completion text at ``response``
    # (not ``message.content``). Usage stays top-level for both surfaces.
    generate_response = trace_ollama_generate(
        _CLIENT.generate,
        model=model,
        prompt=prompt,
        bir_metadata={"recipe": "ollama-01-basics"},
    )

    return {
        "chat": chat_response.message.content,
        "generate": generate_response.response,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lesson 01 — record one local Ollama chat and generate call in a Bir trace."
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
        answers = answer(args.prompt, args.model)
    except Exception as exc:  # pragma: no cover - real-path only
        if smoke:
            raise
        raise SystemExit(
            f"Ollama call failed: {exc}\n"
            f"Ensure Ollama is running and the model is pulled "
            f"(`ollama pull {args.model}`), or run with --smoke."
        ) from exc

    print(f"chat:     {answers['chat']}")
    print(f"generate: {answers['generate']}")

    # Reload from disk and summarize so the run is self-verifying.
    latest = load_traces(trace_path)[-1]
    print(f"\n[bir] trace_id={latest.id}")
    print(f"[bir] events={len(latest.events)}")
    for gen in (event for event in latest.events if event.type == "generation"):
        usage = gen.usage or {}
        print(
            f"[bir] generation {gen.name:<16} model={gen.model}  "
            f"usage: in={usage.get('input_tokens')} "
            f"out={usage.get('output_tokens')} total={usage.get('total_tokens')}"
        )
    print(f"[bir] wrote {trace_path}")


if __name__ == "__main__":
    main()
