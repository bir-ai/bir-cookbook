"""Phase 2 · mistral-chat — trace one Mistral chat call.

Integration wiring only: Phase 1 (the Ollama feature tour) already taught every
SDK feature, so this recipe shows just how to point Bir at Mistral's hosted
API. One `client.chat.complete` call is wrapped with
``bir.integrations.mistral.trace_chat`` inside an ``@observe`` trace and
recorded as a single Bir ``generation``. Note the explicit module import:
``trace_chat`` here is the *Mistral* wrapper, a different function from the
Ollama ``trace_chat`` used in Lesson 01.

The response is OpenAI-shaped: the assistant text lives at
``choices[0].message.content``, the served model at ``model``, and token usage
at ``usage.prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` — the
wrapper reads all three and refines the generation's model from the response.

Run it:
  * offline (no key, no network, deterministic — what CI runs):
      uv run python main.py --smoke
  * real (needs a free La Plateforme key from https://console.mistral.ai/):
      export MISTRAL_API_KEY=...
      uv run python main.py --prompt "In one sentence, what is tracing?"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from bir import configure, load_traces, observe
from bir.integrations.mistral import trace_chat as trace_mistral_chat

DEFAULT_TRACE_PATH = Path(__file__).resolve().parent / ".bir" / "traces.jsonl"
DEFAULT_MODEL = "mistral-small-latest"
DEFAULT_PROMPT = "In one sentence, what is LLM observability?"

# Keep the client module-level so it is never passed as an @observe argument and
# therefore never captured as an input — it holds the API key (see CLAUDE.md).
_CLIENT = None


# --------------------------------------------------------------------------- #
# Offline fake — only used with --smoke. It mirrors exactly what the wrapper
# and this recipe read from a real ``mistralai`` chat response: ``model``, the
# assistant text at ``choices[0].message.content``, token usage at
# ``usage.prompt_tokens`` / ``completion_tokens`` / ``total_tokens``, and a
# ``model_dump()`` returning a plain dict — the path ``_response_output``
# takes, since the real response is a pydantic model. The fake client exposes
# ``chat.complete(model=..., messages=[...])`` to match the real call shape.
# --------------------------------------------------------------------------- #
class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens

    def model_dump(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class _FakeMessage:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def model_dump(self) -> dict:
        return {"role": self.role, "content": self.content}


class _FakeChoice:
    def __init__(self, message: _FakeMessage) -> None:
        self.index = 0
        self.message = message
        self.finish_reason = "stop"

    def model_dump(self) -> dict:
        return {
            "index": self.index,
            "message": self.message.model_dump(),
            "finish_reason": self.finish_reason,
        }


class _FakeChatResponse:
    def __init__(self, *, model: str, content: str, prompt_tokens: int) -> None:
        self.model = model
        self.choices = [_FakeChoice(_FakeMessage("assistant", content))]
        self.usage = _FakeUsage(prompt_tokens, len(content.split()))

    def model_dump(self) -> dict:
        return {
            "model": self.model,
            "choices": [choice.model_dump() for choice in self.choices],
            "usage": self.usage.model_dump(),
        }


class _FakeChat:
    def complete(self, *, model: str, messages: list[dict], **_kwargs) -> _FakeChatResponse:
        prompt = messages[-1]["content"] if messages else ""
        return _FakeChatResponse(
            model=model,
            content=f"(smoke) Observability is watching your LLM app run. You asked: {prompt}",
            prompt_tokens=sum(len(m["content"].split()) for m in messages),
        )


class _FakeMistralClient:
    """Deterministic stand-in for ``mistralai.Mistral`` used only in --smoke mode."""

    def __init__(self) -> None:
        self.chat = _FakeChat()


def _build_client(smoke: bool):
    if smoke:
        return _FakeMistralClient()

    # Real mode: the key comes from the environment only. Import the provider
    # library lazily so --smoke needs neither the package nor a key.
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit(
            "MISTRAL_API_KEY is not set. Get a free key at https://console.mistral.ai/ "
            "(API Keys — the free tier is enough), export it, and rerun. "
            "Or run offline with --smoke."
        )

    try:
        from mistralai import Mistral
    except ModuleNotFoundError as exc:  # pragma: no cover - real-path only
        raise SystemExit(
            "The 'mistralai' package is required for a real run "
            "(`uv add mistralai`), or run offline with --smoke."
        ) from exc

    return Mistral(api_key=api_key)


@observe(name="mistral_chat", capture_inputs=True, capture_outputs=True)
def answer(prompt: str, model: str) -> str:
    """Make one Mistral chat call, recorded as a Bir generation."""

    # The wrapper takes the Mistral callable first, then forwards everything
    # else to it unchanged; its own options are prefixed ``bir_``. The client
    # is read from module scope, never a captured input.
    response = trace_mistral_chat(
        _CLIENT.chat.complete,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        bir_metadata={"recipe": "mistral-chat"},
    )
    return response.choices[0].message.content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 — record one Mistral chat call in a Bir trace."
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--trace-path", default=str(DEFAULT_TRACE_PATH))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Offline mode: in-file fake Mistral client (no key, no network). "
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
            f"Mistral call failed: {exc}\n"
            f"Check MISTRAL_API_KEY and the model name ({args.model}); free-tier "
            f"models are listed at https://docs.mistral.ai/getting-started/models/. "
            f"Or run offline with --smoke."
        ) from exc

    print(f"answer: {text}")

    # Reload from disk and summarize so the run is self-verifying.
    latest = load_traces(trace_path)[-1]
    print(f"\n[bir] trace_id={latest.id}")
    print(f"[bir] events={len(latest.events)}")
    for gen in (event for event in latest.events if event.type == "generation"):
        usage = gen.usage or {}
        print(
            f"[bir] generation {gen.name:<13} model={gen.model}  "
            f"usage: in={usage.get('input_tokens')} "
            f"out={usage.get('output_tokens')} total={usage.get('total_tokens')}"
        )
    print(f"[bir] wrote {trace_path}")


if __name__ == "__main__":
    main()
