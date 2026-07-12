"""Phase 1 · Lesson 03 — prompts & correlation: versions and ids.

Lessons 01–02 taught the trace's *shape*. This lesson answers two questions you
will actually ask in production:

  1. **"Which prompt version produced this output?"** — ``bir.prompt(...)``
     builds a versioned :class:`PromptRecord` (name, version, template,
     variables) that attaches to ``generation(..., prompt=record)``. The Ollama
     wrapper from Lessons 01–02 does not take a prompt record, so this lesson
     uses the manual ``generation(...)`` context manager and its
     ``set_model`` / ``set_output`` / ``set_usage`` setters — the primitive the
     wrappers are built on. The same question is asked through TWO versions of
     the same named prompt, so the reloaded trace shows version → output side
     by side. A ``template_sha256`` is always recorded; the template, variables,
     and rendered text are captured only because the ``capture_*`` flags are
     set (they default to False, so prompt content stays private by default).
  2. **"Which trace does this log line belong to?"** —
     ``get_current_trace_id()`` / ``get_current_span_id()`` return the ids every
     event in the active trace carries. ``bir.logging.install_trace_id_filter``
     wires them into stdlib ``logging`` so ordinary application log lines gain
     ``[trace=… span=…]`` stamps with zero per-call plumbing. After the run,
     one stamped log line's ids are used to find its exact trace *and* event
     in ``./.bir/traces.jsonl``.

Run it:
  * offline (no Ollama, no network, deterministic — what CI runs):
      uv run python main.py --smoke
  * real (needs a local Ollama server + a pulled model):
      ollama pull llama3.2:1b
      uv run python main.py --prompt "Why version prompts at all?"

Ollama is local and keyless, so there is no API key to set.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from bir import (
    configure,
    generation,
    get_current_span_id,
    get_current_trace_id,
    load_traces,
    observe,
    prompt,
)
from bir.logging import install_trace_id_filter

DEFAULT_TRACE_PATH = Path(__file__).resolve().parent / ".bir" / "traces.jsonl"
DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_PROMPT = "What is a trace, and why would I record one?"

# The filter stamps ``bir_trace_id`` / ``bir_span_id`` onto every LogRecord, so
# any ordinary format string can render them — no bir-specific logging calls.
LOG_FORMAT = "%(levelname)s %(name)s [trace=%(bir_trace_id)s span=%(bir_span_id)s] %(message)s"

# Two versions of the SAME named prompt. v2 is the kind of edit teams actually
# ship — a persona and a stricter instruction — and the whole point of
# recording name+version is telling their outputs apart later.
PROMPT_NAME = "cookbook-qa"
PROMPT_VERSIONS: dict[str, dict] = {
    "v1": {
        "template": "Answer in one short sentence: {question}",
        "variables": {},
    },
    "v2": {
        "template": (
            "You are {persona}. Answer in one short sentence and use the word "
            "'trace': {question}"
        ),
        "variables": {"persona": "a patient observability teacher"},
    },
}

# An ordinary application logger — nothing bir-specific about it.
log = logging.getLogger("app")

# Keep the client module-level so it is never captured as an input (see CLAUDE.md).
_CLIENT = None


# --------------------------------------------------------------------------- #
# Offline fake — only used with --smoke. This lesson calls ``client.chat``
# directly (no wrapper), so the fake only needs the fields the lesson reads:
# ``model``, the text at ``message.content``, and token usage at
# ``prompt_eval_count`` / ``eval_count``. Its answer depends on whether the
# rendered prompt carries the v2 persona, so the two versions produce visibly
# different outputs even offline.
# --------------------------------------------------------------------------- #
class _FakeMessage:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


class _FakeChatResponse:
    def __init__(self, *, model: str, content: str, prompt_eval_count: int, eval_count: int) -> None:
        self.model = model
        self.message = _FakeMessage("assistant", content)
        self.prompt_eval_count = prompt_eval_count
        self.eval_count = eval_count
        self.done = True


class _FakeOllamaClient:
    """Deterministic stand-in for ``ollama.Client`` used only in --smoke mode."""

    def chat(self, *, model: str, messages: list[dict], **_kwargs) -> _FakeChatResponse:
        rendered = messages[-1]["content"] if messages else ""
        if rendered.startswith("You are"):
            content = "(smoke) As your teacher: every run leaves a trace you can replay."
        else:
            content = "(smoke) A recorded run of your program."
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
            "Start Ollama and pull a model (`ollama pull llama3.2:1b`), "
            "or run offline with --smoke."
        ) from exc
    return client


@observe(name="ollama_prompts", capture_inputs=True, capture_outputs=True, metadata={"recipe": "ollama-03-prompts"})
def compare_prompt_versions(question: str, model: str) -> dict[str, str]:
    """Ask the same question through both prompt versions, one generation each."""

    # The raw accessors the logging filter is built on. Inside the trace they
    # return the ids every event of this run will carry; outside they are None.
    log.info(
        "get_current_trace_id()=%s get_current_span_id()=%s",
        get_current_trace_id(),
        get_current_span_id(),
    )

    answers: dict[str, str] = {}
    for version, spec in PROMPT_VERSIONS.items():
        # A PromptRecord is inert metadata — it records WHICH prompt ran, it
        # does not run anything. render() fills the template's {placeholders}.
        record = prompt(
            PROMPT_NAME,
            version=version,
            template=spec["template"],
            variables={**spec["variables"], "question": question},
            capture_template=True,
            capture_variables=True,
            capture_rendered=True,
        )
        rendered = record.render()

        # generation(..., prompt=record) is the manual primitive: the record's
        # redacted form lands under the event's metadata["prompt"], and the
        # set_* calls fill in what the wrapper filled in for us in Lesson 01.
        with generation(f"ask_{version}", model=model, input=rendered, prompt=record) as gen:
            # Inside the generation, get_current_span_id() IS this event's id —
            # so this log line points at the exact generation it describes.
            log.info("calling ollama with prompt %s@%s", PROMPT_NAME, version)
            response = _CLIENT.chat(model=model, messages=[{"role": "user", "content": rendered}])
            gen.set_model(response.model)
            gen.set_output(response.message.content)
            if response.prompt_eval_count is not None and response.eval_count is not None:
                gen.set_usage(
                    input_tokens=response.prompt_eval_count,
                    output_tokens=response.eval_count,
                )
        answers[version] = response.message.content
    return answers


# --------------------------------------------------------------------------- #
# Logging wiring. The stream handler shows the stamped lines live; the
# capturing handler keeps (trace_id, span_id, line) so the post-run demo can
# take one line and look its trace up on disk.
# --------------------------------------------------------------------------- #
class _CapturingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[tuple[str | None, str | None, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append((record.bir_trace_id, record.bir_span_id, self.format(record)))


def _setup_logging() -> _CapturingHandler:
    formatter = logging.Formatter(LOG_FORMAT)
    captured = _CapturingHandler()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(), captured):
        handler.setFormatter(formatter)
        # Attach the filter per-handler: a handler stamps every record it
        # emits, including ones propagated from child loggers like ``app``.
        install_trace_id_filter(handler)
        root.addHandler(handler)
    return captured


# --------------------------------------------------------------------------- #
# Self-verification: reload the trace from disk, show which prompt version
# produced which output, then walk one stamped log line back to its trace.
# --------------------------------------------------------------------------- #
def _summarize_versions(events) -> None:
    print("[bir] prompt versions in this trace:")
    for event in events:
        if event.type != "generation":
            continue
        meta = event.metadata.get("prompt", {})
        output = str(event.output)
        snippet = output[:60] + ("…" if len(output) > 60 else "")
        print(
            f"[bir]   {meta.get('name')}@{meta.get('version')}  "
            f"template_sha256={str(meta.get('template_sha256'))[:12]}…  -> {snippet!r}"
        )


def _correlate_log_line(captured: _CapturingHandler, trace_path: Path) -> None:
    # Take the last log line that was emitted inside a trace...
    stamped = next(line for line in reversed(captured.lines) if line[0] is not None)
    line_trace_id, line_span_id, line_text = stamped

    # ...and use ONLY its stamped ids to find the run on disk, the way you
    # would from a production log aggregator: trace_id selects the trace in
    # the JSONL, span_id selects the exact event the line was logged under.
    matching_trace = next(t for t in load_traces(trace_path) if t.id == line_trace_id)
    matching_event = next(e for e in matching_trace.events if e.id == line_span_id)

    print("[bir] log ↔ trace correlation:")
    print(f"[bir]   log line:   {line_text}")
    print(f"[bir]   trace_id -> trace {matching_trace.id} ({len(matching_trace.events)} events) in {trace_path.name}")
    print(
        f"[bir]   span_id  -> {matching_event.type} {matching_event.name!r}"
        + (f" (model={matching_event.model})" if matching_event.model else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lesson 03 — versioned prompt records + log/trace correlation."
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

    captured = _setup_logging()
    # Outside any trace the accessors return None — the stamps show it.
    log.info("lesson 03 starting (smoke=%s)", smoke)

    global _CLIENT
    _CLIENT = _build_client(smoke)

    try:
        answers = compare_prompt_versions(args.prompt, args.model)
    except Exception as exc:  # pragma: no cover - real-path only
        if smoke:
            raise
        raise SystemExit(
            f"Ollama call failed: {exc}\n"
            f"Ensure Ollama is running and the model is pulled "
            f"(`ollama pull {args.model}`), or run with --smoke."
        ) from exc

    for version, answer in answers.items():
        print(f"{PROMPT_NAME}@{version}: {answer}")

    # Reload from disk and summarize so the run is self-verifying.
    latest = load_traces(trace_path)[-1]
    generations = [e for e in latest.events if e.type == "generation"]
    total_tokens = sum((g.usage or {}).get("total_tokens", 0) for g in generations)
    model_name = generations[-1].model if generations else args.model
    print(f"\n[bir] trace_id={latest.id}")
    print(f"[bir] events={len(latest.events)}  model={model_name}  total_tokens={total_tokens}")
    _summarize_versions(latest.events)
    _correlate_log_line(captured, trace_path)
    print(f"[bir] wrote {trace_path}")


if __name__ == "__main__":
    main()
