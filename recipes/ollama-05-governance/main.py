"""Phase 1 · Lesson 05 — governance: production controls.

Lessons 01-04 recorded everything, always. Production needs dials: tag traces
by deployment, turn recording off during an incident, keep only a sample of
high-volume traffic, widen redaction for your own secret shapes, and cap how
much of any one payload reaches disk. All of it lives in ``configure()``, which
mutates one process-global config — so this lesson deliberately calls it
several times, once per part, and prints each call as it happens.

Every part self-verifies by reloading ``./.bir/traces.jsonl`` — counting the
traces on disk before and after a part is the cleanest proof that something
was (or wasn't) recorded:

  A. deployment tags — ``configure(service_name=, environment=, source=)``;
     the reloaded trace ROOT carries ``metadata.service`` (keys ``name`` /
     ``environment``) and ``metadata.source``. Roots only — never children.
  B. the ``enabled`` kill switch — the traced Ollama call still runs and still
     answers, ``get_current_trace_id()`` still returns a live in-process id,
     but the on-disk trace count does not move. Env counterpart: a truthy
     ``BIR_DISABLED`` (an explicit ``configure(enabled=True)`` overrides it).
  C. sampling — exact-name ``sample_rules`` at 1.0 / 0.0 (deterministic:
     always kept vs never kept, and the never-kept call still answers), then
     a fractional global ``sample_rate`` shown statistically over many cheap
     traced calls. The keep/drop decision is made ONCE per trace root and
     inherited by every descendant event.
  D. redaction — ``additional_secret_keys`` / ``additional_redaction_patterns``
     only ever WIDEN the built-in rules, never weaken them. An obviously fake
     secret goes in; ``[redacted]`` is all that reaches disk.
  E. capture limits — ``max_value_length`` / ``max_collection_items`` bound one
     huge payload with a visible ``…[truncated]`` marker. Truncation always
     runs AFTER redaction, so a cut can never expose part of a secret.

Run it:
  * offline (no Ollama, no network, deterministic — what CI runs):
      uv run python main.py --smoke
  * real (needs a local Ollama server + a pulled model):
      ollama pull llama3.2:1b
      uv run python main.py --prompt "Why tag traces with an environment?"

Ollama is local and keyless, so there is no API key to set. Every secret in
this file is fake by construction and exists only to be redacted.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from bir import configure, get_current_trace_id, load_traces, observe
from bir.integrations.ollama import trace_chat as trace_ollama_chat

DEFAULT_TRACE_PATH = Path(__file__).resolve().parent / ".bir" / "traces.jsonl"
DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_PROMPT = "In one sentence, why tag traces with a service name and environment?"

SERVICE_NAME = "support-copilot"
ENVIRONMENT = "staging"
SOURCE = "ollama-05-governance"

# Part C: how many cheap traced calls the fractional-rate demo makes, and the rate.
PING_COUNT = 200
SAMPLE_RATE = 0.25

# Part D/E: obviously fake secrets, built to be redacted. The token matches the
# custom pattern below; the key/fingerprint are caught by mapping-KEY rules, so
# their values never reach disk regardless of content.
FAKE_API_KEY = "obviously-fake-lesson-05-key"
FAKE_FINGERPRINT = "fp-0000-obviously-fake"
FAKE_SESSION_TOKEN = "birdemo-1234-5678"
SESSION_TOKEN_PATTERN = r"birdemo-[0-9]{4}-[0-9]{4}"

# Part E: opt-in capture-size limits.
MAX_VALUE_LENGTH = 80
MAX_COLLECTION_ITEMS = 3
TRUNCATED = "…[truncated]"
REDACTED = "[redacted]"

# Keep the client module-level so it is never passed as an @observe argument and
# therefore never captured as an input (see CLAUDE.md).
_CLIENT = None


def _check(ok: bool, label: str) -> None:
    """Print a visible verification line; any failed check fails the run."""

    print(f"[bir] {'✓' if ok else '✗'} {label}")
    if not ok:
        raise SystemExit(f"self-check failed: {label}")


def _trace_count(trace_path: Path) -> int:
    """Traces currently on disk — the before/after yardstick for every part."""

    return len(load_traces(trace_path))


# --------------------------------------------------------------------------- #
# Offline fake — only used with --smoke. Same response shape as Lesson 01's
# fake (``model``, ``message.content``, ``prompt_eval_count`` / ``eval_count``,
# a ``model_dump``). It echoes the prompt, so Part D's fake token also flows
# through the captured OUTPUT and is redacted there too.
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
        content = f"(smoke) Recording is a governance decision, not a runtime one. You asked: {prompt}"
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


def _ask(prompt: str, model: str) -> str:
    """One Ollama chat call recorded as a generation — must run inside a trace."""

    response = trace_ollama_chat(
        _CLIENT.chat,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        bir_metadata={"recipe": "ollama-05-governance"},
    )
    return response.message.content


# --------------------------------------------------------------------------- #
# The traced entry points. Each part gets its own @observe root because
# sampling rules match trace-root names exactly, and because per-part roots
# keep the on-disk before/after counting honest.
# --------------------------------------------------------------------------- #
@observe(name="tagged_answer", capture_inputs=True, capture_outputs=True)
def tagged_answer(prompt: str, model: str) -> str:
    return _ask(prompt, model)


@observe(name="disabled_answer", capture_inputs=True, capture_outputs=True)
def disabled_answer(prompt: str, model: str) -> tuple[str, str | None]:
    # get_current_trace_id() keeps returning the live in-process id even while
    # recording is disabled, so log correlation works through an incident.
    return _ask(prompt, model), get_current_trace_id()


@observe(name="always_kept", capture_inputs=True, capture_outputs=True)
def always_kept(prompt: str, model: str) -> str:
    return _ask(prompt, model)


@observe(name="never_kept", capture_inputs=True, capture_outputs=True)
def never_kept(prompt: str, model: str) -> str:
    return _ask(prompt, model)


@observe(name="sampled_ping", capture_inputs=True, capture_outputs=True)
def sampled_ping(index: int) -> int:
    # Deliberately no Ollama call: the fractional-rate demo needs volume, and
    # the keep/drop decision happens at the trace ROOT — any cheap traced
    # function demonstrates it. Real calls would prove nothing extra, slowly.
    return index


@observe(name="redacted_answer", capture_inputs=True, capture_outputs=True)
def redacted_answer(request: dict, model: str) -> str:
    return _ask(request["user_prompt"], model)


@observe(name="archive_conversation", capture_inputs=True, capture_outputs=True)
def archive_conversation(note: str, turns: list[str]) -> dict:
    return {"note_chars": len(note), "turns": len(turns)}


# --------------------------------------------------------------------------- #
# The five parts. Each prints the configure() call it makes, runs traced work,
# then reloads ./.bir/traces.jsonl and checks what did (not) reach disk.
# --------------------------------------------------------------------------- #
def part_a_deployment_tags(prompt: str, model: str, trace_path: Path) -> None:
    print("\n== A · deployment tags: service_name / environment / source ==")
    print(f'[bir] configure(service_name="{SERVICE_NAME}", environment="{ENVIRONMENT}", source="{SOURCE}")')
    configure(service_name=SERVICE_NAME, environment=ENVIRONMENT, source=SOURCE)

    before = _trace_count(trace_path)
    print(tagged_answer(prompt, model))

    traces = load_traces(trace_path)
    _check(len(traces) == before + 1, "A: one new trace on disk")
    root = traces[-1].root
    _check(
        root.metadata.get("service") == {"name": SERVICE_NAME, "environment": ENVIRONMENT},
        f"A: root metadata.service == {{'name': '{SERVICE_NAME}', 'environment': '{ENVIRONMENT}'}}",
    )
    _check(root.metadata.get("source") == SOURCE, f"A: root metadata.source == '{SOURCE}'")
    gen = next(event for event in traces[-1].events if event.type == "generation")
    _check(
        "service" not in gen.metadata and "source" not in gen.metadata,
        "A: tags land on trace ROOTS only, never on child events",
    )
    print(f"[bir] trace_id={traces[-1].id}")


def part_b_kill_switch(prompt: str, model: str, trace_path: Path) -> None:
    print("\n== B · the enabled kill switch: code runs, nothing is written ==")
    before = _trace_count(trace_path)

    print("[bir] configure(enabled=False)   # incident toggle; env twin: BIR_DISABLED=1")
    configure(enabled=False)
    answer, live_trace_id = disabled_answer(prompt, model)
    print(answer)
    print("[bir] configure(enabled=True)    # recording resumes for traces started after this")
    configure(enabled=True)

    _check(_trace_count(trace_path) == before, "B: trace count on disk did NOT change while disabled")
    _check(bool(live_trace_id), "B: get_current_trace_id() still returned a live id while disabled")
    _check(
        live_trace_id not in {trace.id for trace in load_traces(trace_path)},
        "B: that live id never reached disk",
    )


def part_c_sampling(model: str, trace_path: Path) -> None:
    print("\n== C · sampling: exact-name rules, then a fractional global rate ==")
    print('[bir] configure(sample_rules={"always_kept": 1.0, "never_kept": 0.0})')
    configure(sample_rules={"always_kept": 1.0, "never_kept": 0.0})

    before = _trace_count(trace_path)
    print(always_kept("In one sentence, why sample traces instead of recording everything?", model))
    # The sampled-out call still runs end to end — you get the answer, Ollama
    # did the work — but its trace root (and every descendant) writes nothing.
    print(never_kept("In one sentence, what happens to work that is sampled out?", model))

    traces = load_traces(trace_path)
    _check(len(traces) == before + 1, "C: two traced calls ran, exactly one trace reached disk")
    _check(traces[-1].root.name == "always_kept", "C: the recorded trace is 'always_kept' (exact-name rule)")

    print(f"[bir] configure(sample_rules={{}}, sample_rate={SAMPLE_RATE})  # {{}} REPLACES (clears) the rule table")
    configure(sample_rules={}, sample_rate=SAMPLE_RATE)
    before = _trace_count(trace_path)
    for index in range(PING_COUNT):
        sampled_ping(index)
    kept = _trace_count(trace_path) - before
    print(
        f"[bir] global sample_rate={SAMPLE_RATE}: kept {kept}/{PING_COUNT} traced calls "
        f"(expected ≈{PING_COUNT * SAMPLE_RATE:.0f}; statistical, varies run to run — no assertion)"
    )

    print("[bir] configure(sample_rate=1.0)  # back to record-everything for the parts below")
    configure(sample_rate=1.0)


def part_d_redaction(model: str, trace_path: Path) -> None:
    print("\n== D · redaction: widen the built-in rules, never weaken them ==")
    print(
        '[bir] configure(additional_secret_keys=["session_fingerprint"], '
        f'additional_redaction_patterns=[r"{SESSION_TOKEN_PATTERN}"])'
    )
    configure(
        additional_secret_keys=["session_fingerprint"],
        additional_redaction_patterns=[SESSION_TOKEN_PATTERN],
    )

    request = {
        "user_prompt": (
            f"The support portal rejected my session token {FAKE_SESSION_TOKEN}. "
            "In one sentence, why might a session token be rejected?"
        ),
        "api_key": FAKE_API_KEY,
        "session-fingerprint": FAKE_FINGERPRINT,
    }
    print(f"[bir] request going in (terminal only, fake by construction): {request}")
    print(redacted_answer(request, model))

    latest = load_traces(trace_path)[-1]
    captured = latest.root.input["request"]
    _check(
        captured["api_key"] == REDACTED,
        'D: built-in KEY rule redacted "api_key" — always on, no configuration needed',
    )
    _check(
        captured["session-fingerprint"] == REDACTED,
        'D: additional_secret_keys matched "session-fingerprint" (whole name, case-insensitive, - == _)',
    )
    gen = next(event for event in latest.events if event.type == "generation")
    gen_prompt = gen.input["messages"][0]["content"]
    _check(
        REDACTED in gen_prompt and FAKE_SESSION_TOKEN not in gen_prompt,
        "D: custom pattern redacted the token inside the captured Ollama generation input",
    )
    raw = json.dumps([event.raw for event in latest.events])
    _check(
        all(fake not in raw for fake in (FAKE_API_KEY, FAKE_FINGERPRINT, FAKE_SESSION_TOKEN)),
        "D: no fake secret appears anywhere in the raw on-disk trace",
    )
    print(f"[bir] captured request on disk: {captured}")


def part_e_capture_limits(trace_path: Path) -> None:
    print("\n== E · capture limits: bound one huge payload, redact before the cut ==")
    print(f"[bir] configure(max_value_length={MAX_VALUE_LENGTH}, max_collection_items={MAX_COLLECTION_ITEMS})")
    configure(max_value_length=MAX_VALUE_LENGTH, max_collection_items=MAX_COLLECTION_ITEMS)

    # The fake token sits inside the first MAX_VALUE_LENGTH characters: if
    # truncation ran first, the cut text would still contain the raw token.
    # Part D's custom pattern is still configured — configure() only changes
    # the fields you pass — so redaction removes it before the cut ever happens.
    note = f"The session token {FAKE_SESSION_TOKEN} was pasted into this support note. " + (
        "More log lines follow. " * 20
    )
    turns = [f"turn-{index:02d}" for index in range(10)]
    print(archive_conversation(note, turns))

    root = load_traces(trace_path)[-1].root
    captured_note = root.input["note"]
    expected_note = note.replace(FAKE_SESSION_TOKEN, REDACTED)[:MAX_VALUE_LENGTH] + TRUNCATED
    _check(
        captured_note == expected_note,
        f"E: string capped at {MAX_VALUE_LENGTH} chars of redacted text + '{TRUNCATED}'",
    )
    _check(
        FAKE_SESSION_TOKEN.split("-")[0] not in captured_note,
        "E: redaction ran BEFORE the cut — no fragment of the token survived",
    )
    _check(
        root.input["turns"] == [f"turn-{index:02d}" for index in range(MAX_COLLECTION_ITEMS)] + [TRUNCATED],
        f"E: list keeps its first {MAX_COLLECTION_ITEMS} items plus one '{TRUNCATED}' marker",
    )
    print(f"[bir] captured note on disk: {captured_note!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lesson 05 — governance: tags, kill switch, sampling, redaction, capture limits."
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Feeds parts A and B; C-E use fixed inputs.")
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
    # Start from a known governance state (enabled=True, sample_rate=1.0
    # override any ambient BIR_DISABLED / BIR_SAMPLE_RATE) so each part below
    # demonstrates exactly the setting it configures.
    configure(
        trace_path=trace_path,
        capture_inputs=True,
        capture_outputs=True,
        enabled=True,
        sample_rate=1.0,
        sample_rules={},
    )

    global _CLIENT
    _CLIENT = _build_client(smoke)

    try:
        part_a_deployment_tags(args.prompt, args.model, trace_path)
        part_b_kill_switch(args.prompt, args.model, trace_path)
        part_c_sampling(args.model, trace_path)
        part_d_redaction(args.model, trace_path)
        part_e_capture_limits(trace_path)
    except Exception as exc:  # pragma: no cover - real-path only
        if smoke:
            raise
        raise SystemExit(
            f"Ollama call failed: {exc}\n"
            f"Ensure Ollama is running and the model is pulled "
            f"(`ollama pull {args.model}`), or run with --smoke."
        ) from exc

    print(f"\n[bir] all governance checks passed — traces on disk: {_trace_count(trace_path)}")
    print(f"[bir] wrote {trace_path}")


if __name__ == "__main__":
    main()
