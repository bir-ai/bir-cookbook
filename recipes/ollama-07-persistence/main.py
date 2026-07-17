"""Phase 1 · Lesson 07 — persistence: files that rotate, a server to send them to.

Everything Bir records lands in one local JSONL file — which grows forever
unless you cap it, and helps nobody else until you ship it. This lesson covers
both halves: opt-in size rotation (``configure(max_bytes=..., backup_count=...)``)
and uploading with ``send_events()`` to a Bir ingestion server.

There is no free hosted Bir server, so this recipe ships its own: an in-file,
in-memory fake (stdlib ``http.server`` on 127.0.0.1, ephemeral port) speaking
the exact wire protocol ``send_events()`` uses — POST ``/v1/events/batch``
answering ``{"accepted": <int>, "event_ids": [...]}`` — and idempotent on
event ids like a real one. The server is the SUBJECT of the lesson, so it runs
in BOTH modes; ``--smoke`` still means what it always means — no EXTERNAL
services — and only the Ollama client gets faked.

Five parts, each self-verified by reloading the local files and querying the
fake server's in-memory store (every asserted check is deterministic in both
modes):

  A. rotation — ``configure(max_bytes=4096, backup_count=3)``, then two real
     wrapped Ollama calls plus many cheap traced journal entries for volume
     (Lesson 05's sampled_ping trick). The active file rotates to ``.1``, the
     old ``.1`` to ``.2``, … — always on whole-line boundaries, so every file
     stays valid JSONL. Default ``load_events()`` reads ONLY the active file;
     ``include_rotated=True`` adds the backups, oldest first.
  B. the retention trade-off — the run writes more than ``backup_count`` files
     can hold, so the oldest backups get dropped: what is retained on disk is
     strictly smaller than what was written (the two REAL calls of this run
     are among the casualties). Also shown (print-only): a trace split across
     a rotation boundary appears in ``load_events()`` but is skipped by
     ``load_traces()`` for any file that lacks its root.
  C. first send — ``send_events(server_url)`` with defaults uploads the active
     file, whole traces root-first; the server ends up holding exactly the
     ``load_events()`` ids.
  D. safe re-sends — a plain re-send re-attempts everything and the idempotent
     server accepts 0 new; ``mark_sent=True`` records acknowledged ids in a
     ``traces.jsonl.sent`` sidecar so the next send attempts 0 events without
     even POSTing.
  E. rotation × sending — a default send strands rotated events;
     ``send_events(include_rotated=True)`` sweeps them up, deduplicated by id.
     Finally, transient failures: the server is scripted to 503 once and the
     send still succeeds on the built-in retry (``retries=2``, exponential
     ``backoff``).

Run it:
  * offline (no Ollama, no external network, deterministic — what CI runs):
      uv run python main.py --smoke
  * real (needs a local Ollama server + the pulled model):
      ollama pull llama3.2:1b
      uv run python main.py --prompt "Why rotate log files?"

Ollama is local and keyless, so there is no API key to set. The fake Bir
server is started and stopped by this script and only ever binds 127.0.0.1.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bir import (
    configure,
    get_current_trace_id,
    load_events,
    load_traces,
    observe,
    send_events,
    span,
)
from bir.integrations.ollama import trace_chat as trace_ollama_chat

DEFAULT_TRACE_PATH = Path(__file__).resolve().parent / ".bir" / "traces.jsonl"
DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_PROMPT = "In one sentence, why rotate log files?"

# The second real call uses a fixed prompt so --prompt only feeds the first.
PROMPT_2 = "In one sentence, why send traces to a central server instead of grepping local files?"

# Rotation knobs for part A: a few-KB cap rotates several times over ~30 KB of
# traced work, and backup_count=3 (the SDK default, made explicit) means the
# oldest backups get DROPPED — which is exactly what part B is about.
MAX_BYTES = 4096
BACKUP_COUNT = 3

# Write volume: 2 wrapped Ollama calls for substance + many cheap traced
# journal entries. Per-trace event counts are fixed by construction and used
# for part B's written-vs-retained accounting:
#   chat trace    = @observe root + the wrapper's generation  -> 2 events
#   journal trace = @observe root + two child spans           -> 3 events
CHAT_CALLS = 2
EVENTS_PER_CHAT_TRACE = 2
JOURNAL_CALLS = 30
EVENTS_PER_JOURNAL_TRACE = 3

# Every root function appends its trace id here, so part B can compare what
# the run WROTE against what rotation RETAINED without re-reading any file.
_WRITTEN_TRACE_IDS: list[str] = []

# Keep the client module-level so it is never passed as an @observe argument and
# therefore never captured as an input (see CLAUDE.md).
_CLIENT = None


def _check(ok: bool, label: str) -> None:
    """Print a visible verification line; any failed check fails the run."""

    print(f"[bir] {'✓' if ok else '✗'} {label}")
    if not ok:
        raise SystemExit(f"self-check failed: {label}")


# --------------------------------------------------------------------------- #
# The fake Bir ingestion server — the lesson's subject, NOT an external
# service: this script starts it on a daemon thread, it binds 127.0.0.1 on an
# ephemeral port, and it is shut down in a finally block. It runs in BOTH
# smoke and real mode (CI runs it too); --smoke only fakes the Ollama client.
# --------------------------------------------------------------------------- #
class _FakeBirHandler(BaseHTTPRequestHandler):
    server: _FakeBirServer

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass  # keep the lesson output readable

    def do_POST(self) -> None:
        if self.path != "/v1/events/batch":
            # A real server without the batch endpoint 404s here and the SDK
            # falls back to one POST per event — not exercised in this lesson.
            self._respond(404, {"error": "not found"})
            return
        srv = self.server
        with srv.lock:
            srv.batch_attempts += 1
            if srv.fail_next_batch_status is not None:
                status = srv.fail_next_batch_status
                srv.fail_next_batch_status = None
                self._respond(status, {"error": "scripted transient failure"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            batch = json.loads(self.rfile.read(length).decode("utf-8"))
            # Idempotent, like a real Bir server: ``accepted`` counts NEW ids
            # only; ``event_ids`` acknowledges every id the server now holds
            # from this batch (that acknowledgment is what mark_sent records).
            accepted = 0
            event_ids: list[str] = []
            for event in batch:
                event_id = event["id"]
                if event_id not in srv.events:
                    srv.events[event_id] = event  # dict preserves arrival order
                    accepted += 1
                event_ids.append(event_id)
        self._respond(200, {"accepted": accepted, "event_ids": event_ids})

    def _respond(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _FakeBirServer(ThreadingHTTPServer):
    """In-memory Bir ingestion server implementing POST /v1/events/batch."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _FakeBirHandler)
        self.events: dict[str, dict] = {}  # id -> raw event, in arrival order
        self.batch_attempts = 0  # every POST to the batch path, failures included
        self.fail_next_batch_status: int | None = None
        self.lock = threading.Lock()
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


# --------------------------------------------------------------------------- #
# Offline fake Ollama — only used with --smoke. Same response shape as Lesson
# 01's fake (``model``, ``message.content``, ``prompt_eval_count`` /
# ``eval_count``, a ``model_dump``).
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
        content = f"(smoke) Rotation bounds the file; send_events ships it. You asked: {prompt}"
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
            "Start Ollama and pull the model (`ollama pull llama3.2:1b`), "
            "or run offline with --smoke."
        ) from exc
    return client


# --------------------------------------------------------------------------- #
# The traced entry points.
# --------------------------------------------------------------------------- #
@observe(name="persisted_answer", capture_inputs=True, capture_outputs=True)
def persisted_answer(prompt: str, model: str) -> str:
    _WRITTEN_TRACE_IDS.append(get_current_trace_id())
    response = trace_ollama_chat(
        _CLIENT.chat,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        bir_name="chat.persisted",
        bir_metadata={"recipe": "ollama-07-persistence"},
    )
    return response.message.content


@observe(name="journal_entry", capture_inputs=True, capture_outputs=True)
def journal_entry(index: int) -> dict:
    # Deliberately no Ollama call: rotation cares about write VOLUME, and any
    # traced work provides it cheaply (Lesson 05's sampled_ping trick). Two
    # child spans make every journal trace multi-line, so a rotation boundary
    # can fall INSIDE a trace — part B shows what that split looks like.
    _WRITTEN_TRACE_IDS.append(get_current_trace_id())
    with span("entry.compose") as compose:
        note = f"journal note {index:03d}: " + "durable " * 10
        compose.set_metadata({"chars": len(note)})
    with span("entry.store") as store:
        store.set_metadata({"index": index, "fsync": True})
    return {"index": index, "chars": len(note)}


# --------------------------------------------------------------------------- #
# Small local helpers.
# --------------------------------------------------------------------------- #
def _fresh_slate(trace_path: Path) -> None:
    """Remove this recipe's output from any previous run.

    Parts A–E assert exact rotation and send accounting, so stale files would
    break them. Only files derived from the trace path are touched: the active
    file, its numeric ``.N`` backups, and the ``.sent`` sidecar. (The SDK's
    hidden ``.{name}.lock`` siblings are empty bookkeeping and stay put.)
    """

    candidates = [trace_path, trace_path.with_name(trace_path.name + ".sent")]
    prefix = trace_path.name + "."
    if trace_path.parent.is_dir():
        for entry in trace_path.parent.iterdir():
            if entry.name.startswith(prefix) and entry.name[len(prefix):].isdigit():
                candidates.append(entry)
    removed = 0
    for candidate in candidates:
        if candidate.exists():
            candidate.unlink()
            removed += 1
    if removed:
        print(f"[bir] cleared {removed} file(s) left by a previous run in {trace_path.parent}")


def _rotated_siblings(trace_path: Path) -> dict[int, Path]:
    """Existing numeric rotated siblings, keyed by suffix (higher = older)."""

    siblings: dict[int, Path] = {}
    prefix = trace_path.name + "."
    for entry in trace_path.parent.iterdir():
        if entry.name.startswith(prefix) and entry.name[len(prefix):].isdigit():
            siblings[int(entry.name[len(prefix):])] = entry
    return siblings


def _trace_files_oldest_first(trace_path: Path) -> list[Path]:
    """The retained files in write order: highest suffix first, active last."""

    siblings = _rotated_siblings(trace_path)
    return [siblings[n] for n in sorted(siblings, reverse=True)] + [trace_path]


def _print_contract_summary(trace_path: Path) -> None:
    """The standard self-verifying line: trace id, event count, model, usage."""

    latest = load_traces(trace_path)[-1]
    gen = next(event for event in latest.events if event.type == "generation")
    print(f"[bir] trace_id={latest.id}  events={len(latest.events)}  model={gen.model}  usage={gen.usage}")


# --------------------------------------------------------------------------- #
# The five parts.
# --------------------------------------------------------------------------- #
def part_a_rotation(prompt: str, model: str, trace_path: Path) -> None:
    print("\n== A · rotation: cap the active file, keep numbered backups ==")
    print(
        f"[bir] configure(max_bytes={MAX_BYTES}, backup_count={BACKUP_COUNT})"
        "  # opt-in; the default is ONE ever-growing file"
    )
    configure(max_bytes=MAX_BYTES, backup_count=BACKUP_COUNT)

    # Substance first: two real wrapped calls. Being the oldest writes of the
    # run, their traces will rotate out entirely — part B's retention point.
    print(persisted_answer(prompt, model))
    _print_contract_summary(trace_path)
    print(persisted_answer(PROMPT_2, model))

    # Volume: enough cheap traced work to rotate several times past the cap.
    for index in range(JOURNAL_CALLS):
        journal_entry(index)

    siblings = _rotated_siblings(trace_path)
    for suffix in sorted(siblings, reverse=True):
        print(f"[bir]   {siblings[suffix].name:<18} {siblings[suffix].stat().st_size:>6} bytes")
    print(f"[bir]   {trace_path.name:<18} {trace_path.stat().st_size:>6} bytes  (active)")

    _check(
        set(siblings) == set(range(1, BACKUP_COUNT + 1)),
        f"A: exactly backup_count={BACKUP_COUNT} rotated files on disk (.1 newest … .{BACKUP_COUNT} oldest)",
    )
    _check(
        all(len(load_events(path)) > 0 for path in siblings.values()),
        "A: every rotated file is valid, non-empty JSONL — rotation only cuts on whole-line boundaries",
    )
    default_count = len(load_events(trace_path))
    full_count = len(load_events(trace_path, include_rotated=True))
    print(f"[bir] load_events() -> {default_count} events   load_events(include_rotated=True) -> {full_count}")
    _check(
        default_count < full_count,
        "A: default load_events() reads ONLY the active file — include_rotated=True adds the backups, oldest first",
    )


def part_b_retention(trace_path: Path) -> None:
    print("\n== B · retention: backups are a bounded cache, not an archive ==")
    assert len(_WRITTEN_TRACE_IDS) == CHAT_CALLS + JOURNAL_CALLS  # accounting sanity
    written_traces = len(_WRITTEN_TRACE_IDS)
    written_events = CHAT_CALLS * EVENTS_PER_CHAT_TRACE + JOURNAL_CALLS * EVENTS_PER_JOURNAL_TRACE

    retained_events = load_events(trace_path, include_rotated=True)
    retained_traces = load_traces(trace_path, include_rotated=True)
    print(f"[bir] written this run:  {written_traces} traces / {written_events} events")
    print(f"[bir] retained on disk:  {len(retained_traces)} traces / {len(retained_events)} events (include_rotated=True)")

    _check(
        {trace.id for trace in retained_traces} < set(_WRITTEN_TRACE_IDS),
        "B: the retained traces are a strict subset of what was written — the oldest backups were dropped",
    )
    _check(
        len(retained_events) < written_events,
        "B: past backup_count, even include_rotated=True cannot see everything — rotation deletes, send first",
    )

    retained_trace_ids = {event.trace_id for event in retained_events}
    survived = [tid for tid in _WRITTEN_TRACE_IDS[:CHAT_CALLS] if tid in retained_trace_ids]
    print(
        f"[bir] the run's {CHAT_CALLS} REAL Ollama traces still on disk: {len(survived)}"
        " — the oldest writes age out first, however precious"
    )

    # Print-only: what a rotation-split trace looks like. Rotation cuts between
    # whole lines, and a trace's root line is written LAST (children first), so
    # a boundary inside a trace leaves head events in an older file and the
    # tail + root in a newer one. Any file holding only the head shows the
    # documented skip: load_events() returns the events, load_traces() drops
    # the whole group because its root is elsewhere — and once that older file
    # ages out past backup_count, those head events vanish for good.
    split_id, split_files = None, []
    seen: dict[str, list[Path]] = {}
    for file in _trace_files_oldest_first(trace_path):
        for event in load_events(file):
            files = seen.setdefault(event.trace_id, [])
            if file not in files:
                files.append(file)
    for tid, files in seen.items():
        if len(files) > 1:
            split_id, split_files = tid, files
            break
    if split_id is None:
        print("[bir] (no trace happened to straddle a rotation boundary this run)")
        return
    head = split_files[0]
    head_events = [event for event in load_events(head) if event.trace_id == split_id]
    head_trace_ids = {trace.id for trace in load_traces(head)}
    print(f"[bir] split trace {split_id} spans {len(split_files)} files; {head.name} holds its first {len(head_events)} event(s)")
    print(
        f"[bir]   load_events('{head.name}') sees those events, but the trace in load_traces('{head.name}')? "
        f"{split_id in head_trace_ids} — the group is silently skipped, its root lives in a newer file"
    )


def part_c_first_send(server: _FakeBirServer, trace_path: Path) -> None:
    print("\n== C · send_events: upload the active file to a Bir server ==")
    active_events = load_events(trace_path)
    result = send_events(server.url)  # path defaults to the configured trace_path
    print(f"[bir] send_events('{server.url}') -> accepted={result.accepted} attempted={result.attempted} skipped={result.skipped}")

    _check(
        result.accepted == result.attempted == len(active_events),
        f"C: the first send accepted everything the active file holds ({len(active_events)} events)",
    )
    _check(
        set(server.events) == {event.id for event in active_events},
        "C: the server's stored ids equal load_events() ids exactly",
    )
    first_received: dict[str, str] = {}
    for event_id, event in server.events.items():  # dict preserves arrival order
        first_received.setdefault(event["trace_id"], event_id)
    _check(
        all(event_id == trace_id for trace_id, event_id in first_received.items()),
        "C: for every trace, the FIRST event the server received was its root — whole traces upload root-first",
    )


def part_d_safe_resends(server: _FakeBirServer, trace_path: Path) -> None:
    print("\n== D · re-sends: idempotent by default, cheap with mark_sent ==")
    active_count = len(load_events(trace_path))

    again = send_events(server.url)
    print(f"[bir] plain re-send -> accepted={again.accepted} attempted={again.attempted} skipped={again.skipped}")
    _check(
        again.attempted == active_count and again.accepted == 0,
        "D: a plain re-send re-attempts every event; the idempotent server accepts 0 new",
    )
    _check(again.skipped == again.attempted, "D: .skipped == attempted - accepted — all duplicates, nothing lost")

    sidecar = trace_path.with_name(trace_path.name + ".sent")
    marked = send_events(server.url, mark_sent=True)
    _check(sidecar.exists(), f"D: mark_sent=True recorded the acknowledged ids in the {sidecar.name} sidecar")
    _check(
        marked.attempted == active_count and marked.accepted == 0,
        "D: the sidecar was empty before this send, so it still attempted everything once",
    )

    posts_before = server.batch_attempts
    marked_again = send_events(server.url, mark_sent=True)
    _check(
        marked_again.attempted == 0 and marked_again.accepted == 0,
        "D: next mark_sent send: every id already recorded — 0 attempted",
    )
    _check(server.batch_attempts == posts_before, "D: …and it never even POSTed — no wire traffic at all")


def part_e_rotated_sends_and_retries(server: _FakeBirServer, trace_path: Path) -> None:
    print("\n== E · rotation × sending, and transient failures ==")
    all_events = load_events(trace_path, include_rotated=True)
    all_ids = {event.id for event in all_events}
    active_ids = {event.id for event in load_events(trace_path)}
    stranded = all_ids - active_ids
    print(f"[bir] events sitting only in rotated backups: {len(stranded)} — every default send so far ignored them")
    _check(
        len(stranded) > 0 and not (stranded & set(server.events)),
        "E: rotated events are stranded — the server has none of them",
    )

    swept = send_events(server.url, include_rotated=True)
    print(f"[bir] send_events(include_rotated=True) -> accepted={swept.accepted} attempted={swept.attempted} skipped={swept.skipped}")
    _check(
        swept.attempted == len(all_events) and swept.accepted == len(stranded),
        "E: the sweep attempted the whole retained set; the server accepted exactly the stranded events",
    )
    _check(
        set(server.events) == all_ids,
        "E: server ids == the include_rotated local ids — deduplicated by id, nothing double-stored",
    )

    # Transient failures: script the server to 503 the NEXT batch request. The
    # send still succeeds because 5xx / timeouts / connection errors are
    # retried up to retries=2 times (kept at its default here), sleeping
    # backoff * 2**attempt between tries — shortened only to keep this snappy.
    server.fail_next_batch_status = 503
    posts_before = server.batch_attempts
    retried = send_events(server.url, backoff=0.1)
    _check(
        server.batch_attempts - posts_before == 2,
        "E: the scripted 503 cost one attempt, the retry succeeded — exactly two batch POSTs on the wire",
    )
    _check(
        retried.attempted == len(active_ids) and retried.accepted == 0,
        "E: the retried send completed normally (everything is a duplicate by now)",
    )
    print("[bir] a 4xx other than 404 would raise immediately instead — permanent rejections are not retried")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lesson 07 — persistence: file rotation (max_bytes/backup_count) and send_events to a Bir server."
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Feeds the first real chat call; the rest is fixed.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--trace-path",
        default=str(DEFAULT_TRACE_PATH),
        help="Cleared at startup (with its .N backups and .sent sidecar) — the rotation and send accounting "
        "in this lesson needs a clean slate.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Offline mode: in-file fake Ollama client (no external network). The loopback Bir server still "
        "runs — the script starts and stops it itself. Also enabled by BIR_COOKBOOK_SMOKE=1.",
    )
    args = parser.parse_args()

    smoke = args.smoke or os.environ.get("BIR_COOKBOOK_SMOKE") == "1"
    trace_path = Path(args.trace_path)
    _fresh_slate(trace_path)
    configure(trace_path=trace_path, capture_inputs=True, capture_outputs=True)

    global _CLIENT
    _CLIENT = _build_client(smoke)

    server = _FakeBirServer()
    server.start()
    print(f"[bir] fake Bir ingestion server listening on {server.url} (in-process, loopback only)")

    try:
        part_a_rotation(args.prompt, args.model, trace_path)
        part_b_retention(trace_path)
        part_c_first_send(server, trace_path)
        part_d_safe_resends(server, trace_path)
        part_e_rotated_sends_and_retries(server, trace_path)
    except Exception as exc:  # pragma: no cover - real-path only
        if smoke:
            raise
        raise SystemExit(
            f"Lesson failed mid-run: {exc}\n"
            f"If this is an Ollama connection error, ensure Ollama is running and the model is pulled "
            f"(`ollama pull {args.model}`), or run with --smoke."
        ) from exc
    finally:
        server.stop()

    server_trace_count = len({event["trace_id"] for event in server.events.values()})
    print(f"\n[bir] all persistence checks passed — the server holds {len(server.events)} events from {server_trace_count} traces")
    print(f"[bir] wrote {trace_path} (+ .1–.{BACKUP_COUNT} backups and the {trace_path.name}.sent sidecar)")


if __name__ == "__main__":
    main()
