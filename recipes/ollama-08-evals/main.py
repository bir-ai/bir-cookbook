"""Phase 1 · Lesson 08 — evals: the offline eval loop, end to end.

Every lesson so far recorded what your LLM code DID; this one measures how WELL
it did. The whole loop is offline and deterministic: a ``Dataset`` of examples,
deterministic evaluator factories, ``run_experiment`` persisting one JSONL row
per example plus a ``.summary.json`` sidecar, ``render_experiment_report`` for
a shareable HTML/Markdown report, and ``compare_experiments`` as the regression
gate a CI job can key on.

The model's prose differs between smoke and real runs, so every asserted check
keys on structure the TASK's code controls — the ``[doc-id]`` citation the code
appends, the ``{"answer", "contexts"}`` mapping it builds — never on live model
text. That is also the lesson's design advice: let code, not the model, own the
fields your evaluators judge.

Eight parts, each self-verified by reloading what was persisted (every asserted
check is deterministic in both modes):

  A. datasets — build a ``Dataset`` in code (an example's ``input`` mapping
     becomes the task's kwargs), round-trip it through ``to_jsonl`` /
     ``from_jsonl`` (redacted by default, like every Bir artifact), and show
     the duplicate-id ``ValueError``.
  B. evaluators standalone — ``evaluator.evaluate()`` needs no experiment: one
     factory of each flavor (expected-from-example fallback, configured
     expected, context-based ``latency_under``/``cost_under``, ``custom_evaluator``
     with bool coercion, and a RAG check) scoring exactly 1.0/0.0 with the
     evidence in ``EvalResult.metadata``.
  C. run_experiment — the grounded-QA task over the whole dataset with
     ``record_traces=True`` (each example runs in its own
     ``experiment.<name>.<example_id>`` trace, scores become score events, and
     the wrapped Ollama call nests inside — the bridge to Lessons 01–07) and
     ``raise_on_error=False`` (one example fails deterministically and becomes a
     ``status="error"`` row while the run continues). Aggregates are recomputed
     from the reloaded rows and must match the summary sidecar. A second, tiny
     run shows the default ``raise_on_error=True``: rows and summary are
     persisted through the failure, then the exception re-raises.
  D. reports — ``render_experiment_report`` in both formats: one self-contained
     string each, written next to the experiment file, byte-identical when
     re-rendered.
  E. the regression gate — the SAME dataset through the baseline task and a
     deliberately degraded candidate (plain string, no RAG mapping, one example
     loses its citation), so the regressed/improved/unchanged sets are exact.
     ``score_tolerances`` absorbs the small drop (boundary inclusive),
     ``tolerance=1.0`` opens the gate, ``missing_score="regress"`` closes it
     again for the evaluator the candidate dropped, and ``per_example=True``
     pins the small drop to the one example that caused it. Finally
     ``list_experiments`` reads every ``.summary.json`` newest-first.
  F. run_experiment_async — the same loop as C, awaited: a coroutine task
     (``trace_chat_async`` around the async Ollama client) runs a two-example
     subset with ``max_concurrency=2``. Results, rows, and aggregates keep
     dataset order regardless of completion order (the smoke fake stalls the
     first call so the first example finishes last), each example still gets
     its own isolated trace, and ``retrieved_context_contains`` checks the
     retrieval side of the RAG output — both retained docs mention "trace".
  G. send_experiment — ship a persisted run (rows + summary sidecar) to a Bir
     server. As in Lesson 07 there is no free hosted server, so an in-file
     loopback fake speaks the exact wire protocol — POST ``/v1/experiments``
     answering ``{"accepted", "id"}`` — in BOTH modes; a scripted 503 shows
     the same retry loop ``send_events`` uses.
  H. capture_traces — unit-test your INSTRUMENTATION the way parts A–G eval
     your outputs: ``bir.testing.capture_traces`` redirects trace writes to a
     private temp file for one ``with`` block (every other configured setting
     is kept) and yields a handle whose ``events()``/``traces()`` read live
     during the block and return a snapshot after it. The lesson's own task
     runs inside the block and is asserted to record one generation with the
     right ``bir_name``, model, and token usage — while the recipe's real
     ``traces.jsonl`` gains nothing and the prior ``configure(...)`` is
     restored on exit.

Run it:
  * offline (no Ollama, no network, deterministic — what CI runs):
      uv run python main.py --smoke
  * real (needs a local Ollama server + the pulled model):
      ollama pull llama3.2:1b
      uv run python main.py --prompt "In one sentence, what does a trace record?"

Ollama is local and keyless, so there is no API key to set. The fake Bir
experiments server in part G is started and stopped by this script and only
ever binds 127.0.0.1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
from collections.abc import Mapping
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bir import configure, load_traces, trace
from bir.evals import (
    Dataset,
    DatasetExample,
    EvaluationContext,
    answer_contains_citation,
    answer_context_overlap,
    compare_experiments,
    contains,
    cost_under,
    custom_evaluator,
    exact_match,
    field_contains,
    field_equals,
    json_valid,
    latency_under,
    list_experiments,
    load_experiment,
    load_experiment_summary,
    numeric_between,
    regex_match,
    render_experiment_report,
    retrieved_context_contains,
    run_experiment,
    run_experiment_async,
    send_experiment,
    similarity_above,
)
from bir.integrations.ollama import (
    trace_chat as trace_ollama_chat,
    trace_chat_async as trace_ollama_chat_async,
)
from bir.testing import capture_traces

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
DEFAULT_PROMPT = "In one sentence, what does a trace record?"

# The tiny corpus the task retrieves from. Retrieval is a plain dict lookup so
# the only nondeterministic step in the whole lesson is the model's wording.
_DOCS = {
    "docs-tracing": (
        "A trace records one run of your program as a tree of events, so any "
        "output can be explained by the steps that produced it."
    ),
    "docs-evals": (
        "An eval runs a task over a dataset and scores every output with "
        "deterministic evaluators, turning quality into numbers you can "
        "compare between runs."
    ),
    "docs-datasets": (
        "A dataset is a list of uniquely identified examples, each holding an "
        "input, an optional expected value, and metadata, stored as one JSON "
        "object per line."
    ),
    "docs-scores": (
        "A score is a named numeric judgment attached to a trace, so quality "
        "checks live next to the recorded run they judged."
    ),
    "docs-gate": (
        "A regression gate compares a candidate run against a baseline and "
        "fails when any shared score drops beyond its tolerance."
    ),
}

# The degraded candidate task in part E drops the citation for exactly this
# example, producing a small, exact aggregate drop (4/5) next to the full ones.
_CITATION_DROPPED_DOC = "docs-datasets"
_CITATION_DROPPED_ID = "e3-datasets"

# Part F's compact async subset. retrieved_context_contains expects ONE fixed
# string across the whole run, and both of these docs mention "trace".
_ASYNC_EXAMPLE_IDS = ("e1-tracing", "e4-scores")

# A generous per-example latency budget keeps the context-evaluator demo
# deterministic: even a cold local model answers well within five minutes.
LATENCY_BUDGET_MS = 300_000

# The seven evaluators every quality run scores with (part C asserts each
# success row scores exactly 1.0 on all of them — they only judge structure
# the task's code controls, never the model's wording).
QUALITY_EVAL_NAMES = (
    "answer_contains_citation",
    "answer_names_doc",
    "cites_right_doc",
    "has_rag_shape",
    "json_valid",
    "latency_under",
    "source_count_ok",
)

# Keep the clients module-level so they are never passed as an @observe/task
# argument and therefore never captured as an input (see CLAUDE.md). The model
# name lives here too: a task's signature is dictated by its examples' input
# keys, so run-wide settings reach it from module scope, not the dataset.
_CLIENT = None
_ASYNC_CLIENT = None
_MODEL = DEFAULT_MODEL


def _check(ok: bool, label: str) -> None:
    """Print a visible verification line; any failed check fails the run."""

    print(f"[bir] {'✓' if ok else '✗'} {label}")
    if not ok:
        raise SystemExit(f"self-check failed: {label}")


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
        question = messages[-1]["content"].rsplit("Question: ", 1)[-1] if messages else ""
        content = f"(smoke) According to the document: {question}"
        return _FakeChatResponse(
            model=model,
            content=content,
            prompt_eval_count=sum(len(m["content"].split()) for m in messages),
            eval_count=len(content.split()),
        )


class _FakeAsyncOllamaClient(_FakeOllamaClient):
    """Deterministic stand-in for ``ollama.AsyncClient`` used only in --smoke mode."""

    def __init__(self) -> None:
        self._calls = 0

    async def chat(self, *, model: str, messages: list[dict], **_kwargs) -> _FakeChatResponse:
        # The FIRST call stalls briefly, so under max_concurrency=2 the first
        # example finishes LAST — letting part F show that results still come
        # back in dataset order, not completion order.
        self._calls += 1
        if self._calls == 1:
            await asyncio.sleep(0.05)
        return super().chat(model=model, messages=messages)


def _build_client(smoke: bool):
    """Return the (sync client, async client) pair — fakes with --smoke."""

    if smoke:
        return _FakeOllamaClient(), _FakeAsyncOllamaClient()

    # Real mode: talk to a local Ollama server. Import the provider library
    # lazily so --smoke needs neither the package nor a running server. Ollama
    # needs no API key — the client honors the OLLAMA_HOST env var if set.
    try:
        from ollama import AsyncClient, Client
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
    return client, AsyncClient()


# --------------------------------------------------------------------------- #
# The fake Bir experiments server — part G's subject, NOT an external service:
# this script starts it on a daemon thread, it binds 127.0.0.1 on an ephemeral
# port, and it is stopped in a finally block. Like Lesson 07's events server it
# runs in BOTH modes (--smoke only fakes the Ollama client), speaking the exact
# wire protocol send_experiment uses: POST /v1/experiments answering
# {"accepted": <int>, "id": <experiment_id>}.
# --------------------------------------------------------------------------- #
class _FakeBirExperimentsHandler(BaseHTTPRequestHandler):
    server: _FakeBirExperimentsServer

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass  # keep the lesson output readable

    def do_POST(self) -> None:
        if self.path != "/v1/experiments":
            self._respond(404, {"error": "not found"})
            return
        srv = self.server
        with srv.lock:
            srv.post_attempts += 1
            if srv.fail_next_status is not None:
                status = srv.fail_next_status
                srv.fail_next_status = None
                self._respond(status, {"error": "scripted transient failure"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            experiment_id = payload["summary"]["experiment_id"]
            srv.experiments[experiment_id] = payload  # re-sends upsert, keyed by id
        self._respond(200, {"accepted": len(payload["results"]), "id": experiment_id})

    def _respond(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _FakeBirExperimentsServer(ThreadingHTTPServer):
    """In-memory Bir server implementing POST /v1/experiments."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _FakeBirExperimentsHandler)
        self.experiments: dict[str, dict] = {}  # experiment_id -> last posted payload
        self.post_attempts = 0  # every POST to the experiments path, failures included
        self.fail_next_status: int | None = None
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
# The tasks under evaluation. An example's ``input`` is the mapping
# ``{"question", "doc_id"}``, so run_experiment calls each task as
# ``task(question=..., doc_id=...)`` — dataset inputs and task signatures are
# designed together. Any non-mapping input would be passed as one positional
# argument instead.
# --------------------------------------------------------------------------- #
def _lookup_doc(doc_id: str) -> str:
    doc = _DOCS.get(doc_id)
    if doc is None:
        # Deterministic failure before any model call: this is how e6 becomes a
        # status="error" row in part C in both smoke and real mode.
        raise LookupError(f"retrieval failed: no document with id {doc_id!r}")
    return doc


def _grounding_messages(question: str, doc: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                "Answer in one short sentence, using only this document.\n\n"
                f"Document: {doc}\n\nQuestion: {question}"
            ),
        }
    ]


def _model_answer(question: str, doc: str) -> str:
    # trace_ollama_chat must run inside an active trace; run_experiment's
    # record_traces=True provides one per example, so the generation lands
    # inside that example's experiment trace.
    response = trace_ollama_chat(
        _CLIENT.chat,
        model=_MODEL,
        messages=_grounding_messages(question, doc),
        bir_name="chat.grounded_answer",
        bir_metadata={"recipe": "ollama-08-evals"},
    )
    return response.message.content.strip()


def _sourced_output(text: str, doc_id: str, doc: str) -> dict:
    contexts = [doc]
    return {
        "answer": f"{text} [{doc_id}]",
        "contexts": contexts,
        "doc_id": doc_id,
        "source_count": len(contexts),
    }


def answer_with_sources(question: str, doc_id: str) -> dict:
    """Baseline task: the RAG shape, with the citation appended by CODE.

    The model writes the prose; the code owns everything the evaluators judge —
    the mapping shape, the ``contexts`` list, the ``doc_id`` and ``source_count``
    fields, and the ``[doc-id]`` citation marker. That is what makes the eval
    deterministic.
    """

    doc = _lookup_doc(doc_id)
    text = _model_answer(question, doc)
    return _sourced_output(text, doc_id, doc)


async def answer_with_sources_async(question: str, doc_id: str) -> dict:
    """Part F's task: the same RAG shape as the baseline, awaited.

    run_experiment_async awaits whatever the task produces, so a coroutine
    function just works; trace_chat_async records the generation inside the
    per-example experiment trace exactly like the sync wrapper does.
    """

    doc = _lookup_doc(doc_id)
    response = await trace_ollama_chat_async(
        _ASYNC_CLIENT.chat,
        model=_MODEL,
        messages=_grounding_messages(question, doc),
        bir_name="chat.grounded_answer",
        bir_metadata={"recipe": "ollama-08-evals"},
    )
    return _sourced_output(response.message.content.strip(), doc_id, doc)


def casual_answer(question: str, doc_id: str) -> str:
    """Degraded candidate for part E: a plain string instead of the RAG shape.

    The ``answer: `` prefix guarantees the string is never valid JSON, brackets
    from the model's own text are stripped so a stray ``[1]`` can't fake a
    citation, and one designated example loses its citation entirely — every
    delta the gate sees is exact and code-controlled.
    """

    doc = _lookup_doc(doc_id)
    text = _model_answer(question, doc).replace("[", "(").replace("]", ")")
    if doc_id == _CITATION_DROPPED_DOC:
        return f"answer: {text}"
    return f"answer: {text} [{doc_id}]"


def _has_rag_shape(output, expected) -> bool:
    del expected
    return (
        isinstance(output, Mapping)
        and isinstance(output.get("answer"), str)
        and isinstance(output.get("contexts"), list)
    )


def _quality_evaluators() -> list:
    """The seven structural checks every quality run scores with.

    ``field_equals("doc_id")`` and ``field_contains("answer")`` configure no
    expected value, so each falls back to its example's ``expected`` (the doc
    the answer must be grounded in) and would raise if an example had none —
    the code-appended ``[doc-id]`` citation is what puts that id in the answer
    text. ``numeric_between`` bounds the ``source_count`` field the task's code
    sets. ``custom_evaluator`` returns a bool that is coerced to exactly
    1.0/0.0.
    """

    return [
        answer_contains_citation(),
        json_valid(),
        custom_evaluator("has_rag_shape", _has_rag_shape),
        field_equals("doc_id", name="cites_right_doc"),
        field_contains("answer", name="answer_names_doc"),
        numeric_between(1, 3, field="source_count", name="source_count_ok"),
        latency_under(LATENCY_BUDGET_MS),
    ]


def _plain_string_evaluator():
    # 0.0 on the baseline's mappings, 1.0 on the candidate's strings — a
    # deterministic "improved" evaluator for part E's diff.
    return custom_evaluator("is_plain_string", lambda output, expected: isinstance(output, str))


def _build_examples(prompt: str) -> list[DatasetExample]:
    return [
        DatasetExample(
            id="e1-tracing",
            input={"question": prompt, "doc_id": "docs-tracing"},
            expected="docs-tracing",
            metadata={"topic": "tracing"},
        ),
        DatasetExample(
            id="e2-evals",
            input={"question": "Why run evals offline?", "doc_id": "docs-evals"},
            expected="docs-evals",
            metadata={"topic": "evals"},
        ),
        DatasetExample(
            id=_CITATION_DROPPED_ID,
            input={"question": "What is in a dataset example?", "doc_id": _CITATION_DROPPED_DOC},
            expected=_CITATION_DROPPED_DOC,
            metadata={"topic": "datasets"},
        ),
        DatasetExample(
            id="e4-scores",
            input={"question": "Where do scores live?", "doc_id": "docs-scores"},
            expected="docs-scores",
            metadata={"topic": "scores"},
        ),
        DatasetExample(
            id="e5-gate",
            input={"question": "When does a regression gate fail?", "doc_id": "docs-gate"},
            expected="docs-gate",
            metadata={"topic": "gate"},
        ),
        # The poisoned example: its doc_id matches nothing, so the task raises
        # before any model call — deterministically, in both modes.
        DatasetExample(
            id="e6-missing-doc",
            input={"question": "What happens when retrieval breaks?", "doc_id": "docs-nonexistent"},
            expected="docs-nonexistent",
            metadata={"topic": "failure"},
        ),
    ]


# --------------------------------------------------------------------------- #
# Small local helpers.
# --------------------------------------------------------------------------- #
def _fresh_slate(trace_path: Path, experiments_dir: Path, datasets_dir: Path) -> None:
    """Remove this recipe's output from any previous run.

    Parts C–G assert exact trace and experiment counts (down to what
    ``list_experiments`` returns), so stale files would break them. Only this
    recipe's own outputs are touched: the active trace file, the experiment
    results/summaries/reports, and the exported dataset.
    """

    candidates = [trace_path]
    for directory, patterns in (
        (experiments_dir, ("*.jsonl", "*.summary.json", "*.report.html", "*.report.md")),
        (datasets_dir, ("*.jsonl",)),
    ):
        if directory.is_dir():
            for pattern in patterns:
                candidates.extend(sorted(directory.glob(pattern)))
    removed = 0
    for candidate in candidates:
        if candidate.exists():
            candidate.unlink()
            removed += 1
    if removed:
        print(f"[bir] cleared {removed} file(s) left by a previous run under {trace_path.parent}")


def _recompute_aggregates(path: Path) -> dict[str, float]:
    """Recompute per-evaluator means from the persisted rows — the long way."""

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in load_experiment(path).results:
        for score in row.scores:
            totals[score.name] = totals.get(score.name, 0.0) + score.value
            counts[score.name] = counts.get(score.name, 0) + 1
    return {name: totals[name] / counts[name] for name in sorted(totals)}


# --------------------------------------------------------------------------- #
# The seven parts.
# --------------------------------------------------------------------------- #
def part_a_datasets(prompt: str, datasets_dir: Path) -> Dataset:
    print("\n== A · datasets: uniquely identified examples, JSONL round-trip ==")
    examples = _build_examples(prompt)
    dataset = Dataset(examples)
    _check(
        len(dataset) == 6 and [example.id for example in dataset] == [example.id for example in examples],
        "A: datasets are sized and iterable, in example order",
    )

    # Duplicate ids are rejected at construction, not discovered mid-run.
    try:
        Dataset([examples[0], examples[0]])
        duplicate_error = None
    except ValueError as exc:
        duplicate_error = str(exc)
    print(f"[bir] Dataset with a duplicate id -> ValueError: {duplicate_error}")
    _check(
        duplicate_error is not None and examples[0].id in duplicate_error,
        "A: duplicate example ids raise a ValueError naming the offending id",
    )

    # Round-trip through JSONL. to_jsonl redacts by default — the same safe
    # capture as traces and experiment rows — which changes nothing here
    # because these inputs hold no secrets.
    dataset_path = datasets_dir / "qa.jsonl"
    dataset.to_jsonl(dataset_path)
    reloaded = Dataset.from_jsonl(dataset_path)
    print(f"[bir] first dataset row: {dataset_path.read_text(encoding='utf-8').splitlines()[0]}")
    _check(
        [example.id for example in reloaded] == [example.id for example in dataset]
        and [example.input for example in reloaded] == [example.input for example in dataset]
        and [example.expected for example in reloaded] == [example.expected for example in dataset],
        "A: to_jsonl -> from_jsonl round-trips ids, inputs, and expected values exactly",
    )
    return dataset


def part_b_evaluators() -> None:
    print("\n== B · evaluators standalone: exact 1.0/0.0, evidence in metadata ==")

    # Flavor 1 — expected from the example. With no configured value the
    # factory falls back to the example's expected, and raises without one.
    capital = exact_match()
    hit = capital.evaluate("Ankara", expected="Ankara")
    _check(hit.value == 1.0 and hit.metadata["expected"] == "Ankara", "B: exact_match scores exactly 1.0 on equality")
    try:
        capital.evaluate("Ankara")
        fallback_error = None
    except ValueError as exc:
        fallback_error = str(exc)
    _check(
        fallback_error == "exact_match requires an expected value",
        "B: with no configured or example expected, the evaluator raises instead of guessing",
    )

    # Flavor 2 — configured expected. similarity_above is difflib's
    # SequenceMatcher ratio: deterministic fuzz, threshold boundary inclusive,
    # achieved ratio recorded so a failure is inspectable.
    fuzzy = similarity_above(0.8, expected="tracing records every step", case_sensitive=False)
    near = fuzzy.evaluate("Tracing records each step")
    far = fuzzy.evaluate("completely different words")
    print(f"[bir] similarity ratios: near={near.metadata['ratio']:.3f} far={far.metadata['ratio']:.3f} (threshold 0.8)")
    _check(near.value == 1.0 and far.value == 0.0, "B: similarity_above is binary — the ratio is evidence, not the score")
    _check(
        similarity_above(1.0, expected="same").evaluate("same").value == 1.0,
        "B: the threshold boundary is inclusive — ratio 1.0 passes threshold 1.0",
    )

    # Flavor 3 — context-based. latency_under and cost_under judge the run,
    # not the output text, so they take an EvaluationContext (run_experiment
    # builds it; here we build one by hand).
    fast = latency_under(50).evaluate(None, context=EvaluationContext(example=None, output=None, duration_ms=20.0))
    slow = latency_under(50).evaluate(None, context=EvaluationContext(example=None, output=None, duration_ms=80.0))
    _check(
        fast.value == 1.0 and slow.value == 0.0 and slow.metadata["duration_ms"] == 80.0,
        "B: latency_under judges context.duration_ms and records it as evidence",
    )
    try:
        latency_under(50).evaluate(None)
        context_error = None
    except ValueError as exc:
        context_error = str(exc)
    _check(
        context_error == "latency_under requires an evaluation context",
        "B: a context evaluator refuses to run without a context",
    )
    priced = cost_under(0.01).evaluate(
        None,
        context=EvaluationContext(example=None, output={"cost": {"total_cost": 0.002}}, duration_ms=1.0),
    )
    _check(
        priced.value == 1.0 and priced.metadata["actual"] == 0.002,
        "B: cost_under reads the output's cost field (nested Lesson-06 shape included)",
    )

    # Flavor 4 — custom_evaluator: any callable; a returned bool is coerced to
    # exactly 1.0/0.0 (ints/floats pass through, EvalResult carries metadata).
    is_mapping = custom_evaluator("is_mapping", lambda output, expected: isinstance(output, Mapping))
    said_yes = is_mapping.evaluate({"answer": "x"})
    _check(
        said_yes.value == 1.0 and isinstance(said_yes.value, float) and is_mapping.evaluate("plain").value == 0.0,
        "B: custom_evaluator coerces a returned bool to exactly 1.0/0.0",
    )

    # Flavor 5 — RAG checks judge the {"answer", "contexts"} shape part C's
    # task produces. Word-overlap is a heuristic for unsupported answers.
    overlap = answer_context_overlap(0.5)
    grounded = overlap.evaluate({"answer": "Paris hosts the Louvre", "contexts": ["The Louvre is in Paris"]})
    adrift = overlap.evaluate({"answer": "Berlin has museums", "contexts": ["The Louvre is in Paris"]})
    _check(
        grounded.value == 1.0 and grounded.metadata["overlap_ratio"] == 0.75 and adrift.value == 0.0,
        "B: answer_context_overlap scores word support of the answer by the contexts",
    )
    print(f"[bir] unsupported words recorded as evidence: {adrift.metadata['unsupported_words']}")


def part_c_run_experiment(dataset: Dataset, experiments_dir: Path, trace_path: Path):
    print("\n== C · run_experiment: score the task over the dataset, traced ==")
    quality_path = experiments_dir / "qa-quality.jsonl"
    # Without an explicit path, results land in CWD-relative
    # .bir/experiments/<name>-<uuid>.jsonl — anchor them in the recipe instead.
    result = run_experiment(
        "qa-quality",
        dataset=dataset,
        task=answer_with_sources,
        evaluators=_quality_evaluators(),
        path=quality_path,
        raise_on_error=False,  # record the failure as a row and keep going
        record_traces=True,  # one trace per example; also what trace_ollama_chat needs
    )

    summary_path = quality_path.with_suffix(".summary.json")
    _check(
        quality_path.exists() and summary_path.exists(),
        "C: one JSONL row per example, plus the .summary.json sidecar",
    )
    summary = load_experiment_summary(summary_path)
    error_rows = [row for row in result.results if row.status == "error"]
    _check(
        result.status == "error" and summary.status == "error" and summary.error_count == len(error_rows) == 1,
        "C: raise_on_error=False turned the poisoned example into a status=error row, not a crash",
    )
    error_row = error_rows[0]
    print(f"[bir] {error_row.example_id} -> error: {error_row.error}")
    _check(
        error_row.example_id == "e6-missing-doc" and error_row.scores == [] and error_row.output is None,
        "C: the error row keeps its place with empty scores and no output",
    )

    # The mean-per-evaluator aggregates: recomputed from the reloaded rows, and
    # exact — only success rows count, and every quality check is structural,
    # so each scores 1.0 in smoke and real mode alike.
    recomputed = _recompute_aggregates(quality_path)
    _check(
        recomputed == result.aggregate_scores == summary.aggregate_scores,
        "C: aggregate_scores recomputed from the reloaded rows match result and summary",
    )
    _check(
        result.aggregate_scores == {name: 1.0 for name in QUALITY_EVAL_NAMES},
        "C: every success row scored exactly 1.0 on all seven structural checks",
    )

    # record_traces=True is the bridge to Lessons 01–07: every example ran in
    # its own trace, in the same trace file every other lesson wrote to.
    traces = {trace.id: trace for trace in load_traces(trace_path)}
    _check(len(traces) == len(dataset), "C: record_traces=True wrote exactly one trace per example")
    _check(
        all(
            row.trace_id in traces
            and traces[row.trace_id].name == f"experiment.qa-quality.{row.example_id}"
            for row in result.results
        ),
        "C: every row's trace_id resolves to a trace named experiment.qa-quality.<example_id>",
    )
    _check(
        all(
            traces[row.trace_id].root.metadata.get("kind") == "experiment"
            and traces[row.trace_id].root.metadata.get("experiment_id") == result.id
            and traces[row.trace_id].root.metadata.get("example_id") == row.example_id
            for row in result.results
        ),
        "C: trace roots carry {kind: experiment, experiment_id, example_id} metadata",
    )
    _check(
        all(
            {event.name for event in traces[row.trace_id].events if event.type == "score"}
            == {score.name for score in row.scores}
            and sum(1 for event in traces[row.trace_id].events if event.type == "generation") == 1
            for row in result.results
            if row.status == "success"
        ),
        "C: each success trace holds the task's generation plus one score event per evaluator",
    )
    error_trace = traces[error_row.trace_id]
    _check(
        error_trace.status == "error" and not any(event.type == "score" for event in error_trace.events),
        "C: the failed example's trace closed with error status and no score events",
    )

    # The default raise_on_error=True is fail-fast, not fail-silent: rows and
    # the summary are persisted through the failing example, THEN it re-raises.
    failfast_path = experiments_dir / "qa-failfast.jsonl"
    poisoned = [example for example in dataset if example.id == "e6-missing-doc"]
    try:
        run_experiment(
            "qa-failfast",
            dataset=poisoned,  # any iterable of examples works, not just a Dataset
            task=answer_with_sources,
            evaluators=_quality_evaluators(),
            path=failfast_path,
        )
        reraised = None
    except LookupError as exc:
        reraised = exc
    failfast_summary_path = failfast_path.with_suffix(".summary.json")
    _check(
        reraised is not None and failfast_path.exists() and failfast_summary_path.exists(),
        "C: raise_on_error=True persisted the rows and summary before re-raising the task's exception",
    )
    _check(
        load_experiment_summary(failfast_summary_path).status == "error",
        "C: the fail-fast run's summary still records what happened",
    )
    return result


def part_d_reports(result, experiments_dir: Path) -> None:
    print("\n== D · reports: one self-contained string per format ==")
    html = render_experiment_report(result)  # format="html" is the default
    markdown = render_experiment_report(result, format="markdown")
    html_path = experiments_dir / "qa-quality.report.html"
    markdown_path = experiments_dir / "qa-quality.report.md"
    html_path.write_text(html, encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    print(f"[bir] wrote {html_path.name} ({len(html)} chars) and {markdown_path.name} ({len(markdown)} chars)")

    _check(
        render_experiment_report(result) == html
        and render_experiment_report(result, format="markdown") == markdown,
        "D: re-rendering the same result is byte-identical — reports are deterministic",
    )
    evaluator_names = result.aggregate_scores.keys()
    _check(
        all(name in html and name in markdown for name in evaluator_names),
        "D: every evaluator appears in the aggregates of both formats",
    )
    _check(
        all(row.example_id in html and row.example_id in markdown for row in result.results),
        "D: every example (the errored one included) appears in both per-example tables",
    )


def part_e_regression_gate(gate_dataset: Dataset, experiments_dir: Path) -> None:
    print("\n== E · the regression gate: baseline vs degraded candidate ==")
    baseline_path = experiments_dir / "qa-baseline.jsonl"
    candidate_path = experiments_dir / "qa-candidate.jsonl"
    baseline = run_experiment(
        "qa-baseline",
        dataset=gate_dataset,
        task=answer_with_sources,
        evaluators=_quality_evaluators() + [_plain_string_evaluator()],
        path=baseline_path,
        record_traces=True,
    )
    # The candidate drops cites_right_doc from its evaluator list (baseline-only
    # coverage loss) and adds two candidate-only checks of its own — contains,
    # and a regex_match whose $ anchor a plain substring check can't express.
    # The two field evaluators stay: on a plain string there is no field to
    # resolve, so each scores 0.0 with reason="non_object" as evidence.
    candidate = run_experiment(
        "qa-candidate",
        dataset=gate_dataset,
        task=casual_answer,
        evaluators=[
            answer_contains_citation(),
            json_valid(),
            custom_evaluator("has_rag_shape", _has_rag_shape),
            field_contains("answer", name="answer_names_doc"),
            numeric_between(1, 3, field="source_count", name="source_count_ok"),
            latency_under(LATENCY_BUDGET_MS),
            _plain_string_evaluator(),
            contains("answer: ", name="has_answer_prefix"),
            regex_match(r"\[docs-[a-z]+\]$", name="ends_with_citation"),
        ],
        path=candidate_path,
        record_traces=True,
    )
    _check(
        candidate.aggregate_scores["answer_contains_citation"] == 4 / 5
        and candidate.aggregate_scores["ends_with_citation"] == 4 / 5,
        "E: exactly one candidate example lost its citation — the bracket check and the anchored regex agree on the 0.2 drop",
    )

    diff = compare_experiments(baseline, candidate, per_example=True)
    for name, delta in diff.deltas.items():
        print(f"[bir]   Δ {name:<26} {delta:+.2f}")
    _check(
        diff.regressed
        == {"answer_contains_citation", "answer_names_doc", "has_rag_shape", "json_valid", "source_count_ok"},
        "E: regressed == exactly the five checks the degraded task breaks",
    )
    _check(
        diff.improved == {"is_plain_string"} and diff.unchanged == {"latency_under"},
        "E: improved and unchanged are exact too — no flakiness in the diff",
    )
    _check(
        diff.baseline_only == {"cites_right_doc"}
        and diff.candidate_only == {"ends_with_citation", "has_answer_prefix"},
        "E: dropped and added evaluators are reported, and candidate_only never regresses",
    )
    _check(diff.has_regressions, "E: has_regressions — the boolean a CI gate keys on — is True")

    citation_deltas = diff.example_deltas["answer_contains_citation"]
    for example_id, delta in citation_deltas.items():
        print(f"[bir]   {example_id:<12} answer_contains_citation Δ {delta:+.1f}")
    _check(
        citation_deltas[_CITATION_DROPPED_ID] == -1.0
        and all(delta == 0.0 for example_id, delta in citation_deltas.items() if example_id != _CITATION_DROPPED_ID),
        "E: per_example=True pins the small drop to the one example that lost its citation",
    )

    # Tolerances. compare_experiments also accepts paths, so a CI job can diff
    # two persisted runs without re-running anything.
    absorbed = compare_experiments(baseline_path, candidate_path, score_tolerances={"answer_contains_citation": 0.2})
    _check(
        absorbed.regressed == {"answer_names_doc", "has_rag_shape", "json_valid", "source_count_ok"},
        "E: score_tolerances absorbs the 0.2 citation drop (boundary inclusive) without loosening the rest",
    )
    try:
        compare_experiments(baseline, candidate, score_tolerances={"citation_typo": 0.2})
        typo_error = None
    except ValueError as exc:
        typo_error = str(exc)
    _check(
        typo_error is not None and "citation_typo" in typo_error,
        "E: a tolerance for a non-shared evaluator raises instead of being silently ignored",
    )
    lenient = compare_experiments(baseline, candidate, tolerance=1.0)
    _check(
        lenient.regressed == frozenset() and not lenient.has_regressions,
        "E: tolerance=1.0 absorbs every drop — the gate opens",
    )
    strict_missing = compare_experiments(baseline, candidate, tolerance=1.0, missing_score="regress")
    _check(
        strict_missing.has_regressions
        and strict_missing.regression_reasons["cites_right_doc"] == "baseline_only",
        "E: missing_score='regress' closes it again — dropping an evaluator is a coverage regression",
    )

    # Every run left a .summary.json sidecar; list_experiments reads them all.
    summaries = list_experiments(experiments_dir)
    print(f"[bir] list_experiments -> {[summary.name for summary in summaries]}")
    _check(
        [summary.name for summary in summaries] == ["qa-candidate", "qa-baseline", "qa-failfast", "qa-quality"],
        "E: list_experiments returns every run's summary, newest first",
    )


def part_f_run_experiment_async(dataset: Dataset, experiments_dir: Path, trace_path: Path) -> None:
    print("\n== F · run_experiment_async: the same eval loop, awaited ==")
    async_dataset = Dataset([example for example in dataset if example.id in _ASYNC_EXAMPLE_IDS])
    async_path = experiments_dir / "qa-async.jsonl"
    result = asyncio.run(
        run_experiment_async(
            "qa-async",
            dataset=async_dataset,
            task=answer_with_sources_async,  # a coroutine function, awaited per example
            evaluators=_quality_evaluators()
            + [retrieved_context_contains("trace", name="context_mentions_trace")],
            path=async_path,
            record_traces=True,
            max_concurrency=2,  # both examples in flight at once
        )
    )

    # In smoke mode the fake async client stalls the FIRST call, so the first
    # example finishes last — yet results, rows, and aggregates stay in
    # dataset order (an SDK guarantee, asserted in both modes).
    completion_order = [row.example_id for row in sorted(result.results, key=lambda row: row.end_time)]
    persisted_order = [row.example_id for row in result.results]
    print(f"[bir] completion order this run: {completion_order}; persisted order: {persisted_order}")
    _check(
        persisted_order == list(_ASYNC_EXAMPLE_IDS),
        "F: results, rows, and aggregates keep dataset order regardless of completion order",
    )
    first, second = result.results
    _check(
        datetime.fromisoformat(second.start_time) < datetime.fromisoformat(first.end_time),
        "F: max_concurrency=2 really overlapped the examples — the second started before the first finished",
    )

    _check(
        result.aggregate_scores == {name: 1.0 for name in QUALITY_EVAL_NAMES + ("context_mentions_trace",)},
        "F: the async run scores 1.0 on all seven quality checks plus retrieved_context_contains",
    )
    _check(
        all(
            next(score for score in row.scores if score.name == "context_mentions_trace").metadata["matched_index"]
            == 0
            for row in result.results
        ),
        "F: retrieved_context_contains records WHICH context matched as evidence",
    )

    reloaded = load_experiment(async_path)
    _check(
        reloaded.aggregate_scores == result.aggregate_scores and async_path.with_suffix(".summary.json").exists(),
        "F: the async run persists the same JSONL rows + .summary.json schema as the sync path",
    )
    traces = {trace.id: trace for trace in load_traces(trace_path)}
    _check(
        all(
            row.trace_id in traces
            and traces[row.trace_id].name == f"experiment.qa-async.{row.example_id}"
            and sum(1 for event in traces[row.trace_id].events if event.type == "generation") == 1
            for row in result.results
        ),
        "F: even run concurrently, each example got its own isolated trace holding its own generation",
    )
    summaries = list_experiments(experiments_dir)
    _check(
        len(summaries) == 5 and summaries[0].name == "qa-async",
        "F: list_experiments now leads with the async run — sync and async persist alike, newest first",
    )


def part_g_send_experiment(result, experiments_dir: Path) -> None:
    print("\n== G · send_experiment: ship a persisted run to a Bir server ==")
    quality_path = experiments_dir / "qa-quality.jsonl"
    server = _FakeBirExperimentsServer()
    server.start()
    print(f"[bir] fake Bir experiments server listening on {server.url} (in-process, loopback only)")
    try:
        sent = send_experiment(quality_path, server.url)
        print(
            f"[bir] send_experiment('{quality_path.name}') -> "
            f"accepted={sent.accepted} experiment_id={sent.experiment_id}"
        )
        _check(
            sent.experiment_id == result.id and sent.accepted == len(result.results),
            "G: SendExperimentResult echoes the run's id; accepted counts every persisted row, the error row included",
        )

        stored = server.experiments[result.id]
        local_summary = json.loads(quality_path.with_suffix(".summary.json").read_text(encoding="utf-8"))
        _check(
            stored["summary"] == local_summary,
            "G: the POSTed summary equals the local .summary.json sidecar exactly",
        )
        _check(
            [row["example_id"] for row in stored["results"]] == [row.example_id for row in result.results],
            "G: the payload carries one result row per example, in dataset order",
        )

        # Transient failures ride the same retry loop as send_events: 5xx /
        # timeouts / connection errors retry up to retries=2 times, sleeping
        # backoff * 2**attempt between tries — shortened only to keep this
        # snappy. A missing file or a 4xx raises immediately instead.
        server.fail_next_status = 503
        posts_before = server.post_attempts
        resent = send_experiment(quality_path, server.url, backoff=0.1)
        _check(
            server.post_attempts - posts_before == 2 and resent.experiment_id == result.id,
            "G: the scripted 503 cost one attempt, the retry succeeded — exactly two POSTs on the wire",
        )
    finally:
        server.stop()


def part_h_capture_traces(trace_path: Path) -> None:
    print("\n== H · capture_traces: unit-test your instrumentation ==")
    # Parts A–G scored the task's OUTPUTS; this scores its TRACES. In a real
    # test suite the with-block below is the body of a pytest function:
    # capture_traces() swaps the active trace_path to a private temp file for
    # the block and restores the previous configure(...) in full on exit. Only
    # WHERE events are written changes — capture flags, sampling, and redaction
    # stay as configured — so a captured event is exactly what a real
    # traces.jsonl write would contain.
    traces_before = len(load_traces(trace_path))

    with capture_traces() as captured:
        with trace("test.answer_with_sources"):
            answer_with_sources(question="What does a trace record?", doc_id="docs-tracing")
        live_types = [event.type for event in captured.events()]  # live mid-block read

    events = captured.events()  # after the block: the exit snapshot
    _check(
        live_types == [event.type for event in events] == ["generation", "trace"],
        "H: events() reads live inside the block and the snapshot after — in write order, children close first",
    )
    gen = next(event for event in events if event.type == "generation")
    print(f"[bir] captured generation: name={gen.name} model={gen.model} usage={gen.usage}")
    _check(
        gen.name == "chat.grounded_answer" and gen.model == _MODEL,
        "H: the task recorded one generation under its bir_name, with the run's model",
    )
    usage = gen.usage or {}
    _check(
        usage.get("input_tokens", 0) > 0
        and usage.get("output_tokens", 0) > 0
        and usage.get("total_tokens") == usage["input_tokens"] + usage["output_tokens"],
        "H: token usage was recorded, with the total derived from its halves",
    )
    (captured_trace,) = captured.traces()
    _check(
        captured_trace.name == "test.answer_with_sources"
        and captured_trace.status == "success"
        and [event.type for event in captured_trace.events] == ["trace", "generation"],
        "H: traces() groups the same events into one successful trace, root first",
    )
    _check(
        captured.trace_path != trace_path
        and not captured.trace_path.exists()
        and len(load_traces(trace_path)) == traces_before,
        "H: isolation held — the temp file is gone and the recipe's traces.jsonl gained nothing",
    )
    print(
        f"[bir] capture_traces: {len(events)} events asserted in isolation; "
        f"{trace_path.name} still holds {traces_before} experiment traces"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lesson 08 — evals: datasets, evaluators, run_experiment(_async), reports, "
        "compare_experiments, send_experiment, and bir.testing.capture_traces."
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Seeds the question of the first dataset example (e1-tracing); the rest is fixed.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--trace-path",
        default=str(DEFAULT_TRACE_PATH),
        help="Cleared at startup (along with the sibling experiments/ and datasets/ output) — the lesson "
        "asserts exact trace and experiment counts.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Offline mode: in-file fake Ollama client (no server, no network). "
        "Also enabled by BIR_COOKBOOK_SMOKE=1.",
    )
    args = parser.parse_args()

    smoke = args.smoke or os.environ.get("BIR_COOKBOOK_SMOKE") == "1"
    trace_path = Path(args.trace_path)
    experiments_dir = trace_path.parent / "experiments"
    datasets_dir = trace_path.parent / "datasets"
    _fresh_slate(trace_path, experiments_dir, datasets_dir)
    configure(trace_path=trace_path, capture_inputs=True, capture_outputs=True)

    global _CLIENT, _ASYNC_CLIENT, _MODEL
    _CLIENT, _ASYNC_CLIENT = _build_client(smoke)
    _MODEL = args.model

    try:
        dataset = part_a_datasets(args.prompt, datasets_dir)
        part_b_evaluators()
        result = part_c_run_experiment(dataset, experiments_dir, trace_path)
        part_d_reports(result, experiments_dir)
        gate_dataset = Dataset([example for example in dataset if example.id != "e6-missing-doc"])
        part_e_regression_gate(gate_dataset, experiments_dir)
        part_f_run_experiment_async(dataset, experiments_dir, trace_path)
        part_g_send_experiment(result, experiments_dir)
        part_h_capture_traces(trace_path)
    except Exception as exc:  # pragma: no cover - real-path only
        if smoke:
            raise
        raise SystemExit(
            f"Lesson failed mid-run: {exc}\n"
            f"If this is an Ollama connection error, ensure Ollama is running and the model is pulled "
            f"(`ollama pull {args.model}`), or run with --smoke."
        ) from exc

    # The standard self-verifying summary line, from the last experiment trace.
    latest = load_traces(trace_path)[-1]
    gen = next((event for event in latest.events if event.type == "generation"), None)
    model_name = gen.model if gen is not None else args.model
    usage = (gen.usage if gen is not None else None) or {}
    print(
        f"\n[bir] all eval checks passed — {len(list_experiments(experiments_dir))} experiments, "
        f"{len(load_traces(trace_path))} experiment traces"
    )
    print(f"[bir] trace_id={latest.id}  events={len(latest.events)}  model={model_name}  usage={usage}")
    print(f"[bir] wrote {trace_path} and {experiments_dir}/ (results, summaries, reports)")


if __name__ == "__main__":
    main()
