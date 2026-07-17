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

Five parts, each self-verified by reloading what was persisted (every asserted
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

Run it:
  * offline (no Ollama, no network, deterministic — what CI runs):
      uv run python main.py --smoke
  * real (needs a local Ollama server + the pulled model):
      ollama pull llama3.2:1b
      uv run python main.py --prompt "In one sentence, what does a trace record?"

Ollama is local and keyless, so there is no API key to set. Not covered here:
``run_experiment_async`` (Lesson 04 taught the async story) and
``send_experiment`` (Lesson 07 taught the server story).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path

from bir import configure, load_traces
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
    field_equals,
    json_valid,
    latency_under,
    list_experiments,
    load_experiment,
    load_experiment_summary,
    render_experiment_report,
    run_experiment,
    similarity_above,
)
from bir.integrations.ollama import trace_chat as trace_ollama_chat

DEFAULT_TRACE_PATH = Path(__file__).resolve().parent / ".bir" / "traces.jsonl"
DEFAULT_MODEL = "llama3.2:1b"
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

# A generous per-example latency budget keeps the context-evaluator demo
# deterministic: even a cold local model answers well within five minutes.
LATENCY_BUDGET_MS = 300_000

# The five evaluators every quality run scores with (part C asserts each
# success row scores exactly 1.0 on all of them — they only judge structure
# the task's code controls, never the model's wording).
QUALITY_EVAL_NAMES = (
    "answer_contains_citation",
    "cites_right_doc",
    "has_rag_shape",
    "json_valid",
    "latency_under",
)

# Keep the client module-level so it is never passed as an @observe/task
# argument and therefore never captured as an input (see CLAUDE.md). The model
# name lives here too: a task's signature is dictated by its examples' input
# keys, so run-wide settings reach it from module scope, not the dataset.
_CLIENT = None
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


def _model_answer(question: str, doc: str) -> str:
    # trace_ollama_chat must run inside an active trace; run_experiment's
    # record_traces=True provides one per example, so the generation lands
    # inside that example's experiment trace.
    response = trace_ollama_chat(
        _CLIENT.chat,
        model=_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Answer in one short sentence, using only this document.\n\n"
                    f"Document: {doc}\n\nQuestion: {question}"
                ),
            }
        ],
        bir_name="chat.grounded_answer",
        bir_metadata={"recipe": "ollama-08-evals"},
    )
    return response.message.content.strip()


def answer_with_sources(question: str, doc_id: str) -> dict:
    """Baseline task: the RAG shape, with the citation appended by CODE.

    The model writes the prose; the code owns everything the evaluators judge —
    the mapping shape, the ``contexts`` list, the ``doc_id`` field, and the
    ``[doc-id]`` citation marker. That is what makes the eval deterministic.
    """

    doc = _lookup_doc(doc_id)
    text = _model_answer(question, doc)
    return {
        "answer": f"{text} [{doc_id}]",
        "contexts": [doc],
        "doc_id": doc_id,
    }


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
    """The five structural checks every quality run scores with.

    ``field_equals("doc_id")`` configures no expected value, so it falls back to
    each example's ``expected`` (the doc the answer must be grounded in) and
    would raise if an example had none. ``custom_evaluator`` returns a bool that
    is coerced to exactly 1.0/0.0.
    """

    return [
        answer_contains_citation(),
        json_valid(),
        custom_evaluator("has_rag_shape", _has_rag_shape),
        field_equals("doc_id", name="cites_right_doc"),
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

    Parts C–E assert exact trace and experiment counts (down to what
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
# The five parts.
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
        "C: every success row scored exactly 1.0 on all five structural checks",
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
    # coverage loss) and adds a candidate-only check of its own.
    candidate = run_experiment(
        "qa-candidate",
        dataset=gate_dataset,
        task=casual_answer,
        evaluators=[
            answer_contains_citation(),
            json_valid(),
            custom_evaluator("has_rag_shape", _has_rag_shape),
            latency_under(LATENCY_BUDGET_MS),
            _plain_string_evaluator(),
            contains("answer: ", name="has_answer_prefix"),
        ],
        path=candidate_path,
        record_traces=True,
    )
    _check(
        candidate.aggregate_scores["answer_contains_citation"] == 4 / 5,
        "E: exactly one candidate example lost its citation — a small, exact 0.2 drop",
    )

    diff = compare_experiments(baseline, candidate, per_example=True)
    for name, delta in diff.deltas.items():
        print(f"[bir]   Δ {name:<26} {delta:+.2f}")
    _check(
        diff.regressed == {"answer_contains_citation", "has_rag_shape", "json_valid"},
        "E: regressed == exactly the three checks the degraded task breaks",
    )
    _check(
        diff.improved == {"is_plain_string"} and diff.unchanged == {"latency_under"},
        "E: improved and unchanged are exact too — no flakiness in the diff",
    )
    _check(
        diff.baseline_only == {"cites_right_doc"} and diff.candidate_only == {"has_answer_prefix"},
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
        absorbed.regressed == {"has_rag_shape", "json_valid"},
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lesson 08 — evals: datasets, evaluators, run_experiment, reports, and compare_experiments."
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

    global _CLIENT, _MODEL
    _CLIENT = _build_client(smoke)
    _MODEL = args.model

    try:
        dataset = part_a_datasets(args.prompt, datasets_dir)
        part_b_evaluators()
        result = part_c_run_experiment(dataset, experiments_dir, trace_path)
        part_d_reports(result, experiments_dir)
        gate_dataset = Dataset([example for example in dataset if example.id != "e6-missing-doc"])
        part_e_regression_gate(gate_dataset, experiments_dir)
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
    print(f"\n[bir] all eval checks passed — 4 experiments, {len(load_traces(trace_path))} experiment traces")
    print(f"[bir] trace_id={latest.id}  events={len(latest.events)}  model={model_name}  usage={usage}")
    print(f"[bir] wrote {trace_path} and {experiments_dir}/ (results, summaries, reports)")


if __name__ == "__main__":
    main()
