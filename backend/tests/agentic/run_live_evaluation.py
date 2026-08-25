from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver


ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
CASES_PATH = Path(__file__).with_name("cases.json")
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "agentic-ai"

sys.path.insert(0, str(BACKEND))
logging.getLogger("dotenv.main").setLevel(logging.ERROR)


class EvaluationCheckpointStore:
    """Keep evaluation conversations isolated from the production database."""

    @asynccontextmanager
    async def create_async_checkpointer(self):
        yield InMemorySaver()


def _load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def _find_book(store, requested_title: str | None) -> dict:
    books = store.list_book_titles()
    if not books:
        raise RuntimeError("The configured Chroma collection contains no books.")
    if not requested_title:
        if len(books) == 1:
            return books[0]
        names = ", ".join(book.get("book_title", "") for book in books)
        raise RuntimeError(f"Multiple books are available; pass --book-title. Available: {names}")

    requested = requested_title.casefold()
    for book in books:
        if requested in book.get("book_title", "").casefold():
            return book
    raise RuntimeError(f"Book not found in Chroma: {requested_title}")


def _connect_book_store(requested_title: str | None, attempts: int = 3):
    from rag.chroma.chroma_store import ChromaStore

    def raise_timeout(_signum, _frame):
        raise TimeoutError("Chroma connection exceeded 45 seconds")

    previous_handler = signal.signal(signal.SIGALRM, raise_timeout)
    last_error = None
    try:
        for attempt in range(1, attempts + 1):
            print(f"Connecting to Chroma Cloud ({attempt}/{attempts})...", flush=True)
            try:
                signal.alarm(45)
                store = ChromaStore()
                book = _find_book(store, requested_title)
                return store, book
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(2)
            finally:
                signal.alarm(0)
    finally:
        signal.signal(signal.SIGALRM, previous_handler)
    raise RuntimeError(f"Unable to load an evaluation book from Chroma after {attempts} attempts: {last_error}")


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _contains_abstention(answer: str) -> bool:
    normalized = answer.casefold()
    markers = [
        "cannot be directly answered",
        "does not mention",
        "does not specifically mention",
        "does not specify",
        "is not mentioned",
        "not mentioned",
        "insufficient evidence",
        "无法直接回答",
        "未提及",
        "没有提及",
        "没有被提及",
        "无法提供相关信息",
        "证据不足",
    ]
    return any(marker in normalized for marker in markers)


def _ordered_subset(expected: list[str], actual: list[str]) -> bool:
    position = 0
    for item in actual:
        if position < len(expected) and item == expected[position]:
            position += 1
    return position == len(expected)


async def _run_router_case(router, case: dict) -> dict:
    started = time.perf_counter()
    try:
        actual = await router.aroute(case["input"], locale=case["locale"])
        expected_route = case["expected"]["route"]
        return {
            "case_id": case["id"],
            "category": case["category"],
            "input": case["input"],
            "expected_route": expected_route,
            "actual_route": actual.get("intent"),
            "route_pass": actual.get("intent") == expected_route,
            "passed": actual.get("intent") == expected_route,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "error": None,
        }
    except Exception as exc:
        return {
            "case_id": case["id"],
            "category": case["category"],
            "input": case["input"],
            "expected_route": case["expected"]["route"],
            "actual_route": None,
            "route_pass": False,
            "passed": False,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


async def _run_with_trace(name: str, tags: list[str], operation) -> dict:
    from langsmith import traceable

    async def execute(run_tree=None):
        result = await operation()
        if run_tree is not None:
            result["trace_id"] = str(run_tree.trace_id)
            try:
                result["trace_url"] = run_tree.get_url()
            except Exception:
                result["trace_url"] = None
        return result

    traced = traceable(name=name, run_type="chain", tags=tags)(execute)
    return await traced()


async def _run_deepread_case(store, llm, case: dict, book: dict, run_number: int = 1) -> dict:
    from agents.critic_agent import CriticAgent
    from agents.deepread_agent import DeepReadAgent, DeepReadConfig
    from agents.followup_agent import FollowupAgent

    started = time.perf_counter()
    agent = DeepReadAgent(
        store=store,
        checkpoint_store=EvaluationCheckpointStore(),
        llm=llm,
        config=DeepReadConfig(enable_graph_retrieval=False),
    )
    tools: list[str] = []
    tool_calls: list[dict] = []
    answer_parts: list[str] = []
    completion: dict = {"citations": [], "docs_count": 0}

    try:
        async for event_type, data in agent.astream_events(
            query=case["input"],
            book_source=book["source"],
            book_id=book["book_id"],
            conversation_id=f"agent-eval-{case['id']}-run-{run_number}",
            locale=case["locale"],
            selected_text=case.get("selected_text"),
            include_trace_details=True,
        ):
            if event_type == "tool":
                tools.append(str(data))
            elif event_type == "tool_trace":
                tool_calls.append(data)
            elif event_type == "token":
                answer_parts.append(str(data))
            elif event_type == "done":
                completion = data

        answer = "".join(answer_parts).strip()
        citations = completion.get("citations", [])
        expected = case["expected"]
        required_tools = expected.get("required_tools", ["search_book_content"])
        forbidden_tools = expected.get("forbidden_tools", [])
        tool_pass = _ordered_subset(required_tools, tools) and not any(
            tool in tools for tool in forbidden_tools
        )
        tool_argument_pass = all(
            call.get("input") not in (None, "", {})
            for call in tool_calls
            if call.get("name") in required_tools
        )
        citation_required = bool(expected.get("citation_required"))
        citation_pass = bool(citations) if citation_required else True
        abstention_pass = _contains_abstention(answer) if expected.get("behavior") == "abstain" else True

        critic_chunks = []
        critic = CriticAgent(llm=llm)
        evidence_text = "\n".join(str(item.get("snippet", "")) for item in citations)
        async for chunk in critic.aevaluate(case["input"], evidence_text, answer, locale=case["locale"]):
            critic_chunks.append(chunk)

        followups = []
        if case.get("reference"):
            followups = await FollowupAgent(llm=llm).agenerate(
                case["input"], answer, locale=case["locale"]
            )

        return {
            "case_id": case["id"],
            "run_number": run_number,
            "category": case["category"],
            "input": case["input"],
            "reference": case.get("reference"),
            "answer": answer,
            "tools": tools,
            "tool_calls": tool_calls,
            "citations": citations,
            "retrieved_contexts": completion.get("retrieved_contexts", []),
            "critic_result": "".join(critic_chunks).strip() or "PASS",
            "followups": followups,
            "tool_pass": tool_pass,
            "tool_argument_pass": tool_argument_pass,
            "citation_pass": citation_pass,
            "abstention_pass": abstention_pass,
            "passed": bool(answer) and tool_pass and tool_argument_pass and citation_pass and abstention_pass,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "error": None,
        }
    except Exception as exc:
        return {
            "case_id": case["id"],
            "run_number": run_number,
            "category": case["category"],
            "input": case["input"],
            "reference": case.get("reference"),
            "answer": "".join(answer_parts).strip(),
            "tools": tools,
            "tool_calls": tool_calls,
            "citations": completion.get("citations", []),
            "retrieved_contexts": [],
            "critic_result": "",
            "followups": [],
            "tool_pass": False,
            "tool_argument_pass": False,
            "citation_pass": False,
            "abstention_pass": False,
            "passed": False,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


async def _run_trajectory_case(router, store, llm, case: dict, book: dict) -> dict:
    from agents import notion_tools

    class SyntheticNoteSearch:
        @staticmethod
        def invoke(query):
            return (
                "Synthetic current-user note for evaluation: the reader associates "
                f"{query} with grief, memory, and the difficulty of moving forward."
            )

    original_search = notion_tools.search_vault_for_concept
    notion_tools.search_vault_for_concept = SyntheticNoteSearch()
    try:
        route = await router.aroute(case["input"], locale=case["locale"])
        result = await _run_deepread_case(store, llm, case, book)
    finally:
        notion_tools.search_vault_for_concept = original_search

    actual_steps = ["RouterAgent", "DeepReadAgent", *result.get("tools", []), "CriticAgent"]
    if result.get("followups"):
        actual_steps.append("FollowupAgent")
    expected_steps = case["expected"].get("required_steps", ["RouterAgent", "DeepReadAgent", "CriticAgent"])
    result.update(
        {
            "actual_route": route.get("intent"),
            "route_pass": route.get("intent") == "book_qa",
            "actual_steps": actual_steps,
            "trajectory_pass": _ordered_subset(expected_steps, actual_steps),
            "note_source": "synthetic_current_user_note",
        }
    )
    result["passed"] = bool(result["passed"] and result["route_pass"] and result["trajectory_pass"])
    return result


def _average_boolean(rows: list[dict], field: str) -> float | None:
    values = [row[field] for row in rows if field in row]
    if not values:
        return None
    return round(sum(bool(value) for value in values) / len(values), 4)


def _rescore_results(rows: list[dict], cases: list[dict]) -> None:
    cases_by_id = {case["id"]: case for case in cases}
    for row in rows:
        case = cases_by_id.get(row.get("case_id"))
        if not case:
            continue
        expected = case["expected"]
        if case["category"] == "routing":
            row["route_pass"] = row.get("actual_route") == expected.get("route")
            row["passed"] = row["route_pass"]
            continue

        tools = row.get("tools", [])
        required_tools = expected.get("required_tools", ["search_book_content"])
        forbidden_tools = expected.get("forbidden_tools", [])
        row["tool_pass"] = _ordered_subset(required_tools, tools) and not any(
            tool in tools for tool in forbidden_tools
        )
        if row.get("tool_calls") is not None:
            row["tool_argument_pass"] = all(
                call.get("input") not in (None, "", {})
                for call in row.get("tool_calls", [])
                if call.get("name") in required_tools
            )
        citation_required = bool(expected.get("citation_required"))
        row["citation_pass"] = bool(row.get("citations")) if citation_required else True
        row["abstention_pass"] = (
            _contains_abstention(row.get("answer", ""))
            if expected.get("behavior") == "abstain"
            else True
        )
        row["passed"] = bool(row.get("answer")) and all(
            [
                row.get("tool_pass", True),
                row.get("tool_argument_pass", True),
                row.get("citation_pass", True),
                row.get("abstention_pass", True),
            ]
        )


def _summary(rows: list[dict]) -> dict:
    latencies = [float(row["latency_seconds"]) for row in rows]
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)
    repeated = [group for group in grouped.values() if len(group) > 1]
    stable_repeated = [
        group
        for group in repeated
        if len({bool(row.get("passed")) for row in group}) == 1
        and len({tuple(row.get("tools", [])) for row in group}) == 1
        and len({bool(row.get("citations")) for row in group}) == 1
        and len({row.get("abstention_pass") for row in group}) == 1
    ]
    return {
        "runs": len(rows),
        "passed": sum(bool(row.get("passed")) for row in rows),
        "failed": sum(not bool(row.get("passed")) for row in rows),
        "routing_accuracy": _average_boolean(rows, "route_pass"),
        "required_tool_compliance": _average_boolean(rows, "tool_pass"),
        "tool_argument_compliance": _average_boolean(rows, "tool_argument_pass"),
        "trajectory_compliance": _average_boolean(rows, "trajectory_pass"),
        "citation_compliance": _average_boolean(rows, "citation_pass"),
        "correct_abstention_rate": _average_boolean(
            [row for row in rows if row.get("case_id") in {"grounding-005", "grounding-006"}],
            "abstention_pass",
        ),
        "repeatability_case_count": len(repeated),
        "behavioral_repeatability": (
            round(len(stable_repeated) / len(repeated), 4) if repeated else None
        ),
        "average_latency_seconds": round(sum(latencies) / len(latencies), 3),
        "p95_latency_seconds": round(_percentile(latencies, 0.95) or 0, 3),
    }


def _write_live_outputs(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "live-agent-results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = payload["results"]
    with (output_dir / "live-agent-results.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "case_id", "run_number", "category", "expected_route", "actual_route",
            "route_pass", "tools", "tool_calls", "tool_pass", "tool_argument_pass",
            "citation_pass", "abstention_pass", "passed", "latency_seconds",
            "trace_id", "trace_url", "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["tools"] = ", ".join(row.get("tools", []))
            csv_row["tool_calls"] = json.dumps(row.get("tool_calls", []), ensure_ascii=False)
            writer.writerow(csv_row)

    summary = payload["summary"]
    lines = [
        "# Live Agentic AI Evaluation",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Model: {payload['model']}",
        f"- Book: {payload['book']['book_title']}",
        f"- LangSmith project: {payload['langsmith_project']}",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Total runs | {summary['runs']} |",
        f"| Passed | {summary['passed']} |",
        f"| Failed | {summary['failed']} |",
        f"| Routing accuracy | {summary['routing_accuracy']} |",
        f"| Required-tool compliance | {summary['required_tool_compliance']} |",
        f"| Tool-argument compliance | {summary['tool_argument_compliance']} |",
        f"| Trajectory compliance | {summary['trajectory_compliance']} |",
        f"| Citation compliance | {summary['citation_compliance']} |",
        f"| Correct abstention rate | {summary['correct_abstention_rate']} |",
        f"| Behavioural repeatability | {summary['behavioral_repeatability']} ({summary['repeatability_case_count']} cases) |",
        f"| Average latency | {summary['average_latency_seconds']} s |",
        f"| P95 latency | {summary['p95_latency_seconds']} s |",
        "",
        "## Failed Cases",
        "",
    ]
    failures = [row for row in rows if not row.get("passed")]
    if failures:
        for row in failures:
            lines.append(f"- `{row['case_id']}` run {row.get('run_number', 1)}: {row.get('error') or 'acceptance criteria not met'}")
    else:
        lines.append("No failed cases in this run.")

    ragas_summary = payload.get("ragas_summary")
    if ragas_summary:
        lines.extend(
            [
                "",
                "## Ragas Scores",
                "",
                "| Metric | Score |",
                "|---|---:|",
            ]
        )
        for name, value in ragas_summary.items():
            display = "Unavailable" if value is None else f"{value:.4f}"
            lines.append(f"| {name} | {display} |")
        lines.extend(
            [
                "",
                "Low context precision indicates that the retriever returned several passages that were not necessary for the reference answer. "
                "This is recorded as a retrieval-quality finding rather than hidden by the overall pass rate.",
            ]
        )

    lines.extend(
        [
            "",
            "## Scope Note",
            "",
            "This is a live-model evaluation using the configured OpenAI-compatible model and the current Chroma book collection. "
            "Routing is evaluated directly. Deep-reading cases record tool events, citations, CriticAgent output, follow-up questions, and latency. "
            "Ragas scores are stored separately.",
            "",
        ]
    )
    (output_dir / "live-agent-summary.md").write_text("\n".join(lines), encoding="utf-8")


def _build_payload(settings, book: dict, results: list[dict]) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.openai_model,
        "book": {
            "book_title": book.get("book_title", ""),
            "author": book.get("author", ""),
            "book_id": book.get("book_id", ""),
        },
        "langsmith_project": settings.langchain_project,
        "summary": _summary(results),
        "results": results,
    }


def _write_langsmith_trace_index(settings, output_dir: Path) -> dict:
    from langsmith import Client

    client = Client(api_url=settings.langchain_endpoint, api_key=settings.langchain_api_key)
    project = client.read_project(project_name=settings.langchain_project, include_stats=True)
    runs = list(client.list_runs(project_name=settings.langchain_project, is_root=True, limit=100))
    records = [
        {
            "run_id": str(run.id),
            "trace_id": str(run.trace_id),
            "name": run.name,
            "status": run.status,
            "start_time": run.start_time.isoformat(),
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "url": run.url,
        }
        for run in runs
    ]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_id": str(project.id),
        "project_name": project.name,
        "root_run_count": len(records),
        "successful_root_runs": sum(run["status"] == "success" for run in records),
        "failed_root_runs": sum(run["status"] == "error" for run in records),
        "runs": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "langsmith-trace-index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _run_ragas(rows: list[dict], output_dir: Path, *, max_workers: int, timeout: int) -> dict:
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    from llm.openai_client import get_embeddings, get_llm

    records = [
        {
            "user_input": row["input"],
            "response": row["answer"],
            "retrieved_contexts": row["retrieved_contexts"],
            "reference": row["reference"],
        }
        for row in rows
        if row.get("run_number", 1) == 1
        and row.get("reference")
        and row.get("answer")
        and row.get("retrieved_contexts")
    ]
    if not records:
        raise RuntimeError("No completed reference cases are available for Ragas evaluation.")

    metrics = [
        Faithfulness(),
        ResponseRelevancy(strictness=1),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        AnswerCorrectness(),
    ]
    result = evaluate(
        Dataset.from_list(records),
        metrics=metrics,
        llm=LangchainLLMWrapper(get_llm(temperature=0.0)),
        embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
        run_config=RunConfig(timeout=timeout, max_retries=2, max_workers=max_workers),
        raise_exceptions=False,
        show_progress=True,
    )
    frame = result.to_pandas()
    frame.to_csv(output_dir / "ragas-live-results.csv", index=False)

    metric_names = [metric.name for metric in metrics]
    summary = {}
    for name in metric_names:
        values = [float(value) for value in frame[name].tolist() if value is not None and not math.isnan(float(value))]
        summary[name] = round(sum(values) / len(values), 4) if values else None

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(records),
        "metrics": metric_names,
        "summary": summary,
        "samples": frame.to_dict(orient="records"),
    }
    (output_dir / "ragas-live-results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run live Agentic AI evaluation for Kant.")
    parser.add_argument("--book-title", default="Norwegian wood")
    parser.add_argument(
        "--categories",
        default="routing,grounding",
        help="Comma-separated case categories to run.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run one routing and one grounded-answer case.")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--ragas-only", action="store_true", help="Score the existing live results without rerunning the agent.")
    parser.add_argument("--refresh-summary", action="store_true", help="Rebuild summary files from existing JSON artifacts.")
    parser.add_argument("--trace-index-only", action="store_true", help="Export metadata for recent LangSmith root traces.")
    parser.add_argument("--ragas-workers", type=int, default=1)
    parser.add_argument("--ragas-timeout", type=int, default=300)
    parser.add_argument("--repeat-count", type=int, default=1, help="Runs for the three repeatability cases.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    from agents.router_agent import RouterAgent
    from config import get_settings
    from llm.openai_client import get_llm

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for live evaluation.")

    if args.trace_index_only:
        if not settings.langchain_api_key:
            raise RuntimeError("LANGCHAIN_API_KEY is required to export the trace index.")
        print(json.dumps(_write_langsmith_trace_index(settings, args.output_dir), ensure_ascii=False, indent=2))
        return

    if args.refresh_summary:
        result_path = args.output_dir / "live-agent-results.json"
        if not result_path.exists():
            raise RuntimeError(f"Live evaluation results not found: {result_path}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        _rescore_results(payload["results"], _load_cases())
        payload["summary"] = _summary(payload["results"])
        ragas_path = args.output_dir / "ragas-live-results.json"
        if ragas_path.exists():
            payload["ragas_summary"] = json.loads(ragas_path.read_text(encoding="utf-8"))["summary"]
        _write_live_outputs(args.output_dir, payload)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        return

    if args.ragas_only:
        result_path = args.output_dir / "live-agent-results.json"
        if not result_path.exists():
            raise RuntimeError(f"Live evaluation results not found: {result_path}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        ragas_payload = _run_ragas(
            payload["results"],
            args.output_dir,
            max_workers=args.ragas_workers,
            timeout=args.ragas_timeout,
        )
        payload["ragas_summary"] = ragas_payload["summary"]
        _write_live_outputs(args.output_dir, payload)
        print(json.dumps({"ragas": payload["ragas_summary"]}, ensure_ascii=False, indent=2))
        return

    cases = _load_cases()
    categories = {item.strip() for item in args.categories.split(",") if item.strip()}
    selected = [case for case in cases if case["category"] in categories]
    if args.smoke:
        selected = [case for case in selected if case["id"] in {"routing-001", "grounding-001"}]

    store, book = _connect_book_store(args.book_title)
    llm = get_llm(temperature=0.0)
    router = RouterAgent(llm=llm)
    results: list[dict] = []

    for case in selected:
        print(f"Running {case['id']}...")
        if case["category"] == "routing":
            result = await _run_with_trace(
                f"Kant Agent Eval {case['id']}",
                ["kant-agentic-eval", case["category"], case["id"]],
                lambda case=case: _run_router_case(router, case),
            )
        elif case["category"] == "trajectory":
            result = await _run_with_trace(
                f"Kant Agent Eval {case['id']}",
                ["kant-agentic-eval", case["category"], case["id"]],
                lambda case=case: _run_trajectory_case(router, store, llm, case, book),
            )
        else:
            result = await _run_with_trace(
                f"Kant Agent Eval {case['id']}",
                ["kant-agentic-eval", case["category"], case["id"]],
                lambda case=case: _run_deepread_case(store, llm, case, book),
            )
        results.append(result)
        _write_live_outputs(args.output_dir, _build_payload(settings, book, results))

    if not args.smoke and args.repeat_count > 1:
        repeated_ids = {"grounding-001", "grounding-005", "grounding-008"}
        repeated_cases = [case for case in selected if case["id"] in repeated_ids]
        for run_number in range(2, args.repeat_count + 1):
            for case in repeated_cases:
                print(f"Repeating {case['id']} run {run_number}...")
                result = await _run_with_trace(
                    f"Kant Agent Eval {case['id']} run {run_number}",
                    ["kant-agentic-eval", "repeatability", case["id"]],
                    lambda case=case, run_number=run_number: _run_deepread_case(
                        store, llm, case, book, run_number
                    ),
                )
                results.append(result)
                _write_live_outputs(args.output_dir, _build_payload(settings, book, results))

    payload = _build_payload(settings, book, results)
    _write_live_outputs(args.output_dir, payload)

    if not args.skip_ragas:
        ragas_payload = _run_ragas(
            results,
            args.output_dir,
            max_workers=args.ragas_workers,
            timeout=args.ragas_timeout,
        )
        payload["ragas_summary"] = ragas_payload["summary"]
        _write_live_outputs(args.output_dir, payload)

    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    if payload.get("ragas_summary"):
        print(json.dumps({"ragas": payload["ragas_summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
