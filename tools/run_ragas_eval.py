from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
OUT_DIR = ROOT / "docs" / "evidence_screenshots"

sys.path.insert(0, str(BACKEND))


EVAL_SET = [
    {
        "user_input": "《万尼亚舅舅》中，万尼亚为什么对教授感到幻灭和痛苦？",
        "reference": (
            "万尼亚痛苦的核心在于他多年牺牲自己的劳动和青春来供养、崇拜教授，"
            "后来却发现教授并没有他想象中的伟大价值。他感到自己的人生被浪费，"
            "再加上对叶列娜无望的感情和教授出售庄园的提议，使他的幻灭和愤怒集中爆发。"
        ),
    },
    {
        "user_input": "《三姊妹》中反复出现的“到莫斯科去”象征什么？",
        "reference": (
            "“到莫斯科去”象征三姊妹对更有意义、更有文化和精神价值生活的向往。"
            "它不只是地理上的返乡愿望，也表现人物对现实停滞、庸常生活和理想不可达的痛苦。"
        ),
    },
    {
        "user_input": "《樱桃园》中樱桃园被卖掉反映了什么社会变化？",
        "reference": (
            "樱桃园被卖掉反映旧贵族阶层的衰落和新兴商业阶层的上升。"
            "柳苞芹娜等旧庄园主人无法适应新的经济现实，而罗巴辛买下庄园则象征社会权力和财富结构的转移。"
        ),
    },
    {
        "user_input": "《万尼亚舅舅》结尾索尼雅是如何安慰万尼亚的？",
        "reference": (
            "索尼雅劝万尼亚继续忍耐和工作，把眼前的痛苦交给时间。"
            "她以宗教式的希望安慰他，相信人在死后会得到安息，看见现在的苦难获得意义。"
        ),
    },
]


def _require_cloud_chroma() -> None:
    from config import get_settings

    settings = get_settings()
    missing = []
    if not settings.chroma_api_key:
        missing.append("CHROMA_API_KEY")
    if settings.chroma_tenant == "default_tenant":
        missing.append("CHROMA_TENANT")
    if settings.chroma_database == "default_database":
        missing.append("CHROMA_DATABASE")
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if missing:
        raise RuntimeError("Missing required cloud evaluation settings: " + ", ".join(missing))


def _docs_to_contexts(docs, limit: int = 4) -> list[str]:
    contexts: list[str] = []
    for doc in docs[:limit]:
        meta = doc.metadata or {}
        location = meta.get("section_title") or meta.get("chapter_title") or ""
        title = meta.get("book_title") or ""
        prefix = " / ".join(x for x in [title, location] if x)
        text = (doc.page_content or "").strip()
        contexts.append(f"{prefix}\n{text}" if prefix else text)
    return contexts


def build_rag_dataset() -> tuple[Dataset, list[dict]]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from llm.openai_client import get_llm
    from rag.chroma.chroma_store import ChromaStore

    store = ChromaStore()
    stats = store.get_stats()
    if stats.get("total_chunks", 0) <= 0:
        raise RuntimeError("Cloud Chroma collection has no chunks to evaluate.")
    llm = get_llm(temperature=0.0)

    records: list[dict] = []
    raw_rows: list[dict] = []
    for item in EVAL_SET:
        question = item["user_input"]
        retrieved_docs = store.similarity_search(question, k=4)
        contexts = _docs_to_contexts(retrieved_docs)
        context_block = "\n\n---\n\n".join(contexts)
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "你是一个基于证据的中文文学阅读助手。只根据给定的检索上下文回答问题；"
                        "如果上下文不足，请明确说明证据不足，不要编造。"
                    )
                ),
                HumanMessage(
                    content=(
                        f"问题：{question}\n\n"
                        f"检索上下文：\n{context_block}\n\n"
                        "请用一段中文回答，并尽量指出答案依据。"
                    )
                ),
            ]
        )
        answer = str(response.content).strip()

        row = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": item["reference"],
        }
        records.append(row)
        raw_rows.append(
            {
                **row,
                "retrieved_context_count": len(contexts),
                "response_preview": answer[:220],
            }
        )

    return Dataset.from_list(records), raw_rows


def _evaluation_to_rows(result) -> list[dict]:
    try:
        df = result.to_pandas()
        return df.to_dict(orient="records")
    except Exception:
        return json.loads(result.to_json())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _require_cloud_chroma()

    from llm.openai_client import get_embeddings, get_llm

    dataset, raw_rows = build_rag_dataset()
    evaluator_llm = LangchainLLMWrapper(get_llm(temperature=0.0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
        show_progress=True,
    )

    rows = _evaluation_to_rows(result)
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    ]
    summary = {}
    for metric in metric_names:
        vals = [r.get(metric) for r in rows if isinstance(r.get(metric), (int, float))]
        summary[metric] = round(sum(vals) / len(vals), 4) if vals else None

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(raw_rows),
        "metrics": metric_names,
        "summary": summary,
        "samples": rows,
        "raw_inputs": raw_rows,
    }

    json_path = OUT_DIR / "ragas_eval_results.json"
    csv_path = OUT_DIR / "ragas_eval_results.csv"
    md_path = OUT_DIR / "ragas_eval_summary.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["user_input", "response", "reference", *metric_names]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    lines = [
        "# RAGAS Evaluation Summary",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Dataset size: {payload['dataset_size']}",
        "- Data source: Cloud Chroma collection via project ChromaStore",
        "",
        "| Metric | Average Score |",
        "|---|---:|",
    ]
    for metric, value in summary.items():
        display = "" if value is None else f"{value:.4f}"
        lines.append(f"| {metric} | {display} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "The evaluation uses a small curated Chinese benchmark for the Chekhov book currently indexed in the cloud vector store. "
            "It measures answer faithfulness, answer relevancy, context precision, context recall, and answer correctness. "
            "Future work should expand the benchmark with more books, adversarial poisoned-content cases, and empty-retrieval cases.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"summary": summary, "json": str(json_path), "csv": str(csv_path), "md": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
