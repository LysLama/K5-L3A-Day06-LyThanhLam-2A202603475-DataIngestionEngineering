"""End-to-end Day 06 Data Ingestion Engineering pipeline and benchmark runner."""
from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .models import Chunk, DocumentPage
from .store import PgVectorStore


def _chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    lengths = sorted(len(c.text) for c in chunks)
    if not lengths:
        return {"count": 0, "total_chars": 0}
    p95_idx = min(int(0.95 * (len(lengths) - 1)), len(lengths) - 1)
    return {
        "count": len(lengths),
        "total_chars": sum(lengths),
        "min_chars": lengths[0],
        "mean_chars": round(sum(lengths) / len(lengths), 1),
        "median_chars": round(statistics.median(lengths), 1),
        "p95_chars": lengths[p95_idx],
        "max_chars": lengths[-1],
    }


def _validate_inputs(pages: list[DocumentPage], questions: list[dict[str, Any]]) -> None:
    if len(pages) < 30:
        raise ValueError(f"Corpus must have at least 30 content pages, got {len(pages)}")
    if len(questions) < 20:
        raise ValueError(f"Benchmark must have at least 20 questions, got {len(questions)}")

    dev_qs = [q for q in questions if q["split"] == "dev"]
    test_qs = [q for q in questions if q["split"] == "test"]
    flt_qs = [q for q in questions if q.get("filter")]
    flt_test_qs = [q for q in test_qs if q.get("filter")]

    if len(dev_qs) < 15 or len(test_qs) < 5:
        raise ValueError(f"Expected >=15 dev and >=5 test questions, got {len(dev_qs)}/{len(test_qs)}")
    if len(flt_qs) < 3 or len(flt_test_qs) < 1:
        raise ValueError(f"Expected >=3 filter questions (>=1 in test), got {len(flt_qs)}/{len(flt_test_qs)}")


def _build_manual_audit_template(
    pages: list[DocumentPage],
    chunks_by_strategy: dict[str, list[Chunk]],
    traces_by_strategy: dict[str, list[dict[str, Any]]],
    existing_completed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare deterministic samples for manual Coherence (10 locations), Metadata, and Grounding audits."""
    # Fixed pre-selected 10 source locations: first 5 pages of wh_vol1 and first 5 pages of wh_vol2
    selected_pages = (
        [(p.document_id, p.page, p.metadata.get("section")) for p in pages if p.document_id == "wh_vol1"][:5]
        + [(p.document_id, p.page, p.metadata.get("section")) for p in pages if p.document_id == "wh_vol2"][:5]
    )

    existing_coh: dict[tuple[str, int, str], dict[str, Any]] = {}
    if existing_completed:
        for item in existing_completed.get("coherence_10_locations", []):
            doc_id_item = item.get("document_id")
            page_item = item.get("page")
            for strat_key, strat_val in item.get("strategies", {}).items():
                if doc_id_item and page_item is not None:
                    existing_coh[(doc_id_item, int(page_item), strat_key)] = strat_val

    coherence_samples: list[dict[str, Any]] = []
    for doc_id, page_num, section in selected_pages:
        entry: dict[str, Any] = {
            "location": f"{doc_id} p.{page_num} ({section})",
            "document_id": doc_id,
            "page": page_num,
            "section": section,
            "strategies": {},
        }
        for strat, chunks in chunks_by_strategy.items():
            page_chunks = [c for c in chunks if c.document_id == doc_id and c.page == page_num]
            # Pick the middle body chunk (or first if only 1) representing the page prose at that location
            target = page_chunks[min(1, len(page_chunks) - 1)] if page_chunks else None
            target_id = target.chunk_id if target else None
            prev = existing_coh.get((doc_id, page_num, strat))
            keep_prev = prev is not None and prev.get("chunk_id") == target_id
            prev_score = prev.get("score_0_to_2", prev.get("score")) if keep_prev else None
            entry["strategies"][strat] = {
                "chunk_id": target_id,
                "chars": len(target.text) if target else 0,
                "text": target.text if target else "",
                "score": prev_score,
                "score_0_to_2": prev_score,
                "rationale": prev.get("rationale", "") if keep_prev else "",
            }
        coherence_samples.append(entry)

    existing_test: dict[tuple[str, str], dict[str, Any]] = {}
    if existing_completed:
        for strat_key, items in existing_completed.get("test_set_top3_for_metadata_and_grounding", {}).items():
            for item in items:
                existing_test[(strat_key, item["id"])] = item

    test_audits: dict[str, list[dict[str, Any]]] = {}
    for strat, traces in traces_by_strategy.items():
        test_items = [t for t in traces if t["split"] == "test"]
        test_audits[strat] = []
        for t in test_items:
            prev = existing_test.get((strat, t["id"]), {})
            test_audits[strat].append(
                {
                    "id": t["id"],
                    "question": t["question"],
                    "expected_answer": t["expected_answer"],
                    "gold_source_id": t["gold_source_id"],
                    "gold_page": t["gold_page"],
                    "filter_used": t["filter_used"],
                    "filter_compliant": t["filter_compliant"],
                    "hit_at_1": t["hit_at_1"],
                    "hit_at_3": t["hit_at_3"],
                    "top_3": [
                        {
                            "rank": h["rank"],
                            "chunk_id": h["chunk_id"],
                            "document_id": h["document_id"],
                            "page": h["page"],
                            "score": h["score"],
                            "metadata": h["metadata"],
                            "text_preview": h["text"][:300],
                        }
                        for h in t["top_k_hits"]
                    ],
                    "grounding_score_0_to_2": prev.get("grounding_score_0_to_2"),
                    "metadata_ok_count_out_of_3": prev.get("metadata_ok_count_out_of_3"),
                    "rationale": prev.get("rationale", ""),
                }
            )

    result: dict[str, Any] = {
        "coherence_10_locations": coherence_samples,
        "test_set_top3_for_metadata_and_grounding": test_audits,
    }
    if existing_completed:
        for key in ("annotator", "coherence_summary", "metadata_audit_test_split", "grounding_audit_test_split"):
            if key in existing_completed:
                result[key] = existing_completed[key]
    return result


def main() -> None:
    starter_dir = Path(__file__).resolve().parents[1]
    repo_root = starter_dir.parent
    load_dotenv(starter_dir / ".env")

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL missing in starter/.env")

    model_name = os.getenv("EMBEDDING_MODEL", "").strip() or "openai/text-embedding-3-small"
    manifest = json.loads((repo_root / "data" / "manifest.json").read_text(encoding="utf-8"))
    benchmark = json.loads((repo_root / "benchmark" / "questions.json").read_text(encoding="utf-8"))
    questions: list[dict[str, Any]] = benchmark["questions"]

    # 1. Ingestion & normalization
    t_ingest_start = time.perf_counter()
    error_log: list[dict[str, Any]] = []
    pages: list[DocumentPage] = []
    doc_stats: list[dict[str, Any]] = []

    for doc in manifest["documents"]:
        pdf_path = repo_root / doc["source"]
        s_page, e_page = doc.get("excerpt_pdf_page_range", doc["selected_page_range"])
        page_offset = int(doc.get("original_page_offset", 0))
        sec_ranges = [(int(a), int(b), str(lbl)) for a, b, lbl in doc["section_ranges"]]
        doc_pages = load_pdf(
            path=pdf_path,
            document_id=doc["document_id"],
            source=doc["source"],
            start_page=int(s_page),
            end_page=int(e_page),
            base_metadata=doc["metadata"],
            section_ranges=sec_ranges,
            error_log=error_log,
            page_offset=page_offset,
        )
        pages.extend(doc_pages)
        doc_stats.append(
            {
                "document_id": doc["document_id"],
                "loaded_pages": len(doc_pages),
                "total_normalized_chars": sum(len(p.text) for p in doc_pages),
            }
        )

    ingest_elapsed = round(time.perf_counter() - t_ingest_start, 3)
    _validate_inputs(pages, questions)

    # 2. Chunking across strategies A, B, C
    chunks_a = [c for p in pages for c in fixed_size(p, size=500, overlap=50)]
    chunks_b = [c for p in pages for c in by_paragraph(p, max_size=900)]
    chunks_c = [
        Chunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            source=c.source,
            page=c.page,
            text=c.text,
            metadata=dict(c.metadata),
        )
        for c in chunks_b
    ]
    chunks_by_strategy = {"A": chunks_a, "B": chunks_b, "C": chunks_c}

    # 3. Embed & store in pgvector
    embedder = MyEmbedder(model_name=model_name, dimensions=1536, batch_size=50)
    store = PgVectorStore(
        database_url=database_url,
        dimensions=embedder.dimensions,
        table_name=f"rag_chunks_{embedder.dimensions}",
    )
    store.create_schema()

    strategy_timings: dict[str, dict[str, Any]] = {}

    # Strategy A
    t0 = time.perf_counter()
    vecs_a = embedder.embed_documents([c.text for c in chunks_a])
    t_embed_a = time.perf_counter() - t0
    t1 = time.perf_counter()
    store.upsert("A", chunks_a, vecs_a)
    t_db_a = time.perf_counter() - t1
    strategy_timings["A"] = {
        "embed_seconds": round(t_embed_a, 3),
        "db_upsert_seconds": round(t_db_a, 3),
        "total_index_seconds": round(t_embed_a + t_db_a, 3),
    }

    # Strategy B
    t0 = time.perf_counter()
    vecs_b = embedder.embed_documents([c.text for c in chunks_b])
    t_embed_b = time.perf_counter() - t0
    t1 = time.perf_counter()
    store.upsert("B", chunks_b, vecs_b)
    t_db_b = time.perf_counter() - t1
    strategy_timings["B"] = {
        "embed_seconds": round(t_embed_b, 3),
        "db_upsert_seconds": round(t_db_b, 3),
        "total_index_seconds": round(t_embed_b + t_db_b, 3),
    }

    # Strategy C shares identical chunks & vectors with B (only retrieval filter differs)
    t1 = time.perf_counter()
    store.upsert("C", chunks_c, vecs_b)
    t_db_c = time.perf_counter() - t1
    strategy_timings["C"] = {
        "embed_seconds": round(t_embed_b, 3),
        "db_upsert_seconds": round(t_db_c, 3),
        "total_index_seconds": round(t_embed_b + t_db_c, 3),
        "note": "Reuses Strategy B embeddings (identical chunks); stored under strategy='C' in PostgreSQL.",
    }

    # Re-ingestion idempotency check on Strategy B
    count_before_reingest = store.count_chunks("B")
    store.upsert("B", chunks_b, vecs_b)
    count_after_reingest = store.count_chunks("B")
    if count_before_reingest != count_after_reingest:
        raise RuntimeError(
            f"Idempotency failed: Strategy B count changed {count_before_reingest} -> {count_after_reingest}"
        )

    # 4. Embed queries once (shared across all strategies) and evaluate top-3
    query_vectors = {q["id"]: embedder.embed_query(q["question"]) for q in questions}

    results_dir = starter_dir / "results"
    root_results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    root_results_dir.mkdir(parents=True, exist_ok=True)

    def _write_result_json(filename: str, payload: Any) -> None:
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        (results_dir / filename).write_text(text, encoding="utf-8")
        (root_results_dir / filename).write_text(text, encoding="utf-8")

    all_traces: list[dict[str, Any]] = []
    traces_by_strategy: dict[str, list[dict[str, Any]]] = {}

    for strat in ("A", "B", "C"):
        t_search_start = time.perf_counter()
        strat_traces: list[dict[str, Any]] = []
        for q in questions:
            flt = q.get("filter") if strat == "C" else None
            hits = store.search(strategy=strat, query_vector=query_vectors[q["id"]], top_k=3, filters=flt)
            trace = evaluate_query(q, hits, strategy=strat, filter_used=flt)
            strat_traces.append(trace)
            all_traces.append(trace)

        search_elapsed = round(time.perf_counter() - t_search_start, 3)
        strategy_timings[strat]["search_seconds"] = search_elapsed
        traces_by_strategy[strat] = strat_traces
        _write_result_json(f"traces_{strat}.json", strat_traces)

    summary = summarize(all_traces)
    _write_result_json("benchmark_summary.json", summary)

    ingestion_report = {
        "corpus_pages_loaded": len(pages),
        "ingestion_seconds": ingest_elapsed,
        "documents": doc_stats,
        "error_or_empty_pages": error_log,
        "embedder": {
            "provider": embedder.provider,
            "model_name": embedder.model_name,
            "dimensions": embedder.dimensions,
            "api_requests": embedder.api_requests,
            "embedded_texts": embedder.embedded_texts,
            "embedded_chars": embedder.embedded_chars,
            "total_tokens": embedder.total_tokens,
            "estimated_cost_usd": round(embedder.estimated_cost_usd, 6),
        },
        "chunk_stats": {s: _chunk_stats(cs) for s, cs in chunks_by_strategy.items()},
        "db_counts": {s: store.count_chunks(s) for s in ("A", "B", "C")},
        "idempotency_check_B": {
            "count_before": count_before_reingest,
            "count_after": count_after_reingest,
            "passed": count_before_reingest == count_after_reingest,
        },
        "timings_by_strategy": strategy_timings,
    }
    _write_result_json("ingestion_stats.json", ingestion_report)

    completed_path = results_dir / "manual_audit_completed.json"
    existing_completed = json.loads(completed_path.read_text(encoding="utf-8")) if completed_path.exists() else None
    audit_template = _build_manual_audit_template(pages, chunks_by_strategy, traces_by_strategy, existing_completed)
    _write_result_json("manual_audit_samples.json", audit_template)

    print("Pipeline complete!")
    print(f"Loaded pages: {len(pages)} | Errors/empty: {len(error_log)}")
    for strat in ("A", "B", "C"):
        print(f"Strategy {strat} chunks: {ingestion_report['chunk_stats'][strat]}")
        print(f"Strategy {strat} dev : {summary[strat]['dev']}")
        print(f"Strategy {strat} test: {summary[strat]['test']}")


if __name__ == "__main__":
    main()
