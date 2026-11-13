"""Evaluate retrieved evidence and preserve full per-query top-3 traces."""
from __future__ import annotations

import re
from typing import Any


def _canonical(text: str) -> str:
    cleaned = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _matches_gold_location(hit: dict[str, Any], question: dict[str, Any]) -> bool:
    hit_doc = hit.get("document_id") or hit.get("metadata", {}).get("document_id")
    hit_page = hit.get("page") or hit.get("metadata", {}).get("page")

    valid_locations = question.get("valid_locations")
    if valid_locations:
        return any(loc["source_id"] == hit_doc and int(loc["page"]) == int(hit_page) for loc in valid_locations)

    return hit_doc == question["gold_source_id"] and int(hit_page) == int(question["gold_page"])


def _matches_evidence(hit_text: str, question: dict[str, Any]) -> bool:
    norm_hit = _canonical(hit_text)
    quotes = [question["evidence_quote"], *question.get("alt_evidence_quotes", [])]
    return any(_canonical(q) in norm_hit for q in quotes if q)


def _matches_filter(hit: dict[str, Any], filter_used: dict[str, Any] | None) -> bool:
    if not filter_used:
        return True
    meta = hit.get("metadata", {})
    for key, expected in filter_used.items():
        actual = hit.get(key, meta.get(key))
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def evaluate_query(
    question: dict[str, Any],
    hits: list[dict[str, Any]],
    strategy: str = "",
    filter_used: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check source/page and verified gold evidence; keep full hit@1 and hit@3 trace."""
    evaluated_hits: list[dict[str, Any]] = []
    matched_rank: int | None = None
    all_follow_filter = True

    for rank, hit in enumerate(hits, start=1):
        loc_ok = _matches_gold_location(hit, question)
        ev_ok = _matches_evidence(hit.get("text", ""), question)
        flt_ok = _matches_filter(hit, filter_used)
        if not flt_ok:
            all_follow_filter = False
        is_hit = loc_ok and ev_ok
        if is_hit and matched_rank is None:
            matched_rank = rank

        evaluated_hits.append(
            {
                "rank": rank,
                "chunk_id": hit["chunk_id"],
                "document_id": hit["document_id"],
                "source": hit["source"],
                "page": hit["page"],
                "score": hit["score"],
                "cosine_distance": hit.get("cosine_distance"),
                "metadata": hit.get("metadata", {}),
                "text": hit["text"],
                "matches_source_page": loc_ok,
                "matches_evidence": ev_ok,
                "matches_filter": flt_ok,
                "is_hit": is_hit,
            }
        )

    return {
        "strategy": strategy or question.get("strategy", ""),
        "id": question["id"],
        "split": question["split"],
        "type": question.get("type", ""),
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "gold_source_id": question["gold_source_id"],
        "gold_page": question["gold_page"],
        "evidence_quote": question["evidence_quote"],
        "declared_filter": question.get("filter"),
        "filter_used": filter_used,
        "filter_compliant": all_follow_filter if filter_used else None,
        "hit_at_1": matched_rank == 1,
        "hit_at_3": matched_rank is not None and matched_rank <= 3,
        "matched_rank": matched_rank,
        "top_k_hits": evaluated_hits,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for trace in traces:
        key = (trace.get("strategy", "default"), trace["split"])
        grouped.setdefault(key, []).append(trace)

    summary: dict[str, Any] = {}
    for (strategy, split), items in sorted(grouped.items()):
        total = len(items)
        hit1 = sum(1 for item in items if item["hit_at_1"])
        hit3 = sum(1 for item in items if item["hit_at_3"])
        filtered_items = [item for item in items if item.get("filter_used")]
        flt_ok = sum(1 for item in filtered_items if item.get("filter_compliant"))

        strat_bucket = summary.setdefault(strategy, {})
        strat_bucket[split] = {
            "total": total,
            "hit_at_1_count": hit1,
            "hit_at_1_ratio": f"{hit1}/{total}",
            "hit_at_1_rate": round(hit1 / total, 4) if total else 0.0,
            "hit_at_3_count": hit3,
            "hit_at_3_ratio": f"{hit3}/{total}",
            "hit_at_3_rate": round(hit3 / total, 4) if total else 0.0,
            "filter_queries": len(filtered_items),
            "filter_compliant_ratio": f"{flt_ok}/{len(filtered_items)}" if filtered_items else "N/A",
        }
    return summary
