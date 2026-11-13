"""Implement fixed-size and structure-aware (paragraph) chunking."""
from __future__ import annotations

import re

from .models import Chunk, DocumentPage

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split page text into fixed-size character windows with overlap while retaining metadata."""
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    if overlap < 0 or overlap >= size:
        raise ValueError("Overlap must satisfy 0 <= overlap < size.")

    text = page.text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    step = size - overlap
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            meta = dict(page.metadata)
            meta["chunk_strategy"] = "fixed_size"
            meta["chunk_index"] = idx
            meta["char_start"] = start
            meta["char_end"] = end
            chunks.append(
                Chunk(
                    chunk_id=f"{page.document_id}_p{page.page:04d}_fixed_{idx:03d}",
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=piece,
                    metadata=meta,
                )
            )
            idx += 1
        if end == len(text):
            break
        start += step

    return chunks


def _split_long_unit(text: str, max_size: int, separator_re: re.Pattern[str] | None) -> list[str]:
    units = separator_re.split(text) if separator_re else text.split(" ")
    pieces: list[str] = []
    current = ""

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > max_size:
            if current:
                pieces.append(current)
                current = ""
            if separator_re is not None:
                pieces.extend(_split_long_unit(unit, max_size, None))
            else:
                for pos in range(0, len(unit), max_size):
                    pieces.append(unit[pos : pos + max_size])
            continue

        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) <= max_size:
            current = candidate
        else:
            pieces.append(current)
            current = unit

    if current:
        pieces.append(current)
    return pieces


def by_paragraph(page: DocumentPage, max_size: int = 900) -> list[Chunk]:
    """Group paragraphs up to max_size, split oversized paragraphs on sentence/word boundaries."""
    if max_size <= 0:
        raise ValueError("max_size must be positive.")

    raw_paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
    if not raw_paragraphs:
        return []

    units: list[str] = []
    for para in raw_paragraphs:
        if len(para) <= max_size:
            units.append(para)
        else:
            units.extend(_split_long_unit(para, max_size, _SENTENCE_SPLIT_RE))

    grouped: list[str] = []
    buffer: list[str] = []
    current_len = 0

    for unit in units:
        extra = len(unit) + (2 if buffer else 0)
        if buffer and current_len + extra > max_size:
            grouped.append("\n\n".join(buffer))
            buffer = [unit]
            current_len = len(unit)
        else:
            buffer.append(unit)
            current_len += extra

    if buffer:
        grouped.append("\n\n".join(buffer))

    chunks: list[Chunk] = []
    for idx, piece in enumerate(grouped):
        meta = dict(page.metadata)
        meta["chunk_strategy"] = "by_paragraph"
        meta["chunk_index"] = idx
        chunks.append(
            Chunk(
                chunk_id=f"{page.document_id}_p{page.page:04d}_para_{idx:03d}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=piece,
                metadata=meta,
            )
        )

    return chunks
