"""Read and normalize source PDF documents while preserving page traceability."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import pypdf

from .models import DocumentPage

_OPENSTAX_HEADER_RE = re.compile(r"^\d+\s+\d+\s+•\s+.*$")
_OPENSTAX_SECTION_HEADER_RE = re.compile(r"^\d+\.\d+\s+•\s+.*\s+\d+$")


def normalize_text(text: str) -> str:
    """Normalize Unicode (NFC) and whitespace while preserving paragraph boundaries."""
    normalized = unicodedata.normalize("NFC", text)
    cleaned_lines: list[str] = []

    for raw_line in normalized.splitlines():
        stripped = raw_line.strip()
        if _OPENSTAX_HEADER_RE.match(stripped) or _OPENSTAX_SECTION_HEADER_RE.match(stripped):
            continue
        if stripped.lower() == "access for free at openstax.org":
            continue
        cleaned_lines.append(stripped)

    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        joined = buffer[0]
        for part in buffer[1:]:
            if joined.endswith("-") and not joined.endswith(" -"):
                joined += part
            else:
                joined += " " + part
        paragraphs.append(joined)
        buffer.clear()

    for line in cleaned_lines:
        if not line:
            flush_buffer()
            continue
        compact = re.sub(r"[ \t]{2,}", " ", line)
        if compact.startswith("•") and buffer:
            flush_buffer()
        buffer.append(compact)

    flush_buffer()
    return "\n\n".join(paragraphs)


def _resolve_section(page_num: int, section_ranges: list[tuple[int, int, str]] | None) -> str | None:
    if not section_ranges:
        return None
    for start, end, label in section_ranges:
        if start <= page_num <= end:
            return label
    return None


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    start_page: int = 1,
    end_page: int | None = None,
    base_metadata: dict[str, Any] | None = None,
    section_ranges: list[tuple[int, int, str]] | None = None,
    error_log: list[dict[str, Any]] | None = None,
    page_offset: int = 0,
) -> list[DocumentPage]:
    """Extract and normalize physical PDF pages, recording empty or unreadable pages."""
    if not path.exists():
        raise FileNotFoundError(f"Source PDF not found: {path}")

    reader = pypdf.PdfReader(str(path))
    total_pages = len(reader.pages)
    last_page = min(end_page, total_pages) if end_page is not None else total_pages
    if start_page < 1 or start_page > last_page:
        raise ValueError(f"Invalid page range {start_page}..{last_page} for {path.name} ({total_pages} pages)")

    pages: list[DocumentPage] = []
    for page_num in range(start_page, last_page + 1):
        canonical_page = page_num + page_offset
        try:
            raw_text = reader.pages[page_num - 1].extract_text(extraction_mode="layout") or ""
        except Exception as exc:
            if error_log is not None:
                error_log.append(
                    {
                        "document_id": document_id,
                        "source": source,
                        "page": canonical_page,
                        "error": f"extract_failed: {exc}",
                    }
                )
            continue

        clean_text = normalize_text(raw_text)
        if not clean_text.strip():
            if error_log is not None:
                error_log.append(
                    {
                        "document_id": document_id,
                        "source": source,
                        "page": canonical_page,
                        "error": "empty_text_after_normalization",
                    }
                )
            continue

        meta = dict(base_metadata or {})
        section = _resolve_section(canonical_page, section_ranges)
        if section is not None:
            meta["section"] = section

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=canonical_page,
                text=clean_text,
                metadata=meta,
            )
        )

    return pages
