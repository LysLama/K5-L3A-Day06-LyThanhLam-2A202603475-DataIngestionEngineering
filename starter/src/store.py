"""Store and search vectors using PostgreSQL + pgvector."""
from __future__ import annotations

import json
import math
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .models import Chunk

_ALLOWED_FILTER_COLUMNS = frozenset({"document_id", "volume", "chapter", "section", "page"})


def _vector_literal(vec: list[float], expected_dim: int) -> str:
    if len(vec) != expected_dim:
        raise ValueError(f"Vector dimension mismatch: expected {expected_dim}, got {len(vec)}")
    for val in vec:
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError("Vector contains non-finite value.")
    return "[" + ",".join(f"{float(v):.8f}" for v in vec) + "]"


class PgVectorStore:
    """Persistent PostgreSQL + pgvector store with idempotent upsert and filtered cosine search."""

    def __init__(self, database_url: str, dimensions: int, table_name: str = "rag_chunks") -> None:
        if not database_url:
            raise ValueError("database_url is required.")
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        if not table_name.replace("_", "").isalnum():
            raise ValueError(f"Invalid table_name: {table_name}")

        self.database_url = database_url
        self.dimensions = dimensions
        self.table_name = table_name

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self.database_url, connect_timeout=10)

    def create_schema(self) -> None:
        """Create table and metadata indexes if missing; verify pgvector extension and dimension."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                if not cur.fetchone():
                    raise RuntimeError("pgvector extension is not enabled in the database.")

                cur.execute(
                    """
                    SELECT format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute a
                    JOIN pg_class c ON a.attrelid = c.oid
                    JOIN pg_namespace n ON c.relnamespace = n.oid
                    WHERE n.nspname = current_schema()
                      AND c.relname = %s
                      AND a.attname = 'embedding'
                      AND NOT a.attisdropped
                    """,
                    (self.table_name,),
                )
                existing_type = cur.fetchone()
                expected_type = f"vector({self.dimensions})"
                if existing_type and existing_type[0] != expected_type:
                    raise RuntimeError(
                        f"Existing table {self.table_name} has embedding type {existing_type[0]}, "
                        f"expected {expected_type}."
                    )

                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        strategy TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        volume TEXT,
                        chapter TEXT,
                        section TEXT,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({self.dimensions}) NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (strategy, chunk_id)
                    )
                    """
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_strat_doc "
                    f"ON {self.table_name} (strategy, document_id)"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_strat_vol_sec "
                    f"ON {self.table_name} (strategy, volume, section)"
                )
            conn.commit()

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Validate lengths/dimensions, upsert stable chunk IDs, and remove obsolete chunks."""
        if len(chunks) != len(vectors):
            raise ValueError(f"Chunk/vector length mismatch: {len(chunks)} chunks vs {len(vectors)} vectors")
        if not chunks:
            return

        doc_chunk_ids: dict[str, list[str]] = {}
        rows: list[tuple[Any, ...]] = []
        for chunk, vec in zip(chunks, vectors):
            vec_lit = _vector_literal(vec, self.dimensions)
            doc_chunk_ids.setdefault(chunk.document_id, []).append(chunk.chunk_id)
            meta = dict(chunk.metadata)
            rows.append(
                (
                    strategy,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    int(chunk.page),
                    meta.get("volume"),
                    meta.get("chapter"),
                    meta.get("section"),
                    chunk.text,
                    Jsonb(meta),
                    vec_lit,
                )
            )

        with self._connect() as conn:
            with conn.cursor() as cur:
                # Remove chunks from earlier runs of the same document that no longer exist
                for doc_id, active_ids in doc_chunk_ids.items():
                    cur.execute(
                        f"""
                        DELETE FROM {self.table_name}
                        WHERE strategy = %s
                          AND document_id = %s
                          AND NOT (chunk_id = ANY(%s))
                        """,
                        (strategy, doc_id, active_ids),
                    )

                cur.executemany(
                    f"""
                    INSERT INTO {self.table_name} (
                        strategy, chunk_id, document_id, source, page,
                        volume, chapter, section, text, metadata, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        source = EXCLUDED.source,
                        page = EXCLUDED.page,
                        volume = EXCLUDED.volume,
                        chapter = EXCLUDED.chapter,
                        section = EXCLUDED.section,
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        updated_at = NOW()
                    """,
                    rows,
                )
            conn.commit()

    def prune_removed_documents(self, strategy: str, active_document_ids: list[str]) -> int:
        """Delete chunks belonging to documents no longer present in the corpus."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {self.table_name}
                    WHERE strategy = %s
                      AND NOT (document_id = ANY(%s))
                    """,
                    (strategy, active_document_ids),
                )
                deleted = cur.rowcount
            conn.commit()
        return deleted

    def count_chunks(self, strategy: str) -> int:
        """Return total stored chunks for a strategy."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name} WHERE strategy = %s", (strategy,))
                row = cur.fetchone()
        return int(row[0]) if row else 0

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter whitelisted metadata in SQL, rank by cosine distance, and return top-k hits."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        qvec_lit = _vector_literal(query_vector, self.dimensions)
        where_clauses = ["strategy = %s"]
        filter_params: list[Any] = [strategy]

        if filters:
            for key, value in filters.items():
                if key not in _ALLOWED_FILTER_COLUMNS:
                    raise ValueError(f"Unsupported metadata filter key: {key}")
                if value is None:
                    continue
                if isinstance(value, list):
                    where_clauses.append(f"{key} = ANY(%s)")
                    filter_params.append(value)
                else:
                    where_clauses.append(f"{key} = %s")
                    filter_params.append(value)

        where_sql = " AND ".join(where_clauses)
        sql = f"""
            SELECT
                chunk_id,
                document_id,
                source,
                page,
                volume,
                chapter,
                section,
                text,
                metadata,
                (embedding <=> %s::vector) AS cosine_distance
            FROM {self.table_name}
            WHERE {where_sql}
            ORDER BY embedding <=> %s::vector ASC, chunk_id ASC
            LIMIT %s
        """
        params = [qvec_lit, *filter_params, qvec_lit, top_k]

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            meta = row[8] if isinstance(row[8], dict) else json.loads(row[8])
            meta.setdefault("document_id", row[1])
            meta.setdefault("source", row[2])
            meta.setdefault("page", row[3])
            if row[4] is not None:
                meta.setdefault("volume", row[4])
            if row[5] is not None:
                meta.setdefault("chapter", row[5])
            if row[6] is not None:
                meta.setdefault("section", row[6])

            cosine_dist = float(row[9])
            results.append(
                {
                    "chunk_id": row[0],
                    "document_id": row[1],
                    "source": row[2],
                    "page": int(row[3]),
                    "text": row[7],
                    "cosine_distance": round(cosine_dist, 6),
                    "score": round(1.0 - cosine_dist, 6),
                    "metadata": meta,
                }
            )
        return results
