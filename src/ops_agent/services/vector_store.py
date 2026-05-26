from __future__ import annotations

import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ops_agent.config import settings
from ops_agent.models import Chunk, RetrievalHit
from ops_agent.services.database_service import DatabaseService
from ops_agent.services.embedding_service import EmbeddingModel, create_embedding_model

TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


class LocalVectorStore:
    """SQLite 本地向量存储，接口与 pgvector 实现保持一致。"""

    def __init__(
        self,
        index_file: Path = settings.vector_store_path,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.index_file = index_file
        self.embedding_model = embedding_model or create_embedding_model()
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self._migrate_legacy_json_if_needed()

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        # 同一个 source 重新入库时，先查询旧 chunk 并软删除，再插入新 chunk。
        document_ids = sorted({chunk.document_id for chunk in chunks})
        sources = sorted({_chunk_source(chunk) for chunk in chunks if _chunk_source(chunk)})
        ingestion_run_id = _utc_timestamp()
        with self._connect() as connection:
            old_chunk_ids = self._active_chunk_ids_for_sources(connection, document_ids, sources)
            self._mark_chunks_deleted(connection, old_chunk_ids)
            rows = [
                (
                    _fresh_chunk_id(chunk.chunk_id, old_chunk_ids, ingestion_run_id),
                    chunk.document_id,
                    chunk.title,
                    chunk.text,
                    chunk.start_char,
                    chunk.end_char,
                    json.dumps(chunk.metadata, ensure_ascii=False),
                    json.dumps(self.embedding_model.embed(chunk.text)),
                    ingestion_run_id,
                )
                for chunk in chunks
            ]
            connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id,
                    document_id,
                    title,
                    text,
                    start_char,
                    end_char,
                    metadata_json,
                    embedding_json,
                    deleted,
                    deleted_at,
                    ingestion_run_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    title = excluded.title,
                    text = excluded.text,
                    start_char = excluded.start_char,
                    end_char = excluded.end_char,
                    metadata_json = excluded.metadata_json,
                    embedding_json = excluded.embedding_json,
                    deleted = 0,
                    deleted_at = NULL,
                    ingestion_run_id = excluded.ingestion_run_id
                """,
                rows,
            )
            self._record_ingestion_log(
                connection=connection,
                ingestion_run_id=ingestion_run_id,
                source=sources[0] if len(sources) == 1 else ",".join(sources),
                old_chunk_count=len(old_chunk_ids),
                new_chunk_count=len(chunks),
                status="success",
                message="",
            )

    def search(self, query: str, top_k: int = settings.top_k) -> list[RetrievalHit]:
        query_embedding = self.embedding_model.embed(query)
        hits: list[RetrievalHit] = []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, document_id, title, text, start_char, end_char, metadata_json, embedding_json
                FROM chunks
                WHERE deleted = 0
                """
            ).fetchall()

        for row in rows:
            chunk = Chunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                title=row["title"],
                text=row["text"],
                start_char=row["start_char"],
                end_char=row["end_char"],
                metadata=json.loads(row["metadata_json"]),
            )
            score = cosine_similarity(query_embedding, json.loads(row["embedding_json"]))
            hits.append(RetrievalHit(chunk=chunk, score=score))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def keyword_search(self, query: str, top_k: int = settings.top_k) -> list[RetrievalHit]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, document_id, title, text, start_char, end_char, metadata_json
                FROM chunks
                WHERE deleted = 0
                """
            ).fetchall()

        chunks = [
            Chunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                title=row["title"],
                text=row["text"],
                start_char=row["start_char"],
                end_char=row["end_char"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]
        hits = _bm25_rank(query_tokens, chunks)
        return hits[:top_k]

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM chunks WHERE deleted = 0").fetchone()
        return int(row["total"])

    def mark_deleted_by_source(self, source: str) -> int:
        with self._connect() as connection:
            old_chunk_ids = self._active_chunk_ids_for_sources(connection, [], [source])
            self._mark_chunks_deleted(connection, old_chunk_ids)
            return len(old_chunk_ids)

    def clear_all(self) -> int:
        # 清空知识库采用软删除，保留历史记录和入库日志，避免误删后无法审计。
        with self._connect() as connection:
            rows = connection.execute("SELECT chunk_id FROM chunks WHERE deleted = 0").fetchall()
            chunk_ids = [str(row["chunk_id"]) for row in rows]
            self._mark_chunks_deleted(connection, chunk_ids)
            connection.execute(
                """
                INSERT INTO ingestion_logs (
                    ingestion_run_id,
                    source,
                    old_chunk_count,
                    new_chunk_count,
                    status,
                    message,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (_utc_timestamp(), "*", len(chunk_ids), 0, "cleared", "clear all active chunks", _utc_timestamp()),
            )
            return len(chunk_ids)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.index_file)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    deleted_at TEXT,
                    ingestion_run_id TEXT
                )
                """
            )
            self._ensure_column(connection, "chunks", "deleted", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(connection, "chunks", "deleted_at", "TEXT")
            self._ensure_column(connection, "chunks", "ingestion_run_id", "TEXT")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ingestion_run_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    old_chunk_count INTEGER NOT NULL,
                    new_chunk_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_deleted ON chunks(deleted)")

    def _ensure_column(self, connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _active_chunk_ids_for_sources(
        self,
        connection: sqlite3.Connection,
        document_ids: list[str],
        sources: list[str],
    ) -> list[str]:
        rows = connection.execute(
            """
            SELECT chunk_id, document_id, metadata_json
            FROM chunks
            WHERE deleted = 0
            """
        ).fetchall()
        matched: list[str] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            row_source = str(metadata.get("source") or metadata.get("source_path") or "")
            if row["document_id"] in document_ids or (row_source and row_source in sources):
                matched.append(str(row["chunk_id"]))
        return matched

    def _mark_chunks_deleted(self, connection: sqlite3.Connection, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        deleted_at = _utc_timestamp()
        connection.executemany(
            "UPDATE chunks SET deleted = 1, deleted_at = ? WHERE chunk_id = ?",
            [(deleted_at, chunk_id) for chunk_id in chunk_ids],
        )

    def _record_ingestion_log(
        self,
        connection: sqlite3.Connection,
        ingestion_run_id: str,
        source: str,
        old_chunk_count: int,
        new_chunk_count: int,
        status: str,
        message: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO ingestion_logs (
                ingestion_run_id,
                source,
                old_chunk_count,
                new_chunk_count,
                status,
                message,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ingestion_run_id, source, old_chunk_count, new_chunk_count, status, message, _utc_timestamp()),
        )

    def _migrate_legacy_json_if_needed(self) -> None:
        if self.index_file.suffix.lower() != ".db" or self.count() > 0:
            return

        legacy_file = self.index_file.with_suffix(".json")
        if not legacy_file.exists():
            return

        records = json.loads(legacy_file.read_text(encoding="utf-8"))
        rows = [
            (
                record["chunk"]["chunk_id"],
                record["chunk"]["document_id"],
                record["chunk"]["title"],
                record["chunk"]["text"],
                record["chunk"]["start_char"],
                record["chunk"]["end_char"],
                json.dumps(record["chunk"]["metadata"], ensure_ascii=False),
                json.dumps(record["embedding"]),
                _utc_timestamp(),
            )
            for record in records
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id,
                    document_id,
                    title,
                    text,
                    start_char,
                    end_char,
                    metadata_json,
                    embedding_json,
                    ingestion_run_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    title = excluded.title,
                    text = excluded.text,
                    start_char = excluded.start_char,
                    end_char = excluded.end_char,
                    metadata_json = excluded.metadata_json,
                    embedding_json = excluded.embedding_json,
                    deleted = 0,
                    deleted_at = NULL,
                    ingestion_run_id = excluded.ingestion_run_id
                """,
                rows,
            )


class PgVectorStore:
    """PostgreSQL + pgvector implementation for production knowledge retrieval."""

    def __init__(self, embedding_model: EmbeddingModel | None = None) -> None:
        self.embedding_model = embedding_model or create_embedding_model()
        self.database = DatabaseService()
        if settings.database_url:
            self.database.ensure_knowledge_indexes()

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        document_ids = sorted({chunk.document_id for chunk in chunks})
        sources = sorted({_chunk_source(chunk) for chunk in chunks if _chunk_source(chunk)})
        ingestion_run_id = _utc_timestamp()
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                old_chunk_ids = self._active_chunk_ids_for_sources(cursor, document_ids, sources)
                if old_chunk_ids:
                    cursor.execute(
                        "UPDATE knowledge_chunks SET deleted = true, deleted_at = now() WHERE chunk_id = ANY(%s)",
                        (old_chunk_ids,),
                    )
                rows = [
                    (
                        _fresh_chunk_id(chunk.chunk_id, old_chunk_ids, ingestion_run_id),
                        chunk.document_id,
                        chunk.title,
                        chunk.text,
                        chunk.start_char,
                        chunk.end_char,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                        self.embedding_model.embed(chunk.text),
                        ingestion_run_id,
                    )
                    for chunk in chunks
                ]
                cursor.executemany(
                    """
                    INSERT INTO knowledge_chunks (
                        chunk_id,
                        document_id,
                        title,
                        text,
                        start_char,
                        end_char,
                        metadata_json,
                        embedding,
                        deleted,
                        deleted_at,
                        ingestion_run_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector, false, NULL, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        document_id = excluded.document_id,
                        title = excluded.title,
                        text = excluded.text,
                        start_char = excluded.start_char,
                        end_char = excluded.end_char,
                        metadata_json = excluded.metadata_json,
                        embedding = excluded.embedding,
                        deleted = false,
                        deleted_at = NULL,
                        ingestion_run_id = excluded.ingestion_run_id
                    """,
                    rows,
                )
                cursor.execute(
                    """
                    INSERT INTO ingestion_logs (
                        ingestion_run_id,
                        source,
                        old_chunk_count,
                        new_chunk_count,
                        status,
                        message
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        ingestion_run_id,
                        sources[0] if len(sources) == 1 else ",".join(sources),
                        len(old_chunk_ids),
                        len(chunks),
                        "success",
                        "",
                    ),
                )
            connection.commit()

    def search(self, query: str, top_k: int = settings.top_k) -> list[RetrievalHit]:
        embedding = self.embedding_model.embed(query)
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        chunk_id,
                        document_id,
                        title,
                        text,
                        start_char,
                        end_char,
                        metadata_json,
                        1 - (embedding <=> %s::vector) AS score
                    FROM knowledge_chunks
                    WHERE deleted = false
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding, embedding, top_k),
                )
                rows = cursor.fetchall()

        hits: list[RetrievalHit] = []
        for row in rows:
            metadata = row[6] if isinstance(row[6], dict) else json.loads(row[6])
            chunk = Chunk(
                chunk_id=row[0],
                document_id=row[1],
                title=row[2],
                text=row[3],
                start_char=row[4],
                end_char=row[5],
                metadata=metadata,
            )
            hits.append(RetrievalHit(chunk=chunk, score=float(row[7])))
        return hits

    def keyword_search(self, query: str, top_k: int = settings.top_k) -> list[RetrievalHit]:
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        chunk_id,
                        document_id,
                        title,
                        text,
                        start_char,
                        end_char,
                        metadata_json,
                        ts_rank_cd(to_tsvector('simple', text), websearch_to_tsquery('simple', %s)) AS score
                    FROM knowledge_chunks
                    WHERE deleted = false
                        AND to_tsvector('simple', text) @@ websearch_to_tsquery('simple', %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (query, query, top_k),
                )
                rows = cursor.fetchall()

        hits: list[RetrievalHit] = []
        for row in rows:
            metadata = row[6] if isinstance(row[6], dict) else json.loads(row[6])
            chunk = Chunk(
                chunk_id=row[0],
                document_id=row[1],
                title=row[2],
                text=row[3],
                start_char=row[4],
                end_char=row[5],
                metadata=metadata,
            )
            hits.append(RetrievalHit(chunk=chunk, score=float(row[7])))
        return hits

    def count(self) -> int:
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM knowledge_chunks WHERE deleted = false")
                return int(cursor.fetchone()[0])

    def mark_deleted_by_source(self, source: str) -> int:
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                old_chunk_ids = self._active_chunk_ids_for_sources(cursor, [], [source])
                if old_chunk_ids:
                    cursor.execute(
                        "UPDATE knowledge_chunks SET deleted = true, deleted_at = now() WHERE chunk_id = ANY(%s)",
                        (old_chunk_ids,),
                    )
            connection.commit()
        return len(old_chunk_ids)

    def clear_all(self) -> int:
        with self.database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM knowledge_chunks WHERE deleted = false")
                deleted_count = int(cursor.fetchone()[0])
                cursor.execute("UPDATE knowledge_chunks SET deleted = true, deleted_at = now() WHERE deleted = false")
                cursor.execute(
                    """
                    INSERT INTO ingestion_logs (
                        ingestion_run_id,
                        source,
                        old_chunk_count,
                        new_chunk_count,
                        status,
                        message
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (_utc_timestamp(), "*", deleted_count, 0, "cleared", "clear all active chunks"),
                )
            connection.commit()
        return deleted_count

    def _active_chunk_ids_for_sources(self, cursor, document_ids: list[str], sources: list[str]) -> list[str]:
        cursor.execute(
            """
            SELECT chunk_id
            FROM knowledge_chunks
            WHERE deleted = false
                AND (
                    document_id = ANY(%s)
                    OR metadata_json->>'source' = ANY(%s)
                    OR metadata_json->>'source_path' = ANY(%s)
                )
            """,
            (document_ids, sources, sources),
        )
        return [str(row[0]) for row in cursor.fetchall()]


def create_vector_store() -> LocalVectorStore | PgVectorStore:
    if settings.vector_provider == "pgvector":
        return PgVectorStore()
    return LocalVectorStore()


def _tokens(text: str) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    cjk_chars = [char for char in text if "\u4e00" <= char <= "\u9fff"]
    tokens.extend(a + b for a, b in zip(cjk_chars, cjk_chars[1:]))
    return tokens


def _chunk_source(chunk: Chunk) -> str:
    return str(chunk.metadata.get("source") or chunk.metadata.get("source_path") or "")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fresh_chunk_id(chunk_id: str, old_chunk_ids: list[str], ingestion_run_id: str) -> str:
    if chunk_id not in old_chunk_ids:
        return chunk_id
    suffix = re.sub(r"[^0-9A-Za-z]+", "", ingestion_run_id)[-14:]
    return f"{chunk_id}-{suffix}"


def _bm25_rank(query_tokens: list[str], chunks: list[Chunk]) -> list[RetrievalHit]:
    if not chunks:
        return []
    tokenized = [_tokens(chunk.text) for chunk in chunks]
    avg_len = sum(len(tokens) for tokens in tokenized) / max(len(tokenized), 1)
    document_frequency: dict[str, int] = {}
    for tokens in tokenized:
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    hits: list[RetrievalHit] = []
    total = len(chunks)
    k1 = 1.5
    b = 0.75
    for chunk, tokens in zip(chunks, tokenized):
        if not tokens:
            continue
        term_counts: dict[str, int] = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1
        score = 0.0
        for token in query_tokens:
            tf = term_counts.get(token, 0)
            if tf == 0:
                continue
            df = document_frequency.get(token, 0)
            idf = math.log(1 + ((total - df + 0.5) / (df + 0.5)))
            denominator = tf + k1 * (1 - b + b * (len(tokens) / max(avg_len, 1)))
            score += idf * ((tf * (k1 + 1)) / denominator)
        if score > 0:
            hits.append(RetrievalHit(chunk=chunk, score=score))
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits
