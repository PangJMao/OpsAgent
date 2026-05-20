from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ops_agent.core.config import settings
from ops_agent.retrieval.embeddings import HashingEmbeddingModel, cosine_similarity
from ops_agent.schemas import Chunk, RetrievalHit


class LocalVectorStore:
    """切换到 pgvector 前使用的 SQLite 本地向量存储方案。"""

    def __init__(
        self,
        index_file: Path = settings.vector_store_path,
        embedding_model: HashingEmbeddingModel | None = None,
    ) -> None:
        self.index_file = index_file
        self.embedding_model = embedding_model or HashingEmbeddingModel()
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self._migrate_legacy_json_if_needed()

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        # 同一文档重新入库时先清理旧 chunk，避免切分策略变更后新旧片段同时被检索到。
        document_ids = sorted({chunk.document_id for chunk in chunks})
        rows = [
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.title,
                chunk.text,
                chunk.start_char,
                chunk.end_char,
                json.dumps(chunk.metadata, ensure_ascii=False),
                json.dumps(self.embedding_model.embed(chunk.text)),
            )
            for chunk in chunks
        ]
        with self._connect() as connection:
            connection.executemany(
                "DELETE FROM chunks WHERE document_id = ?",
                [(document_id,) for document_id in document_ids],
            )
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
                    embedding_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    title = excluded.title,
                    text = excluded.text,
                    start_char = excluded.start_char,
                    end_char = excluded.end_char,
                    metadata_json = excluded.metadata_json,
                    embedding_json = excluded.embedding_json
                """,
                rows,
            )

    def search(self, query: str, top_k: int = settings.top_k) -> list[RetrievalHit]:
        query_embedding = self.embedding_model.embed(query)
        hits: list[RetrievalHit] = []

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    chunk_id,
                    document_id,
                    title,
                    text,
                    start_char,
                    end_char,
                    metadata_json,
                    embedding_json
                FROM chunks
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

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM chunks").fetchone()
        return int(row["total"])

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
                    embedding_json TEXT NOT NULL
                )
                """
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
                    embedding_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    title = excluded.title,
                    text = excluded.text,
                    start_char = excluded.start_char,
                    end_char = excluded.end_char,
                    metadata_json = excluded.metadata_json,
                    embedding_json = excluded.embedding_json
                """,
                rows,
            )
