import json
import sqlite3
from pathlib import Path
from typing import Dict, List

import numpy as np

from rag_kit.bm25 import BM25Index, normalize
from rag_kit.embeddings import EmbeddingModel, cosine_scores
from rag_kit.models import Chunk, Document, SearchHit


class SQLiteKnowledgeStore:
    def __init__(self, database_url: str, embedding_model: EmbeddingModel) -> None:
        self.database_path = _sqlite_path(database_url)
        self.embedding_model = embedding_model

    def init_schema(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.database_path)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    object_key TEXT,
                    meta_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    meta_json TEXT NOT NULL,
                    vector_json TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")

    def build(self, documents: List[Document], chunks: List[Chunk], object_keys: Dict[str, str]) -> None:
        self.init_schema()
        vectors = self.embedding_model.embed([chunk.text for chunk in chunks]) if chunks else np.empty((0, 0))
        with sqlite3.connect(str(self.database_path)) as connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")

            seen_document_ids = set()
            for document in documents:
                if document.id in seen_document_ids:
                    continue
                seen_document_ids.add(document.id)
                source = document.metadata.get("source", "")
                connection.execute(
                    "INSERT INTO documents(id, source, object_key, meta_json) VALUES (?, ?, ?, ?)",
                    (
                        document.id,
                        source,
                        object_keys.get(source),
                        json.dumps(document.metadata, ensure_ascii=False),
                    ),
                )

            for chunk, vector in zip(chunks, vectors):
                connection.execute(
                    """
                    INSERT INTO chunks(
                        id, document_id, source, page, chunk_index, text, meta_json, vector_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.id,
                        chunk.metadata.get("document_id", ""),
                        chunk.metadata.get("source", ""),
                        chunk.metadata.get("page"),
                        int(chunk.metadata.get("chunk_index", 0)),
                        chunk.text,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                        json.dumps(vector.tolist()),
                    ),
                )

    def count(self) -> int:
        self.init_schema()
        with sqlite3.connect(str(self.database_path)) as connection:
            row = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0])

    def list_documents(self, limit: int = 50) -> List[dict]:
        self.init_schema()
        with sqlite3.connect(str(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT id, source, object_key, meta_json FROM documents ORDER BY source LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_chunks(self, limit: int = 20) -> List[dict]:
        self.init_schema()
        with sqlite3.connect(str(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, document_id, source, page, chunk_index, text, meta_json
                FROM chunks
                ORDER BY source, chunk_index
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def search(self, query: str, top_k: int, vector_weight: float, bm25_weight: float) -> List[SearchHit]:
        records = self._load_chunks()
        if not records:
            return []

        chunks = [
            Chunk(id=row["id"], text=row["text"], metadata=json.loads(row["meta_json"]))
            for row in records
        ]
        matrix = np.array([json.loads(row["vector_json"]) for row in records], dtype=np.float32)
        query_vector = self.embedding_model.embed([query])[0]
        vector_scores = normalize(cosine_scores(query_vector, matrix))
        bm25_scores = normalize(BM25Index([chunk.text for chunk in chunks]).search(query))
        hybrid_scores = vector_weight * vector_scores + bm25_weight * bm25_scores

        ranked = hybrid_scores.argsort()[::-1][:top_k]
        hits: List[SearchHit] = []
        for index in ranked:
            hits.append(
                SearchHit(
                    chunk=chunks[int(index)],
                    score=float(hybrid_scores[index]),
                    vector_score=float(vector_scores[index]),
                    bm25_score=float(bm25_scores[index]),
                )
            )
        return hits

    def _load_chunks(self) -> List[dict]:
        self.init_schema()
        with sqlite3.connect(str(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT id, text, meta_json, vector_json FROM chunks"
            ).fetchall()
        return [dict(row) for row in rows]


def _sqlite_path(database_url: str) -> Path:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError("SQLiteKnowledgeStore requires a sqlite:/// database URL")
    raw_path = database_url[len(prefix) :]
    path = Path(raw_path)
    return path if path.is_absolute() else Path.cwd() / path
