import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from rag_kit.models import SearchHit


class TraceStore:
    def __init__(self, database_url: str) -> None:
        self.database_path = _sqlite_path(database_url)

    def init_schema(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.database_path)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS query_traces (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    storage_backend TEXT NOT NULL,
                    hits_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_query_traces_created_at ON query_traces(created_at)")

    def record(
        self,
        question: str,
        answer: str,
        top_k: int,
        latency_ms: float,
        storage_backend: str,
        hits: List[SearchHit],
    ) -> str:
        self.init_schema()
        trace_id = uuid.uuid4().hex[:16]
        payload = [_hit_to_dict(hit) for hit in hits]
        with sqlite3.connect(str(self.database_path)) as connection:
            connection.execute(
                """
                INSERT INTO query_traces(
                    id, question, answer, top_k, latency_ms, storage_backend, hits_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    question,
                    answer,
                    top_k,
                    latency_ms,
                    storage_backend,
                    json.dumps(payload, ensure_ascii=False),
                    datetime.utcnow().isoformat(timespec="seconds") + "Z",
                ),
            )
        return trace_id

    def list_traces(self, limit: int = 20) -> List[Dict]:
        self.init_schema()
        with sqlite3.connect(str(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, question, top_k, latency_ms, storage_backend, created_at
                FROM query_traces
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_trace(self, trace_id: str) -> Optional[Dict]:
        self.init_schema()
        with sqlite3.connect(str(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute("SELECT * FROM query_traces WHERE id = ?", (trace_id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        item["hits"] = json.loads(item.pop("hits_json"))
        return item


def _hit_to_dict(hit: SearchHit) -> Dict:
    return {
        "id": hit.chunk.id,
        "score": round(hit.score, 4),
        "vector_score": round(hit.vector_score, 4),
        "bm25_score": round(hit.bm25_score, 4),
        "source": hit.chunk.metadata.get("source"),
        "page": hit.chunk.metadata.get("page"),
        "chunk_index": hit.chunk.metadata.get("chunk_index"),
        "preview": hit.chunk.text[:360],
    }


def _sqlite_path(database_url: str) -> Path:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        return Path.cwd() / "data" / "rag_traces.db"
    raw_path = database_url[len(prefix) :]
    path = Path(raw_path)
    return path if path.is_absolute() else Path.cwd() / path

