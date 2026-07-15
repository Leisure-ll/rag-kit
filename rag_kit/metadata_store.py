from datetime import datetime
from typing import Iterable, List, Optional

from sqlalchemy import JSON, Column, DateTime, Integer, MetaData, String, Table, Text, create_engine, func, select
from sqlalchemy.engine import Engine

from rag_kit.models import Chunk, Document


metadata = MetaData()

documents_table = Table(
    "documents",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("source", String(1024), nullable=False),
    Column("object_key", String(1024), nullable=True),
    Column("meta", JSON, nullable=False),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
)

chunks_table = Table(
    "chunks",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("document_id", String(64), nullable=False, index=True),
    Column("source", String(1024), nullable=False),
    Column("chunk_index", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("meta", JSON, nullable=False),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
)


class MetadataStore:
    def __init__(self, database_url: str) -> None:
        self.engine: Engine = create_engine(database_url, pool_pre_ping=True, future=True)

    def init_schema(self) -> None:
        metadata.create_all(self.engine)

    def replace_all(self, documents: Iterable[Document], chunks: Iterable[Chunk], object_keys: dict) -> None:
        self.init_schema()
        with self.engine.begin() as connection:
            connection.execute(chunks_table.delete())
            connection.execute(documents_table.delete())

            document_rows = []
            seen_document_ids = set()
            for document in documents:
                if document.id in seen_document_ids:
                    continue
                seen_document_ids.add(document.id)
                source = document.metadata.get("source", "")
                document_rows.append(
                    {
                        "id": document.id,
                        "source": source,
                        "object_key": object_keys.get(source),
                        "meta": document.metadata,
                        "created_at": datetime.utcnow(),
                    }
                )
            if document_rows:
                connection.execute(documents_table.insert(), document_rows)

            chunk_rows = [
                {
                    "id": chunk.id,
                    "document_id": chunk.metadata.get("document_id", ""),
                    "source": chunk.metadata.get("source", ""),
                    "chunk_index": int(chunk.metadata.get("chunk_index", 0)),
                    "text": chunk.text,
                    "meta": chunk.metadata,
                    "created_at": datetime.utcnow(),
                }
                for chunk in chunks
            ]
            if chunk_rows:
                connection.execute(chunks_table.insert(), chunk_rows)

    def count_chunks(self) -> int:
        self.init_schema()
        with self.engine.connect() as connection:
            return int(connection.execute(select(func.count()).select_from(chunks_table)).scalar_one())

    def list_chunks(self, limit: int = 20) -> List[dict]:
        self.init_schema()
        with self.engine.connect() as connection:
            rows = connection.execute(select(chunks_table).limit(limit)).mappings().all()
        return [dict(row) for row in rows]


class NullMetadataStore:
    def count_chunks(self) -> int:
        return 0

    def list_chunks(self, limit: int = 20) -> List[dict]:
        return []

