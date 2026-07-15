import json
from pathlib import Path
from typing import List

import numpy as np

from rag_kit.embeddings import EmbeddingModel, cosine_scores
from rag_kit.models import Chunk


class LocalVectorStore:
    def __init__(self, index_dir: Path, embedding_model: EmbeddingModel) -> None:
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self.chunks: List[Chunk] = []
        self.vectors = np.empty((0, embedding_model.dim), dtype=np.float32)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        self.vectors = self.embedding_model.embed(texts) if texts else np.empty((0, self.embedding_model.dim), dtype=np.float32)
        self.save()

    def search_scores(self, query: str) -> np.ndarray:
        if self.is_empty:
            return np.array([], dtype=np.float32)
        query_vector = self.embedding_model.embed([query])[0]
        return cosine_scores(query_vector, self.vectors)

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / "vectors.npy", self.vectors)
        records = [
            {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata}
            for chunk in self.chunks
        ]
        (self.index_dir / "chunks.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> bool:
        vectors_path = self.index_dir / "vectors.npy"
        chunks_path = self.index_dir / "chunks.json"
        if not vectors_path.exists() or not chunks_path.exists():
            return False
        self.vectors = np.load(vectors_path)
        records = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk(id=item["id"], text=item["text"], metadata=item.get("metadata", {})) for item in records]
        return True
