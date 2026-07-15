from typing import List

from rag_kit.bm25 import BM25Index, normalize
from rag_kit.models import Chunk, SearchHit
from rag_kit.vector_store import LocalVectorStore


class HybridRetriever:
    def __init__(
        self,
        vector_store: LocalVectorStore,
        vector_weight: float = 0.65,
        bm25_weight: float = 0.35,
    ) -> None:
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.bm25 = BM25Index([chunk.text for chunk in vector_store.chunks])

    @property
    def chunks(self) -> List[Chunk]:
        return self.vector_store.chunks

    def refresh(self) -> None:
        self.bm25 = BM25Index([chunk.text for chunk in self.vector_store.chunks])

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        if self.vector_store.is_empty:
            return []

        vector_scores = normalize(self.vector_store.search_scores(query))
        bm25_scores = normalize(self.bm25.search(query))
        hybrid_scores = self.vector_weight * vector_scores + self.bm25_weight * bm25_scores

        ranked = hybrid_scores.argsort()[::-1][:top_k]
        hits: List[SearchHit] = []
        for index in ranked:
            hits.append(
                SearchHit(
                    chunk=self.vector_store.chunks[int(index)],
                    score=float(hybrid_scores[index]),
                    vector_score=float(vector_scores[index]),
                    bm25_score=float(bm25_scores[index]),
                )
            )
        return hits
