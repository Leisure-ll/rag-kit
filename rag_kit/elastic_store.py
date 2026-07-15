from typing import List

import numpy as np
from elasticsearch import Elasticsearch, helpers

from rag_kit.bm25 import normalize
from rag_kit.embeddings import EmbeddingModel
from rag_kit.models import Chunk, SearchHit


class ElasticsearchChunkStore:
    def __init__(self, url: str, index_name: str, embedding_model: EmbeddingModel) -> None:
        self.client = Elasticsearch(url)
        self.index_name = index_name
        self.embedding_model = embedding_model

    def ensure_index(self) -> None:
        if self.client.indices.exists(index=self.index_name):
            return
        self.client.indices.create(
            index=self.index_name,
            mappings={
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                    "text": {"type": "text", "analyzer": "standard"},
                    "metadata": {"type": "object", "enabled": True},
                    "vector": {
                        "type": "dense_vector",
                        "dims": self.embedding_model.dim,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            },
        )

    def build(self, chunks: List[Chunk]) -> None:
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
        self.ensure_index()

        vectors = self.embedding_model.embed([chunk.text for chunk in chunks]) if chunks else np.empty((0, 0))
        actions = []
        for chunk, vector in zip(chunks, vectors):
            actions.append(
                {
                    "_index": self.index_name,
                    "_id": chunk.id,
                    "_source": {
                        "chunk_id": chunk.id,
                        "document_id": chunk.metadata.get("document_id"),
                        "source": chunk.metadata.get("source"),
                        "page": chunk.metadata.get("page"),
                        "chunk_index": chunk.metadata.get("chunk_index", 0),
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "vector": vector.tolist(),
                    },
                }
            )
        if actions:
            helpers.bulk(self.client, actions)
            self.client.indices.refresh(index=self.index_name)

    def count(self) -> int:
        if not self.client.indices.exists(index=self.index_name):
            return 0
        return int(self.client.count(index=self.index_name)["count"])

    def search(self, query: str, top_k: int, vector_weight: float, bm25_weight: float) -> List[SearchHit]:
        if not self.client.indices.exists(index=self.index_name):
            return []

        vector_hits = self._vector_search(query, top_k=max(top_k * 4, 20))
        bm25_hits = self._bm25_search(query, top_k=max(top_k * 4, 20))
        merged_ids = list(dict.fromkeys([hit["_id"] for hit in vector_hits] + [hit["_id"] for hit in bm25_hits]))
        if not merged_ids:
            return []

        vector_score_by_id = {hit["_id"]: float(hit["_score"] or 0.0) for hit in vector_hits}
        bm25_score_by_id = {hit["_id"]: float(hit["_score"] or 0.0) for hit in bm25_hits}

        vector_scores = normalize(np.array([vector_score_by_id.get(chunk_id, 0.0) for chunk_id in merged_ids], dtype=np.float32))
        bm25_scores = normalize(np.array([bm25_score_by_id.get(chunk_id, 0.0) for chunk_id in merged_ids], dtype=np.float32))
        hybrid_scores = vector_weight * vector_scores + bm25_weight * bm25_scores

        docs = self.client.mget(index=self.index_name, ids=merged_ids)["docs"]
        source_by_id = {doc["_id"]: doc["_source"] for doc in docs if doc.get("found")}

        ranked = hybrid_scores.argsort()[::-1][:top_k]
        hits: List[SearchHit] = []
        for index in ranked:
            chunk_id = merged_ids[int(index)]
            source = source_by_id.get(chunk_id)
            if not source:
                continue
            chunk = Chunk(
                id=chunk_id,
                text=source["text"],
                metadata=source.get("metadata", {}),
            )
            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=float(hybrid_scores[index]),
                    vector_score=float(vector_scores[index]),
                    bm25_score=float(bm25_scores[index]),
                )
            )
        return hits

    def _vector_search(self, query: str, top_k: int) -> List[dict]:
        query_vector = self.embedding_model.embed([query])[0].tolist()
        response = self.client.search(
            index=self.index_name,
            size=top_k,
            query={
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {"query_vector": query_vector},
                    },
                }
            },
            source=False,
        )
        return response["hits"]["hits"]

    def _bm25_search(self, query: str, top_k: int) -> List[dict]:
        response = self.client.search(
            index=self.index_name,
            size=top_k,
            query={"match": {"text": query}},
            source=False,
        )
        return response["hits"]["hits"]

