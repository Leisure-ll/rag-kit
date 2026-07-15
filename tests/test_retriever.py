from pathlib import Path

from rag_kit.embeddings import HashingEmbedding
from rag_kit.models import Chunk
from rag_kit.retriever import HybridRetriever
from rag_kit.vector_store import LocalVectorStore


def test_hybrid_retriever_returns_relevant_chunk(tmp_path: Path):
    store = LocalVectorStore(tmp_path, HashingEmbedding(dim=128))
    store.build(
        [
            Chunk(id="1", text="RAG 使用向量检索和 BM25 混合召回。", metadata={"source": "a.md"}),
            Chunk(id="2", text="Redis 常用于缓存热点数据。", metadata={"source": "b.md"}),
        ]
    )
    retriever = HybridRetriever(store)

    hits = retriever.search("BM25 和向量检索", top_k=1)

    assert hits[0].chunk.id == "1"

