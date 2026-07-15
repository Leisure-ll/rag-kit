from pathlib import Path
from typing import AsyncIterator, List, Optional

from rag_kit.config import Settings, get_settings
from rag_kit.embeddings import create_embedding_model
from rag_kit.llm import ChatClient
from rag_kit.loaders import load_path
from rag_kit.models import IngestResponse, QueryResponse, SearchHit, Source
from rag_kit.retriever import HybridRetriever
from rag_kit.splitter import RecursiveTextSplitter
from rag_kit.vector_store import LocalVectorStore


class RAGService:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.embedding_model = create_embedding_model(
            self.settings.embedding_backend,
            self.settings.sentence_transformer_model,
        )
        self.vector_store = LocalVectorStore(self.settings.index_dir, self.embedding_model)
        self.vector_store.load()
        self.retriever = HybridRetriever(
            self.vector_store,
            vector_weight=self.settings.vector_weight,
            bm25_weight=self.settings.bm25_weight,
        )
        self.chat_client = ChatClient(
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
            model=self.settings.llm_model,
            timeout=self.settings.llm_timeout,
        )

    def ingest(self, path: Path) -> IngestResponse:
        documents = load_path(path)
        splitter = RecursiveTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        self.vector_store.build(chunks)
        self.retriever.refresh()
        return IngestResponse(
            documents=len(documents),
            chunks=len(chunks),
            index_dir=str(self.settings.index_dir),
        )

    def search(self, question: str, top_k: Optional[int] = None) -> List[SearchHit]:
        return self.retriever.search(question, top_k=top_k or self.settings.top_k)

    async def ask(self, question: str, top_k: Optional[int] = None) -> QueryResponse:
        hits = self.search(question, top_k)
        answer = await self.chat_client.complete(question, hits)
        return QueryResponse(answer=answer, sources=_to_sources(hits))

    async def ask_stream(self, question: str, top_k: Optional[int] = None) -> AsyncIterator[str]:
        hits = self.search(question, top_k)
        async for token in self.chat_client.stream(question, hits):
            yield token


def _to_sources(hits: List[SearchHit]) -> List[Source]:
    sources: List[Source] = []
    for hit in hits:
        text = hit.chunk.text.strip().replace("\n", " ")
        preview = text[:220] + ("..." if len(text) > 220 else "")
        sources.append(
            Source(
                id=hit.chunk.id,
                score=round(hit.score, 4),
                source=hit.chunk.metadata.get("source"),
                page=hit.chunk.metadata.get("page"),
                preview=preview,
            )
        )
    return sources
