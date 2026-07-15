from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

from rag_kit.config import Settings, get_settings
from rag_kit.elastic_store import ElasticsearchChunkStore
from rag_kit.embeddings import create_embedding_model
from rag_kit.llm import ChatClient
from rag_kit.loaders import load_path
from rag_kit.metadata_store import MetadataStore
from rag_kit.models import IngestResponse, QueryResponse, SearchHit, Source
from rag_kit.object_store import ObjectStore
from rag_kit.retriever import HybridRetriever
from rag_kit.sqlite_store import SQLiteKnowledgeStore
from rag_kit.splitter import RecursiveTextSplitter
from rag_kit.vector_store import LocalVectorStore


class RAGService:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.embedding_model = create_embedding_model(
            self.settings.embedding_backend,
            self.settings.sentence_transformer_model,
        )
        self.storage_backend = self.settings.storage_backend.lower()
        self.vector_store = LocalVectorStore(self.settings.index_dir, self.embedding_model)
        self.vector_store.load()
        self.retriever = HybridRetriever(
            self.vector_store,
            vector_weight=self.settings.vector_weight,
            bm25_weight=self.settings.bm25_weight,
        )
        self.metadata_store: Optional[MetadataStore] = None
        self.object_store: Optional[ObjectStore] = None
        self.elasticsearch_store: Optional[ElasticsearchChunkStore] = None
        self.sqlite_store: Optional[SQLiteKnowledgeStore] = None
        if self.storage_backend in {"external", "sqlite"}:
            self.object_store = ObjectStore(
                endpoint=self.settings.minio_endpoint,
                access_key=self.settings.minio_access_key,
                secret_key=self.settings.minio_secret_key,
                bucket=self.settings.minio_bucket,
                secure=self.settings.minio_secure,
            )
        if self.storage_backend == "sqlite":
            self.sqlite_store = SQLiteKnowledgeStore(self.settings.database_url, self.embedding_model)
        if self.storage_backend == "external":
            self.metadata_store = MetadataStore(self.settings.database_url)
            self.elasticsearch_store = ElasticsearchChunkStore(
                url=self.settings.elasticsearch_url,
                index_name=self.settings.elasticsearch_index,
                embedding_model=self.embedding_model,
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

        if self.storage_backend in {"external", "sqlite"}:
            object_keys: Dict[str, str] = {}
            if self.object_store:
                object_keys = self.object_store.upload_sources(
                    document.metadata.get("source", "") for document in documents
                )
            if self.storage_backend == "sqlite" and self.sqlite_store:
                self.sqlite_store.build(documents, chunks, object_keys)
                return IngestResponse(
                    documents=len(documents),
                    chunks=len(chunks),
                    index_dir=f"sqlite:{self.settings.database_url} + minio:{self.settings.minio_bucket}",
                )
            if self.metadata_store:
                self.metadata_store.replace_all(documents, chunks, object_keys)
            if self.elasticsearch_store:
                self.elasticsearch_store.build(chunks)
            return IngestResponse(
                documents=len(documents),
                chunks=len(chunks),
                index_dir=f"mysql + elasticsearch:{self.settings.elasticsearch_index} + minio:{self.settings.minio_bucket}",
            )

        self.vector_store.build(chunks)
        self.retriever.refresh()
        return IngestResponse(
            documents=len(documents),
            chunks=len(chunks),
            index_dir=str(self.settings.index_dir),
        )

    def search(self, question: str, top_k: Optional[int] = None) -> List[SearchHit]:
        if self.storage_backend == "sqlite" and self.sqlite_store:
            return self.sqlite_store.search(
                question,
                top_k=top_k or self.settings.top_k,
                vector_weight=self.settings.vector_weight,
                bm25_weight=self.settings.bm25_weight,
            )
        if self.storage_backend == "external" and self.elasticsearch_store:
            return self.elasticsearch_store.search(
                question,
                top_k=top_k or self.settings.top_k,
                vector_weight=self.settings.vector_weight,
                bm25_weight=self.settings.bm25_weight,
            )
        return self.retriever.search(question, top_k=top_k or self.settings.top_k)

    async def ask(self, question: str, top_k: Optional[int] = None) -> QueryResponse:
        hits = self.search(question, top_k)
        answer = await self.chat_client.complete(question, hits)
        return QueryResponse(answer=answer, sources=_to_sources(hits))

    async def ask_stream(self, question: str, top_k: Optional[int] = None) -> AsyncIterator[str]:
        hits = self.search(question, top_k)
        async for token in self.chat_client.stream(question, hits):
            yield token

    def stats(self) -> Dict:
        if self.storage_backend == "sqlite":
            return {
                "storage_backend": "sqlite",
                "sqlite_database": self.settings.database_url,
                "sqlite_chunks": self.sqlite_store.count() if self.sqlite_store else 0,
                "minio_bucket": self.settings.minio_bucket,
                "llm_enabled": self.chat_client.enabled,
            }
        if self.storage_backend == "external":
            mysql_chunks = self.metadata_store.count_chunks() if self.metadata_store else 0
            es_chunks = self.elasticsearch_store.count() if self.elasticsearch_store else 0
            return {
                "storage_backend": "external",
                "mysql_chunks": mysql_chunks,
                "elasticsearch_chunks": es_chunks,
                "minio_bucket": self.settings.minio_bucket,
                "llm_enabled": self.chat_client.enabled,
            }
        return {
            "storage_backend": "local",
            "chunks": len(self.vector_store.chunks),
            "index_dir": str(self.settings.index_dir),
            "llm_enabled": self.chat_client.enabled,
        }


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
