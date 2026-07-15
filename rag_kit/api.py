import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from rag_kit.models import IngestResponse, QueryRequest, QueryResponse, Source
from rag_kit.service import RAGService


app = FastAPI(title="RAG Kit", version="0.1.0")
_service = RAGService()


def get_service() -> RAGService:
    return _service


@app.get("/health")
def health() -> Dict:
    stats = _service.stats()
    stats["status"] = "ok"
    return stats


@app.get("/stats")
def stats(service: RAGService = Depends(get_service)) -> Dict:
    return service.stats()


@app.post("/ingest", response_model=IngestResponse)
def ingest_path(path: str = Form(...), service: RAGService = Depends(get_service)) -> IngestResponse:
    target = Path(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"path not found: {path}")
    return service.ingest(target)


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...), service: RAGService = Depends(get_service)) -> IngestResponse:
    suffix = Path(file.filename or "upload.txt").suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await file.read())
        temp_path = Path(temp_file.name)
    try:
        return service.ingest(temp_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, service: RAGService = Depends(get_service)) -> QueryResponse:
    return await service.ask(request.question, request.top_k)


@app.post("/query/stream")
async def query_stream(request: QueryRequest, service: RAGService = Depends(get_service)) -> StreamingResponse:
    async def event_stream():
        async for token in service.ask_stream(request.question, request.top_k):
            yield f"data: {token}\n\n"
        sources = [source.model_dump(mode="json") for source in _sources(request, service)]
        yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sources(request: QueryRequest, service: RAGService) -> List[Source]:
    hits = service.search(request.question, request.top_k)
    return [
        Source(
            id=hit.chunk.id,
            score=round(hit.score, 4),
            source=hit.chunk.metadata.get("source"),
            page=hit.chunk.metadata.get("page"),
            preview=hit.chunk.text[:220],
        )
        for hit in hits
    ]
