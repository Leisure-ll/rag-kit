from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    vector_score: float
    bm25_score: float


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    index_dir: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    stream: bool = False


class Source(BaseModel):
    id: str
    score: float
    source: Optional[str] = None
    page: Optional[int] = None
    preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
