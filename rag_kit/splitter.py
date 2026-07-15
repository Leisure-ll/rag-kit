import hashlib
import re
from typing import List

from rag_kit.models import Chunk, Document


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class RecursiveTextSplitter:
    def __init__(self, chunk_size: int = 650, chunk_overlap: int = 120) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for document in documents:
            for i, text in enumerate(self.split_text(document.text)):
                chunk_id = hashlib.sha1(f"{document.id}:{i}:{text}".encode("utf-8")).hexdigest()[:16]
                metadata = dict(document.metadata)
                metadata["document_id"] = document.id
                metadata["chunk_index"] = i
                chunks.append(Chunk(id=chunk_id, text=text, metadata=metadata))
        return chunks

    def split_text(self, text: str) -> List[str]:
        text = _clean_text(text)
        if not text:
            return []

        sections = self._split_by_structure(text)
        chunks: List[str] = []
        current = ""

        for section in sections:
            if len(section) > self.chunk_size:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._split_long_section(section))
                continue

            candidate = f"{current}\n\n{section}".strip() if current else section
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                chunks.append(current.strip())
                current = self._with_overlap(current, section)

        if current:
            chunks.append(current.strip())
        return [chunk for chunk in chunks if chunk]

    def _split_by_structure(self, text: str) -> List[str]:
        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        if len(blocks) > 1:
            return blocks
        return [part.strip() for part in re.split(r"(?<=[。！？.!?])\s+", text) if part.strip()]

    def _split_long_section(self, text: str) -> List[str]:
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end].strip())
            if end == len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return chunks

    def _with_overlap(self, previous: str, next_section: str) -> str:
        overlap = previous[-self.chunk_overlap :].strip()
        return f"{overlap}\n\n{next_section}".strip() if overlap else next_section
