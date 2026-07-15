import hashlib
import math
import re
from typing import List, Protocol

import numpy as np


class EmbeddingModel(Protocol):
    dim: int

    def embed(self, texts: List[str]) -> np.ndarray:
        ...


class HashingEmbedding:
    """Dependency-light embedding fallback for demos and offline environments."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _tokenize(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, bucket] += sign
            norm = float(np.linalg.norm(vectors[row]))
            if norm > 0:
                vectors[row] /= norm
        return vectors


class SentenceTransformerEmbedding:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)


def create_embedding_model(backend: str, model_name: str) -> EmbeddingModel:
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedding(model_name)
    return HashingEmbedding()


def _tokenize(text: str) -> List[str]:
    lower = text.lower()
    words = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", lower)
    bigrams = [lower[i : i + 2] for i in range(max(0, len(lower) - 1)) if _has_cjk(lower[i : i + 2])]
    return words + bigrams


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.array([], dtype=np.float32)
    query_norm = np.linalg.norm(query)
    if math.isclose(float(query_norm), 0.0):
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return matrix @ (query / query_norm)
