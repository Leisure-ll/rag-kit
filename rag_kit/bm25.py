import math
import re
from collections import Counter
from typing import List

import numpy as np


class BM25Index:
    def __init__(self, texts: List[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.tokens = [_tokenize(text) for text in texts]
        self.doc_len = np.array([len(tokens) for tokens in self.tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.tokens]
        self.doc_freq: Counter[str] = Counter()
        for freqs in self.term_freqs:
            self.doc_freq.update(freqs.keys())
        self.total_docs = len(self.tokens)

    def search(self, query: str) -> np.ndarray:
        query_terms = _tokenize(query)
        scores = np.zeros(self.total_docs, dtype=np.float32)
        if not query_terms or self.total_docs == 0:
            return scores

        for term in query_terms:
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.total_docs - df + 0.5) / (df + 0.5))
            for index, freqs in enumerate(self.term_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denominator = tf + self.k1 * (1 - self.b + self.b * self.doc_len[index] / (self.avgdl or 1))
                scores[index] += idf * (tf * (self.k1 + 1) / denominator)
        return scores


def normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_score = float(scores.min())
    max_score = float(scores.max())
    if math.isclose(max_score, min_score):
        return np.ones_like(scores) if max_score > 0 else np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)


def _tokenize(text: str) -> List[str]:
    lower = text.lower()
    words = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", lower)
    cjk_bigrams = [lower[i : i + 2] for i in range(max(0, len(lower) - 1)) if _has_cjk(lower[i : i + 2])]
    return words + cjk_bigrams


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)
