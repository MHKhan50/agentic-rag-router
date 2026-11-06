"""Dense retrieval over the pooled corpus.

Scoring is a single matrix product on L2-normalised vectors (cosine
similarity). At the corpus sizes used here (~10^4-10^5 passages) exact search
in NumPy is fast and avoids an approximate-index dependency.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .data import Passage


class DenseRetriever:
    def __init__(self, passages: Sequence[Passage], passage_embeddings: np.ndarray):
        if len(passages) != passage_embeddings.shape[0]:
            raise ValueError("passages and embeddings must align")
        self.passages = list(passages)
        self.matrix = _l2_normalise(passage_embeddings.astype(np.float32))

    def search(self, query_embeddings: np.ndarray, top_k: int) -> List[List[Passage]]:
        q = _l2_normalise(np.atleast_2d(query_embeddings).astype(np.float32))
        scores = q @ self.matrix.T
        k = min(top_k, scores.shape[1])
        idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        results = []
        for row, cand in zip(scores, idx):
            ranked = cand[np.argsort(-row[cand], kind="stable")]
            results.append([self.passages[i] for i in ranked])
        return results


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)
