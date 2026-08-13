from __future__ import annotations

import math
import numpy as np

from .ann_common import ann_score as _score


class IVFFlatIndex:
    """Small deterministic NumPy IVFFlat index."""

    def __init__(self, vectors: list[list[float]], *, metric: str = "cosine",
                 n_lists: int | None = None, iterations: int = 8, seed: int = 42):
        self.metric = metric
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be a 2D matrix")
        n = len(self.vectors)
        self.n_lists = max(1, min(int(n_lists or max(1, math.sqrt(n or 1))), max(n, 1)))
        if n == 0:
            self.centroids = np.empty((0, 0), dtype=np.float32)
            self.lists: list[list[int]] = []
            return

        rng = np.random.default_rng(seed)
        self.centroids = self.vectors[rng.choice(n, self.n_lists, replace=False)].copy()
        labels = np.zeros(n, dtype=np.int32)
        for _ in range(max(1, iterations)):
            scores = np.array([
                [_score(metric, v, c) for c in self.centroids]
                for v in self.vectors
            ], dtype=np.float32)
            new_labels = np.argmax(scores, axis=1).astype(np.int32)
            if np.array_equal(labels, new_labels):
                labels = new_labels
                break
            labels = new_labels
            for i in range(self.n_lists):
                members = self.vectors[labels == i]
                if len(members):
                    self.centroids[i] = members.mean(axis=0)
        self.lists = [np.where(labels == i)[0].astype(int).tolist() for i in range(self.n_lists)]

    def search(self, query: list[float], *, top_k: int = 10, n_probe: int | None = None) -> list[int]:
        if len(self.vectors) == 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.vectors.shape[1],):
            raise ValueError("query vector dimension mismatch")
        probes = max(1, min(int(n_probe or math.ceil(math.sqrt(self.n_lists))), self.n_lists))
        closest = sorted(
            range(self.n_lists),
            key=lambda i: _score(self.metric, q, self.centroids[i]),
            reverse=True,
        )[:probes]
        candidates = [idx for li in closest for idx in self.lists[li]]
        candidates = list(dict.fromkeys(candidates))
        candidates.sort(key=lambda i: _score(self.metric, q, self.vectors[i]), reverse=True)
        return candidates[:top_k]
