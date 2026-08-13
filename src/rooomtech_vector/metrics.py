from __future__ import annotations

import numpy as np

SUPPORTED_METRICS = {"cosine", "dot", "euclidean"}


def _as_vector(value: list[float]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    if arr.size == 0:
        raise ValueError("vector must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("vector contains NaN or infinity")
    return arr


def score(metric: str, query: list[float], candidate: list[float]) -> float:
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"unsupported metric: {metric}")
    q = _as_vector(query)
    c = _as_vector(candidate)
    if q.shape != c.shape:
        raise ValueError(f"dimension mismatch: query={q.size}, candidate={c.size}")

    if metric == "cosine":
        denom = float(np.linalg.norm(q) * np.linalg.norm(c))
        return 0.0 if denom == 0.0 else float(np.dot(q, c) / denom)
    if metric == "dot":
        return float(np.dot(q, c))

    distance = float(np.linalg.norm(q - c))
    return 1.0 / (1.0 + distance)
