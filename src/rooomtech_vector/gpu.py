from __future__ import annotations

from typing import Any


def cupy_available() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def gpu_scores(metric: str, query: list[float], vectors: list[list[float]]) -> list[float]:
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("CuPy is not installed; install Rooomtech[gpu] on a CUDA host") from exc

    if not vectors:
        return []
    q = cp.asarray(query, dtype=cp.float32)
    x = cp.asarray(vectors, dtype=cp.float32)
    if metric == "dot":
        scores = x @ q
    elif metric == "cosine":
        qn = cp.linalg.norm(q)
        xn = cp.linalg.norm(x, axis=1)
        scores = (x @ q) / cp.maximum(xn * qn, 1e-12)
    elif metric == "euclidean":
        dist = cp.linalg.norm(x - q[None, :], axis=1)
        scores = 1.0 / (1.0 + dist)
    else:
        raise ValueError(f"unsupported metric: {metric}")
    return cp.asnumpy(scores).astype(float).tolist()
