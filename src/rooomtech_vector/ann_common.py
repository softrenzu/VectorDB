from __future__ import annotations

import numpy as np


def ann_score(metric: str, q: np.ndarray, x: np.ndarray) -> float:
    if metric == "cosine":
        denom = float(np.linalg.norm(q) * np.linalg.norm(x))
        return 0.0 if denom == 0.0 else float(np.dot(q, x) / denom)
    if metric == "dot":
        return float(np.dot(q, x))
    if metric == "euclidean":
        return 1.0 / (1.0 + float(np.linalg.norm(q - x)))
    raise ValueError(f"unsupported metric: {metric}")
