from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class QuantizedCandidates:
    ids: list[int]
    mode: str
    compression_ratio_estimate: float


class QuantizedIndex:
    """Scalar-int8 or binary candidate index followed by exact re-ranking."""

    def __init__(self, vectors: list[list[float]], *, mode: str = "scalar"):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be a 2D matrix")
        self.mode = mode
        if mode == "scalar":
            mins = np.min(self.vectors, axis=0) if len(self.vectors) else np.array([], dtype=np.float32)
            maxs = np.max(self.vectors, axis=0) if len(self.vectors) else np.array([], dtype=np.float32)
            scales = (maxs - mins) / 255.0
            scales[scales == 0] = 1.0
            self.mins = mins
            self.scales = scales
            self.quantized = np.clip(np.rint((self.vectors - mins) / scales), 0, 255).astype(np.uint8)
        elif mode == "binary":
            self.quantized = np.packbits(self.vectors >= 0, axis=1)
        else:
            raise ValueError("quantization mode must be scalar or binary")

    def search(self, query: list[float], *, candidate_k: int) -> QuantizedCandidates:
        if len(self.vectors) == 0:
            return QuantizedCandidates([], self.mode, 1.0)
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.vectors.shape[1],):
            raise ValueError("query vector dimension mismatch")
        if self.mode == "scalar":
            qq = np.clip(np.rint((q - self.mins) / self.scales), 0, 255).astype(np.uint8)
            diff = self.quantized.astype(np.int16) - qq.astype(np.int16)
            approx = -np.sum(diff * diff, axis=1, dtype=np.int64)
            ids = np.argsort(approx)[::-1][:candidate_k].astype(int).tolist()
            return QuantizedCandidates(ids, self.mode, 4.0)
        qb = np.packbits((q >= 0)[None, :], axis=1)[0]
        xor = np.bitwise_xor(self.quantized, qb)
        bits = np.unpackbits(xor, axis=1)
        hamming = np.sum(bits, axis=1)
        ids = np.argsort(hamming)[:candidate_k].astype(int).tolist()
        return QuantizedCandidates(ids, self.mode, 32.0)
