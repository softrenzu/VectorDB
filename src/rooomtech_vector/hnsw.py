from __future__ import annotations

import heapq
import math
from typing import Iterable

import numpy as np

from .ann_common import ann_score as _score

class HNSWIndex:
    """Compact dependency-free HNSW implementation for small/medium Python deployments.

    It favors readability and deterministic behavior over peak throughput. The original
    vectors remain the source of truth; ANN only selects candidates.
    """

    def __init__(
        self,
        vectors: list[list[float]],
        *,
        metric: str = "cosine",
        m: int = 16,
        ef_construction: int = 64,
        seed: int = 42,
    ):
        if m < 2:
            raise ValueError("m must be >= 2")
        self.metric = metric
        self.m = int(m)
        self.ef_construction = max(int(ef_construction), self.m)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be a 2D matrix")
        self._rng = np.random.default_rng(seed)
        self.levels: list[int] = []
        self.links: list[dict[int, set[int]]] = []
        self.entry_point: int | None = None
        self.max_level = -1
        for idx in range(len(self.vectors)):
            self._insert(idx)

    def _random_level(self) -> int:
        level = 0
        p = 1.0 / math.e
        while level < 32 and float(self._rng.random()) < p:
            level += 1
        return level

    def _ensure_level(self, level: int) -> None:
        while len(self.links) <= level:
            self.links.append({})

    def _neighbors(self, node: int, level: int) -> set[int]:
        if level >= len(self.links):
            return set()
        return self.links[level].get(node, set())

    def _greedy(self, query: np.ndarray, entry: int, level: int) -> int:
        current = entry
        current_score = _score(self.metric, query, self.vectors[current])
        changed = True
        while changed:
            changed = False
            for nb in self._neighbors(current, level):
                s = _score(self.metric, query, self.vectors[nb])
                if s > current_score:
                    current, current_score, changed = nb, s, True
        return current

    def _search_layer(self, query: np.ndarray, entries: Iterable[int], ef: int, level: int) -> list[int]:
        visited: set[int] = set()
        candidates: list[tuple[float, int]] = []
        best: list[tuple[float, int]] = []
        for node in entries:
            if node in visited:
                continue
            visited.add(node)
            s = _score(self.metric, query, self.vectors[node])
            heapq.heappush(candidates, (-s, node))
            heapq.heappush(best, (s, node))

        while candidates:
            neg_s, node = heapq.heappop(candidates)
            candidate_score = -neg_s
            worst = best[0][0] if best else -float("inf")
            if len(best) >= ef and candidate_score < worst:
                break
            for nb in self._neighbors(node, level):
                if nb in visited:
                    continue
                visited.add(nb)
                s = _score(self.metric, query, self.vectors[nb])
                if len(best) < ef or s > best[0][0]:
                    heapq.heappush(candidates, (-s, nb))
                    heapq.heappush(best, (s, nb))
                    if len(best) > ef:
                        heapq.heappop(best)
        return [node for _, node in sorted(best, reverse=True)]

    def _prune(self, node: int, level: int) -> None:
        neighbors = self.links[level].setdefault(node, set())
        if len(neighbors) <= self.m:
            return
        ranked = sorted(
            neighbors,
            key=lambda nb: _score(self.metric, self.vectors[node], self.vectors[nb]),
            reverse=True,
        )[: self.m]
        self.links[level][node] = set(ranked)

    def _insert(self, idx: int) -> None:
        level = self._random_level()
        self.levels.append(level)
        self._ensure_level(level)
        for l in range(level + 1):
            self.links[l].setdefault(idx, set())

        if self.entry_point is None:
            self.entry_point = idx
            self.max_level = level
            return

        query = self.vectors[idx]
        entry = self.entry_point
        for l in range(self.max_level, level, -1):
            entry = self._greedy(query, entry, l)

        for l in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(query, [entry], self.ef_construction, l)
            selected = [n for n in candidates if n != idx][: self.m]
            for nb in selected:
                self.links[l][idx].add(nb)
                self.links[l].setdefault(nb, set()).add(idx)
            self._prune(idx, l)
            if selected:
                entry = selected[0]

        if level > self.max_level:
            self.entry_point = idx
            self.max_level = level

    def search(self, query: list[float], *, top_k: int = 10, ef_search: int = 64) -> list[int]:
        if self.entry_point is None or len(self.vectors) == 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.vectors.shape[1],):
            raise ValueError("query vector dimension mismatch")
        entry = self.entry_point
        for level in range(self.max_level, 0, -1):
            entry = self._greedy(q, entry, level)
        ef = max(int(ef_search), int(top_k))
        candidates = self._search_layer(q, [entry], ef, 0)
        candidates.sort(key=lambda i: _score(self.metric, q, self.vectors[i]), reverse=True)
        return candidates[:top_k]
