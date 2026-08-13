from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .ann import HNSWIndex, IVFFlatIndex, QuantizedIndex
from .bm25 import bm25_scores
from .filters import matches_filter
from .gpu import cupy_available, gpu_scores
from .metrics import SUPPORTED_METRICS, score as vector_score
from .storage import StoredPoint

class EngineHybridMixin:
    @staticmethod
    def _minmax(values: list[float]) -> list[float]:
        if not values:
            return []
        lo, hi = min(values), max(values)
        if hi == lo:
            return [1.0 if hi != 0 else 0.0 for _ in values]
        return [(v - lo) / (hi - lo) for v in values]

    def search_hybrid(
        self,
        collection_name: str,
        *,
        query_vector: list[float],
        query_text: str,
        vector_name: str = "default",
        top_k: int = 10,
        candidate_k: int | None = None,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        text_weight: float = 1.0,
        namespace: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        explain: bool = False,
        fusion: str = "rrf",
        index_mode: str | None = None,
        quantization: str | None = None,
    ) -> list[dict[str, Any]]:
        if fusion not in {"rrf", "weighted"}:
            raise ValueError("fusion must be rrf or weighted")
        candidate_k = candidate_k or max(top_k * 4, 20)
        dense = self.search_dense(
            collection_name,
            query_vector,
            vector_name=vector_name,
            top_k=candidate_k,
            namespace=namespace,
            metadata_filter=metadata_filter,
            explain=False,
            index_mode=index_mode,
            quantization=quantization,
        )
        text = self.search_text(
            collection_name,
            query_text,
            top_k=candidate_k,
            namespace=namespace,
            metadata_filter=metadata_filter,
            explain=False,
        )
        dense_rank = {item["id"] + "\0" + item["namespace"]: (rank, item) for rank, item in enumerate(dense, 1)}
        text_rank = {item["id"] + "\0" + item["namespace"]: (rank, item) for rank, item in enumerate(text, 1)}
        dense_norm = dict(zip(dense_rank.keys(), self._minmax([v[1]["score"] for v in dense_rank.values()])))
        text_norm = dict(zip(text_rank.keys(), self._minmax([v[1]["score"] for v in text_rank.values()])))
        keys = set(dense_rank) | set(text_rank)
        fused: list[tuple[float, dict[str, Any]]] = []
        for key in keys:
            d = dense_rank.get(key)
            t = text_rank.get(key)
            if fusion == "rrf":
                dense_part = dense_weight / (rrf_k + d[0]) if d else 0.0
                text_part = text_weight / (rrf_k + t[0]) if t else 0.0
            else:
                dense_part = dense_weight * dense_norm.get(key, 0.0)
                text_part = text_weight * text_norm.get(key, 0.0)
            fused_score = dense_part + text_part
            base = (d or t)[1].copy()
            base["score"] = fused_score
            if explain:
                base["explain"] = {
                    "mode": "hybrid",
                    "fusion": fusion,
                    "rrf_k": rrf_k if fusion == "rrf" else None,
                    "dense_rank": d[0] if d else None,
                    "text_rank": t[0] if t else None,
                    "dense_score": d[1]["score"] if d else None,
                    "bm25_score": t[1]["score"] if t else None,
                    "dense_contribution": dense_part,
                    "text_contribution": text_part,
                    "fused_score": fused_score,
                }
            fused.append((fused_score, base))
        fused.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in fused[:top_k]]

    def search_multifield(
        self,
        collection_name: str,
        queries: dict[str, list[float]],
        *,
        weights: dict[str, float] | None = None,
        top_k: int = 10,
        candidate_k: int | None = None,
        rrf_k: int = 60,
        namespace: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        index_mode: str | None = None,
        explain: bool = False,
    ) -> list[dict[str, Any]]:
        if not queries:
            return []
        weights = weights or {}
        candidate_k = candidate_k or max(top_k * 4, 20)
        ranked_by_field: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {}
        for vector_name, query in queries.items():
            results = self.search_dense(
                collection_name,
                query,
                vector_name=vector_name,
                top_k=candidate_k,
                namespace=namespace,
                metadata_filter=metadata_filter,
                index_mode=index_mode,
            )
            ranked_by_field[vector_name] = {
                item["id"] + "\0" + item["namespace"]: (rank, item)
                for rank, item in enumerate(results, 1)
            }
        keys: set[str] = set()
        for ranks in ranked_by_field.values():
            keys |= set(ranks)
        fused: list[tuple[float, dict[str, Any]]] = []
        for key in keys:
            score = 0.0
            contributions: dict[str, Any] = {}
            base: dict[str, Any] | None = None
            for field, ranks in ranked_by_field.items():
                found = ranks.get(key)
                if not found:
                    contributions[field] = {"rank": None, "weight": weights.get(field, 1.0), "rrf": 0.0}
                    continue
                rank, item = found
                weight = float(weights.get(field, 1.0))
                part = weight / (rrf_k + rank)
                score += part
                base = base or item.copy()
                contributions[field] = {
                    "rank": rank,
                    "raw_score": item["score"],
                    "weight": weight,
                    "rrf": part,
                }
            assert base is not None
            base["score"] = score
            if explain:
                base["explain"] = {
                    "mode": "multifield",
                    "fusion": "rrf",
                    "rrf_k": rrf_k,
                    "fields": contributions,
                    "fused_score": score,
                }
            fused.append((score, base))
        fused.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in fused[:top_k]]

    def search_mmr(
        self,
        collection_name: str,
        query_vector: list[float],
        *,
        vector_name: str = "default",
        top_k: int = 10,
        fetch_k: int = 50,
        lambda_mult: float = 0.7,
        namespace: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        index_mode: str | None = None,
        explain: bool = False,
    ) -> list[dict[str, Any]]:
        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError("lambda_mult must be between 0 and 1")
        fetch_k = max(fetch_k, top_k)
        candidates = self.search_dense(
            collection_name,
            query_vector,
            vector_name=vector_name,
            top_k=fetch_k,
            namespace=namespace,
            metadata_filter=metadata_filter,
            index_mode=index_mode,
        )
        points = {(p.namespace, p.id): p for p in self._index_points(collection_name, namespace, vector_name)}
        selected: list[dict[str, Any]] = []
        remaining = candidates.copy()
        while remaining and len(selected) < top_k:
            best_item = None
            best_score = -float("inf")
            for item in remaining:
                relevance = float(item["score"])
                p = points[(item["namespace"], item["id"])]
                redundancy = 0.0
                if selected:
                    redundancy = max(
                        vector_score(
                            "cosine",
                            p.vectors[vector_name],
                            points[(s["namespace"], s["id"])].vectors[vector_name],
                        )
                        for s in selected
                    )
                mmr = lambda_mult * relevance - (1.0 - lambda_mult) * redundancy
                if mmr > best_score:
                    best_item = item.copy()
                    best_score = mmr
                    if explain:
                        best_item["explain"] = {
                            "mode": "mmr",
                            "relevance": relevance,
                            "max_redundancy": redundancy,
                            "lambda_mult": lambda_mult,
                            "mmr_score": mmr,
                        }
            assert best_item is not None
            best_item["score"] = best_score
            selected.append(best_item)
            remaining = [r for r in remaining if not (r["id"] == best_item["id"] and r["namespace"] == best_item["namespace"])]
        return selected
