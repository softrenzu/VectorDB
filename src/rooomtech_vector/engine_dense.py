from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .ann import HNSWIndex, IVFFlatIndex, QuantizedIndex
from .bm25 import bm25_scores
from .filters import matches_filter
from .gpu import cupy_available, gpu_scores
from .metrics import SUPPORTED_METRICS, score as vector_score
from .storage import StoredPoint

class EngineDenseMixin:
    def search_dense(
        self,
        collection_name: str,
        query_vector: list[float],
        *,
        vector_name: str = "default",
        top_k: int = 10,
        score_threshold: float | None = None,
        namespace: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        explain: bool = False,
        index_mode: str | None = None,
        ef_search: int = 64,
        n_probe: int | None = None,
        quantization: str | None = None,
        candidate_multiplier: int = 8,
        use_gpu: bool = False,
    ) -> list[dict[str, Any]]:
        collection = self._collection(collection_name)
        schema = collection["config"]["vectors"]
        if vector_name not in schema:
            raise ValueError(f"unknown vector name: {vector_name}")
        if len(query_vector) != int(schema[vector_name]["dimension"]):
            raise ValueError("query vector dimension mismatch")
        metric = str(schema[vector_name]["metric"])
        index_cfg = collection["config"]["index"]

        all_points = self._index_points(collection_name, namespace, vector_name)
        filtered_points = [p for p in all_points if matches_filter(p.metadata, metadata_filter)]
        allowed = {(p.namespace, p.id) for p in filtered_points}

        requested_mode = index_mode or str(index_cfg.get("mode", "auto"))
        if requested_mode not in {"auto", "exact", "hnsw", "ivf"}:
            raise ValueError("index_mode must be auto, exact, hnsw, or ivf")
        resolved_mode = requested_mode
        if requested_mode == "auto":
            threshold = int(index_cfg.get("dynamic_threshold", 1000))
            resolved_mode = "exact" if len(filtered_points) < threshold else "hnsw"
        requested_quant = quantization if quantization is not None else str(index_cfg.get("quantization", "none"))
        if requested_quant not in {"none", "scalar", "binary"}:
            raise ValueError("quantization must be none, scalar, or binary")

        planner: dict[str, Any] = {
            "requested_index": requested_mode,
            "resolved_index": resolved_mode,
            "points_total": len(all_points),
            "points_after_filter": len(filtered_points),
            "quantization": requested_quant,
            "gpu": False,
            "filter_fallback_exact": False,
        }

        candidate_points: list[StoredPoint]
        candidate_k = max(top_k * max(1, candidate_multiplier), top_k)
        if requested_quant != "none":
            candidate_k = max(candidate_k, min(256, len(all_points)))
        if requested_quant != "none" and all_points:
            qpoints, qindex = self._get_quantized_index(collection_name, namespace, vector_name, requested_quant)
            qresult = qindex.search(query_vector, candidate_k=min(candidate_k, len(qpoints)))
            candidate_points = [qpoints[i] for i in qresult.ids if (qpoints[i].namespace, qpoints[i].id) in allowed]
            planner["quantization_compression_ratio_estimate"] = qresult.compression_ratio_estimate
            planner["candidate_source"] = f"{requested_quant}_quantized"
        elif resolved_mode == "hnsw" and all_points:
            ipoints, index = self._get_ann_index(collection_name, namespace, vector_name, "hnsw", metric, index_cfg)
            ids = index.search(query_vector, top_k=min(candidate_k, len(ipoints)), ef_search=max(ef_search, candidate_k))
            candidate_points = [ipoints[i] for i in ids if (ipoints[i].namespace, ipoints[i].id) in allowed]
            planner["candidate_source"] = "hnsw"
            planner["ef_search"] = ef_search
        elif resolved_mode == "ivf" and all_points:
            ipoints, index = self._get_ann_index(collection_name, namespace, vector_name, "ivf", metric, index_cfg)
            ids = index.search(query_vector, top_k=min(candidate_k, len(ipoints)), n_probe=n_probe)
            candidate_points = [ipoints[i] for i in ids if (ipoints[i].namespace, ipoints[i].id) in allowed]
            planner["candidate_source"] = "ivf"
            planner["n_probe"] = n_probe
        else:
            candidate_points = filtered_points
            planner["candidate_source"] = "exact"

        if len(candidate_points) < min(top_k, len(filtered_points)) and candidate_points is not filtered_points:
            existing = {(p.namespace, p.id) for p in candidate_points}
            candidate_points.extend(p for p in filtered_points if (p.namespace, p.id) not in existing)
            planner["filter_fallback_exact"] = True

        if use_gpu and candidate_points:
            if not cupy_available():
                raise RuntimeError("GPU search requested but CuPy is not available")
            scores = gpu_scores(metric, query_vector, [p.vectors[vector_name] for p in candidate_points])
            planner["gpu"] = True
            scored = list(zip(scores, candidate_points))
        else:
            scored = [(vector_score(metric, query_vector, p.vectors[vector_name]), p) for p in candidate_points]

        scored = [(s, p) for s, p in scored if score_threshold is None or s >= score_threshold]
        scored.sort(key=lambda item: item[0], reverse=True)
        planner["candidates_scored_exactly"] = len(scored)
        return [
            self._serialize_result(
                p,
                s,
                {
                    "mode": "dense",
                    "metric": metric,
                    "vector_name": vector_name,
                    "dense_score": s,
                    "planner": planner,
                }
                if explain
                else None,
            )
            for s, p in scored[:top_k]
        ]

    def search_text(
        self,
        collection_name: str,
        query_text: str,
        *,
        top_k: int = 10,
        namespace: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        explain: bool = False,
    ) -> list[dict[str, Any]]:
        points = self._filtered_points(collection_name, namespace, metadata_filter)
        scores = bm25_scores(query_text, [p.text for p in points])
        ranked = sorted(zip(scores, points), key=lambda item: item[0], reverse=True)
        ranked = [(s, p) for s, p in ranked if s > 0.0]
        return [
            self._serialize_result(
                p,
                s,
                {"mode": "text", "algorithm": "bm25", "tokenizer": "cjk-bigram-aware", "bm25_score": s}
                if explain
                else None,
            )
            for s, p in ranked[:top_k]
        ]
