from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .ann import HNSWIndex, IVFFlatIndex, QuantizedIndex
from .bm25 import bm25_scores
from .filters import matches_filter
from .gpu import cupy_available, gpu_scores
from .metrics import SUPPORTED_METRICS, score as vector_score
from .storage import StoredPoint

class EngineDataMixin:
    @staticmethod
    def capabilities() -> dict[str, Any]:
        return {
            "dense_vector": True,
            "exact_search": True,
            "named_vectors": True,
            "full_text_bm25": True,
            "cjk_bigram_tokenizer": True,
            "hybrid_search": True,
            "hybrid_fusion": ["rrf", "weighted"],
            "metadata_filter": True,
            "nested_metadata_paths": True,
            "multi_tenant_namespace": True,
            "search_explain": True,
            "snapshot_restore": True,
            "idempotent_upsert": True,
            "sqlite_wal": True,
            "hnsw": True,
            "ivf_flat": True,
            "dynamic_index": True,
            "quantization": ["scalar_int8", "binary"],
            "exact_rerank_after_quantization": True,
            "mmr_diversity": True,
            "multi_field_named_vector_fusion": True,
            "gpu_exact_search": {"optional": True, "available": cupy_available()},
            "replication": False,
            "sharding": False,
        }

    def create_collection(
        self,
        name: str,
        *,
        dimension: int | None = None,
        metric: str = "cosine",
        vectors: dict[str, dict[str, Any]] | None = None,
        if_not_exists: bool = False,
        index_mode: str = "auto",
        dynamic_threshold: int = 1000,
        quantization: str = "none",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
        ivf_lists: int | None = None,
    ) -> dict[str, Any]:
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("collection name must use letters, digits, underscore, or hyphen")
        if index_mode not in {"auto", "exact", "hnsw", "ivf"}:
            raise ValueError("index_mode must be auto, exact, hnsw, or ivf")
        if quantization not in {"none", "scalar", "binary"}:
            raise ValueError("quantization must be none, scalar, or binary")
        if dynamic_threshold < 1:
            raise ValueError("dynamic_threshold must be >= 1")
        if vectors is None:
            if dimension is None or dimension <= 0:
                raise ValueError("dimension must be a positive integer")
            vectors = {"default": {"dimension": dimension, "metric": metric}}
        for vector_name, cfg in vectors.items():
            if not vector_name:
                raise ValueError("vector name must not be empty")
            dim = int(cfg["dimension"])
            met = str(cfg.get("metric", "cosine"))
            if dim <= 0:
                raise ValueError("vector dimension must be positive")
            if met not in SUPPORTED_METRICS:
                raise ValueError(f"unsupported metric: {met}")
            cfg["dimension"] = dim
            cfg["metric"] = met
        existing = self.store.get_collection(name)
        config = {
            "vectors": vectors,
            "index": {
                "mode": index_mode,
                "dynamic_threshold": int(dynamic_threshold),
                "hnsw": {"m": int(hnsw_m), "ef_construction": int(hnsw_ef_construction)},
                "ivf": {"lists": ivf_lists},
                "quantization": quantization,
            },
        }
        if existing:
            if if_not_exists:
                return existing
            raise ValueError(f"collection already exists: {name}")
        self.store.create_collection(name, config)
        return self.store.get_collection(name)  # type: ignore[return-value]

    def _collection(self, name: str) -> dict[str, Any]:
        collection = self.store.get_collection(name)
        if collection is None:
            raise KeyError(f"collection not found: {name}")
        collection["config"].setdefault(
            "index",
            {
                "mode": "auto",
                "dynamic_threshold": 1000,
                "hnsw": {"m": 16, "ef_construction": 64},
                "ivf": {"lists": None},
                "quantization": "none",
            },
        )
        return collection

    def list_collections(self) -> list[dict[str, Any]]:
        return [self._collection(c["name"]) for c in self.store.list_collections()]

    def _invalidate(self, collection_name: str) -> None:
        for key in [k for k in self._index_cache if k and k[0] == collection_name]:
            self._index_cache.pop(key, None)

    def drop_collection(self, name: str) -> None:
        self._collection(name)
        self.store.drop_collection(name)
        self._invalidate(name)

    def _normalize_vectors(self, vectors: Any) -> dict[str, list[float]]:
        if isinstance(vectors, list):
            return {"default": [float(x) for x in vectors]}
        if isinstance(vectors, dict):
            return {str(k): [float(x) for x in v] for k, v in vectors.items()}
        raise ValueError("vectors must be a list or mapping of named vectors")

    def _validate_vectors(self, collection: dict[str, Any], vectors: dict[str, list[float]]) -> None:
        schema = collection["config"]["vectors"]
        unknown = set(vectors) - set(schema)
        if unknown:
            raise ValueError(f"unknown vector names: {sorted(unknown)}")
        for name, value in vectors.items():
            expected = int(schema[name]["dimension"])
            if len(value) != expected:
                raise ValueError(f"dimension mismatch for {name}: expected {expected}, got {len(value)}")

    def upsert(self, collection_name: str, points: list[dict[str, Any]]) -> list[str]:
        collection = self._collection(collection_name)
        normalized: list[dict[str, Any]] = []
        ids: list[str] = []
        for p in points:
            point_id = str(p["id"])
            vectors = self._normalize_vectors(p["vectors"])
            self._validate_vectors(collection, vectors)
            normalized.append(
                {
                    "id": point_id,
                    "namespace": str(p.get("namespace") or ""),
                    "text": str(p.get("text") or ""),
                    "metadata": dict(p.get("metadata") or {}),
                    "vectors": vectors,
                }
            )
            ids.append(point_id)
        self.store.upsert_points(collection_name, normalized)
        self._invalidate(collection_name)
        return ids

    def exists(self, collection_name: str, point_id: str, namespace: str = "") -> bool:
        self._collection(collection_name)
        return self.store.point_exists(collection_name, point_id, namespace)

    def _filtered_points(
        self,
        collection_name: str,
        namespace: str | None,
        metadata_filter: dict[str, Any] | None,
    ) -> list[StoredPoint]:
        self._collection(collection_name)
        points = self.store.list_points(collection_name, namespace)
        return [p for p in points if matches_filter(p.metadata, metadata_filter)]

    @staticmethod
    def _serialize_result(point: StoredPoint, score: float, explain: dict[str, Any] | None) -> dict[str, Any]:
        result = {
            "id": point.id,
            "namespace": point.namespace,
            "text": point.text,
            "metadata": point.metadata,
            "score": float(score),
        }
        if explain is not None:
            result["explain"] = explain
        return result
