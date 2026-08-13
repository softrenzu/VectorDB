from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .ann import HNSWIndex, IVFFlatIndex, QuantizedIndex
from .bm25 import bm25_scores
from .filters import matches_filter
from .gpu import cupy_available, gpu_scores
from .metrics import SUPPORTED_METRICS, score as vector_score
from .storage import StoredPoint

class EngineIndexMixin:
    def _index_points(self, collection_name: str, namespace: str | None, vector_name: str) -> list[StoredPoint]:
        return [p for p in self.store.list_points(collection_name, namespace) if vector_name in p.vectors]

    def _get_ann_index(
        self,
        collection_name: str,
        namespace: str | None,
        vector_name: str,
        mode: str,
        metric: str,
        config: dict[str, Any],
    ) -> tuple[list[StoredPoint], Any]:
        key = (collection_name, namespace, vector_name, mode)
        cached = self._index_cache.get(key)
        if cached:
            return cached
        points = self._index_points(collection_name, namespace, vector_name)
        vectors = [p.vectors[vector_name] for p in points]
        if mode == "hnsw":
            hc = config["hnsw"]
            index = HNSWIndex(vectors, metric=metric, m=int(hc["m"]), ef_construction=int(hc["ef_construction"]))
        elif mode == "ivf":
            index = IVFFlatIndex(vectors, metric=metric, n_lists=config["ivf"].get("lists"))
        else:
            raise ValueError(f"unsupported ANN mode: {mode}")
        self._index_cache[key] = (points, index)
        return points, index

    def _get_quantized_index(
        self,
        collection_name: str,
        namespace: str | None,
        vector_name: str,
        quantization: str,
    ) -> tuple[list[StoredPoint], QuantizedIndex]:
        key = (collection_name, namespace, vector_name, "quant", quantization)
        cached = self._index_cache.get(key)
        if cached:
            return cached  # type: ignore[return-value]
        points = self._index_points(collection_name, namespace, vector_name)
        index = QuantizedIndex([p.vectors[vector_name] for p in points], mode=quantization)
        self._index_cache[key] = (points, index)
        return points, index

    def rebuild_indexes(self, collection_name: str) -> dict[str, Any]:
        collection = self._collection(collection_name)
        self._invalidate(collection_name)
        points = self.store.list_points(collection_name)
        warmed: list[str] = []
        mode = collection["config"]["index"].get("mode", "auto")
        if mode in {"hnsw", "ivf"}:
            for vector_name, cfg in collection["config"]["vectors"].items():
                self._get_ann_index(collection_name, None, vector_name, mode, cfg["metric"], collection["config"]["index"])
                warmed.append(f"{mode}:{vector_name}")
        return {"collection": collection_name, "points": len(points), "warmed": warmed}

    def delete_ids(self, collection_name: str, ids: list[str], namespace: str | None = None) -> int:
        self._collection(collection_name)
        deleted = self.store.delete_ids(collection_name, ids, namespace)
        self._invalidate(collection_name)
        return deleted

    def delete_by_filter(self, collection_name: str, metadata_filter: dict[str, Any], namespace: str | None = None) -> int:
        points = self._filtered_points(collection_name, namespace, metadata_filter)
        if namespace is None:
            deleted = 0
            by_namespace: dict[str, list[str]] = {}
            for p in points:
                by_namespace.setdefault(p.namespace, []).append(p.id)
            for ns, ids in by_namespace.items():
                deleted += self.store.delete_ids(collection_name, ids, ns)
            self._invalidate(collection_name)
            return deleted
        deleted = self.store.delete_ids(collection_name, [p.id for p in points], namespace)
        self._invalidate(collection_name)
        return deleted

    def stats(self, collection_name: str) -> dict[str, Any]:
        collection = self._collection(collection_name)
        points = self.store.list_points(collection_name)
        namespaces = sorted({p.namespace for p in points})
        return {
            "collection": collection_name,
            "points": len(points),
            "namespaces": len(namespaces),
            "namespace_values": namespaces,
            "vectors": collection["config"]["vectors"],
            "index": collection["config"]["index"],
            "cached_indexes": sum(1 for k in self._index_cache if k[0] == collection_name),
        }

    def snapshot(self, collection_name: str) -> dict[str, Any]:
        collection = self._collection(collection_name)
        points = self.store.list_points(collection_name)
        return {
            "format": "rooomtech-vector.snapshot.v2",
            "collection": collection,
            "points": [asdict(p) for p in points],
        }

    def restore(self, snapshot: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
        if snapshot.get("format") not in {"rooomtech-vector.snapshot.v1", "rooomtech-vector.snapshot.v2"}:
            raise ValueError("unsupported snapshot format")
        collection = snapshot["collection"]
        name = collection["name"]
        existing = self.store.get_collection(name)
        if existing and not overwrite:
            raise ValueError(f"collection already exists: {name}")
        if existing:
            self.store.drop_collection(name)
        config = collection["config"]
        config.setdefault(
            "index",
            {
                "mode": "auto",
                "dynamic_threshold": 1000,
                "hnsw": {"m": 16, "ef_construction": 64},
                "ivf": {"lists": None},
                "quantization": "none",
            },
        )
        self.store.create_collection(name, config)
        points = [
            {
                "id": p["id"],
                "namespace": p.get("namespace", ""),
                "text": p.get("text", ""),
                "metadata": p.get("metadata", {}),
                "vectors": p["vectors"],
            }
            for p in snapshot.get("points", [])
        ]
        self.upsert(name, points)
        return self.stats(name)
