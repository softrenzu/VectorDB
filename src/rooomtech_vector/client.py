from __future__ import annotations

from typing import Any

import httpx


class RooomtechVectorClient:
    def __init__(self, base_url: str = "http://localhost:8080", api_key: str | None = None, timeout: float = 30.0):
        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def capabilities(self) -> dict[str, Any]:
        return self._client.get("/v1/capabilities").raise_for_status().json()

    def create_collection(self, name: str, dimension: int, metric: str = "cosine", **kwargs: Any) -> dict[str, Any]:
        return self._client.put(
            f"/v1/collections/{name}", json={"dimension": dimension, "metric": metric, **kwargs}
        ).raise_for_status().json()

    def upsert(self, collection: str, points: list[dict[str, Any]]) -> dict[str, Any]:
        return self._client.post(
            f"/v1/collections/{collection}/points", json={"points": points}
        ).raise_for_status().json()

    def exists(self, collection: str, point_id: str, namespace: str = "") -> bool:
        return bool(self._client.get(
            f"/v1/collections/{collection}/points/{point_id}/exists", params={"namespace": namespace}
        ).raise_for_status().json()["exists"])

    def search_dense(self, collection: str, vector: list[float], **kwargs: Any) -> list[dict[str, Any]]:
        payload = {"vector": vector, **kwargs}
        return self._client.post(
            f"/v1/collections/{collection}/search/dense", json=payload
        ).raise_for_status().json()["results"]

    def search_text(self, collection: str, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        payload = {"query": query, **kwargs}
        return self._client.post(
            f"/v1/collections/{collection}/search/text", json=payload
        ).raise_for_status().json()["results"]

    def search_hybrid(self, collection: str, vector: list[float], query: str, **kwargs: Any) -> list[dict[str, Any]]:
        payload = {"vector": vector, "query": query, **kwargs}
        return self._client.post(
            f"/v1/collections/{collection}/search/hybrid", json=payload
        ).raise_for_status().json()["results"]

    def search_multifield(self, collection: str, queries: dict[str, list[float]], **kwargs: Any) -> list[dict[str, Any]]:
        payload = {"queries": queries, **kwargs}
        return self._client.post(
            f"/v1/collections/{collection}/search/multifield", json=payload
        ).raise_for_status().json()["results"]

    def search_mmr(self, collection: str, vector: list[float], **kwargs: Any) -> list[dict[str, Any]]:
        payload = {"vector": vector, **kwargs}
        return self._client.post(
            f"/v1/collections/{collection}/search/mmr", json=payload
        ).raise_for_status().json()["results"]

    def rebuild_indexes(self, collection: str) -> dict[str, Any]:
        return self._client.post(f"/v1/collections/{collection}/indexes/rebuild").raise_for_status().json()

    def delete_ids(self, collection: str, ids: list[str], namespace: str | None = None) -> int:
        payload: dict[str, Any] = {"ids": ids}
        if namespace is not None:
            payload["namespace"] = namespace
        return int(self._client.post(
            f"/v1/collections/{collection}/delete", json=payload
        ).raise_for_status().json()["deleted"])
