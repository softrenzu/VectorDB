"""Dify BaseVector adapter source template.

This file is intentionally not installed with the standalone package because it imports
Dify internals. It is meant for a Dify provider package/container layer.
"""
from __future__ import annotations

from typing import Any, override

from core.rag.datasource.vdb.vector_base import BaseVector
from core.rag.models.document import Document
from rooomtech_vector.client import RooomtechVectorClient


class RooomtechDifyAdapter(BaseVector):
    def __init__(self, collection_name: str, endpoint: str, api_key: str | None = None):
        super().__init__(collection_name)
        self.client = RooomtechVectorClient(endpoint, api_key=api_key)

    @override
    def get_type(self) -> str:
        # Future official integration should return VectorType.ROOOMTECH_VECTOR.
        # A source-unmodified Dify deployment currently needs an existing provider alias.
        return "rooomtech-vector"

    @override
    def create(self, texts: list[Document], embeddings: list[list[float]], **kwargs):
        if not embeddings:
            return []
        try:
            self.client.create_collection(self._collection_name, len(embeddings[0]), "cosine")
        except Exception:
            pass
        return self.add_texts(texts, embeddings)

    @override
    def add_texts(self, documents: list[Document], embeddings: list[list[float]], **kwargs):
        points = []
        ids = []
        for doc, embedding in zip(documents, embeddings, strict=True):
            metadata = dict(doc.metadata or {})
            point_id = str(metadata.get("doc_id"))
            ids.append(point_id)
            points.append({"id": point_id, "vectors": embedding, "text": doc.page_content, "metadata": metadata})
        self.client.upsert(self._collection_name, points)
        return ids

    @override
    def text_exists(self, id: str) -> bool:
        return self.client.exists(self._collection_name, id)

    @override
    def delete_by_ids(self, ids: list[str]) -> None:
        self.client.delete_ids(self._collection_name, ids)

    @override
    def delete_by_metadata_field(self, key: str, value: str) -> None:
        self.client._client.post(
            f"/v1/collections/{self._collection_name}/delete",
            json={"filter": {key: {"$eq": value}}},
        ).raise_for_status()

    @override
    def search_by_vector(self, query_vector: list[float], **kwargs: Any) -> list[Document]:
        results = self.client.search_dense(
            self._collection_name,
            query_vector,
            top_k=int(kwargs.get("top_k", 4)),
            score_threshold=kwargs.get("score_threshold"),
        )
        return [Document(page_content=r["text"], metadata={**r["metadata"], "score": r["score"]}) for r in results]

    @override
    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        results = self.client.search_text(self._collection_name, query, top_k=int(kwargs.get("top_k", 4)))
        return [Document(page_content=r["text"], metadata={**r["metadata"], "score": r["score"]}) for r in results]

    @override
    def delete(self) -> None:
        self.client._client.delete(f"/v1/collections/{self._collection_name}").raise_for_status()
