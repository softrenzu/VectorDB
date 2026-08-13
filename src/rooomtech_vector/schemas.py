from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class VectorConfig(BaseModel):
    dimension: int = Field(gt=0)
    metric: Literal["cosine", "dot", "euclidean"] = "cosine"


class CreateCollection(BaseModel):
    dimension: int | None = Field(default=None, gt=0)
    metric: Literal["cosine", "dot", "euclidean"] = "cosine"
    vectors: dict[str, VectorConfig] | None = None
    if_not_exists: bool = False
    index_mode: Literal["auto", "exact", "hnsw", "ivf"] = "auto"
    dynamic_threshold: int = Field(default=1000, ge=1)
    quantization: Literal["none", "scalar", "binary"] = "none"
    hnsw_m: int = Field(default=16, ge=2, le=128)
    hnsw_ef_construction: int = Field(default=64, ge=2, le=4096)
    ivf_lists: int | None = Field(default=None, ge=1)


class Point(BaseModel):
    id: str
    vectors: list[float] | dict[str, list[float]]
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    namespace: str = ""


class UpsertRequest(BaseModel):
    points: list[Point]


class DenseSearchRequest(BaseModel):
    vector: list[float]
    vector_name: str = "default"
    top_k: int = Field(default=10, ge=1, le=1000)
    score_threshold: float | None = None
    namespace: str | None = None
    filter: dict[str, Any] | None = None
    explain: bool = False
    index_mode: Literal["auto", "exact", "hnsw", "ivf"] | None = None
    ef_search: int = Field(default=64, ge=1, le=100000)
    n_probe: int | None = Field(default=None, ge=1)
    quantization: Literal["none", "scalar", "binary"] | None = None
    candidate_multiplier: int = Field(default=8, ge=1, le=100)
    use_gpu: bool = False


class TextSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=1000)
    namespace: str | None = None
    filter: dict[str, Any] | None = None
    explain: bool = False


class HybridSearchRequest(BaseModel):
    vector: list[float]
    query: str
    vector_name: str = "default"
    top_k: int = Field(default=10, ge=1, le=1000)
    candidate_k: int | None = Field(default=None, ge=1, le=10000)
    rrf_k: int = Field(default=60, ge=1)
    dense_weight: float = Field(default=1.0, ge=0)
    text_weight: float = Field(default=1.0, ge=0)
    namespace: str | None = None
    filter: dict[str, Any] | None = None
    explain: bool = False
    fusion: Literal["rrf", "weighted"] = "rrf"
    index_mode: Literal["auto", "exact", "hnsw", "ivf"] | None = None
    quantization: Literal["none", "scalar", "binary"] | None = None


class MultiFieldSearchRequest(BaseModel):
    queries: dict[str, list[float]]
    weights: dict[str, float] = Field(default_factory=dict)
    top_k: int = Field(default=10, ge=1, le=1000)
    candidate_k: int | None = Field(default=None, ge=1, le=10000)
    rrf_k: int = Field(default=60, ge=1)
    namespace: str | None = None
    filter: dict[str, Any] | None = None
    index_mode: Literal["auto", "exact", "hnsw", "ivf"] | None = None
    explain: bool = False


class MMRSearchRequest(BaseModel):
    vector: list[float]
    vector_name: str = "default"
    top_k: int = Field(default=10, ge=1, le=1000)
    fetch_k: int = Field(default=50, ge=1, le=10000)
    lambda_mult: float = Field(default=0.7, ge=0.0, le=1.0)
    namespace: str | None = None
    filter: dict[str, Any] | None = None
    index_mode: Literal["auto", "exact", "hnsw", "ivf"] | None = None
    explain: bool = False


class DeleteRequest(BaseModel):
    ids: list[str] | None = None
    filter: dict[str, Any] | None = None
    namespace: str | None = None


class RestoreRequest(BaseModel):
    snapshot: dict[str, Any]
    overwrite: bool = False
