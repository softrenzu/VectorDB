from __future__ import annotations

import os
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException

from .engine import VectorEngine
from .schemas import (
    CreateCollection, DeleteRequest, DenseSearchRequest, HybridSearchRequest,
    MMRSearchRequest, MultiFieldSearchRequest, RestoreRequest, TextSearchRequest,
    UpsertRequest,
)

DB_PATH = os.getenv("ROOOMTECH_VECTOR_DB_PATH", "data/rooomtech_vector.db")
API_KEY = os.getenv("ROOOMTECH_VECTOR_API_KEY")
engine = VectorEngine(DB_PATH)
app = FastAPI(title="Rooomtech VectorDB", version="0.2.0")


def auth(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


def _convert_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/capabilities", dependencies=[Depends(auth)])
def capabilities() -> dict[str, Any]:
    return engine.capabilities()


@app.get("/v1/collections", dependencies=[Depends(auth)])
def list_collections() -> list[dict[str, Any]]:
    return engine.list_collections()


@app.put("/v1/collections/{name}", dependencies=[Depends(auth)])
def create_collection(name: str, request: CreateCollection) -> dict[str, Any]:
    try:
        vectors = {k: v.model_dump() for k, v in request.vectors.items()} if request.vectors else None
        return engine.create_collection(
            name, dimension=request.dimension, metric=request.metric, vectors=vectors,
            if_not_exists=request.if_not_exists, index_mode=request.index_mode,
            dynamic_threshold=request.dynamic_threshold, quantization=request.quantization,
            hnsw_m=request.hnsw_m, hnsw_ef_construction=request.hnsw_ef_construction,
            ivf_lists=request.ivf_lists,
        )
    except Exception as exc:
        raise _convert_error(exc) from exc


@app.delete("/v1/collections/{name}", dependencies=[Depends(auth)])
def drop_collection(name: str) -> dict[str, bool]:
    try:
        engine.drop_collection(name)
        return {"deleted": True}
    except Exception as exc:
        raise _convert_error(exc) from exc


@app.get("/v1/collections/{name}/stats", dependencies=[Depends(auth)])
def stats(name: str) -> dict[str, Any]:
    try:
        return engine.stats(name)
    except Exception as exc:
        raise _convert_error(exc) from exc


@app.post("/v1/collections/{name}/indexes/rebuild", dependencies=[Depends(auth)])
def rebuild_indexes(name: str) -> dict[str, Any]:
    try:
        return engine.rebuild_indexes(name)
    except Exception as exc:
        raise _convert_error(exc) from exc


@app.post("/v1/collections/{name}/points", dependencies=[Depends(auth)])
def upsert(name: str, request: UpsertRequest) -> dict[str, Any]:
    try:
        ids = engine.upsert(name, [p.model_dump() for p in request.points])
        return {"upserted": len(ids), "ids": ids}
    except Exception as exc:
        raise _convert_error(exc) from exc


@app.get("/v1/collections/{name}/points/{point_id}/exists", dependencies=[Depends(auth)])
def exists(name: str, point_id: str, namespace: str = "") -> dict[str, bool]:
    try:
        return {"exists": engine.exists(name, point_id, namespace)}
    except Exception as exc:
        raise _convert_error(exc) from exc


@app.post("/v1/collections/{name}/search/dense", dependencies=[Depends(auth)])
def dense_search(name: str, request: DenseSearchRequest) -> dict[str, Any]:
    try:
        return {"results": engine.search_dense(
            name, request.vector, vector_name=request.vector_name, top_k=request.top_k,
            score_threshold=request.score_threshold, namespace=request.namespace,
            metadata_filter=request.filter, explain=request.explain, index_mode=request.index_mode,
            ef_search=request.ef_search, n_probe=request.n_probe,
            quantization=request.quantization, candidate_multiplier=request.candidate_multiplier,
            use_gpu=request.use_gpu,
        )}
    except Exception as exc:
        raise _convert_error(exc) from exc


@app.post("/v1/collections/{name}/search/text", dependencies=[Depends(auth)])
def text_search(name: str, request: TextSearchRequest) -> dict[str, Any]:
    try:
        return {"results": engine.search_text(
            name, request.query, top_k=request.top_k, namespace=request.namespace,
            metadata_filter=request.filter, explain=request.explain,
        )}
    except Exception as exc:
        raise _convert_error(exc) from exc


@app.post("/v1/collections/{name}/search/hybrid", dependencies=[Depends(auth)])
def hybrid_search(name: str, request: HybridSearchRequest) -> dict[str, Any]:
    try:
        return {"results": engine.search_hybrid(
            name, query_vector=request.vector, query_text=request.query,
            vector_name=request.vector_name, top_k=request.top_k,
            candidate_k=request.candidate_k, rrf_k=request.rrf_k,
            dense_weight=request.dense_weight, text_weight=request.text_weight,
            namespace=request.namespace, metadata_filter=request.filter,
            explain=request.explain, fusion=request.fusion,
            index_mode=request.index_mode, quantization=request.quantization,
        )}
    except Exception as exc:
        raise _convert_error(exc) from exc


@app.post("/v1/collections/{name}/search/multifield", dependencies=[Depends(auth)])
def multifield_search(name: str, request: MultiFieldSearchRequest) -> dict[str, Any]:
    try:
        return {"results": engine.search_multifield(
            name, request.queries, weights=request.weights, top_k=request.top_k,
            candidate_k=request.candidate_k, rrf_k=request.rrf_k,
            namespace=request.namespace, metadata_filter=request.filter,
            index_mode=request.index_mode, explain=request.explain,
        )}
    except Exception as exc:
        raise _convert_error(exc) from exc


@app.post("/v1/collections/{name}/search/mmr", dependencies=[Depends(auth)])
def mmr_search(name: str, request: MMRSearchRequest) -> dict[str, Any]:
    try:
        return {"results": engine.search_mmr(
            name, request.vector, vector_name=request.vector_name, top_k=request.top_k,
            fetch_k=request.fetch_k, lambda_mult=request.lambda_mult,
            namespace=request.namespace, metadata_filter=request.filter,
            index_mode=request.index_mode, explain=request.explain,
        )}
    except Exception as exc:
        raise _convert_error(exc) from exc


@app.post("/v1/collections/{name}/delete", dependencies=[Depends(auth)])
def delete(name: str, request: DeleteRequest) -> dict[str, int]:
    try:
        if request.ids is not None:
            return {"deleted": engine.delete_ids(name, request.ids, request.namespace)}
        if request.filter is not None:
            return {"deleted": engine.delete_by_filter(name, request.filter, request.namespace)}
        raise ValueError("ids or filter is required")
    except Exception as exc:
        raise _convert_error(exc) from exc


@app.get("/v1/collections/{name}/snapshot", dependencies=[Depends(auth)])
def snapshot(name: str) -> dict[str, Any]:
    try:
        return engine.snapshot(name)
    except Exception as exc:
        raise _convert_error(exc) from exc


@app.post("/v1/restore", dependencies=[Depends(auth)])
def restore(request: RestoreRequest) -> dict[str, Any]:
    try:
        return engine.restore(request.snapshot, overwrite=request.overwrite)
    except Exception as exc:
        raise _convert_error(exc) from exc


def run() -> None:
    host = os.getenv("ROOOMTECH_VECTOR_HOST", "0.0.0.0")
    port = int(os.getenv("ROOOMTECH_VECTOR_PORT", "8080"))
    uvicorn.run("rooomtech_vector.api:app", host=host, port=port, reload=False)
