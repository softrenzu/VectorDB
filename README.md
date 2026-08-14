# RooomVector — Vector Database for RAG

Version: `0.4.0`

RooomVector is a source-available Python vector database focused on RAG and Dify workloads.

## Key features

- Filter-aware Dynamic Index: Exact or HNSW selected after metadata filtering
- Japanese-first hybrid retrieval: dense vectors + BM25 + CJK bigrams
- Explainable retrieval plans and score contributions
- HNSW and IVFFlat approximate search
- Scalar int8 and binary quantization with exact reranking
- Optional CUDA/CuPy exact scoring
- RRF and weighted fusion
- MMR diversity search
- Nested metadata filters and namespace multi-tenancy
- Snapshot/restore and SQLite WAL persistence
- API-key authentication
- Python client and Dify adapter template

RooomVector does not claim distributed production parity with Qdrant, Weaviate, or Milvus. Multi-node consensus and distributed storage remain roadmap work.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export ROOOMTECH_VECTOR_DB_PATH=./data/rooomtech_vector.db
rooom-vector
```

`rooomtech-vector` is retained as a compatibility command alias.

Server default: `http://localhost:8080`

Dify integration is documented in `docs/DIFY_INTEGRATION.md`. RooomVector uses its own API rather than implementing the Qdrant API.

## Licensing and support

Version `0.4.0` and later use the ROOOMTECH licensing terms in `LICENSE`. A separate commercial software license agreement and paid maintenance, support, implementation, integration, upgrades, security support, SLA options, private builds, and custom development are available.

Contact: `support@rooomtech.com`

Versions through `0.3.x` retain their published license terms. Third-party software retains its own licenses.
