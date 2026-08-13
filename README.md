# Rooomtech VectorDB

Rooomtech VectorDB is a Python vector database focused on RAG and Dify workloads. It combines modern retrieval patterns with a codebase designed to remain understandable and extensible by a very small team.

Version: `0.3.0`

## What is different

Rooomtech v0.3 is built around five differentiators:

1. **Filter-aware Dynamic Index** — `auto` selects Exact or HNSW from the number of points remaining after namespace/metadata filtering. Small filtered sets stay exact; larger sets move to ANN.
2. **Japanese-first Hybrid RAG** — Dense search + BM25 + CJK bigram tokenization without requiring MeCab or an external search server.
3. **Explainable retrieval** — `explain=true` returns the selected index, quantization mode, candidate counts, fallback behavior, dense score, BM25 contribution, and fusion contribution.
4. **Quality-first approximate search** — HNSW, IVFFlat, scalar int8 candidate compression, and binary quantization rerank candidates with the original vectors.
5. **Built-in diversity** — MMR search is part of the database API instead of requiring application-side post-processing.

## Implemented features

- Exact dense vector search
- HNSW ANN implemented in Python
- IVFFlat ANN implemented with NumPy
- Dynamic Exact -> HNSW query planning
- Cosine, dot product, Euclidean distance
- Named vectors and multi-field RRF fusion
- Scalar int8 candidate quantization + exact rerank
- Binary quantization + exact rerank
- Optional CUDA/CuPy exact scoring
- BM25 full-text search
- CJK bigram tokenizer for Japanese/Chinese/Korean text
- Hybrid Dense + BM25
- RRF and weighted normalized fusion
- MMR diversity search
- Nested metadata filters
- Namespace multi-tenancy
- Exact fallback for selective ANN filters
- Search explain / query-plan output
- Snapshot / restore
- SQLite WAL persistence
- API-key authentication
- Python client
- Dify `BaseVector` adapter template
- Docker / Docker Compose
- pytest coverage for API and engine behavior

## Not claimed yet

Rooomtech v0.3 does **not** claim distributed production parity with Qdrant, Weaviate, or Milvus. These remain future work and are reported as unsupported by `/v1/capabilities`:

- Multi-node sharding
- Multi-node replication / consensus
- Persistent on-disk ANN indexes
- GPU ANN indexes
- Product quantization
- First-class sparse embedding vectors
- Late-interaction multivectors

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export ROOOMTECH_VECTOR_DB_PATH=./data/rooomtech_vector.db
rooomtech-vector
```

Server default: `http://localhost:8080`

Optional authentication:

```bash
export ROOOMTECH_VECTOR_API_KEY=change-me
rooomtech-vector
```

Docker:

```bash
docker compose up --build
```

## Create a collection

```bash
curl -X PUT http://localhost:8080/v1/collections/dify_demo \
  -H 'Content-Type: application/json' \
  -d '{
    "dimension":1536,
    "metric":"cosine",
    "index_mode":"auto",
    "dynamic_threshold":1000,
    "quantization":"scalar"
  }'
```

## Dense search with query-plan explanation

```bash
curl -X POST http://localhost:8080/v1/collections/dify_demo/search/dense \
  -H 'Content-Type: application/json' \
  -d '{
    "vector":[0.1,0.2,0.3],
    "top_k":5,
    "index_mode":"auto",
    "explain":true
  }'
```

## Hybrid search

```bash
curl -X POST http://localhost:8080/v1/collections/dify_demo/search/hybrid \
  -H 'Content-Type: application/json' \
  -d '{
    "vector":[0.1,0.2,0.3],
    "query":"Dify VectorDB",
    "fusion":"rrf",
    "top_k":5,
    "namespace":"tenant-a",
    "explain":true
  }'
```

## API surface

- `GET /health`
- `GET /v1/capabilities`
- `GET /v1/collections`
- `PUT /v1/collections/{name}`
- `DELETE /v1/collections/{name}`
- `GET /v1/collections/{name}/stats`
- `POST /v1/collections/{name}/indexes/rebuild`
- `POST /v1/collections/{name}/points`
- `GET /v1/collections/{name}/points/{id}/exists`
- `POST /v1/collections/{name}/search/dense`
- `POST /v1/collections/{name}/search/text`
- `POST /v1/collections/{name}/search/hybrid`
- `POST /v1/collections/{name}/search/multifield`
- `POST /v1/collections/{name}/search/mmr`
- `POST /v1/collections/{name}/delete`
- `GET /v1/collections/{name}/snapshot`
- `POST /v1/restore`

## Dify

See `docs/DIFY_INTEGRATION.md` and `integrations/dify/dify_vector.py`.

Rooomtech VectorDB does **not** implement the Qdrant API. The Dify adapter calls Rooomtech VectorDB's own API.

## Test

```bash
pip install -e '.[dev]'
pytest -q
```

Current local validation: **8 tests passing**.

## Enterprise support

Paid maintenance, technical support, implementation, integration, upgrades, SLA options, custom development, and a commercial contract option are available from ROOOMTECH. Contact `tasuku.yoshioka@rooomtech.com`.

## License

Apache-2.0.
