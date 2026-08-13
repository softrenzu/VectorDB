from pathlib import Path

from rooomtech_vector.bm25 import tokenize
from rooomtech_vector.engine import VectorEngine
from rooomtech_vector.filters import matches_filter


def engine(tmp_path: Path) -> VectorEngine:
    return VectorEngine(tmp_path / "test.db")


def test_metadata_filter():
    meta = {"year": 2026, "tags": ["rag", "dify"], "acl": {"team": "ai"}}
    assert matches_filter(meta, {"year": {"$gte": 2025}})
    assert matches_filter(meta, {"tags": {"$contains": "dify"}})
    assert matches_filter(meta, {"acl.team": "ai"})
    assert not matches_filter(meta, {"$or": [{"year": {"$lt": 2020}}, {"acl.team": "sales"}]})


def test_cjk_tokenizer():
    tokens = tokenize("東京都の観光情報")
    assert "東京" in tokens
    assert "観光" in tokens


def test_dense_hybrid_namespace_and_snapshot(tmp_path: Path):
    e = engine(tmp_path)
    e.create_collection("docs", dimension=3, metric="cosine")
    e.upsert(
        "docs",
        [
            {"id": "a", "vectors": [1, 0, 0], "text": "Dify vector database integration", "metadata": {"kind": "tech"}, "namespace": "tenant-a"},
            {"id": "b", "vectors": [0.9, 0.1, 0], "text": "Dify hybrid search BM25", "metadata": {"kind": "tech"}, "namespace": "tenant-a"},
            {"id": "c", "vectors": [1, 0, 0], "text": "private other tenant", "metadata": {"kind": "tech"}, "namespace": "tenant-b"},
        ],
    )

    dense = e.search_dense("docs", [1, 0, 0], namespace="tenant-a", top_k=10)
    assert {x["id"] for x in dense} == {"a", "b"}

    hybrid = e.search_hybrid(
        "docs",
        query_vector=[1, 0, 0],
        query_text="BM25",
        namespace="tenant-a",
        top_k=2,
        explain=True,
    )
    assert hybrid[0]["id"] == "b"
    assert hybrid[0]["explain"]["fusion"] == "rrf"

    snapshot = e.snapshot("docs")
    e.drop_collection("docs")
    assert e.list_collections() == []
    restored = e.restore(snapshot)
    assert restored["points"] == 3


def test_named_vectors(tmp_path: Path):
    e = engine(tmp_path)
    e.create_collection(
        "multi",
        vectors={
            "title": {"dimension": 2, "metric": "cosine"},
            "body": {"dimension": 3, "metric": "dot"},
        },
    )
    e.upsert(
        "multi",
        [{"id": "1", "vectors": {"title": [1, 0], "body": [1, 2, 3]}, "text": "hello"}],
    )
    result = e.search_dense("multi", [1, 0], vector_name="title")
    assert result[0]["id"] == "1"


def test_hnsw_ivf_quantization_and_dynamic(tmp_path: Path):
    e = engine(tmp_path)
    e.create_collection(
        "ann",
        dimension=4,
        metric="cosine",
        index_mode="auto",
        dynamic_threshold=10,
    )
    points = []
    for i in range(80):
        if i == 37:
            vec = [1.0, 0.0, 0.0, 0.0]
        else:
            vec = [0.0, 1.0, float(i % 7) / 10.0, float(i % 5) / 10.0]
        points.append({"id": str(i), "vectors": vec, "text": f"point {i}", "metadata": {"group": i % 2}})
    e.upsert("ann", points)

    exact = e.search_dense("ann", [1, 0, 0, 0], top_k=1, index_mode="exact")
    assert exact[0]["id"] == "37"

    hnsw = e.search_dense("ann", [1, 0, 0, 0], top_k=3, index_mode="hnsw", explain=True)
    assert hnsw[0]["id"] == "37"
    assert hnsw[0]["explain"]["planner"]["resolved_index"] == "hnsw"

    ivf = e.search_dense("ann", [1, 0, 0, 0], top_k=3, index_mode="ivf")
    assert ivf[0]["id"] == "37"

    scalar = e.search_dense("ann", [1, 0, 0, 0], top_k=3, quantization="scalar", explain=True)
    assert scalar[0]["id"] == "37"
    assert scalar[0]["explain"]["planner"]["quantization"] == "scalar"

    binary = e.search_dense("ann", [1, 0, 0, 0], top_k=3, quantization="binary")
    assert any(x["id"] == "37" for x in binary)

    auto = e.search_dense("ann", [1, 0, 0, 0], top_k=1, explain=True)
    assert auto[0]["explain"]["planner"]["resolved_index"] == "hnsw"


def test_weighted_hybrid_and_mmr(tmp_path: Path):
    e = engine(tmp_path)
    e.create_collection("rag", dimension=2)
    e.upsert(
        "rag",
        [
            {"id": "a", "vectors": [1, 0], "text": "東京 ホテル AI", "metadata": {"kind": "hotel"}},
            {"id": "b", "vectors": [0.99, 0.01], "text": "東京 ホテル 自動化", "metadata": {"kind": "hotel"}},
            {"id": "c", "vectors": [0, 1], "text": "京都 旅館 AI", "metadata": {"kind": "ryokan"}},
        ],
    )
    weighted = e.search_hybrid(
        "rag",
        query_vector=[1, 0],
        query_text="東京 ホテル",
        top_k=2,
        fusion="weighted",
        explain=True,
    )
    assert weighted[0]["explain"]["fusion"] == "weighted"

    mmr = e.search_mmr("rag", [1, 0], top_k=2, fetch_k=3, lambda_mult=0.4, explain=True)
    assert len(mmr) == 2
    assert mmr[0]["explain"]["mode"] == "mmr"


def test_multifield_named_vector_fusion(tmp_path: Path):
    e = engine(tmp_path)
    e.create_collection(
        "multi2",
        vectors={
            "title": {"dimension": 2, "metric": "cosine"},
            "body": {"dimension": 2, "metric": "cosine"},
        },
    )
    e.upsert(
        "multi2",
        [
            {"id": "a", "vectors": {"title": [1, 0], "body": [0, 1]}, "text": "A"},
            {"id": "b", "vectors": {"title": [0.8, 0.2], "body": [1, 0]}, "text": "B"},
        ],
    )
    result = e.search_multifield(
        "multi2",
        {"title": [1, 0], "body": [1, 0]},
        weights={"body": 2.0, "title": 1.0},
        top_k=2,
        explain=True,
    )
    assert result[0]["id"] == "b"
    assert result[0]["explain"]["mode"] == "multifield"
