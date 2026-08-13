import importlib


def test_api(tmp_path, monkeypatch):
    monkeypatch.setenv("ROOOMTECH_VECTOR_DB_PATH", str(tmp_path / "api.db"))
    import rooomtech_vector.api as api
    importlib.reload(api)
    from fastapi.testclient import TestClient

    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok"}
    caps = client.get("/v1/capabilities").json()
    assert caps["hybrid_search"] is True
    assert caps["hnsw"] is True
    assert caps["ivf_flat"] is True

    r = client.put("/v1/collections/demo", json={"dimension": 2, "metric": "cosine"})
    assert r.status_code == 200
    r = client.post(
        "/v1/collections/demo/points",
        json={"points": [{"id": "1", "vectors": [1, 0], "text": "vector database", "metadata": {"source": "test"}}]},
    )
    assert r.json()["upserted"] == 1
    r = client.post(
        "/v1/collections/demo/search/hybrid",
        json={"vector": [1, 0], "query": "database", "top_k": 1, "explain": True},
    )
    assert r.status_code == 200
    assert r.json()["results"][0]["id"] == "1"
