from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class StoredPoint:
    collection: str
    id: str
    namespace: str
    text: str
    metadata: dict[str, Any]
    vectors: dict[str, list[float]]
    created_at: str
    updated_at: str


class SQLiteStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS collections (
                name TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS points (
                collection TEXT NOT NULL,
                id TEXT NOT NULL,
                namespace TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                vectors_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (collection, namespace, id),
                FOREIGN KEY(collection) REFERENCES collections(name) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_points_collection_namespace
              ON points(collection, namespace);
            """
        )
        self._conn.commit()

    def create_collection(self, name: str, config: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO collections(name, config_json, created_at) VALUES(?,?,?)",
                (name, json.dumps(config, ensure_ascii=False), _now()),
            )
            self._conn.commit()

    def upsert_collection(self, name: str, config: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO collections(name, config_json, created_at) VALUES(?,?,?)
                   ON CONFLICT(name) DO UPDATE SET config_json=excluded.config_json""",
                (name, json.dumps(config, ensure_ascii=False), _now()),
            )
            self._conn.commit()

    def get_collection(self, name: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT name, config_json, created_at FROM collections WHERE name=?", (name,)
        ).fetchone()
        if row is None:
            return None
        return {"name": row["name"], "config": json.loads(row["config_json"]), "created_at": row["created_at"]}

    def list_collections(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT name, config_json, created_at FROM collections ORDER BY name"
        ).fetchall()
        return [
            {"name": r["name"], "config": json.loads(r["config_json"]), "created_at": r["created_at"]}
            for r in rows
        ]

    def drop_collection(self, name: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM points WHERE collection=?", (name,))
            self._conn.execute("DELETE FROM collections WHERE name=?", (name,))
            self._conn.commit()

    def upsert_points(self, collection: str, points: list[dict[str, Any]]) -> None:
        now = _now()
        with self._lock:
            for p in points:
                namespace = p.get("namespace") or ""
                existing = self._conn.execute(
                    "SELECT created_at FROM points WHERE collection=? AND namespace=? AND id=?",
                    (collection, namespace, p["id"]),
                ).fetchone()
                created = existing["created_at"] if existing else now
                self._conn.execute(
                    """INSERT INTO points(collection,id,namespace,text,metadata_json,vectors_json,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?)
                       ON CONFLICT(collection,namespace,id) DO UPDATE SET
                         text=excluded.text,
                         metadata_json=excluded.metadata_json,
                         vectors_json=excluded.vectors_json,
                         updated_at=excluded.updated_at""",
                    (
                        collection,
                        p["id"],
                        namespace,
                        p.get("text", ""),
                        json.dumps(p.get("metadata", {}), ensure_ascii=False),
                        json.dumps(p["vectors"], ensure_ascii=False),
                        created,
                        now,
                    ),
                )
            self._conn.commit()

    def list_points(self, collection: str, namespace: str | None = None) -> list[StoredPoint]:
        if namespace is None:
            rows = self._conn.execute(
                "SELECT * FROM points WHERE collection=? ORDER BY id", (collection,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM points WHERE collection=? AND namespace=? ORDER BY id",
                (collection, namespace),
            ).fetchall()
        return [
            StoredPoint(
                collection=r["collection"],
                id=r["id"],
                namespace=r["namespace"],
                text=r["text"],
                metadata=json.loads(r["metadata_json"]),
                vectors=json.loads(r["vectors_json"]),
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def point_exists(self, collection: str, point_id: str, namespace: str = "") -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM points WHERE collection=? AND namespace=? AND id=?",
            (collection, namespace, point_id),
        ).fetchone()
        return row is not None

    def delete_ids(self, collection: str, ids: list[str], namespace: str | None = None) -> int:
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        params: list[Any] = [collection]
        sql = f"DELETE FROM points WHERE collection=? AND id IN ({placeholders})"
        params.extend(ids)
        if namespace is not None:
            sql += " AND namespace=?"
            params.append(namespace)
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return int(cur.rowcount)
