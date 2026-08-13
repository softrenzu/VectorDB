from __future__ import annotations

from pathlib import Path
from typing import Any

from .engine_data import EngineDataMixin
from .engine_dense import EngineDenseMixin
from .engine_hybrid import EngineHybridMixin
from .engine_indexing import EngineIndexMixin
from .storage import SQLiteStore, StoredPoint


class VectorEngine(EngineDataMixin, EngineIndexMixin, EngineDenseMixin, EngineHybridMixin):
    """RAG-oriented vector DB engine with adaptive exact/ANN retrieval."""

    def __init__(self, db_path: str | Path = "data/rooomtech_vector.db"):
        self.store = SQLiteStore(db_path)
        self._index_cache: dict[tuple[Any, ...], tuple[list[StoredPoint], Any]] = {}
