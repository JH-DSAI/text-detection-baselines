"""On-disk cache of what the expensive detectors have scored.

A detector whose scoring is slow or paid for -- a local LLM, a remote endpoint --
stores each result here as soon as it has it, keyed by everything that determined
it, and looks there before scoring anything. A rerun therefore skips whatever an
earlier run scored, including a run that failed or was interrupted partway, which
makes the cache the checkpoint too.

A detector stores its raw result, such as a probability or an endpoint's verdict
records, rather than its :class:`~.base.ModelOutput`. The cheap steps that turn the
one into the other, such as OOD flagging, run again on every read, so a change to
them takes effect without discarding what is cached.

The cache is a single SQLite file, so concurrent runs can share it, and each write
is committed at once.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..settings import ENV_PREFIX, Settings, user_cache_dir

#: Environment variable that overrides where the cache is kept. It populates
#: :attr:`~text_detection_baselines.settings.Settings.score_cache_path`, so it can
#: also be set in ``.env``.
SCORE_CACHE_PATH_ENV_VAR = f"{ENV_PREFIX}SCORE_CACHE_PATH"

#: File name of the cache in the user cache directory.
SCORE_CACHE_FILENAME = "scores.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (namespace, key)
)
"""


def resolve_score_cache_path() -> Path:
    """Return where the score cache lives: :data:`SCORE_CACHE_PATH_ENV_VAR`, else the user cache directory."""
    override = Settings().score_cache_path
    if override is not None:
        return override.expanduser()
    return user_cache_dir() / SCORE_CACHE_FILENAME


def _digest(key: Mapping[str, Any]) -> str:
    """Hash *key* so that equal mappings, in any order, give the same digest."""
    canonical = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ScoreCache:
    """A persistent store of JSON values, keyed by what produced them.

    Each entry belongs to a *namespace*, one per kind of result a detector
    stores. The *key* must name everything that determined the value, such as
    the model and its version, the settings that affect its output, and the
    input itself, since nothing else invalidates an entry. The key is stored
    only as a hash, so the inputs it names are not in the file; the values
    detectors store are.

    The file is created on the first read or write, not on construction, so a
    run whose detectors cache nothing leaves nothing behind.
    """

    def __init__(self, path: Path, *, refresh: bool = False) -> None:
        """Initialize the cache.

        Args:
            path:    The SQLite file. It and its parent directories are created
                as needed.
            refresh: Whether to ignore what is stored. Every lookup then misses,
                and what is scored instead overwrites the stored entry.
        """
        self.path = path
        self.refresh = refresh
        #: Number of entries read so far.
        self.hits = 0
        #: Number of entries written so far.
        self.writes = 0
        self._connection: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Autocommit, so that each entry is saved the moment it is written;
            # the timeout lets concurrent runs wait out one another's writes.
            connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
            connection.execute(_SCHEMA)
            self._connection = connection
        return self._connection

    def get(self, namespace: str, key: Mapping[str, Any]) -> Any | None:
        """Return the value stored under *key*, or None if there is none or the cache is refreshing."""
        if self.refresh:
            return None
        row = (
            self._connect()
            .execute("SELECT value FROM entries WHERE namespace = ? AND key = ?", (namespace, _digest(key)))
            .fetchone()
        )
        if row is None:
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, namespace: str, key: Mapping[str, Any], value: Any) -> None:
        """Store *value*, which must be JSON-serializable and not None, under *key*."""
        self._connect().execute(
            "INSERT OR REPLACE INTO entries (namespace, key, value, created_at) VALUES (?, ?, ?, ?)",
            (namespace, _digest(key), json.dumps(value), datetime.now(timezone.utc).isoformat()),
        )
        self.writes += 1

    def close(self) -> None:
        """Close the file, if it was opened. The cache reopens it if used again."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
