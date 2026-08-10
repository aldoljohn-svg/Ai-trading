"""DB-API 2.0 connection management for SQLite (default) and PostgreSQL.

All SQL in the code base is written with ``?`` placeholders; the executor
rewrites them to ``%s`` for PostgreSQL.  A single connection guarded by a
re-entrant lock is used: the trading loop is single-writer by design and SQLite
serialises writes anyway, so a pool would add complexity without benefit.

Blocking DB work is pushed off the event loop with :func:`Database.run` which
wraps ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import urlparse

from app.database.models import TABLES_BY_NAME, schema_sql
from app.logger import get_logger

log = get_logger(__name__)

_PLACEHOLDER_RE = re.compile(r"\?")


class DatabaseError(RuntimeError):
    pass


def _now() -> int:
    return int(time.time())


class Database:
    """Thin, dialect-aware DB-API wrapper."""

    def __init__(self, url: str, root: Path | None = None) -> None:
        self.url = url
        self._root = root or Path.cwd()
        self._lock = threading.RLock()
        self._conn: Any = None
        self.dialect = "sqlite"
        self._connect()

    # -- connection -------------------------------------------------------

    def _connect(self) -> None:
        url = self.url
        if url.startswith("sqlite"):
            self.dialect = "sqlite"
            path_part = url.split("///", 1)[1] if "///" in url else url.split("://", 1)[-1]
            if path_part in {":memory:", ""}:
                self.path = ":memory:"
            else:
                path = Path(path_part)
                if not path.is_absolute():
                    path = self._root / path
                path.parent.mkdir(parents=True, exist_ok=True)
                self.path = str(path)
            self._conn = sqlite3.connect(
                self.path, check_same_thread=False, timeout=30.0
            )
            self._conn.row_factory = sqlite3.Row
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
            self._conn.commit()
        elif url.startswith("postgres"):
            self.dialect = "postgres"
            driver = None
            try:  # psycopg 3
                import psycopg  # type: ignore

                driver = psycopg
                self._conn = psycopg.connect(url, autocommit=False)
            except Exception:
                try:  # psycopg2
                    import psycopg2  # type: ignore
                    import psycopg2.extras  # type: ignore

                    driver = psycopg2
                    self._conn = psycopg2.connect(url)
                except Exception as exc:  # pragma: no cover - env dependent
                    raise DatabaseError(
                        "PostgreSQL URL configured but no psycopg/psycopg2 "
                        f"driver is installed: {exc}"
                    ) from exc
            self._driver = driver
            parsed = urlparse(url)
            self.path = f"{parsed.hostname}:{parsed.port or 5432}{parsed.path}"
        else:
            raise DatabaseError(
                f"unsupported DATABASE_URL scheme: {url!r} "
                "(expected sqlite:/// or postgresql://)"
            )

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    # -- sql helpers ------------------------------------------------------

    def _adapt(self, sql: str) -> str:
        if self.dialect == "postgres":
            return _PLACEHOLDER_RE.sub("%s", sql)
        return sql

    def _rows_to_dicts(self, cursor: Any) -> list[dict[str, Any]]:
        if cursor.description is None:
            return []
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Run a write statement; returns ``lastrowid`` when available."""

        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute(self._adapt(sql), tuple(params))
                rowid = getattr(cur, "lastrowid", None) or 0
                self._conn.commit()
                return int(rowid)
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def execute_many(self, sql: str, rows: Iterable[Sequence[Any]]) -> int:
        batch = [tuple(r) for r in rows]
        if not batch:
            return 0
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.executemany(self._adapt(sql), batch)
                self._conn.commit()
                return len(batch)
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute(self._adapt(sql), tuple(params))
                return self._rows_to_dicts(cur)
            finally:
                cur.close()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params: Sequence[Any] = (), default: Any = None) -> Any:
        row = self.query_one(sql, params)
        if not row:
            return default
        value = next(iter(row.values()))
        return default if value is None else value

    # -- upsert -----------------------------------------------------------

    def insert(self, table: str, values: dict[str, Any]) -> int:
        cols = list(values)
        placeholders = ", ".join("?" for _ in cols)
        sql = (
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
        )
        if self.dialect == "postgres":
            sql += " RETURNING id"
            rows = self.query(sql, [values[c] for c in cols])
            return int(rows[0]["id"]) if rows else 0
        return self.execute(sql, [values[c] for c in cols])

    def upsert(self, table: str, values: dict[str, Any], conflict: Sequence[str]) -> int:
        cols = list(values)
        placeholders = ", ".join("?" for _ in cols)
        updates = [c for c in cols if c not in conflict]
        set_clause = ", ".join(f"{c}=excluded.{c}" for c in updates) or None
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
        if set_clause:
            sql += (
                f" ON CONFLICT ({', '.join(conflict)}) DO UPDATE SET {set_clause}"
            )
        else:
            sql += f" ON CONFLICT ({', '.join(conflict)}) DO NOTHING"
        return self.execute(sql, [values[c] for c in cols])

    def upsert_many(
        self, table: str, rows: Sequence[dict[str, Any]], conflict: Sequence[str]
    ) -> int:
        if not rows:
            return 0
        cols = list(rows[0])
        placeholders = ", ".join("?" for _ in cols)
        updates = [c for c in cols if c not in conflict]
        set_clause = ", ".join(f"{c}=excluded.{c}" for c in updates)
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
        sql += (
            f" ON CONFLICT ({', '.join(conflict)}) DO UPDATE SET {set_clause}"
            if set_clause
            else f" ON CONFLICT ({', '.join(conflict)}) DO NOTHING"
        )
        return self.execute_many(sql, [[r.get(c) for c in cols] for r in rows])

    def update(self, table: str, values: dict[str, Any], where: dict[str, Any]) -> None:
        set_clause = ", ".join(f"{c}=?" for c in values)
        where_clause = " AND ".join(f"{c}=?" for c in where)
        sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        self.execute(sql, list(values.values()) + list(where.values()))

    # -- schema -----------------------------------------------------------

    def migrate(self) -> None:
        """Create tables and indexes, and add any missing columns."""

        with self._lock:
            cur = self._conn.cursor()
            try:
                for statement in schema_sql(self.dialect):
                    cur.execute(statement)
                self._conn.commit()
            finally:
                cur.close()
        self._add_missing_columns()
        log.info("database ready (%s: %s)", self.dialect, self.path)

    def _existing_columns(self, table: str) -> set[str]:
        if self.dialect == "sqlite":
            rows = self.query(f"PRAGMA table_info({table})")
            return {str(r["name"]) for r in rows}
        rows = self.query(
            "SELECT column_name AS name FROM information_schema.columns "
            "WHERE table_name = ?",
            (table,),
        )
        return {str(r["name"]) for r in rows}

    def _add_missing_columns(self) -> None:
        for name, spec in TABLES_BY_NAME.items():
            existing = self._existing_columns(name)
            if not existing:
                continue
            for column in spec.columns:
                if column.type == "PK" or column.name in existing:
                    continue
                ddl = column.ddl(self.dialect)
                # A NOT NULL column cannot be added without a default.
                if not column.null and column.default is None:
                    ddl = ddl.replace(" NOT NULL", "")
                log.info("migrating: adding %s.%s", name, column.name)
                self.execute(f"ALTER TABLE {name} ADD COLUMN {ddl}")

    # -- async ------------------------------------------------------------

    async def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a blocking DB call in a worker thread."""

        return await asyncio.to_thread(fn, *args, **kwargs)

    # -- maintenance ------------------------------------------------------

    def vacuum(self) -> None:
        if self.dialect == "sqlite":
            with self._lock:
                self._conn.execute("VACUUM")
                self._conn.commit()

    def prune_candles(self, keep_per_symbol: int = 2000) -> int:
        """Bound candle storage so a long-running VPS deployment stays small."""

        if self.dialect != "sqlite":
            return 0
        removed = self.execute(
            """
            DELETE FROM candles WHERE id IN (
                SELECT id FROM (
                    SELECT id, ROW_NUMBER() OVER (
                        PARTITION BY symbol, timeframe ORDER BY ts DESC
                    ) AS rn FROM candles
                ) WHERE rn > ?
            )
            """,
            (keep_per_symbol,),
        )
        return removed

    def health(self) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            self.scalar("SELECT 1")
            latency_ms = (time.perf_counter() - started) * 1000
            return {"ok": True, "latency_ms": round(latency_ms, 2), "dialect": self.dialect}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "dialect": self.dialect}


def dumps(value: Any) -> str:
    """JSON encode for a ``JSON`` column, tolerating non-serialisable values."""

    return json.dumps(value, default=str, separators=(",", ":"))


def loads(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


_database: Database | None = None


def get_database(url: str | None = None, root: Path | None = None) -> Database:
    global _database
    if _database is None:
        if url is None:
            from app.config import get_settings

            settings = get_settings()
            url = settings.database_url
            root = root or settings.resolve_path(".")
        _database = Database(url, root=root)
        _database.migrate()
    return _database


def set_database(database: Database | None) -> None:
    global _database
    _database = database


__all__ = [
    "Database",
    "DatabaseError",
    "get_database",
    "set_database",
    "dumps",
    "loads",
]
