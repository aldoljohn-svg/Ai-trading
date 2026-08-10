"""Persistence layer.

The default backend is SQLite via the standard library ``sqlite3`` module; the
same schema and the same SQL run on PostgreSQL through any DB-API 2.0 driver
(``psycopg`` or ``psycopg2``).  Working directly against DB-API keeps the
default deployment dependency-free and keeps one single source of truth for the
schema (:mod:`app.database.models`).
"""

from app.database.database import Database, get_database, set_database
from app.database.models import TABLES, TableSpec

__all__ = ["Database", "get_database", "set_database", "TABLES", "TableSpec"]
