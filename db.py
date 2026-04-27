"""Database connection helper for the dehydration risk report.

Reads DATABASE_URL from .env in the same directory.
Provides get_connection() returning a psycopg2 connection to VDP Postgres.

Usage:
    conn = db.get_connection()
    df = pd.read_sql(query, conn)
    conn.close()

The Fly.io proxy tunnel must be running before connecting:
    fly proxy 16380:5432 -a <app-name>
"""

import os
from pathlib import Path
from urllib.parse import urlparse, unquote

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)


def _parse_db_params() -> dict:
    """Parse DATABASE_URL into psycopg2.connect keyword arguments."""
    url = os.getenv("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. "
            "Copy .env.example to .env and fill in the VDP connection string."
        )

    parsed = urlparse(url)
    qs = dict(pair.split("=", 1) for pair in (parsed.query or "").split("&") if "=" in pair)

    return {
        "host": parsed.hostname or "127.0.0.1",
        "port": parsed.port or 5432,
        "database": (parsed.path or "/").lstrip("/"),
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "connect_timeout": 15,
        "sslmode": qs.get("sslmode", "prefer"),
    }


def get_connection():
    """Return an open psycopg2 connection to VDP Postgres.

    Raises RuntimeError if DATABASE_URL is not configured.
    Raises psycopg2.OperationalError if the Fly tunnel is not running.
    """
    import psycopg2  # imported lazily so missing dep gives a clear error

    params = _parse_db_params()
    return psycopg2.connect(**params)
