"""Share the password-attempt window across workers on one backend instance."""

import hashlib
import math
import sqlite3
import time
from contextlib import closing

LIMIT = 30
WINDOW_SECONDS = 300


def reserve_attempt(path, email, client_ip):
    """Return (reservation, retry_after). Pending requests count until resolved."""
    bucket = hashlib.sha256(f"{email.casefold()}\0{client_ip}".encode()).digest()
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "CREATE TABLE IF NOT EXISTS attempts "
            "(reservation TEXT PRIMARY KEY, bucket BLOB NOT NULL, expires REAL NOT NULL)"
        )
        db.execute("CREATE INDEX IF NOT EXISTS attempts_bucket ON attempts (bucket)")
        db.execute("CREATE INDEX IF NOT EXISTS attempts_expiry ON attempts (expires)")
        now = time.time()
        db.execute("DELETE FROM attempts WHERE expires <= ?", (now,))
        count, earliest = db.execute(
            "SELECT count(*), min(expires) FROM attempts WHERE bucket = ?", (bucket,)
        ).fetchone()
        if count >= LIMIT:
            return None, max(1, math.ceil(earliest - now))
        reservation = db.execute("SELECT lower(hex(randomblob(16)))").fetchone()[0]
        db.execute("INSERT INTO attempts VALUES (?, ?, ?)", (reservation, bucket, now + WINDOW_SECONDS))
        return reservation, None


def release_attempt(path, reservation):
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("DELETE FROM attempts WHERE reservation = ?", (reservation,))
