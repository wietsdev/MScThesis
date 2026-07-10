import hashlib
import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / ".cache" / "responses.db"


def _key(model_id: str, prompt: str, params: dict) -> str:
    payload = json.dumps(
        {"model": model_id, "prompt": prompt, "params": params},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, response TEXT)"
    )
    conn.commit()
    return conn


def get(model_id: str, prompt: str, params: dict) -> str | None:
    key = _key(model_id, prompt, params)
    with _connect() as conn:
        row = conn.execute(
            "SELECT response FROM responses WHERE key = ?", (key,)
        ).fetchone()
    return row[0] if row else None


def set(model_id: str, prompt: str, params: dict, response: str) -> None:
    key = _key(model_id, prompt, params)
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO responses (key, response) VALUES (?, ?)",
            (key, response),
        )
