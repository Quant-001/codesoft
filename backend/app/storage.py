import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("FRAUD_DB_PATH", BASE_DIR / "data" / "fraud_ops.sqlite3"))


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                amount REAL NOT NULL,
                merchant TEXT NOT NULL,
                category TEXT NOT NULL,
                city TEXT NOT NULL,
                state TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                label TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                model_source TEXT NOT NULL,
                model_version TEXT NOT NULL,
                threshold REAL NOT NULL,
                analyst_decision TEXT NOT NULL DEFAULT 'Pending',
                decision_note TEXT,
                decided_at TEXT,
                request_payload TEXT NOT NULL,
                risk_factors TEXT NOT NULL,
                explanations TEXT NOT NULL
            )
            """
        )


def save_prediction(transaction: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO transactions (
                created_at, amount, merchant, category, city, state, prediction, label,
                fraud_probability, risk_level, model_source, model_version, threshold,
                request_payload, risk_factors, explanations
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                float(transaction["amt"]),
                str(transaction["merchant"]),
                str(transaction["category"]),
                str(transaction["city"]),
                str(transaction["state"]),
                int(result["prediction"]),
                str(result["label"]),
                float(result["fraud_probability"]),
                str(result["risk_level"]),
                str(result["model_source"]),
                str(result["model_version"]),
                float(result["threshold"]),
                json.dumps(transaction),
                json.dumps(result["risk_factors"]),
                json.dumps(result["explanations"]),
            ),
        )
        row_id = cursor.lastrowid

    return get_transaction(int(row_id))


def list_transactions(limit: int = 25) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM transactions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [serialize_row(row) for row in rows]


def get_transaction(transaction_id: int) -> dict[str, Any]:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM transactions WHERE id = ?",
            (transaction_id,),
        ).fetchone()

    if row is None:
        raise KeyError(transaction_id)
    return serialize_row(row)


def update_decision(
    transaction_id: int, decision: str, note: str | None = None
) -> dict[str, Any]:
    decided_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE transactions
            SET analyst_decision = ?, decision_note = ?, decided_at = ?
            WHERE id = ?
            """,
            (decision, note, decided_at, transaction_id),
        )

    if cursor.rowcount == 0:
        raise KeyError(transaction_id)
    return get_transaction(transaction_id)


def serialize_row(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    record["request_payload"] = json.loads(record["request_payload"])
    record["risk_factors"] = json.loads(record["risk_factors"])
    record["explanations"] = json.loads(record["explanations"])
    return record
