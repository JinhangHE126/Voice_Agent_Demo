from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from repair_order.domain.repair_order import RepairOrderDraft


class OrderRepository:
    """In-memory + optional SQLite persistence."""

    def __init__(self, sqlite_path: Path | None = None) -> None:
        self.saved_orders: list[dict] = []
        self.sqlite_path = sqlite_path
        if sqlite_path is not None:
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(sqlite_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS repair_orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        call_id TEXT NOT NULL,
                        customer_name TEXT,
                        customer_phone TEXT,
                        customer_address TEXT,
                        repair_description TEXT,
                        payload_json TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cols = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(repair_orders)").fetchall()
                }
                if "customer_address" not in cols:
                    conn.execute(
                        "ALTER TABLE repair_orders ADD COLUMN customer_address TEXT"
                    )
                conn.commit()

    def save(self, call_id: str, draft: RepairOrderDraft) -> dict:
        record = {
            "call_id": call_id,
            "order": asdict(draft),
        }
        # Enum values are not JSON-serializable by default in asdict path for sqlite.
        order_payload = {
            "customer_name": draft.customer_name,
            "customer_phone": draft.customer_phone,
            "customer_address": draft.customer_address,
            "repair_description": draft.repair_description,
            "confirmed": draft.confirmed,
            "state": draft.state.value,
        }
        record["order"] = order_payload
        self.saved_orders.append(record)

        if self.sqlite_path is not None:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute(
                    """
                    INSERT INTO repair_orders
                    (call_id, customer_name, customer_phone, customer_address,
                     repair_description, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        call_id,
                        draft.customer_name,
                        draft.customer_phone,
                        draft.customer_address,
                        draft.repair_description,
                        json.dumps(order_payload, ensure_ascii=False),
                    ),
                )
                conn.commit()
        return record
