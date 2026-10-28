from __future__ import annotations

import sqlite3

from repair_order.domain.repair_order import RepairOrderDraft
from repair_order.repositories.order_repository import OrderRepository


def test_sqlite_schema_is_lazy_until_first_save(tmp_path) -> None:
    db = tmp_path / "orders.sqlite"
    repo = OrderRepository(sqlite_path=db)

    assert not db.exists()

    draft = RepairOrderDraft(
        customer_name="陈先生",
        customer_phone="91234567",
        customer_address="香港湾仔测试道一号",
        repair_description="冷气坏咗",
    )
    repo.save("call-1", draft)

    assert db.exists()
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT call_id, customer_name FROM repair_orders"
        ).fetchone()
    assert row == ("call-1", "陈先生")
