from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repair_order.domain.repair_order import RepairOrderDraft
from repair_order.repositories.order_repository import OrderRepository
from repair_order.services.dialog_manager import DialogManager


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    call_id = "text-call-001"
    db_path = ROOT / "output" / "repair_orders.sqlite"
    repo = OrderRepository(sqlite_path=db_path)
    manager = DialogManager(repository=repo)
    draft = RepairOrderDraft()

    greeting = manager.greeting(draft)
    print(f"AI[{greeting.prompt_id}]: {greeting.text}")
    print("输入 q 退出。\n")

    while True:
        try:
            user = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if not user:
            continue
        if user.lower() in {"q", "quit", "exit"}:
            break

        turn = manager.process_user_text(call_id, draft, user)
        print(
            "草稿:",
            {
                "name": draft.customer_name,
                "phone": draft.customer_phone,
                "repair": draft.repair_description,
                "state": draft.state.value,
                "awaiting": draft.awaiting_field,
            },
        )
        print(f"AI[{turn.plan.prompt_id}]: {turn.plan.text}")
        if turn.saved:
            print(f"已保存到 SQLite: {db_path}")
            break


if __name__ == "__main__":
    main()
