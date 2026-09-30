from __future__ import annotations

import pytest

from repair_order.domain.repair_order import RepairOrderDraft
from repair_order.repositories.order_repository import OrderRepository
from repair_order.services.dialog_manager import DialogManager


@pytest.mark.asyncio
async def test_process_user_text_async_short_answers() -> None:
    repo = OrderRepository()
    manager = DialogManager(repository=repo)
    draft = RepairOrderDraft()
    manager.greeting(draft)

    t1 = await manager.process_user_text_async("c1", draft, "屋企冷气唔冻")
    assert t1.plan.prompt_id == "ask_phone"

    t2 = await manager.process_user_text_async("c1", draft, "陈先生")
    assert draft.customer_name == "陈先生"
    assert t2.plan.prompt_id == "ask_phone"
