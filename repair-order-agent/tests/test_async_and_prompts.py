from __future__ import annotations

import asyncio

import pytest

from repair_order.domain.repair_order import RepairOrderDraft
from repair_order.repositories.order_repository import OrderRepository
from repair_order.services.dialog_manager import DialogManager
from repair_order.services import info_extractor


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


@pytest.mark.asyncio
async def test_async_extractor_skips_llm_for_expected_phone(monkeypatch) -> None:
    called = False

    async def fake_llm_extract(text: str, *, expected_field: str | None = None):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(info_extractor, "_llm_extract_async", fake_llm_extract)

    out = await info_extractor.extract_info_async(
        "九一二三四五六七",
        expected_field="customer_phone",
    )

    assert out.customer_phone == "91234567"
    assert called is False


@pytest.mark.asyncio
async def test_async_extractor_timeout_falls_back_to_rules(monkeypatch) -> None:
    async def slow_llm_extract(text: str, *, expected_field: str | None = None):
        await asyncio.sleep(0.1)
        return {"repair": "不应返回"}

    monkeypatch.setattr(info_extractor, "_llm_extract_async", slow_llm_extract)
    monkeypatch.setattr(info_extractor, "_llm_timeout_s", lambda: 0.01)

    out = await info_extractor.extract_info_async("帮我处理一下")

    assert out.intent == "provide_info"
    assert out.repair_description is None
