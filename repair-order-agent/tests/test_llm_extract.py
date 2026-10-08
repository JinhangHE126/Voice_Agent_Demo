from __future__ import annotations

import asyncio

from repair_order.services import info_extractor as ie
from repair_order.services.info_extractor import ExtractResult, extract_info, extract_info_async


def test_apply_llm_obj_only_accepts_repair_and_address() -> None:
    result = ie._apply_llm_obj(
        ExtractResult(),
        {
            "name": "陈先生",
            "phone": "91234567",
            "customer_address": "旺角弥敦道100号",
            "address_evidence": ["旺角弥敦道100号"],
            "repair_description": "电脑主机进水",
            "repair_evidence": ["进水"],
            "intent": "confirm",
            "confidence": {"repair": 0.9, "address": 0.8, "name": 0.9},
        },
        "电脑主机进水，地址係旺角弥敦道100号",
        expected_field="repair_description",
    )
    assert result.customer_name is None
    assert result.customer_phone is None
    assert result.intent == "other"
    assert result.customer_address == "旺角弥敦道100号"
    assert result.repair_description == "电脑主机进水"
    assert result.confidence["repair"] == 0.9
    assert "name" not in result.confidence


def test_apply_llm_obj_rejects_ungrounded_address() -> None:
    result = ie._apply_llm_obj(
        ExtractResult(),
        {
            "customer_address": "尖沙咀北京道88号",
            "address_evidence": ["尖沙咀北京道88号"],
            "repair_description": None,
        },
        "地址係旺角弥敦道100号",
        expected_field="customer_address",
    )
    assert result.customer_address is None


def test_rules_skip_llm_for_name_and_phone(monkeypatch) -> None:
    called = {"n": 0}

    def _fake_llm(text: str, *, expected_field: str | None = None):
        called["n"] += 1
        return {"repair_description": "不应调用"}

    monkeypatch.setattr(ie, "_llm_extract", _fake_llm)
    monkeypatch.setenv("REPAIR_ORDER_USE_LLM_EXTRACTOR", "1")

    name_out = extract_info("叫我何先生", expected_field="customer_name")
    phone_out = extract_info("91234567", expected_field="customer_phone")

    assert name_out.customer_name == "何先生"
    assert phone_out.customer_phone == "91234567"
    assert called["n"] == 0


def test_repair_field_prefers_llm_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("REPAIR_ORDER_USE_LLM_EXTRACTOR", "1")
    monkeypatch.setattr(
        ie,
        "_llm_extract",
        lambda text, *, expected_field=None: {
            "repair_description": "电脑主机进水，冒烟",
            "repair_evidence": ["进水", "冒烟"],
            "customer_address": None,
            "confidence": {"repair": 0.92},
        },
    )

    out = extract_info(
        "喂你好，我电脑主机不小心进水了，还有点冒烟",
        expected_field="repair_description",
    )
    assert out.repair_description == "电脑主机进水，冒烟"
    assert out.confidence.get("repair") == 0.92


def test_llm_timeout_falls_back_to_rules(monkeypatch) -> None:
    monkeypatch.setenv("REPAIR_ORDER_USE_LLM_EXTRACTOR", "1")
    monkeypatch.setenv("REPAIR_ORDER_EXTRACT_TIMEOUT_S", "0.5")

    async def _slow(*_args, **_kwargs):
        await asyncio.sleep(2.0)
        return {"repair_description": "不应出现"}

    monkeypatch.setattr(ie, "_llm_extract_async", _slow)

    out = asyncio.run(
        extract_info_async(
            "我要维修电脑主机",
            expected_field="repair_description",
        )
    )
    assert out.repair_description == "我要维修电脑主机"


def test_redact_phone_before_llm() -> None:
    redacted = ie._redact_for_llm("电话是9123-4567，电脑坏了")
    assert "9123" not in redacted
    assert "<PHONE>" in redacted
    assert "电脑坏了" in redacted


def test_confirming_short_reply_skips_llm(monkeypatch) -> None:
    called = {"n": 0}

    def _fake_llm(text: str, *, expected_field: str | None = None):
        called["n"] += 1
        return {"repair_description": "不应调用"}

    monkeypatch.setattr(ie, "_llm_extract", _fake_llm)
    monkeypatch.setenv("REPAIR_ORDER_USE_LLM_EXTRACTOR", "1")

    out = extract_info("可以", expected_field=None)
    assert out.is_confirm is True
    assert called["n"] == 0


def test_confirming_unknown_short_reply_still_skips_llm(monkeypatch) -> None:
    called = {"n": 0}

    def _fake_llm(text: str, *, expected_field: str | None = None):
        called["n"] += 1
        return {"repair_description": "不应调用"}

    monkeypatch.setattr(ie, "_llm_extract", _fake_llm)
    monkeypatch.setenv("REPAIR_ORDER_USE_LLM_EXTRACTOR", "1")

    out = extract_info("啊啊", expected_field=None)
    assert out.is_confirm is False
    assert called["n"] == 0


def test_confirming_address_correction_still_calls_llm(monkeypatch) -> None:
    called = {"n": 0}

    def _fake_llm(text: str, *, expected_field: str | None = None):
        called["n"] += 1
        return {
            "customer_address": "旺角弥敦道100号",
            "address_evidence": ["旺角弥敦道100号"],
            "repair_description": None,
        }

    monkeypatch.setattr(ie, "_llm_extract", _fake_llm)
    monkeypatch.setenv("REPAIR_ORDER_USE_LLM_EXTRACTOR", "1")

    out = extract_info("地址改成旺角弥敦道100号", expected_field=None)
    assert called["n"] == 1
    assert out.customer_address == "旺角弥敦道100号"


def test_low_information_repair_turn_skips_llm(monkeypatch) -> None:
    called = {"n": 0}

    def _fake_llm(text: str, *, expected_field: str | None = None):
        called["n"] += 1
        return {"repair_description": "不应调用"}

    monkeypatch.setattr(ie, "_llm_extract", _fake_llm)
    monkeypatch.setenv("REPAIR_ORDER_USE_LLM_EXTRACTOR", "1")

    for text in ("喂。", "Oi.", "嗯，说话。"):
        out = extract_info(text, expected_field="repair_description")
        assert out.repair_description is None
    assert called["n"] == 0
