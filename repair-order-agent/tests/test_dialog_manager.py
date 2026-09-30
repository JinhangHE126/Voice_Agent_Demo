from __future__ import annotations

from repair_order.domain.repair_order import OrderState, RepairOrderDraft
from repair_order.repositories.order_repository import OrderRepository
from repair_order.services.dialog_manager import DialogManager
from repair_order.services.info_extractor import extract_info


def test_next_prompt_asks_language_first_when_unknown() -> None:
    manager = DialogManager()
    draft = RepairOrderDraft()
    plan = manager.next_prompt(draft)
    assert plan.prompt_id == "ask_language"
    assert plan.expected_field is None


def test_next_prompt_goes_to_confirm_when_all_fields_ready() -> None:
    manager = DialogManager()
    draft = RepairOrderDraft(
        customer_name="陈先生",
        customer_phone="91234567",
        customer_address="旺角弥敦道100号",
        repair_description="冷气不制冷",
    )
    plan = manager.next_prompt(draft)
    assert plan.prompt_id == "confirm_order"
    assert plan.dynamic is True
    assert "旺角弥敦道100号" in plan.text


def test_process_user_text_collect_confirm_save() -> None:
    repo = OrderRepository()
    manager = DialogManager(repository=repo)
    draft = RepairOrderDraft()

    turn1 = manager.process_user_text(
        call_id="call-001",
        draft=draft,
        user_text="我叫陈先生，电话9123 4567，地址旺角弥敦道100号，冷气唔冻要维修。",
    )
    assert turn1.plan.prompt_id == "confirm_order"
    assert draft.customer_name is not None
    assert draft.customer_phone == "91234567"
    assert draft.customer_address == "旺角弥敦道100号"
    assert draft.state == OrderState.CONFIRMING

    turn2 = manager.process_user_text(
        call_id="call-001",
        draft=draft,
        user_text="啱",
    )
    assert turn2.saved is True
    assert turn2.plan.prompt_id == "order_saved"
    assert draft.state == OrderState.COMPLETED
    assert len(repo.saved_orders) == 1


def test_process_user_text_deny_and_correct_phone() -> None:
    manager = DialogManager()
    draft = RepairOrderDraft(
        customer_name="陈先生",
        customer_phone="91234567",
        customer_address="旺角弥敦道100号",
        repair_description="冷气唔冻",
        state=OrderState.CONFIRMING,
    )

    turn = manager.process_user_text(
        call_id="call-002",
        draft=draft,
        user_text="唔啱，电话係 9876 5432",
    )
    assert draft.customer_phone == "98765432"
    assert turn.plan.prompt_id == "confirm_order"
    assert draft.state == OrderState.CONFIRMING


def test_confirming_deny_asks_which_field_when_unspecified() -> None:
    manager = DialogManager()
    draft = RepairOrderDraft(
        customer_name="陈先生",
        customer_phone="91234567",
        customer_address="旺角弥敦道100号",
        repair_description="冷气唔冻",
        state=OrderState.CONFIRMING,
    )
    turn = manager.process_user_text(
        call_id="call-003",
        draft=draft,
        user_text="唔啱",
    )
    assert turn.plan.prompt_id == "ask_correct_field"
    assert draft.customer_name == "陈先生"
    assert draft.customer_phone == "91234567"
    assert draft.customer_address == "旺角弥敦道100号"


def test_extract_name_requires_self_intro_context() -> None:
    text = "嗯，好嘅，明白小姐。我想問你機器點樣壞。"
    out = extract_info(text)
    assert out.customer_name is None


def test_extract_name_from_self_intro() -> None:
    text = "我叫陈先生，电话係 9123 4567。"
    out = extract_info(text)
    assert out.customer_name == "陈先生"


def test_expected_name_strips_call_me_prefix() -> None:
    out = extract_info("叫我何先生", expected_field="customer_name")
    assert out.customer_name == "何先生"


def test_expected_name_strips_conversational_suffix() -> None:
    out = extract_info("叫我胡先生就行", expected_field="customer_name")
    assert out.customer_name == "胡先生"


def test_expected_name_rejects_incomplete_call_me_phrase() -> None:
    out = extract_info("你可以叫我", expected_field="customer_name")
    assert out.customer_name is None


def test_incomplete_name_phrase_keeps_asking_for_name() -> None:
    manager = DialogManager()
    draft = RepairOrderDraft(
        customer_phone="51464523",
        customer_address="香港数码城二楼",
        repair_description="电脑主机进水",
        awaiting_field="customer_name",
    )

    turn = manager.process_user_text(
        call_id="incomplete-name",
        draft=draft,
        user_text="你可以叫我",
    )

    assert draft.customer_name is None
    assert turn.plan.prompt_id == "ask_name"


def test_expected_repair_rejects_generic_scaffold() -> None:
    out = extract_info("我维修的", expected_field="repair_description")
    assert out.repair_description is None


def test_expected_repair_accepts_actual_device_detail() -> None:
    out = extract_info("我要维修电脑主机", expected_field="repair_description")
    assert out.repair_description == "我要维修电脑主机"


def test_long_repair_description_is_summarized() -> None:
    out = extract_info(
        "你好，我今天早上电脑主机突然打不开了，"
        "可能是昨天晚上进水了，需要来修一下",
        expected_field="repair_description",
    )
    assert out.repair_description == "电脑主机打不开，进水了"


def test_extract_intent_and_fields_to_correct() -> None:
    out = extract_info("唔啱，电话要改做 9876 5432")
    assert out.intent == "deny"
    assert "customer_phone" in out.fields_to_correct
    assert out.customer_phone == "98765432"


def test_extract_address_from_phrase() -> None:
    out = extract_info("地址係旺角弥敦道100号")
    assert out.customer_address == "旺角弥敦道100号"


def test_repair_address_phrase_does_not_become_repair_content() -> None:
    out = extract_info(
        "我的维修地址是香港数码城",
        expected_field="customer_address",
    )
    assert out.customer_address == "香港数码城"
    assert out.repair_description is None


def test_address_turn_does_not_overwrite_existing_repair() -> None:
    manager = DialogManager()
    draft = RepairOrderDraft(
        customer_phone="12345678",
        repair_description="电脑主机坏了",
        awaiting_field="customer_address",
    )

    manager.process_user_text(
        call_id="address-preserves-repair",
        draft=draft,
        user_text="我的维修地址是香港数码城",
    )

    assert draft.customer_address == "香港数码城"
    assert draft.repair_description == "电脑主机坏了"


def test_expected_phone_recovers_digit_like_asr_substitution() -> None:
    out = extract_info(
        "五公二六六二三七",
        expected_field="customer_phone",
    )
    assert out.customer_phone == "59266237"


def test_expected_phone_recovers_asr_substitution_after_phone_cue() -> None:
    out = extract_info(
        "联系电话是五公二六六二三七",
        expected_field="customer_phone",
    )
    assert out.customer_phone == "59266237"


def test_phone_fuzzy_substitution_is_not_applied_to_prose() -> None:
    out = extract_info("公司电话稍后再讲")
    assert out.customer_phone is None
    assert out.phone_candidate is None


def test_multi_turn_short_answers_with_expected_field() -> None:
    repo = OrderRepository()
    manager = DialogManager(repository=repo)
    draft = RepairOrderDraft()

    greeting = manager.greeting(draft)
    assert greeting.prompt_id == "greeting"

    t1 = manager.process_user_text("c1", draft, "屋企冷气唔冻")
    assert draft.repair_description is not None
    assert t1.plan.prompt_id == "ask_phone"

    t2 = manager.process_user_text("c1", draft, "九一二三四五六七")
    assert draft.customer_phone == "91234567"
    assert t2.plan.prompt_id == "ask_address"

    t3 = manager.process_user_text("c1", draft, "旺角弥敦道100号")
    assert draft.customer_address == "旺角弥敦道100号"
    assert t3.plan.prompt_id == "ask_name"

    t4 = manager.process_user_text("c1", draft, "陈先生")
    assert draft.customer_name == "陈先生"
    assert t4.plan.prompt_id == "confirm_order"

    t5 = manager.process_user_text("c1", draft, "啱")
    assert t5.saved is True
    assert draft.state == OrderState.COMPLETED
