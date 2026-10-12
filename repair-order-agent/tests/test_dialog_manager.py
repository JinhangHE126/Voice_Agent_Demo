from __future__ import annotations

from repair_order.domain.repair_order import OrderState, RepairOrderDraft
from repair_order.repositories.order_repository import OrderRepository
from repair_order.services.dialog_manager import DialogManager
from repair_order.services.info_extractor import extract_info


def test_next_prompt_asks_name_first() -> None:
    manager = DialogManager()
    draft = RepairOrderDraft()
    plan = manager.next_prompt(draft)
    assert plan.prompt_id == "ask_name"
    assert plan.expected_field == "customer_name"


def test_next_prompt_goes_to_confirm_when_all_fields_ready() -> None:
    manager = DialogManager()
    draft = RepairOrderDraft(
        customer_name="陈先生",
        customer_phone="91234567",
        customer_address="旺角弥敦道100号",
        repair_description="冷气不制冷",
        repair_confirmed=True,
        address_confirmed=True,
    )
    plan = manager.next_prompt(draft)
    assert plan.prompt_id == "confirm_order"
    assert plan.dynamic is True
    assert "旺角弥敦道100号" in plan.text


def test_empty_input_during_confirmation_uses_short_local_prompt() -> None:
    manager = DialogManager()

    for state in (
        OrderState.CONFIRMING_REPAIR,
        OrderState.CONFIRMING_ADDRESS,
        OrderState.CONFIRMING_ORDER,
    ):
        draft = RepairOrderDraft(state=state)
        plan = manager.empty_input_reprompt(draft)

        assert plan.prompt_id == "ask_confirm_yesno"
        assert plan.dynamic is False


def test_process_user_text_collect_confirm_save() -> None:
    repo = OrderRepository()
    manager = DialogManager(repository=repo)
    draft = RepairOrderDraft()

    turn1 = manager.process_user_text(
        call_id="call-001",
        draft=draft,
        user_text="我叫陈先生，电话9123 4567，地址旺角弥敦道100号，冷气唔冻要维修。",
    )
    assert turn1.plan.prompt_id == "confirm_repair"
    assert draft.customer_name is not None
    assert draft.customer_phone == "91234567"
    assert draft.customer_address == "旺角弥敦道100号"
    assert draft.state == OrderState.CONFIRMING_REPAIR

    turn2 = manager.process_user_text(
        call_id="call-001",
        draft=draft,
        user_text="啱",
    )
    assert turn2.saved is False
    assert turn2.plan.prompt_id == "confirm_address"
    assert draft.state == OrderState.CONFIRMING_ADDRESS
    assert draft.repair_confirmed is True

    turn3 = manager.process_user_text(
        call_id="call-001",
        draft=draft,
        user_text="啱",
    )
    assert turn3.saved is False
    assert turn3.plan.prompt_id == "confirm_order"
    assert draft.state == OrderState.CONFIRMING_ORDER
    assert draft.address_confirmed is True

    turn4 = manager.process_user_text(
        call_id="call-001",
        draft=draft,
        user_text="啱",
    )
    assert turn4.saved is True
    assert turn4.plan.prompt_id == "order_saved"
    assert draft.state == OrderState.COMPLETED
    assert len(repo.saved_orders) == 1


def test_process_user_text_deny_and_correct_phone() -> None:
    manager = DialogManager()
    draft = RepairOrderDraft(
        customer_name="陈先生",
        customer_phone="91234567",
        customer_address="旺角弥敦道100号",
        repair_description="冷气唔冻",
        repair_confirmed=True,
        address_confirmed=True,
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
        repair_confirmed=True,
        address_confirmed=True,
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


def test_repair_denial_clears_only_repair_and_collects_it_again() -> None:
    manager = DialogManager()
    draft = RepairOrderDraft(
        customer_name="陈先生",
        repair_description="冷气唔冻",
        state=OrderState.CONFIRMING_REPAIR,
    )

    turn = manager.process_user_text("repair-deny", draft, "唔啱")

    assert draft.customer_name == "陈先生"
    assert draft.repair_description is None
    assert draft.repair_confirmed is False
    assert turn.plan.prompt_id == "ask_repair"


def test_address_denial_clears_only_address_and_collects_it_again() -> None:
    manager = DialogManager()
    draft = RepairOrderDraft(
        customer_name="陈先生",
        customer_phone="91234567",
        customer_address="旺角弥敦道100号",
        repair_description="冷气唔冻",
        repair_confirmed=True,
        state=OrderState.CONFIRMING_ADDRESS,
    )

    turn = manager.process_user_text("address-deny", draft, "唔啱")

    assert draft.customer_name == "陈先生"
    assert draft.customer_phone == "91234567"
    assert draft.customer_address is None
    assert draft.address_confirmed is False
    assert draft.repair_confirmed is True
    assert turn.plan.prompt_id == "ask_address"
    assert "九龙" in turn.plan.text or "九龍" in turn.plan.text or "香港" in turn.plan.text


def test_extract_name_requires_self_intro_context() -> None:
    text = "嗯，好嘅，明白小姐。我想問你機器點樣壞。"
    out = extract_info(text)
    assert out.customer_name is None


def test_extract_name_from_self_intro() -> None:
    text = "我叫陈先生，电话係 9123 4567。"
    out = extract_info(text)
    assert out.customer_name == "陈先生"


def test_expected_name_accepts_spoken_single_surname() -> None:
    assert (
        extract_info(
            "哎，你好，我 姓何。",
            expected_field="customer_name",
        ).customer_name
        == "何"
    )
    assert (
        extract_info(
            "哎，我姓何。",
            expected_field="customer_name",
        ).customer_name
        == "何"
    )


def test_spoken_single_surname_asks_preferred_title() -> None:
    manager = DialogManager()
    draft = RepairOrderDraft(awaiting_field="customer_name")

    turn = manager.process_user_text(
        "surname",
        draft,
        "哎，你好，我 姓何。",
    )

    assert draft.customer_name == "何"
    assert draft.customer_name_needs_title is True
    assert turn.plan.prompt_id == "ask_name_title"
    assert turn.plan.expected_field == "customer_name_title"


def test_standalone_title_completes_spoken_surname() -> None:
    manager = DialogManager()
    draft = RepairOrderDraft(
        customer_name="何",
        customer_name_needs_title=True,
        awaiting_field="customer_name_title",
    )

    turn = manager.process_user_text("title", draft, "先生就可以了")

    assert draft.customer_name == "何先生"
    assert draft.customer_name_needs_title is False
    assert turn.plan.prompt_id == "ask_repair"


def test_expected_name_strips_call_me_prefix() -> None:
    out = extract_info("叫我何先生", expected_field="customer_name")
    assert out.customer_name == "何先生"


def test_expected_name_strips_conversational_suffix() -> None:
    out = extract_info("叫我胡先生就行", expected_field="customer_name")
    assert out.customer_name == "胡先生"


def test_expected_name_strips_asr_conversational_tail() -> None:
    out = extract_info("你叫我何先生著好了", expected_field="customer_name")
    assert out.customer_name == "何先生"


def test_name_extraction_prefers_complete_titled_self_introduction() -> None:
    out = extract_info(
        "哎，我姓何，您叫我何先生就行。",
        expected_field="customer_name",
    )
    assert out.customer_name == "何先生"


def test_address_sentence_is_not_extracted_as_customer_name() -> None:
    out = extract_info(
        "我是九龙半岛油尖旺区佐敦文乐街文苑楼九幺幺B室。",
        expected_field="customer_address",
    )
    assert out.customer_name is None
    assert out.customer_address is not None


def test_expected_name_rejects_greeting_and_laughter_noise() -> None:
    out = extract_info("好，呵呵吸", expected_field="customer_name")
    assert out.customer_name is None


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


def test_expected_repair_rejects_greeting_only() -> None:
    out = extract_info("喂，你好", expected_field="repair_description")
    assert out.repair_description is None


def test_repair_strips_leading_greeting() -> None:
    out = extract_info(
        "喂，你好，我电脑主机坏了",
        expected_field="repair_description",
    )
    assert out.repair_description == "我电脑主机坏了"


def test_long_repair_description_is_summarized() -> None:
    out = extract_info(
        "你好，我今天早上电脑主机突然打不开了，"
        "可能是昨天晚上进水了，需要来修一下",
        expected_field="repair_description",
    )
    assert out.repair_description == "电脑主机打不开，进水了"


def test_repair_summary_keeps_cause_hazard_and_current_state() -> None:
    out = extract_info(
        "喂，你好，我电脑主机坏掉了，它昨晚不小心进了一点水，"
        "然后还冒烟了，我就不敢动，然后那个今天就打不开了。",
        expected_field="repair_description",
    )
    assert out.repair_description == "电脑主机坏掉了，进水，冒烟了，打不开"


def test_extract_intent_and_fields_to_correct() -> None:
    out = extract_info("唔啱，电话要改做 9876 5432")
    assert out.intent == "deny"
    assert "customer_phone" in out.fields_to_correct
    assert out.customer_phone == "98765432"


def test_extract_address_from_phrase() -> None:
    out = extract_info("地址係旺角弥敦道100号")
    assert out.customer_address == "旺角弥敦道100号"
    assert out.is_confirm is False
    assert out.intent == "provide_info"


def test_address_strips_spoken_residence_prefix() -> None:
    out = extract_info(
        "哎好，我住九龙半岛油尖旺区佐敦文苑街文苑楼九座九幺幺D室。",
        expected_field="customer_address",
    )
    assert out.customer_address is not None
    assert not out.customer_address.startswith("哎")
    assert "我住" not in out.customer_address
    assert "九龙半岛油尖旺区佐敦文苑街文苑楼" in out.customer_address


def test_address_strips_wo_shi_zai_prefix() -> None:
    out = extract_info(
        "对，我是在香港九龙半岛油尖旺区佐敦吴苑街吴苑楼。",
        expected_field="confirmation",
    )
    assert out.customer_address is not None
    assert not out.customer_address.startswith("我是")
    assert "香港九龙半岛油尖旺区佐敦吴苑街吴苑楼" in out.customer_address


def test_address_strips_wo_men_zai_prefix() -> None:
    out = extract_info(
        "我们在香港九龙半岛酒店望去。",
        expected_field="customer_address",
    )
    assert out.customer_address is not None
    assert not out.customer_address.startswith("我们在")
    assert "香港九龙半岛" in out.customer_address


def test_repair_strips_leading_hao_filler() -> None:
    out = extract_info(
        "好，我电脑主机坏了，需要去清洗",
        expected_field="repair_description",
    )
    assert out.repair_description is not None
    assert not out.repair_description.startswith("好")
    assert "电脑主机坏了" in out.repair_description


def test_repair_keeps_water_and_cleaning_details() -> None:
    out = extract_info(
        "好，我的电脑主机坏了，需要去清洗一下灰尘，它昨天进了很多水。",
        expected_field="repair_description",
    )
    repair = out.repair_description or ""
    assert "电脑主机" in repair or "主机" in repair
    assert "进水" in repair or "进了很多水" in repair
    assert "清洗" in repair or "灰尘" in repair


def test_repair_correction_strips_denial_scaffold() -> None:
    out = extract_info(
        "嗯，不对，维维修的东西是他电脑主机坏了，他进了很多水。",
        expected_field="confirmation",
    )
    repair = out.repair_description or ""
    assert repair
    assert not repair.startswith("不对")
    assert "维维修" not in repair
    assert "维修的东西是" not in repair
    assert "电脑主机" in repair
    assert "进水" in repair or "进了很多水" in repair


def test_confirming_repair_merges_short_supplement() -> None:
    manager = DialogManager()
    draft = RepairOrderDraft(
        customer_name="何先生",
        repair_description="电脑主机坏了",
        state=OrderState.CONFIRMING_REPAIR,
    )

    turn = manager.process_user_text(
        "repair-supplement",
        draft,
        "对对，然后它还进了很多水。",
    )

    assert draft.repair_confirmed is False
    assert "电脑主机坏了" in (draft.repair_description or "")
    assert "进水" in (draft.repair_description or "") or "进了很多水" in (
        draft.repair_description or ""
    )
    assert turn.plan.prompt_id == "confirm_repair"


def test_copula_in_information_is_not_confirmation() -> None:
    assert extract_info("电话係91234567").is_confirm is False
    assert extract_info("问题係冷气唔冻").is_confirm is False


def test_short_cantonese_confirmation_still_works() -> None:
    assert extract_info("係").is_confirm is True
    assert extract_info("嗯，係啊").is_confirm is True
    assert extract_info("啱").is_confirm is True
    assert extract_info("可以").is_confirm is True
    assert extract_info("就这样").is_confirm is True
    assert extract_info("冇问题").is_confirm is True
    assert extract_info("好的").is_confirm is True
    assert extract_info("对的对的，对的。对的对的。喂。").is_confirm is True
    assert extract_info("嗯，对的对的 。喂。").is_confirm is True
    # ASR 把「对，对的」的逗号误识别成句号 → 去标点后「对对」粘连。
    assert extract_info("对。对的，对的。").is_confirm is True
    assert extract_info("Yes, Nala.").is_confirm is True
    assert extract_info("就是。").is_confirm is True
    assert extract_info("就係。").is_confirm is True
    assert extract_info("就系。").is_confirm is True
    assert extract_info("Uh yes.").is_confirm is True
    assert extract_info("可以了。").is_confirm is True
    assert extract_info("嗯，安了，不需要改。").is_confirm is True
    assert extract_info("阿拉。").is_confirm is True
    assert extract_info("赔了。").is_confirm is True
    assert extract_info("需要改，可以了。").is_confirm is False
    assert extract_info("It's.").is_confirm is False


def test_no_change_phrase_confirms_address() -> None:
    manager = DialogManager()
    draft = RepairOrderDraft(
        customer_name="何",
        customer_phone="51266237",
        customer_address="九龙油尖旺区佐敦吴苑街吴苑楼九楼九二一C室",
        repair_description="电脑主机进水，无法开机",
        repair_confirmed=True,
        state=OrderState.CONFIRMING_ADDRESS,
    )

    turn = manager.process_user_text(
        "confirm-address",
        draft,
        "嗯，安了，不需要改。",
    )

    assert draft.address_confirmed is True
    assert turn.plan.prompt_id == "confirm_order"


def test_address_correction_does_not_overwrite_existing_customer_name() -> None:
    manager = DialogManager()
    draft = RepairOrderDraft(
        customer_name="何先生",
        customer_phone="51266237",
        customer_address="旧地址",
        repair_description="电脑主机进水",
        repair_confirmed=True,
        state=OrderState.CONFIRMING_ADDRESS,
    )

    manager.process_user_text(
        "address-correction",
        draft,
        "我是九龙半岛油尖旺区佐敦文乐街文苑楼九幺幺B室。",
    )

    assert draft.customer_name == "何先生"
    assert draft.customer_address != "旧地址"


def test_expected_name_rejects_asr_sentence() -> None:
    out = extract_info("来搞我开了。", expected_field="customer_name")
    assert out.customer_name is None


def test_expected_name_accepts_conservative_standalone_names() -> None:
    assert extract_info("陈大文", expected_field="customer_name").customer_name == "陈大文"
    assert extract_info("Peter", expected_field="customer_name").customer_name == "Peter"


def test_repair_summary_preserves_all_independent_faults() -> None:
    out = extract_info(
        "你好，我电脑主机坏掉了，它昨天晚上进了点水，然后主机有点冒烟，"
        "现在风扇也不转了，然后开机也没有反应，耳机也没有声音。",
        expected_field="repair_description",
    )
    repair = out.repair_description or ""
    for fact in ("进了点水", "冒烟", "风扇也不转", "开机也没有反应", "耳机也没有声音"):
        assert fact in repair


def test_expected_address_rejects_greeting_only() -> None:
    out = extract_info("喂，你好", expected_field="customer_address")
    assert out.customer_address is None


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


def test_expected_phone_strips_spoken_filler_and_spaces() -> None:
    out = extract_info(
        "呃，五幺二 六六二三七。",
        expected_field="customer_phone",
    )
    assert out.customer_phone == "51266237"
    assert out.phone_candidate == "51266237"


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

    t1 = manager.process_user_text("c1", draft, "陈先生")
    assert draft.customer_name == "陈先生"
    assert t1.plan.prompt_id == "ask_repair"

    t2 = manager.process_user_text("c1", draft, "屋企冷气唔冻")
    assert draft.repair_description is not None
    assert t2.plan.prompt_id == "confirm_repair"

    t3 = manager.process_user_text("c1", draft, "啱")
    assert draft.repair_confirmed is True
    assert t3.plan.prompt_id == "ask_phone"

    t4 = manager.process_user_text("c1", draft, "九一二三四五六七")
    assert draft.customer_phone == "91234567"
    assert t4.plan.prompt_id == "ask_address"

    t5 = manager.process_user_text("c1", draft, "旺角弥敦道100号")
    assert draft.customer_address == "旺角弥敦道100号"
    assert t5.plan.prompt_id == "confirm_address"

    t6 = manager.process_user_text("c1", draft, "啱")
    assert draft.address_confirmed is True
    assert t6.plan.prompt_id == "confirm_order"

    t7 = manager.process_user_text("c1", draft, "啱")
    assert t7.saved is True
    assert draft.state == OrderState.COMPLETED
