from __future__ import annotations

from dataclasses import dataclass

from repair_order.domain.repair_order import OrderState, RepairOrderDraft
from repair_order.repositories.order_repository import OrderRepository
from repair_order.services.info_extractor import extract_info, extract_info_async


@dataclass
class ResponsePlan:
    text: str
    prompt_id: str
    dynamic: bool = False
    expected_field: str | None = None


@dataclass
class TurnResult:
    plan: ResponsePlan
    saved: bool = False
    saved_record: dict | None = None
    updated_fields: list[str] | None = None
    phone_candidate: str | None = None
    phone_error: str | None = None


def _clean_repair_text(text: str | None) -> str:
    return (text or "").strip().rstrip("，。,.!！?？；; ")


class DialogManager:
    _FIELD_PRIORITY = (
        "repair_description",
        "customer_phone",
        "customer_address",
        "customer_name",
    )

    def __init__(self, repository: OrderRepository | None = None) -> None:
        self.repository = repository

    def _lang_of(self, draft: RepairOrderDraft) -> str:
        return draft.session_language or "yue"

    def _say(self, draft: RepairOrderDraft, key: str) -> str:
        lang = self._lang_of(draft)
        yue = {
            "greeting": "你好，广东话、普通话或者 English 都可以。请问有什么可以帮到你？",
            "ask_repair": "请简单讲下要维修嘅内容。",
            "ask_phone": "请讲一下你嘅联系电话。",
            "ask_address": "请讲一下维修地址。",
            "ask_name": "请问点称呼你？",
            "ask_confirm_yesno": "请讲啱或者唔啱，如果唔啱可以直接讲要改边项资料。",
            "ask_correct_field": "明白，边项要改？姓名、电话、地址，定维修内容？",
            "invalid_phone_length": "你啱啱讲嘅电话号码位数唔啱，请逐个数字再讲一次。",
            "invalid_phone_generic": "唔好意思，我确认唔到个电话号码，请讲慢少少再讲一次。",
            "order_saved": "已经帮你登记维修订单，我哋会尽快安排同事联络你。",
            "completed": "订单已经确认，多谢你来电。",
            "confirm": (
                "确认一下：你係{customer_name}，电话係{customer_phone}，"
                "地址係{customer_address}，需要维修内容係{repair_description}，啱唔啱？"
            ),
            "ask_language": "请问想用广东话、普通话，还是英文？",
        }
        cmn = {
            "greeting": "您好，粤语、普通话或 English 都可以。请问有什么可以帮您？",
            "ask_repair": "请简要描述需要维修的内容。",
            "ask_phone": "请提供您的联系电话。",
            "ask_address": "请提供维修地址。",
            "ask_name": "请问怎么称呼您？",
            "ask_confirm_yesno": "请回答对或不对，如果不对请告诉我哪项需要修改。",
            "ask_correct_field": "明白，请问要修改姓名、电话、地址，还是维修内容？",
            "invalid_phone_length": "您刚才说的电话号码位数不正确，请逐位再说一次。",
            "invalid_phone_generic": "抱歉，我没确认到电话号码，请放慢一点再说一次。",
            "order_saved": "已为您登记维修工单，我们会尽快安排同事联系您。",
            "completed": "订单已经确认，感谢来电。",
            "confirm": (
                "确认一下：您是{customer_name}，电话是{customer_phone}，"
                "地址是{customer_address}，维修内容是{repair_description}，对吗？"
            ),
            "ask_language": "请问您想用粤语、普通话，还是英文？",
        }
        en = {
            "greeting": "Hello, Cantonese, Mandarin, or English are all okay. How can I help you?",
            "ask_repair": "Please briefly describe what needs repair.",
            "ask_phone": "Please provide your contact phone number.",
            "ask_address": "Please provide the service address.",
            "ask_name": "May I have your name, please?",
            "ask_confirm_yesno": "Please say yes or no. If no, tell me which field to update.",
            "ask_correct_field": "Sure. Which part should I update: name, phone, address, or repair details?",
            "invalid_phone_length": "The phone number length seems invalid. Please say each digit again.",
            "invalid_phone_generic": "Sorry, I could not confirm the phone number. Please say it slowly again.",
            "order_saved": "Your repair order has been created. Our colleague will contact you soon.",
            "completed": "Your order is confirmed. Thank you for calling.",
            "confirm": (
                "Please confirm: your name is {customer_name}, phone is {customer_phone}, "
                "address is {customer_address}, and repair request is {repair_description}. "
                "Is that correct?"
            ),
            "ask_language": "Which language do you prefer: Cantonese, Mandarin, or English?",
        }
        table = {"yue": yue, "cmn": cmn, "en": en}.get(lang, yue)
        return table[key]

    def greeting(self, draft: RepairOrderDraft) -> ResponsePlan:
        draft.awaiting_field = "repair_description"
        return ResponsePlan(
            text=self._say(draft, "greeting"),
            prompt_id="greeting",
            dynamic=False,
            expected_field="repair_description",
        )

    def _apply_extract(
        self,
        call_id: str,
        draft: RepairOrderDraft,
        extracted,
    ) -> TurnResult:
        updated_fields: list[str] = []
        conf = extracted.confidence or {}

        def conf_ok(field: str, default: float = 0.7) -> bool:
            return float(conf.get(field, default)) >= 0.55

        if extracted.customer_name:
            if conf_ok("name", 0.75):
                draft.customer_name = extracted.customer_name
                updated_fields.append("customer_name")
        if extracted.customer_phone:
            if conf_ok("phone", 0.8):
                draft.customer_phone = extracted.customer_phone
                updated_fields.append("customer_phone")
        if extracted.customer_address:
            if conf_ok("address", 0.75):
                draft.customer_address = extracted.customer_address.strip()
                updated_fields.append("customer_address")
        if extracted.repair_description:
            if conf_ok("repair", 0.75):
                draft.repair_description = _clean_repair_text(
                    extracted.repair_description
                )
                updated_fields.append("repair_description")

        # Session language tracks customer language. Use explicit language
        # selection immediately; otherwise require moderate confidence.
        if extracted.intent == "set_language":
            draft.session_language = extracted.lang
            draft.language_source = "explicit"
            draft.language_confidence = 1.0
        elif extracted.lang in {"yue", "cmn", "en"} and extracted.lang_confidence >= 0.55:
            if draft.session_language is None or extracted.lang_confidence >= draft.language_confidence:
                draft.session_language = extracted.lang
                draft.language_source = "detected"
                draft.language_confidence = extracted.lang_confidence

        if (
            draft.state == OrderState.CONFIRMING
            and extracted.is_confirm
            and draft.is_complete()
        ):
            draft.confirmed = True
            draft.state = OrderState.SAVING
            saved_record = (
                self.repository.save(call_id, draft) if self.repository else None
            )
            draft.state = OrderState.COMPLETED
            draft.awaiting_field = None
            return TurnResult(
                plan=ResponsePlan(
                    text=self._say(draft, "order_saved"),
                    prompt_id="order_saved",
                    dynamic=False,
                    expected_field=None,
                ),
                saved=True,
                saved_record=saved_record,
                updated_fields=updated_fields,
                phone_candidate=extracted.phone_candidate,
                phone_error=extracted.phone_error,
            )

        if draft.state == OrderState.CONFIRMING and extracted.is_deny:
            draft.confirmed = False
            draft.state = OrderState.COLLECTING
            fields_to_correct = [f for f in extracted.fields_to_correct if f]
            if fields_to_correct:
                # If user says a field is wrong but doesn't provide a replacement,
                # clear only that slot and collect again.
                for field_name in fields_to_correct:
                    if field_name == "customer_name" and "customer_name" not in updated_fields:
                        draft.customer_name = None
                    elif field_name == "customer_phone" and "customer_phone" not in updated_fields:
                        draft.customer_phone = None
                    elif field_name == "customer_address" and "customer_address" not in updated_fields:
                        draft.customer_address = None
                    elif field_name == "repair_description" and "repair_description" not in updated_fields:
                        draft.repair_description = None
            elif not updated_fields:
                return TurnResult(
                    plan=ResponsePlan(
                        text=self._say(draft, "ask_correct_field"),
                        prompt_id="ask_correct_field",
                        dynamic=False,
                        expected_field=None,
                    ),
                    updated_fields=[],
                    phone_candidate=extracted.phone_candidate,
                    phone_error=extracted.phone_error,
                )

        if draft.state == OrderState.CONFIRMING and not (
            extracted.is_confirm or extracted.is_deny
        ):
            if not updated_fields:
                return TurnResult(
                    plan=ResponsePlan(
                        text=self._say(draft, "ask_confirm_yesno"),
                        prompt_id="ask_confirm_yesno",
                        dynamic=False,
                        expected_field=None,
                    ),
                    updated_fields=updated_fields,
                    phone_candidate=extracted.phone_candidate,
                    phone_error=extracted.phone_error,
                )
            draft.state = OrderState.COLLECTING

        # In collecting mode, if we are waiting for phone and heard an invalid
        # candidate, return a targeted prompt instead of generic ask_phone.
        if (
            draft.state == OrderState.COLLECTING
            and draft.awaiting_field == "customer_phone"
            and not draft.customer_phone
            and extracted.phone_candidate
        ):
            prompt_id = (
                "invalid_phone_length"
                if extracted.phone_error == "invalid_length"
                else "invalid_phone_generic"
            )
            return TurnResult(
                plan=ResponsePlan(
                    text=self._say(draft, prompt_id),
                    prompt_id=prompt_id,
                    dynamic=False,
                    expected_field="customer_phone",
                ),
                updated_fields=updated_fields,
                phone_candidate=extracted.phone_candidate,
                phone_error=extracted.phone_error,
            )

        plan = self.next_prompt(draft)
        return TurnResult(
            plan=plan,
            updated_fields=updated_fields,
            phone_candidate=extracted.phone_candidate,
            phone_error=extracted.phone_error,
        )

    def process_user_text(
        self,
        call_id: str,
        draft: RepairOrderDraft,
        user_text: str,
    ) -> TurnResult:
        extracted = extract_info(user_text, expected_field=draft.awaiting_field)
        return self._apply_extract(call_id, draft, extracted)

    async def process_user_text_async(
        self,
        call_id: str,
        draft: RepairOrderDraft,
        user_text: str,
    ) -> TurnResult:
        extracted = await extract_info_async(
            user_text, expected_field=draft.awaiting_field
        )
        return self._apply_extract(call_id, draft, extracted)

    def next_prompt(self, draft: RepairOrderDraft) -> ResponsePlan:
        if draft.state == OrderState.COMPLETED:
            draft.awaiting_field = None
            return ResponsePlan(
                text=self._say(draft, "completed"),
                prompt_id="completed",
                dynamic=False,
                expected_field=None,
            )

        if (
            draft.session_language is None
            and not draft.language_asked
            and not draft.repair_description
            and not draft.customer_phone
            and not draft.customer_address
            and not draft.customer_name
        ):
            draft.language_asked = True
            return ResponsePlan(
                text=self._say(draft, "ask_language"),
                prompt_id="ask_language",
                dynamic=False,
                expected_field=None,
            )

        missing = set(draft.missing_fields())
        if missing:
            field = next((f for f in self._FIELD_PRIORITY if f in missing), None)
            if field is None:
                field = "repair_description"
            draft.awaiting_field = field
            draft.state = OrderState.COLLECTING
            if field == "repair_description":
                return ResponsePlan(
                    text=self._say(draft, "ask_repair"),
                    prompt_id="ask_repair",
                    dynamic=False,
                    expected_field=field,
                )
            if field == "customer_phone":
                return ResponsePlan(
                    text=self._say(draft, "ask_phone"),
                    prompt_id="ask_phone",
                    dynamic=False,
                    expected_field=field,
                )
            if field == "customer_address":
                return ResponsePlan(
                    text=self._say(draft, "ask_address"),
                    prompt_id="ask_address",
                    dynamic=False,
                    expected_field=field,
                )
            if field == "customer_name":
                return ResponsePlan(
                    text=self._say(draft, "ask_name"),
                    prompt_id="ask_name",
                    dynamic=False,
                    expected_field=field,
                )

        draft.state = OrderState.CONFIRMING
        draft.awaiting_field = None
        repair = _clean_repair_text(draft.repair_description)
        return ResponsePlan(
            text=self._say(draft, "confirm").format(
                customer_name=draft.customer_name,
                customer_phone=draft.customer_phone,
                customer_address=draft.customer_address,
                repair_description=repair,
            ),
            prompt_id="confirm_order",
            dynamic=True,
            expected_field=None,
        )
