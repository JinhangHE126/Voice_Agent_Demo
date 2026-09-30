from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI, OpenAI

_PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s]{6,}\d)")
_DIGIT_ONLY_RE = re.compile(r"^[\d\s\-]+$")
_SELF_NAME_RE = re.compile(
    r"(?:我叫|我係|我是|姓名(?:係|是)?|my name is)\s*([^\s，。,.!！?？]{1,16})",
    re.IGNORECASE,
)
_TITLE_NAME_RE = re.compile(r"([^\s，。,.!！?？]{1,12}(?:先生|小姐|太太|女士|生))")
_SELF_NAME_CUES = ("我叫", "我係", "我是", "姓名", "my name is")
_NON_NAME_TOKENS = ("明白", "小姐", "先生", "女士", "太太", "客户", "客戶", "喂", "你好")

_CN_DIGIT_MAP = {
    "零": "0",
    "〇": "0",
    "一": "1",
    "壹": "1",
    "二": "2",
    "两": "2",
    "兩": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "洞": "0",
}

_CONFIRM_WORDS = {
    "啱",
    "係",
    "是",
    "正确",
    "正確",
    "没错",
    "無錯",
    "无错",
    "yes",
    "ok",
    "okay",
    "对",
    "對",
}
_DENY_WORDS = {
    "唔啱",
    "唔係",
    "不是",
    "不对",
    "不對",
    "錯",
    "错",
    "no",
}
_REPAIR_MARKERS = (
    "维修",
    "維修",
    "整",
    "坏",
    "壞",
    "故障",
    "唔冻",
    "唔凍",
    "不制冷",
    "漏水",
    "跳闸",
    "识别",
    "識別",
    "锁",
    "鎖",
    "满",
    "滿",
)
_REPAIR_TOO_VAGUE = {"坏咗", "壞咗", "坏了", "壞咗啊", "唔得", "有问题", "有問題"}

_ADDRESS_MARKERS = (
    "地址",
    "住址",
    "屋企",
    "家里",
    "家裡",
    "大厦",
    "大廈",
    "楼",
    "樓",
    "街",
    "路",
    "村",
    "号",
    "號",
    "室",
    "座",
    "flat",
    "floor",
    "room",
    "building",
)
_ADDRESS_TOO_VAGUE = {"屋企", "家里", "家裡", "这里", "這裡", "那边", "那邊"}

_LLM_EXTRACTOR_PROMPT = (
    "你是电话维修工单的结构化抽取器。只输出一个JSON对象，不要解释。\n"
    "字段要求：name, phone, address, repair, intent, fields_to_correct, lang, confidence, evidence。\n"
    "name/phone/address/repair 无法确定时必须是 null。\n"
    "intent 只能是 provide_info|confirm|deny|use_ani|transfer|other。\n"
    "fields_to_correct 只允许 customer_name/customer_phone/customer_address/repair_description。\n"
    "confidence 为0到1的小数；evidence给出原文证据片段。"
)

_llm_client: OpenAI | None = None
_async_llm_client: AsyncOpenAI | None = None


@dataclass
class ExtractResult:
    customer_name: str | None = None
    customer_phone: str | None = None
    phone_candidate: str | None = None
    phone_error: str | None = None  # not_provided|not_understood|invalid_length|invalid_format
    customer_address: str | None = None
    repair_description: str | None = None
    is_confirm: bool = False
    is_deny: bool = False
    intent: str = "other"
    fields_to_correct: list[str] = field(default_factory=list)
    lang: str = "yue"
    lang_confidence: float = 0.0
    confidence: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)


def _read_env_value(env_path: Path, key: str) -> str:
    if not env_path.exists():
        return ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return ""


def _extract_json_object(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _resolved_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if api_key:
        return api_key
    fallback_env = Path(__file__).resolve().parents[3] / "voice-agent-demo" / ".env"
    return _read_env_value(fallback_env, "DASHSCOPE_API_KEY")


def _llm_enabled() -> bool:
    return os.getenv("REPAIR_ORDER_USE_LLM_EXTRACTOR", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _llm_client_or_none() -> OpenAI | None:
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    if not _llm_enabled():
        return None
    api_key = _resolved_api_key()
    if not api_key:
        return None
    _llm_client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=15.0,
    )
    return _llm_client


def _async_llm_client_or_none() -> AsyncOpenAI | None:
    global _async_llm_client
    if _async_llm_client is not None:
        return _async_llm_client
    if not _llm_enabled():
        return None
    api_key = _resolved_api_key()
    if not api_key:
        return None
    _async_llm_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=15.0,
    )
    return _async_llm_client


def _llm_messages(text: str, expected_field: str | None) -> list[dict[str, str]]:
    hint = ""
    if expected_field:
        hint = (
            f"\n当前系统正在等待字段: {expected_field}。"
            "若用户只回答该字段，请优先填写它。"
        )
    return [
        {"role": "system", "content": _LLM_EXTRACTOR_PROMPT + hint},
        {"role": "user", "content": text},
    ]


def _llm_extract(text: str, *, expected_field: str | None = None) -> dict | None:
    client = _llm_client_or_none()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=os.getenv("REPAIR_ORDER_EXTRACT_MODEL", "qwen-plus"),
            messages=_llm_messages(text, expected_field),
            temperature=0,
            max_tokens=280,
        )
    except Exception:
        return None
    content = (resp.choices[0].message.content or "").strip()
    return _extract_json_object(content)


async def _llm_extract_async(
    text: str, *, expected_field: str | None = None
) -> dict | None:
    client = _async_llm_client_or_none()
    if client is None:
        return None
    try:
        resp = await client.chat.completions.create(
            model=os.getenv("REPAIR_ORDER_EXTRACT_MODEL", "qwen-plus"),
            messages=_llm_messages(text, expected_field),
            temperature=0,
            max_tokens=280,
        )
    except Exception:
        return None
    content = (resp.choices[0].message.content or "").strip()
    return _extract_json_object(content)


def _spoken_digits_to_phone(text: str) -> str:
    chars: list[str] = []
    for ch in text:
        if ch.isdigit():
            chars.append(ch)
        elif ch in _CN_DIGIT_MAP:
            chars.append(_CN_DIGIT_MAP[ch])
    return "".join(chars)


def _is_valid_phone_digits(digits: str) -> bool:
    if not digits:
        return False
    if len(digits) == 8:
        return True
    if len(digits) == 11 and digits.startswith("1"):
        return True
    if len(digits) == 11 and digits.startswith("852"):
        return True
    if len(digits) == 13 and digits.startswith("86"):
        return True
    return False


def _extract_phone_candidate(text: str) -> str | None:
    match = _PHONE_RE.search(text)
    if match:
        digits = re.sub(r"\D", "", match.group(0))
        return digits or None
    spoken = _spoken_digits_to_phone(text)
    return spoken or None


def _normalize_phone(text: str) -> str | None:
    candidate = _extract_phone_candidate(text)
    if candidate and _is_valid_phone_digits(candidate):
        return candidate
    return None


def _phone_error_for(candidate: str | None) -> str | None:
    if candidate is None:
        return None
    if not candidate.isdigit():
        return "invalid_format"
    if not _is_valid_phone_digits(candidate):
        return "invalid_length"
    return None


def _extract_name(text: str) -> str | None:
    match = _SELF_NAME_RE.search(text)
    if match:
        return match.group(1).strip()
    title = _TITLE_NAME_RE.search(text.strip(" ，。,.!！?？"))
    if title:
        return title.group(1).strip()
    return None


def _extract_repair(text: str) -> str | None:
    stripped = text.strip(" ，。,.!！?？")
    if not stripped:
        return None
    if any(marker in stripped for marker in _REPAIR_MARKERS):
        return stripped
    return None


def _normalize_name(
    text: str | None,
    full_text: str,
    *,
    expected_field: str | None = None,
) -> str | None:
    candidate = (text or "").strip(" ，。,.!！?？")
    if not candidate:
        return None
    if len(candidate) <= 1 and not full_text.startswith("我姓"):
        return None
    if candidate in _NON_NAME_TOKENS:
        return None
    if candidate in {"明白小姐", "先生", "小姐"}:
        return None

    if expected_field == "customer_name":
        if len(candidate) > 20:
            return None
        if any(x in candidate for x in ("电话", "電話", "维修", "維修")):
            return None
        return candidate

    if _TITLE_NAME_RE.fullmatch(candidate) and not any(
        cue in full_text for cue in _SELF_NAME_CUES
    ):
        # Accept short standalone reply like "陈先生" even if we're asking another slot.
        if full_text.strip(" ，。,.!！?？") == candidate:
            return candidate
        return None
    return candidate


def _detect_lang(text: str) -> tuple[str, float]:
    t = (text or "").strip().lower()
    if t in {"广东话", "廣東話", "粤语", "粵語", "cantonese"}:
        return "yue", 0.99
    if t in {"普通话", "普通話", "国语", "國語", "mandarin"}:
        return "cmn", 0.99
    if t in {"english", "eng", "英文", "英语", "英語"}:
        return "en", 0.99

    if re.search(r"[A-Za-z]{3,}", text) and not re.search(r"[\u4e00-\u9fff]", text):
        return "en", 0.95
    if any(x in text for x in ("唔", "冇", "喺", "咗", "佢哋", "咁", "咩")):
        return "yue", 0.82
    if re.search(r"[\u4e00-\u9fff]", text):
        return "cmn", 0.58
    return "cmn", 0.4


def _extract_address(text: str) -> str | None:
    stripped = text.strip(" ，。,.!！?？")
    if not stripped:
        return None
    m = re.search(
        r"(?:地址(?:係|是|在)?|住址(?:係|是|在)?)\s*([^\n，。,.!！?？]{2,80})",
        stripped,
    )
    if m:
        return m.group(1).strip()
    if any(marker.lower() in stripped.lower() for marker in _ADDRESS_MARKERS):
        # Avoid treating pure repair sentences as address.
        if any(marker in stripped for marker in _REPAIR_MARKERS) and not any(
            x in stripped for x in ("地址", "住址", "街", "路", "大厦", "大廈", "号", "號")
        ):
            return None
        return stripped
    return None


def _address_valid(text: str | None) -> bool:
    t = (text or "").strip(" ，。,.!！?？")
    if not t:
        return False
    if t in _ADDRESS_TOO_VAGUE:
        return False
    if len(t) < 2:
        return False
    if t.isdigit():
        return False
    return True


def _heuristic_corrections(text: str) -> list[str]:
    out: list[str] = []
    lowered = text.lower()
    if any(k in text for k in ("电话", "電話", "号码", "號碼", "手機")):
        out.append("customer_phone")
    if any(k in text for k in ("姓名", "称呼", "稱呼", "我叫", "我係", "我是")):
        out.append("customer_name")
    if any(k in text for k in ("地址", "住址", "地点", "地點", "位置")):
        out.append("customer_address")
    if any(k in text for k in ("维修", "維修", "故障", "壞", "坏", "症状", "情況", "情况")):
        out.append("repair_description")
    if "use_ani" in lowered or "来电号码" in text or "來電號碼" in text:
        if "customer_phone" not in out:
            out.append("customer_phone")
    # de-dup keep order
    seen: set[str] = set()
    ordered = []
    for item in out:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _detect_intent(text: str, is_confirm: bool, is_deny: bool) -> str:
    lowered = text.lower()
    if lowered.strip() in {
        "广东话",
        "廣東話",
        "粤语",
        "粵語",
        "cantonese",
        "普通话",
        "普通話",
        "国语",
        "國語",
        "mandarin",
        "english",
        "eng",
    }:
        return "set_language"
    if any(x in text for x in ("转人工", "轉人工", "经理", "經理", "投诉", "投訴")):
        return "transfer"
    if any(x in text for x in ("来电号码", "來電號碼", "就用呢个号码", "就用这个号码")):
        return "use_ani"
    if is_confirm:
        return "confirm"
    if is_deny:
        return "deny"
    if text.strip():
        return "provide_info"
    return "other"


def _normalize_conf(conf: object) -> dict[str, float]:
    if not isinstance(conf, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in conf.items():
        try:
            fv = float(v)
        except Exception:
            continue
        if fv < 0:
            fv = 0.0
        if fv > 1:
            fv = 1.0
        out[str(k)] = fv
    return out


def _normalize_evidence(ev: object) -> dict[str, str]:
    if not isinstance(ev, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in ev.items():
        if v is None:
            continue
        out[str(k)] = str(v).strip()
    return out


def _repair_valid(text: str | None) -> bool:
    t = (text or "").strip(" ，。,.!！?？")
    if not t:
        return False
    if t in _REPAIR_TOO_VAGUE:
        return False
    if len(t) < 3:
        return False
    return True


def _apply_expected_field_fallback(
    result: ExtractResult,
    text: str,
    expected_field: str | None,
) -> ExtractResult:
    if not expected_field:
        return result
    stripped = text.strip(" ，。,.!！?？")
    if expected_field == "customer_name" and not result.customer_name:
        title = _TITLE_NAME_RE.fullmatch(stripped)
        if title:
            result.customer_name = title.group(1)
        elif 1 < len(stripped) <= 12 and not _DIGIT_ONLY_RE.match(stripped):
            result.customer_name = _normalize_name(
                stripped, text, expected_field="customer_name"
            )
    elif expected_field == "customer_phone" and not result.customer_phone:
        result.customer_phone = _normalize_phone(stripped)
    elif expected_field == "customer_address" and not result.customer_address:
        if _address_valid(stripped):
            result.customer_address = stripped
    elif expected_field == "repair_description" and not result.repair_description:
        if 2 <= len(stripped) <= 80:
            result.repair_description = stripped
    return result


def _apply_llm_obj(
    result: ExtractResult,
    llm_obj: dict | None,
    normalized: str,
    expected_field: str | None,
) -> ExtractResult:
    if not llm_obj:
        return result
    # allow both new and old schemas
    name = llm_obj.get("name", llm_obj.get("customer_name"))
    phone = llm_obj.get("phone", llm_obj.get("customer_phone"))
    address = llm_obj.get("address", llm_obj.get("customer_address"))
    repair = llm_obj.get("repair", llm_obj.get("repair_description"))
    result.customer_name = (
        _normalize_name(str(name), normalized, expected_field=expected_field)
        if name
        else None
    )
    result.customer_phone = _normalize_phone(str(phone)) if phone else None
    if phone:
        result.phone_candidate = _extract_phone_candidate(str(phone))
        result.phone_error = _phone_error_for(result.phone_candidate)
    if address:
        candidate = str(address).strip()
        result.customer_address = candidate if _address_valid(candidate) else None
    if repair:
        result.repair_description = str(repair).strip()

    intent = llm_obj.get("intent")
    if isinstance(intent, str) and intent.strip():
        result.intent = intent.strip()
    fields = llm_obj.get("fields_to_correct")
    if isinstance(fields, list):
        result.fields_to_correct = [str(x) for x in fields if str(x).strip()]
    lang = llm_obj.get("lang")
    if isinstance(lang, str) and lang.strip():
        result.lang = lang.strip()
    result.confidence = _normalize_conf(llm_obj.get("confidence"))
    result.evidence = _normalize_evidence(llm_obj.get("evidence"))
    if "lang" in result.confidence:
        result.lang_confidence = float(result.confidence["lang"])
    return result


def _finalize_extract(
    result: ExtractResult,
    normalized: str,
    expected_field: str | None,
) -> ExtractResult:
    lowered = normalized.lower()

    result.is_deny = any(word in lowered for word in _DENY_WORDS)
    result.is_confirm = (not result.is_deny) and any(
        word in lowered for word in _CONFIRM_WORDS
    )
    if result.intent == "other":
        result.intent = _detect_intent(normalized, result.is_confirm, result.is_deny)
    if not result.fields_to_correct and result.is_deny:
        result.fields_to_correct = _heuristic_corrections(normalized)

    candidate_from_text = _extract_phone_candidate(normalized)
    if result.phone_candidate is None:
        result.phone_candidate = candidate_from_text
    if not result.customer_phone:
        result.customer_phone = _normalize_phone(normalized)
    if result.phone_error is None:
        if result.customer_phone is None:
            result.phone_error = _phone_error_for(result.phone_candidate)
        else:
            result.phone_error = None
    if not result.customer_name:
        result.customer_name = _normalize_name(
            _extract_name(normalized),
            normalized,
            expected_field=expected_field,
        )
    if not result.customer_address:
        result.customer_address = _extract_address(normalized)
    if not result.repair_description:
        result.repair_description = _extract_repair(normalized)

    result = _apply_expected_field_fallback(result, normalized, expected_field)

    if result.customer_phone and not _is_valid_phone_digits(result.customer_phone):
        result.customer_phone = None
    if result.customer_address and not _address_valid(result.customer_address):
        result.customer_address = None
    if result.repair_description and not _repair_valid(result.repair_description):
        result.repair_description = None

    if "phone" not in result.confidence and result.customer_phone:
        result.confidence["phone"] = 0.85
    if result.customer_phone is None and result.phone_candidate and "phone" not in result.confidence:
        result.confidence["phone"] = 0.35
    if "name" not in result.confidence and result.customer_name:
        result.confidence["name"] = 0.8
    if "address" not in result.confidence and result.customer_address:
        result.confidence["address"] = 0.8
    if "repair" not in result.confidence and result.repair_description:
        result.confidence["repair"] = 0.8

    if "phone" not in result.evidence and result.customer_phone:
        result.evidence["phone"] = result.customer_phone
    if "name" not in result.evidence and result.customer_name:
        result.evidence["name"] = result.customer_name
    if "address" not in result.evidence and result.customer_address:
        result.evidence["address"] = result.customer_address
    if "repair" not in result.evidence and result.repair_description:
        result.evidence["repair"] = result.repair_description

    lang_guess, lang_conf = _detect_lang(normalized)
    if result.lang not in {"yue", "cmn", "en"}:
        result.lang = lang_guess
    if result.lang_confidence <= 0:
        result.lang_confidence = lang_conf
    if "lang" not in result.confidence:
        result.confidence["lang"] = result.lang_confidence
    if result.phone_candidate is None and result.phone_error is None:
        result.phone_error = "not_provided"
    return result


def extract_info(
    text: str,
    *,
    expected_field: str | None = None,
) -> ExtractResult:
    normalized = (text or "").strip()
    result = ExtractResult()
    if not normalized:
        return result
    llm_obj = _llm_extract(normalized, expected_field=expected_field)
    result = _apply_llm_obj(result, llm_obj, normalized, expected_field)
    return _finalize_extract(result, normalized, expected_field)


async def extract_info_async(
    text: str,
    *,
    expected_field: str | None = None,
) -> ExtractResult:
    normalized = (text or "").strip()
    result = ExtractResult()
    if not normalized:
        return result
    llm_obj = await _llm_extract_async(normalized, expected_field=expected_field)
    result = _apply_llm_obj(result, llm_obj, normalized, expected_field)
    return _finalize_extract(result, normalized, expected_field)
