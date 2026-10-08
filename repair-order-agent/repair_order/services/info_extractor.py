from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)

_PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s]{6,}\d)")
_PHONE_CUE_RE = re.compile(
    r"(?:联系电话|聯絡電話|电话号码|電話號碼|电话|電話|号码|號碼)"
    r"(?:係|是|为|為)?\s*[:：]?\s*(.+)$"
)
_DIGIT_ONLY_RE = re.compile(r"^[\d\s\-]+$")
_SELF_NAME_RE = re.compile(
    r"(?:我叫|叫我|称呼我|稱呼我|我係|我是|姓名(?:係|是)?|my name is)\s*"
    r"([^\s，。,.!！?？]{1,16})",
    re.IGNORECASE,
)
_TITLE_NAME_RE = re.compile(r"([^\s，。,.!！?？]{1,12}(?:先生|小姐|太太|女士|生))")
_SELF_NAME_CUES = (
    "我叫",
    "叫我",
    "称呼我",
    "稱呼我",
    "我係",
    "我是",
    "姓名",
    "my name is",
)
_NON_NAME_TOKENS = ("明白", "小姐", "先生", "女士", "太太", "客户", "客戶", "喂", "你好")
_INCOMPLETE_NAME_PHRASES = {
    "你可以叫我",
    "可以叫我",
    "叫我",
    "我叫",
    "你叫我",
    "称呼我",
    "稱呼我",
    "随便",
    "隨便",
    "都可以",
}
_NAME_TRAILING_FILLERS = (
    "著好了",
    "着好了",
    "就好了",
    "就好啦",
    "就好喇",
    "就好",
    "就可以了",
    "就可以",
    "就行了",
    "就得啦",
    "就得了",
    "就行",
    "就得",
    "可以了",
    "得啦",
)
_NAME_REJECT_SUBSTRINGS = (
    "呵呵",
    "哈哈",
    "你好",
    "您好",
    "明白",
    "收到",
)

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

# Common ASR substitutions seen while callers dictate a phone number. These
# are only enabled for an expected phone field and a digit-like utterance.
_PHONE_FUZZY_DIGIT_MAP = {
    "幺": "1",
    "么": "1",
    "拐": "7",
    "公": "9",
    "勾": "9",
    "狗": "9",
    "够": "9",
    "夠": "9",
}
_PHONE_SPOKEN_SEPARATORS = set(" \t\r\n，。,.!！?？、-—")

_CONFIRM_FULL_UTTERANCES = {
    "啱",
    "啱啊",
    "啱呀",
    "啱啦",
    "係",
    "係啊",
    "係呀",
    "係啦",
    "係喇",
    "是",
    "是的",
    "对",
    "對",
    "对的",
    "對的",
    "正确",
    "正確",
    "没错",
    "無錯",
    "无错",
    "yes",
    "yep",
    "yeah",
    "sure",
    "ok",
    "okay",
    "冇问题",
    "冇問題",
    "冇问题啊",
    "冇問題啊",
    "没问题",
    "沒問題",
    "没有问题",
    "沒有問題",
    "可以",
    "可以啊",
    "可以呀",
    "可以啦",
    "就这样",
    "就這樣",
    "就咁",
    "就咁啦",
    "就咁啊",
    "行",
    "行啊",
    "行啦",
    "得",
    "得啦",
    "好",
    "好啊",
    "好呀",
    "好啦",
    "好的",
    "嗯",
    "嗯嗯",
    "确认",
    "確認",
    "没问题啊",
    "沒問題啊",
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
    "冒烟",
    "冒煙",
    "烧焦",
    "燒焦",
    "漏电",
    "漏電",
    "跳闸",
    "识别",
    "識別",
    "锁",
    "鎖",
    "满",
    "滿",
)
_NON_REPAIR_PHRASES = (
    "维修地址",
    "維修地址",
    "报修地址",
    "報修地址",
    "维修地点",
    "維修地點",
)
_REPAIR_TOO_VAGUE = {
    "坏咗",
    "壞咗",
    "坏了",
    "壞咗啊",
    "唔得",
    "有问题",
    "有問題",
    "我维修的",
    "我維修的",
    "要维修",
    "要維修",
    "维修一下",
    "維修一下",
}
_REPAIR_GENERIC_WORDS = (
    "有问题",
    "有問題",
    "帮我",
    "幫我",
    "我想",
    "需要",
    "维修",
    "維修",
    "修理",
    "坏咗",
    "壞咗",
    "坏了",
    "壞了",
    "唔得",
    "一下",
    "的",
    "嘅",
    "我",
    "要",
)
_REPAIR_FAULT_MARKERS = (
    "打不开",
    "打唔开",
    "開唔到",
    "开不了",
    "開不了",
    "无法",
    "無法",
    "进水",
    "進水",
    "进了水",
    "進了水",
    "进了点水",
    "進了點水",
    "入水",
    "入咗水",
    "漏水",
    "冒烟",
    "冒煙",
    "烧焦",
    "燒焦",
    "漏电",
    "漏電",
    "跳闸",
    "跳掣",
    "异响",
    "異響",
    "不转",
    "不轉",
    "唔转",
    "唔轉",
    "没有声音",
    "沒有聲音",
    "冇声音",
    "冇聲音",
    "无声",
    "無聲",
    "没反应",
    "没有反应",
    "沒有反應",
    "无反应",
    "無反應",
    "冇反應",
    "唔冻",
    "唔凍",
    "不制冷",
    "坏",
    "壞",
    "故障",
)
_REPAIR_DEVICE_MARKERS = (
    "电脑",
    "電腦",
    "主机",
    "主機",
    "笔记本",
    "筆記本",
    "冷气",
    "冷氣",
    "空调",
    "空調",
    "冰箱",
    "洗衣机",
    "洗衣機",
    "电视",
    "電視",
    "打印机",
    "打印機",
    "门锁",
    "門鎖",
    "机器",
    "機器",
    "设备",
    "設備",
)
_REPAIR_HAZARD_MARKERS = (
    "冒烟",
    "冒煙",
    "烧焦",
    "燒焦",
    "焦味",
    "异味",
    "異味",
    "漏电",
    "漏電",
    "火花",
)
_REPAIR_CAUSE_MARKERS = (
    "进水",
    "進水",
    "进了水",
    "進了水",
    "进了点水",
    "進了點水",
    "入水",
    "入咗水",
    "受潮",
    "摔",
    "跌",
    "撞",
)
_REPAIR_REACTION_ONLY_MARKERS = (
    "不敢动",
    "不敢碰",
    "冇敢郁",
    "没敢动",
    "先放着",
    "先放住",
)
_REPAIR_TIME_RE = re.compile(
    r"(?:今天|今日|昨天|尋日|昨晚|琴晚)"
    r"(?:早上|朝早|上午|中午|下午|晚上|夜晚)?"
)
_REPAIR_LEADING_FILLER_RE = re.compile(
    r"^(?:(?:你好|您好|喂|哎|唉|好的|好嘅|嗯|嗱)[呀啊哦喔]*"
    r"[\s，。,.!！?？]*)+"
)

_SOCIAL_ONLY_TOKENS = (
    "hello",
    "hi",
    "喂",
    "你好",
    "您好",
    "嗨",
    "嗯",
    "哦",
    "噢",
    "啊",
    "呀",
    "呵呵",
    "哈哈",
    "好的",
    "好嘅",
    "好",
    "明白",
    "收到",
    "谢谢",
    "謝謝",
    "唔该",
    "唔該",
)

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
    "只抽取两个字段：repair_description 与 customer_address。\n"
    "输出JSON示例："
    '{"repair_description":"电脑主机进水并冒烟，风扇不转，无法开机，耳机无声",'
    '"repair_evidence":["进了点水","主机有点冒烟","风扇也不转了","开机也没有反应","耳机也没有声音"],'
    '"customer_address":null,"address_evidence":null,"confidence":{"repair":0.9,"address":0.0}}。\n'
    "规则：\n"
    "1. 无法确定时该字段必须是 null，对应 evidence 也是 null。\n"
    "2. 禁止猜测、补写、改写地址中的数字/楼层/房号。\n"
    "3. repair_description 必须保留原文中每一个独立故障、原因、危险现象和当前状态；"
    "不得为了简短而遗漏任何一项，只去掉寒暄与“帮我修一下”等脚手架。\n"
    "4. customer_address 只保留地址本身，去掉“地址是/住址在”等前缀。\n"
    "5. repair_evidence 必须逐项列出每个独立故障/原因/危险现象的原文连续片段，"
    "非空字段必须有 evidence。\n"
    "6. 不要输出姓名、电话、intent、确认/否认等其他字段。"
)

_llm_client: OpenAI | None = None
_async_llm_client: AsyncOpenAI | None = None
_LLM_FIELDS = frozenset({"repair_description", "customer_address"})


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


def _fallback_env_path() -> Path:
    return Path(__file__).resolve().parents[3] / "voice-agent-demo" / ".env"


def _extract_provider() -> str:
    return os.getenv("REPAIR_ORDER_EXTRACT_PROVIDER", "deepseek").strip().lower() or "deepseek"


def _extract_base_url() -> str:
    configured = os.getenv("REPAIR_ORDER_EXTRACT_BASE_URL", "").strip()
    if configured:
        return configured
    if _extract_provider() == "dashscope":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return "https://api.deepseek.com"


def _extract_model() -> str:
    configured = os.getenv("REPAIR_ORDER_EXTRACT_MODEL", "").strip()
    if configured:
        return configured
    if _extract_provider() == "dashscope":
        return "qwen-plus"
    return "deepseek-flash"


def _resolved_api_key() -> str:
    for key in (
        "REPAIR_ORDER_EXTRACT_API_KEY",
        "DEEPSEEK_API_KEY",
        "DASHSCOPE_API_KEY",
    ):
        api_key = os.getenv(key, "").strip()
        if api_key:
            return api_key
    fallback_env = _fallback_env_path()
    for key in (
        "REPAIR_ORDER_EXTRACT_API_KEY",
        "DEEPSEEK_API_KEY",
        "DASHSCOPE_API_KEY",
    ):
        api_key = _read_env_value(fallback_env, key)
        if api_key:
            return api_key
    return ""


def _llm_enabled() -> bool:
    return os.getenv("REPAIR_ORDER_USE_LLM_EXTRACTOR", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _llm_timeout_s() -> float:
    try:
        return max(
            0.5,
            float(os.getenv("REPAIR_ORDER_EXTRACT_TIMEOUT_S", "3.0")),
        )
    except ValueError:
        return 3.0


def reset_llm_clients() -> None:
    """Test helper: drop cached OpenAI clients after env changes."""
    global _llm_client, _async_llm_client
    _llm_client = None
    _async_llm_client = None


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
        base_url=_extract_base_url(),
        timeout=_llm_timeout_s(),
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
        base_url=_extract_base_url(),
        timeout=_llm_timeout_s(),
    )
    return _async_llm_client


def _redact_for_llm(text: str) -> str:
    """Hide phone-like spans before sending text to the extract LLM."""
    return _PHONE_RE.sub("<PHONE>", text or "")


def _llm_messages(text: str, expected_field: str | None) -> list[dict[str, str]]:
    hint = ""
    if expected_field in _LLM_FIELDS:
        hint = (
            f"\n当前系统正在等待字段: {expected_field}。"
            "若用户只回答该字段，请优先填写它；另一字段保持 null。"
        )
    elif expected_field:
        hint = (
            f"\n当前系统正在等待字段: {expected_field}。"
            "该字段由规则处理；你仍只输出 repair_description / customer_address，"
            "若原文没有则保持 null。"
        )
    return [
        {"role": "system", "content": _LLM_EXTRACTOR_PROMPT + hint},
        {"role": "user", "content": _redact_for_llm(text)},
    ]


def _llm_create_kwargs(text: str, expected_field: str | None) -> dict:
    kwargs: dict = {
        "model": _extract_model(),
        "messages": _llm_messages(text, expected_field),
        "temperature": 0,
        "max_tokens": 280,
        "response_format": {"type": "json_object"},
    }
    if _extract_provider() != "dashscope":
        # DeepSeek thinking adds latency; extraction does not need it.
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return kwargs


def _llm_extract(text: str, *, expected_field: str | None = None) -> dict | None:
    client = _llm_client_or_none()
    if client is None:
        logger.warning(
            "extract_llm skipped provider=%s model=%s reason=disabled_or_missing_key",
            _extract_provider(),
            _extract_model(),
        )
        return None
    started = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            **_llm_create_kwargs(text, expected_field)
        )
    except Exception as exc:
        logger.warning(
            "extract_llm failed provider=%s model=%s expected=%s elapsed_ms=%.1f error=%s",
            _extract_provider(),
            _extract_model(),
            expected_field,
            (time.perf_counter() - started) * 1000.0,
            type(exc).__name__,
        )
        return None
    content = (resp.choices[0].message.content or "").strip()
    obj = _extract_json_object(content)
    logger.info(
        "extract_llm completed provider=%s model=%s expected=%s elapsed_ms=%.1f "
        "valid_json=%s repair=%s address=%s",
        _extract_provider(),
        _extract_model(),
        expected_field,
        (time.perf_counter() - started) * 1000.0,
        obj is not None,
        bool(obj and obj.get("repair_description")),
        bool(obj and obj.get("customer_address")),
    )
    return obj


async def _llm_extract_async(
    text: str, *, expected_field: str | None = None
) -> dict | None:
    client = _async_llm_client_or_none()
    if client is None:
        logger.warning(
            "extract_llm skipped provider=%s model=%s reason=disabled_or_missing_key",
            _extract_provider(),
            _extract_model(),
        )
        return None
    started = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            **_llm_create_kwargs(text, expected_field)
        )
    except asyncio.CancelledError:
        logger.warning(
            "extract_llm cancelled provider=%s model=%s expected=%s elapsed_ms=%.1f",
            _extract_provider(),
            _extract_model(),
            expected_field,
            (time.perf_counter() - started) * 1000.0,
        )
        raise
    except Exception as exc:
        logger.warning(
            "extract_llm failed provider=%s model=%s expected=%s elapsed_ms=%.1f error=%s",
            _extract_provider(),
            _extract_model(),
            expected_field,
            (time.perf_counter() - started) * 1000.0,
            type(exc).__name__,
        )
        return None
    content = (resp.choices[0].message.content or "").strip()
    obj = _extract_json_object(content)
    logger.info(
        "extract_llm completed provider=%s model=%s expected=%s elapsed_ms=%.1f "
        "valid_json=%s repair=%s address=%s",
        _extract_provider(),
        _extract_model(),
        expected_field,
        (time.perf_counter() - started) * 1000.0,
        obj is not None,
        bool(obj and obj.get("repair_description")),
        bool(obj and obj.get("customer_address")),
    )
    return obj


def _spoken_digits_to_phone(text: str, *, allow_fuzzy: bool = False) -> str:
    chars: list[str] = []
    for ch in text:
        if ch.isdigit():
            chars.append(ch)
        elif ch in _CN_DIGIT_MAP:
            chars.append(_CN_DIGIT_MAP[ch])
    exact = "".join(chars)
    if not allow_fuzzy:
        return exact

    fuzzy_text = text
    cue_match = _PHONE_CUE_RE.search(text)
    if cue_match:
        fuzzy_text = cue_match.group(1)

    # Never fuzzy-map arbitrary prose (for example, "公司电话..."). Accept
    # only a compact sequence made entirely of digits, known digit words,
    # separators, and a small set of observed phone-number substitutions.
    # A recognized phone-field prefix may be safely removed first.
    fuzzy_chars: list[str] = []
    used_fuzzy = False
    for ch in fuzzy_text:
        if ch in _PHONE_SPOKEN_SEPARATORS:
            continue
        if ch.isdigit():
            fuzzy_chars.append(ch)
        elif ch in _CN_DIGIT_MAP:
            fuzzy_chars.append(_CN_DIGIT_MAP[ch])
        elif ch in _PHONE_FUZZY_DIGIT_MAP:
            fuzzy_chars.append(_PHONE_FUZZY_DIGIT_MAP[ch])
            used_fuzzy = True
        else:
            return exact
    fuzzy = "".join(fuzzy_chars)
    if used_fuzzy and _is_valid_phone_digits(fuzzy):
        return fuzzy
    return exact


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


def _extract_phone_candidate(
    text: str,
    *,
    allow_fuzzy: bool = False,
) -> str | None:
    match = _PHONE_RE.search(text)
    if match:
        digits = re.sub(r"\D", "", match.group(0))
        return digits or None
    spoken = _spoken_digits_to_phone(text, allow_fuzzy=allow_fuzzy)
    return spoken or None


def _normalize_phone(text: str, *, allow_fuzzy: bool = False) -> str | None:
    candidate = _extract_phone_candidate(text, allow_fuzzy=allow_fuzzy)
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


def _is_social_only(text: str | None) -> bool:
    compact = re.sub(r"[\s，。,.!！?？、:：;；\-—]+", "", (text or "").lower())
    if not compact:
        return True
    previous = None
    while compact and compact != previous:
        previous = compact
        for token in sorted(_SOCIAL_ONLY_TOKENS, key=len, reverse=True):
            compact = compact.replace(token, "")
    return not compact


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
    marker_text = stripped
    for phrase in _NON_REPAIR_PHRASES:
        marker_text = marker_text.replace(phrase, "")
    if any(marker in marker_text for marker in _REPAIR_MARKERS):
        return stripped
    return None


def _standalone_name_valid(candidate: str) -> bool:
    """Conservative validation for a short answer to the name question."""
    if re.fullmatch(r"[\u3400-\u9fff]{2,4}", candidate):
        # Prefer asking again over persisting an ASR sentence as a name.
        return not any(
            token in candidate
            for token in (
                "我",
                "你",
                "佢",
                "叫",
                "係",
                "是",
                "了",
                "啦",
                "啊",
                "呀",
                "喂",
                "说",
                "說",
                "话",
                "話",
                "搞",
                "开",
                "開",
                "拍",
            )
        )
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z' -]{1,29}", candidate))


def _normalize_name(
    text: str | None,
    full_text: str,
    *,
    expected_field: str | None = None,
) -> str | None:
    candidate = (text or "").strip(" ，。,.!！?？")
    if not candidate:
        return None
    while candidate:
        removed = False
        for suffix in _NAME_TRAILING_FILLERS:
            if candidate.endswith(suffix):
                candidate = candidate[: -len(suffix)].strip(" ，。,.!！?？")
                removed = True
                break
        if not removed:
            break

    # When the answer contains a clear title, keep only the actual name and
    # discard ASR/conversational tails such as "何先生著好了".
    title = _TITLE_NAME_RE.search(candidate)
    if title and (
        expected_field == "customer_name"
        or any(cue in full_text for cue in _SELF_NAME_CUES)
    ):
        candidate = title.group(1).strip()

    if len(candidate) <= 1 and not full_text.startswith("我姓"):
        return None
    if _is_social_only(candidate):
        return None
    if any(token in candidate for token in _NAME_REJECT_SUBSTRINGS):
        return None
    if re.search(r"[，。,.!！?？、:：;；]", candidate):
        return None
    if candidate in _NON_NAME_TOKENS:
        return None
    if candidate in _INCOMPLETE_NAME_PHRASES:
        return None
    if candidate in {"明白小姐", "先生", "小姐"}:
        return None

    if expected_field == "customer_name":
        has_self_cue = any(cue in full_text for cue in _SELF_NAME_CUES)
        has_title = bool(_TITLE_NAME_RE.fullmatch(candidate))
        if not (has_self_cue or has_title or _standalone_name_valid(candidate)):
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
    if _is_social_only(t):
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


def _is_confirm_utterance(text: str) -> bool:
    """Accept confirmation replies, but never treat copulas inside facts as yes."""
    compact = re.sub(r"[\s，。,.!！?？、]+", "", text.lower())
    for filler in ("喂", "嗯", "啊", "呀", "哦"):
        compact = compact.removesuffix(filler)
    if compact in _CONFIRM_FULL_UTTERANCES:
        return True
    # Natural acknowledgements such as “嗯，係啊” and “好，啱”.
    for prefix in ("嗯嗯", "嗯", "好的", "好啊", "好呀", "好啦", "好"):
        if (
            compact.startswith(prefix)
            and compact[len(prefix) :] in _CONFIRM_FULL_UTTERANCES
        ):
            return True
    # ASR often repeats a short confirmation: “对的对的，对的。喂。”
    remainder = compact
    consumed = False
    repeatable = sorted(
        (
            token
            for token in _CONFIRM_FULL_UTTERANCES
            if token not in {"好", "嗯", "嗯嗯", "係", "是", "对", "對"}
        ),
        key=len,
        reverse=True,
    )
    while remainder:
        matched = next((token for token in repeatable if remainder.startswith(token)), None)
        if matched is None:
            break
        consumed = True
        remainder = remainder[len(matched) :]
    if consumed and not remainder:
        return True
    return False


def _utterance_may_need_llm_fields(text: str) -> bool:
    """Only call DeepSeek when the turn likely carries repair/address content."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    if any(marker in stripped for marker in _ADDRESS_MARKERS):
        return True
    if any(marker in stripped for marker in _REPAIR_MARKERS) and not any(
        phrase in stripped for phrase in _NON_REPAIR_PHRASES
    ):
        return True
    # Longer free-form replies may still hide soft repair/address wording.
    return len(stripped) >= 16


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
    if _is_social_only(t):
        return False
    t = _REPAIR_LEADING_FILLER_RE.sub("", t).strip(" ，。,.!！?？")
    if not t:
        return False
    if t in _REPAIR_TOO_VAGUE:
        return False
    if len(t) < 3:
        return False
    detail = t
    for word in _REPAIR_GENERIC_WORDS:
        detail = detail.replace(word, "")
    return bool(detail.strip(" ，。,.!！?？"))


def _clean_repair_clause(text: str) -> str:
    clause = text.strip()
    clause = _REPAIR_LEADING_FILLER_RE.sub("", clause)
    clause = _REPAIR_TIME_RE.sub("", clause)
    clause = re.sub(r"^(?:我嘅|我的|我)\s*", "", clause)
    clause = re.sub(
        r"^(?:(?:然后|然後|跟住|之后|之後|后来|後來|嗰个|那個|那个|"
        r"仲|还|還|它|佢|就)\s*)+",
        "",
        clause,
    )
    for filler in ("可能是", "可能係", "应该是", "應該係", "突然"):
        clause = clause.replace(filler, "")
    clause = re.sub(
        r"(?:不小心)?(?:进|進|入)了?(?:一点|一點|啲)?水",
        "进水",
        clause,
    )
    clause = clause.replace("打不开了", "打不开")
    clause = clause.replace("打唔开咗", "打唔开")
    return clause.strip(" ，。,.!！?？；; ")


def _repair_clause_score(clause: str) -> int:
    if any(marker in clause for marker in _REPAIR_HAZARD_MARKERS):
        return 100
    if any(marker in clause for marker in _REPAIR_FAULT_MARKERS):
        return 80
    if any(marker in clause for marker in _REPAIR_CAUSE_MARKERS):
        return 70
    if any(marker in clause for marker in _REPAIR_DEVICE_MARKERS):
        return 50
    return 0


def _summarize_repair_text(text: str | None) -> str:
    original = (text or "").strip(" ，。,.!！?？；; ")
    original = _REPAIR_LEADING_FILLER_RE.sub("", original).strip(
        " ，。,.!！?？；; "
    )
    if len(original) <= 24:
        return original

    clauses = re.split(r"[，。；;,.!?！？]+", original)
    scored_parts: list[tuple[int, str, int]] = []
    for index, raw_clause in enumerate(clauses):
        clause = _clean_repair_clause(raw_clause)
        if not clause:
            continue

        request_only = (
            any(x in clause for x in ("维修", "維修", "修一下", "整一下"))
            and not any(x in clause for x in _REPAIR_FAULT_MARKERS)
        )
        if request_only:
            continue
        if (
            any(x in clause for x in _REPAIR_REACTION_ONLY_MARKERS)
            and _repair_clause_score(clause) == 0
        ):
            continue
        score = _repair_clause_score(clause)
        if score > 0 and clause not in {item[1] for item in scored_parts}:
            scored_parts.append((index, clause, score))

    # Preserve all independent facts. Repair records value completeness over
    # aggressive shortening; cap only pathological transcripts.
    summary_parts = [item[1] for item in scored_parts[:8]]
    summary = "，".join(summary_parts) or original
    if len(summary) > 100:
        summary = summary[:100].rstrip(" ，。,.!！?？；; ")
    return summary


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
        elif _standalone_name_valid(stripped):
            result.customer_name = _normalize_name(
                stripped, text, expected_field="customer_name"
            )
    elif expected_field == "customer_phone" and not result.customer_phone:
        result.customer_phone = _normalize_phone(stripped, allow_fuzzy=True)
    elif expected_field == "customer_address" and not result.customer_address:
        if _address_valid(stripped):
            result.customer_address = stripped
    elif expected_field == "repair_description" and not result.repair_description:
        if 2 <= len(stripped) <= 80:
            result.repair_description = stripped
    return result


def _evidence_snippets(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def _grounded_in_source(value: str, evidence: object, source: str) -> bool:
    """Reject LLM values that cannot be traced back to the user utterance."""
    if not value or not source:
        return False
    if value in source:
        return True
    return any(snippet in source for snippet in _evidence_snippets(evidence))


def _apply_llm_obj(
    result: ExtractResult,
    llm_obj: dict | None,
    normalized: str,
    expected_field: str | None,
) -> ExtractResult:
    """Apply LLM output for repair/address only. Name/phone/intent stay rule-owned."""
    del expected_field  # reserved for future field-specific post-filters
    if not llm_obj:
        return result

    address = llm_obj.get("customer_address", llm_obj.get("address"))
    repair = llm_obj.get("repair_description", llm_obj.get("repair"))
    address_evidence = llm_obj.get(
        "address_evidence",
        (llm_obj.get("evidence") or {}).get("address")
        if isinstance(llm_obj.get("evidence"), dict)
        else None,
    )
    repair_evidence = llm_obj.get(
        "repair_evidence",
        (llm_obj.get("evidence") or {}).get("repair")
        if isinstance(llm_obj.get("evidence"), dict)
        else None,
    )

    if address:
        candidate = str(address).strip()
        if _address_valid(candidate) and _grounded_in_source(
            candidate, address_evidence, normalized
        ):
            result.customer_address = candidate
            snippets = _evidence_snippets(address_evidence)
            result.evidence["address"] = snippets[0] if snippets else candidate

    if repair:
        candidate = str(repair).strip()
        if candidate and _grounded_in_source(candidate, repair_evidence, normalized):
            result.repair_description = candidate
            snippets = _evidence_snippets(repair_evidence)
            result.evidence["repair"] = snippets[0] if snippets else candidate

    conf = _normalize_conf(llm_obj.get("confidence"))
    for key in ("repair", "address"):
        if key in conf:
            result.confidence[key] = conf[key]
    return result


def _finalize_extract(
    result: ExtractResult,
    normalized: str,
    expected_field: str | None,
) -> ExtractResult:
    lowered = normalized.lower()

    result.is_deny = any(word in lowered for word in _DENY_WORDS)
    result.is_confirm = (not result.is_deny) and _is_confirm_utterance(normalized)
    if result.intent == "other":
        result.intent = _detect_intent(normalized, result.is_confirm, result.is_deny)
    if not result.fields_to_correct and result.is_deny:
        result.fields_to_correct = _heuristic_corrections(normalized)

    allow_fuzzy_phone = expected_field == "customer_phone"
    candidate_from_text = _extract_phone_candidate(
        normalized,
        allow_fuzzy=allow_fuzzy_phone,
    )
    if result.phone_candidate is None:
        result.phone_candidate = candidate_from_text
    if not result.customer_phone:
        result.customer_phone = _normalize_phone(
            normalized,
            allow_fuzzy=allow_fuzzy_phone,
        )
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
    elif result.repair_description:
        result.repair_description = _summarize_repair_text(
            result.repair_description
        )

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


def _rules_are_sufficient(
    result: ExtractResult,
    expected_field: str | None,
    text: str = "",
) -> bool:
    # Control intents are deterministic and should never wait on a network
    # model. Denials also carry heuristic fields_to_correct when provided.
    if result.intent in {"confirm", "deny", "set_language", "use_ani", "transfer"}:
        return True

    # Name/phone are rule-owned. Never call the extract LLM for them.
    if expected_field in {"customer_name", "customer_phone"}:
        return True

    # An invalid phone candidate is already actionable: return the targeted
    # validation prompt immediately instead of asking an LLM to reinterpret it.
    if (
        expected_field == "customer_phone"
        and result.phone_candidate
        and result.phone_error in {"invalid_length", "invalid_format"}
    ):
        return True

    # repair / address turns: prefer DeepSeek for record quality.
    if expected_field in _LLM_FIELDS:
        return False

    # Confirming / free-form (expected_field is None): skip DeepSeek unless the
    # utterance looks like it may correct repair or address content.
    if result.repair_description and result.customer_address:
        return True
    if not _utterance_may_need_llm_fields(text):
        return True
    return False


def extract_info(
    text: str,
    *,
    expected_field: str | None = None,
) -> ExtractResult:
    normalized = (text or "").strip()
    result = ExtractResult()
    if not normalized:
        return result
    rule_result = _finalize_extract(ExtractResult(), normalized, expected_field)
    if _rules_are_sufficient(rule_result, expected_field, normalized):
        return rule_result
    if not _llm_enabled():
        return rule_result
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
    rule_result = _finalize_extract(ExtractResult(), normalized, expected_field)
    if _rules_are_sufficient(rule_result, expected_field, normalized):
        return rule_result
    if not _llm_enabled():
        return rule_result
    try:
        llm_obj = await asyncio.wait_for(
            _llm_extract_async(normalized, expected_field=expected_field),
            timeout=_llm_timeout_s(),
        )
    except asyncio.TimeoutError:
        logger.warning(
            "extract_llm fallback expected=%s reason=timeout timeout_s=%.1f",
            expected_field,
            _llm_timeout_s(),
        )
        return rule_result
    result = _apply_llm_obj(result, llm_obj, normalized, expected_field)
    return _finalize_extract(result, normalized, expected_field)
