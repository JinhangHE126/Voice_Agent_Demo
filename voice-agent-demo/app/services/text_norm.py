from __future__ import annotations

import re


_SENTENCE_END = re.compile(r"[，。！？,!.?、；;]")


def split_for_tts(text: str, *, first_flush: int = 6, later_flush: int = 16) -> list[str]:
    """Split streaming/full text into short speakable clauses."""
    text = (text or "").strip()
    if not text:
        return []
    parts: list[str] = []
    buf = ""
    first = True
    for ch in text:
        buf += ch
        limit = first_flush if first else later_flush
        if _SENTENCE_END.search(ch) or len(buf) >= limit:
            piece = buf.strip()
            if piece:
                parts.append(piece)
                first = False
            buf = ""
    if buf.strip():
        parts.append(buf.strip())
    return parts


def detect_reply_lang(user_text: str, asr_lang: str = "yue") -> str:
    t = user_text or ""
    if re.search(r"[A-Za-z]{3,}", t) and not re.search(r"[\u4e00-\u9fff]", t):
        return "en"
    if asr_lang in {"yue", "zh-HK"}:
        return "yue"
    if asr_lang in {"zh", "zh-CN"}:
        return "zh"
    # Heuristic Cantonese particles
    if any(x in t for x in ("唔", "冇", "喺", "嘅", "咗", "嚟", "哋")):
        return "yue"
    return "zh"


def language_boost_for(reply_lang: str) -> str:
    if reply_lang == "yue":
        return "Chinese,Yue"
    if reply_lang == "en":
        return "English"
    return "Chinese"
