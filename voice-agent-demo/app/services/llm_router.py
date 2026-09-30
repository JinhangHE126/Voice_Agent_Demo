from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from app.services.text_norm import detect_reply_lang

FirstTokenCallback = Callable[[], Awaitable[None]]


class FAQMatcher:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._items: list[dict[str, Any]] = []
        self.reload()

    def reload(self) -> None:
        if not self.path.exists():
            self._items = []
            return
        self._items = json.loads(self.path.read_text(encoding="utf-8"))

    def match(self, text: str, reply_lang: str) -> str | None:
        t = (text or "").strip().lower()
        if not t:
            return None
        best: dict[str, Any] | None = None
        best_hits = 0
        for item in self._items:
            keys = [str(k).lower() for k in item.get("keywords") or []]
            hits = sum(1 for k in keys if k and k in t)
            if hits > best_hits:
                best_hits = hits
                best = item
        if not best or best_hits <= 0:
            return None
        # Require at least one solid keyword hit; prefer multi-hit for hours.
        if best_hits < 1:
            return None
        if reply_lang == "en":
            return best.get("reply_en") or best.get("reply_zh")
        if reply_lang == "yue":
            return best.get("reply_yue") or best.get("reply_zh")
        return best.get("reply_zh") or best.get("reply_yue")


def _needs_complex(text: str) -> bool:
    t = text or ""
    if len(t) >= 40:
        return True
    markers = ("为什么", "點解", "比较", "比較", "分析", "如果", "假设", "假設", "详细", "詳細")
    return any(m in t for m in markers)


class GradedLLMRouter:
    """
    L0 FAQ (instant) → L1 fast cloud LLM → L2 Bonsai (complex only).
    Always streams short clauses for TTS.
    """

    def __init__(
        self,
        *,
        faq_path: Path,
        system_prompt: str,
        fast_base_url: str,
        fast_api_key: str,
        fast_model: str,
        fast_max_tokens: int = 60,
        fast_temperature: float = 0.4,
        fast_timeout_s: float = 3.0,
        bonsai_base_url: str = "",
        bonsai_api_key: str = "",
        bonsai_model: str = "bonsai-2-27b",
        bonsai_max_tokens: int = 80,
        bonsai_temperature: float = 0.5,
        bonsai_disable_thinking: bool = True,
        mock: bool = False,
    ) -> None:
        self.faq = FAQMatcher(faq_path)
        self.system_prompt = system_prompt
        self.fast_timeout_s = fast_timeout_s
        self.mock = mock
        self.bonsai_disable_thinking = bonsai_disable_thinking
        self._fast: AsyncOpenAI | None = None
        self._bonsai: AsyncOpenAI | None = None
        self.fast_model = fast_model
        self.fast_max_tokens = fast_max_tokens
        self.fast_temperature = fast_temperature
        self.bonsai_model = bonsai_model
        self.bonsai_max_tokens = bonsai_max_tokens
        self.bonsai_temperature = bonsai_temperature
        self._lock = asyncio.Lock()

        if not mock and fast_api_key:
            self._fast = AsyncOpenAI(
                base_url=fast_base_url,
                api_key=fast_api_key,
                timeout=fast_timeout_s + 2.0,
            )
        if not mock and bonsai_api_key and bonsai_base_url:
            self._bonsai = AsyncOpenAI(
                base_url=bonsai_base_url,
                api_key=bonsai_api_key,
                timeout=8.0,
            )

    def _messages(
        self, user_text: str, history: list[dict[str, str]] | None
    ) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        if history:
            msgs.extend(history[-6:])
        msgs.append({"role": "user", "content": user_text})
        return msgs

    async def warm(self) -> None:
        if self.mock or not self._fast:
            return
        try:
            async with self._lock:
                await self._fast.chat.completions.create(
                    model=self.fast_model,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                    temperature=0,
                    stream=False,
                )
        except Exception:
            pass

    async def stream_reply(
        self,
        user_text: str,
        *,
        asr_lang: str = "yue",
        history: list[dict[str, str]] | None = None,
        on_first_token: FirstTokenCallback | None = None,
        cancel_event: asyncio.Event | None = None,
        route_out: dict[str, str] | None = None,
    ) -> AsyncIterator[str]:
        reply_lang = detect_reply_lang(user_text, asr_lang)

        # ---- L0 FAQ ----
        faq = self.faq.match(user_text, reply_lang)
        if faq:
            if route_out is not None:
                route_out["route"] = "L0_faq"
            if on_first_token:
                await on_first_token()
            yield faq
            return

        if self.mock:
            if route_out is not None:
                route_out["route"] = "mock"
            async for part in self._mock_stream(user_text, reply_lang, on_first_token, cancel_event):
                yield part
            return

        use_bonsai = _needs_complex(user_text) and self._bonsai is not None
        client = self._bonsai if use_bonsai else self._fast
        model = self.bonsai_model if use_bonsai else self.fast_model
        max_tokens = self.bonsai_max_tokens if use_bonsai else self.fast_max_tokens
        temperature = self.bonsai_temperature if use_bonsai else self.fast_temperature
        extra = None
        if use_bonsai and self.bonsai_disable_thinking:
            extra = {"chat_template_kwargs": {"enable_thinking": False}}

        if route_out is not None:
            route_out["route"] = "L2_bonsai" if use_bonsai else "L1_fast"

        if client is None:
            if route_out is not None:
                route_out["route"] = "fallback"
            if on_first_token:
                await on_first_token()
            yield "请稍后再试。" if reply_lang != "en" else "Please try again."
            return

        messages = self._messages(user_text, history)
        marked = False

        async def mark() -> None:
            nonlocal marked
            if not marked:
                marked = True
                if on_first_token:
                    await on_first_token()

        async with self._lock:
            try:
                stream = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                        extra_body=extra,
                    ),
                    timeout=self.fast_timeout_s,
                )
                async for event in stream:
                    if cancel_event and cancel_event.is_set():
                        return
                    if not event.choices:
                        continue
                    text = getattr(event.choices[0].delta, "content", None)
                    if text:
                        await mark()
                        yield text
                if marked:
                    return
            except Exception:
                pass

            if cancel_event and cancel_event.is_set():
                return

            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                    extra_body=extra,
                )
                text = (resp.choices[0].message.content or "").strip()
            except Exception:
                text = ""
            if not text:
                text = "请稍后再试。" if reply_lang != "en" else "Please try again."
            await mark()
            yield text

    async def _mock_stream(
        self,
        user_text: str,
        reply_lang: str,
        on_first_token: FirstTokenCallback | None,
        cancel_event: asyncio.Event | None,
    ) -> AsyncIterator[str]:
        if reply_lang == "en":
            parts = ["We close at 8 PM", " on Sunday."]
        elif reply_lang == "zh":
            parts = ["星期天我们", "晚上八点关门。"]
        else:
            parts = ["星期日我哋", "夜晚八点关门。"]
        await asyncio.sleep(0.08)
        if on_first_token:
            await on_first_token()
        for part in parts:
            if cancel_event and cancel_event.is_set():
                return
            yield part
            await asyncio.sleep(0.04)
