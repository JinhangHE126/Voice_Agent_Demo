from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from collections.abc import Awaitable, Callable

PartialCallback = Callable[[str], Awaitable[None]]
FinalCallback = Callable[[str, str], Awaitable[None]]  # text, language


class MockStreamingASR:
    """Architecture mock: emits staged Cantonese partials then a final."""

    def __init__(
        self,
        *,
        on_partial: PartialCallback | None = None,
        on_final: FinalCallback | None = None,
        language: str = "yue",
    ) -> None:
        self.on_partial = on_partial
        self.on_final = on_final
        self.language = language
        self._started = False
        self._closed = False
        self._finalized = False
        self._partials = ["我想知", "我想知星期日", "我想知星期日几点关门"]
        self._partial_idx = 0
        self._last_partial_at = 0.0

    async def start(self) -> None:
        self._started = True
        self._closed = False
        self._finalized = False
        self._partial_idx = 0
        self._last_partial_at = time.perf_counter()

    async def push_pcm16(self, pcm: bytes) -> None:
        if not self._started or self._closed or self._finalized:
            return
        now = time.perf_counter()
        if (
            self.on_partial
            and self._partial_idx < len(self._partials)
            and now - self._last_partial_at >= 0.35
        ):
            text = self._partials[self._partial_idx]
            self._partial_idx += 1
            self._last_partial_at = now
            await self.on_partial(text)

    async def end_utterance(self) -> None:
        if self._finalized or self._closed:
            return
        self._finalized = True
        await asyncio.sleep(0.06)
        if self.on_final:
            await self.on_final(self._partials[-1], self.language)

    async def close(self) -> None:
        self._closed = True
        self._started = False


class QwenRealtimeASR:
    """
    Qwen3-ASR-Flash-Realtime over DashScope WebSocket.

    Uses manual endpointing. Local VAD owns the utterance boundary so a
    short natural pause does not let cloud VAD finalize half a sentence.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "qwen3-asr-flash-realtime",
        language: str = "yue",
        ws_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
        silence_ms: int = 400,
        final_timeout_ms: int = 2500,
        on_partial: PartialCallback | None = None,
        on_final: FinalCallback | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for Qwen ASR")
        self.api_key = api_key
        self.model = model
        self.language = language
        self.ws_url = ws_url
        self.silence_ms = silence_ms
        self.final_timeout_ms = max(500, int(final_timeout_ms))
        self.on_partial = on_partial
        self.on_final = on_final
        self._ws = None
        self._reader_task: asyncio.Task | None = None
        self._closed = False
        self._final_event = asyncio.Event()
        self._last_final = ""
        self._partial = ""
        self._final_dispatched = False
        self._dispatch_lock = asyncio.Lock()

    async def _dispatch_final(self, text: str, lang: str) -> None:
        async with self._dispatch_lock:
            if self._final_dispatched:
                return
            self._final_dispatched = True
            self._last_final = text
            self._partial = ""
            self._final_event.set()
            if self.on_final:
                await self.on_final(text, lang)

    def _event_id(self) -> str:
        return f"evt_{uuid.uuid4().hex[:12]}"

    async def start(self) -> None:
        import websockets

        url = f"{self.ws_url}?model={self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        self._ws = await websockets.connect(url, additional_headers=headers)
        self._closed = False
        self._final_event.clear()
        self._last_final = ""
        self._partial = ""
        await self._ws.send(
            json.dumps(
                {
                    "event_id": self._event_id(),
                    "type": "session.update",
                    "session": {
                        "input_audio_format": "pcm",
                        "sample_rate": 16000,
                        # An empty language enables Qwen's automatic language
                        # detection, allowing Cantonese, Mandarin and English
                        # to share one realtime session.
                        "input_audio_transcription": (
                            {"language": self.language}
                            if self.language
                            else {}
                        ),
                        "turn_detection": None,
                    },
                }
            )
        )
        self._reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if self._closed:
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                event = msg.get("type") or ""
                if event == "conversation.item.input_audio_transcription.delta":
                    text = (
                        msg.get("delta")
                        or msg.get("text")
                        or msg.get("transcript")
                        or ""
                    )
                    if text:
                        self._partial = text if len(text) > len(self._partial) else (
                            self._partial + text
                        )
                        # Prefer cumulative when provided
                        if msg.get("transcript"):
                            self._partial = msg["transcript"]
                        if self.on_partial:
                            await self.on_partial(self._partial)
                elif event == "conversation.item.input_audio_transcription.completed":
                    text = (
                        msg.get("transcript")
                        or msg.get("text")
                        or self._partial
                        or ""
                    ).strip()
                    lang = (msg.get("language") or self.language or "")
                    await self._dispatch_final(text, lang)
                elif event in {"error", "session.finished"}:
                    pass
        except Exception:
            pass

    async def push_pcm16(self, pcm: bytes) -> None:
        if not self._ws or self._closed or not pcm:
            return
        await self._ws.send(
            json.dumps(
                {
                    "event_id": self._event_id(),
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }
            )
        )

    async def end_utterance(self) -> None:
        """Commit one utterance and provide a timeout fallback final callback."""
        if not self._ws or self._closed:
            return
        self._final_event.clear()
        self._final_dispatched = False
        try:
            await self._ws.send(
                json.dumps(
                    {
                        "event_id": self._event_id(),
                        "type": "input_audio_buffer.commit",
                    }
                )
            )
        except Exception:
            return
        # Avoid endless ENDPOINTING if cloud ASR never sends completed.
        try:
            await asyncio.wait_for(
                self._final_event.wait(),
                timeout=self.final_timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            fallback = (self._partial or "").strip()
            await self._dispatch_final(fallback, self.language or "yue")

    async def close(self) -> None:
        self._closed = True
        if self._ws:
            try:
                await self._ws.send(
                    json.dumps(
                        {"event_id": self._event_id(), "type": "session.finish"}
                    )
                )
            except Exception:
                pass
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
