from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

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
        self._connect_lock = asyncio.Lock()
        self._last_connect_attempt = 0.0
        self._reconnect_cooldown_s = 1.0

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

    async def _open_connection(self) -> None:
        import websockets

        url = f"{self.ws_url}?model={self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        ws = await websockets.connect(
            url,
            additional_headers=headers,
            open_timeout=15.0,
            ping_interval=20.0,
            ping_timeout=20.0,
        )
        await ws.send(
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
        self._ws = ws
        self._final_event.clear()
        self._last_final = ""
        self._partial = ""
        self._reader_task = asyncio.create_task(self._read_loop(ws))

    async def _drop_connection(self, ws) -> None:
        if self._ws is ws:
            self._ws = None
        reader = self._reader_task
        if (
            reader
            and reader is not asyncio.current_task()
            and not reader.done()
        ):
            reader.cancel()
            try:
                await reader
            except asyncio.CancelledError:
                pass
        if self._reader_task is reader:
            self._reader_task = None
        if ws:
            try:
                await ws.close()
            except Exception:
                pass

    async def _reconnect(self, failed_ws=None, *, force: bool = False) -> bool:
        async with self._connect_lock:
            if self._closed:
                return False
            if (
                failed_ws is not None
                and self._ws is not None
                and self._ws is not failed_ws
            ):
                return True
            now = time.monotonic()
            if (
                not force
                and self._ws is None
                and now - self._last_connect_attempt < self._reconnect_cooldown_s
            ):
                return False
            self._last_connect_attempt = now
            old_ws = self._ws or failed_ws
            await self._drop_connection(old_ws)
            try:
                await self._open_connection()
                logger.info("Qwen ASR WebSocket connected")
                return True
            except Exception as exc:
                logger.warning("Qwen ASR reconnect failed: %r", exc)
                self._ws = None
                return False

    async def start(self) -> None:
        self._closed = False
        connected = await self._reconnect(force=True)
        if not connected:
            # Keep the browser call alive. Audio pushes will retry after the
            # reconnect cooldown, which handles transient handshake failures
            # without tearing down the client WebSocket.
            logger.warning(
                "Qwen ASR initial connection unavailable; "
                "will retry when audio arrives"
            )

    async def _read_loop(self, ws) -> None:
        try:
            async for raw in ws:
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
                elif event == "error":
                    logger.warning("Qwen ASR error event: %s", msg)
                    break
                elif event == "session.finished":
                    logger.info("Qwen ASR session finished by server")
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed:
                logger.warning("Qwen ASR read loop disconnected: %r", exc)
        finally:
            if self._ws is ws:
                self._ws = None
            if self._reader_task is asyncio.current_task():
                self._reader_task = None

    async def push_pcm16(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        payload = json.dumps(
            {
                "event_id": self._event_id(),
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )
        ws = self._ws
        if ws is None:
            if not await self._reconnect():
                return
            ws = self._ws
        try:
            await ws.send(payload)
            return
        except Exception as exc:
            logger.warning("Qwen ASR audio send failed; reconnecting: %r", exc)

        if not await self._reconnect(ws):
            return
        retry_ws = self._ws
        if retry_ws is None:
            return
        try:
            await retry_ws.send(payload)
        except Exception as exc:
            logger.warning("Qwen ASR audio retry failed: %r", exc)
            await self._drop_connection(retry_ws)

    async def _finish_failed_utterance(self) -> None:
        fallback = (self._partial or "").strip()
        await self._dispatch_final(fallback, self.language or "yue")

    async def end_utterance(self) -> None:
        """Commit one utterance and provide a timeout fallback final callback."""
        if self._closed:
            return
        self._final_event.clear()
        self._final_dispatched = False
        ws = self._ws
        if ws is None:
            await self._finish_failed_utterance()
            return
        try:
            await ws.send(
                json.dumps(
                    {
                        "event_id": self._event_id(),
                        "type": "input_audio_buffer.commit",
                    }
                )
            )
        except Exception as exc:
            logger.warning("Qwen ASR commit failed: %r", exc)
            await self._drop_connection(ws)
            await self._finish_failed_utterance()
            return
        # Avoid endless ENDPOINTING if cloud ASR never sends completed.
        try:
            await asyncio.wait_for(
                self._final_event.wait(),
                timeout=self.final_timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            await self._finish_failed_utterance()

    async def close(self) -> None:
        self._closed = True
        ws = self._ws
        if ws:
            try:
                await ws.send(
                    json.dumps(
                        {"event_id": self._event_id(), "type": "session.finish"}
                    )
                )
            except Exception:
                pass
        await self._drop_connection(ws)
