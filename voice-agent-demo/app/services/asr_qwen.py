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

    @property
    def is_ready(self) -> bool:
        return self._started and not self._closed

    async def wait_until_ready(self, timeout_s: float) -> bool:
        if not self.is_ready and timeout_s > 0:
            await asyncio.sleep(0)
        return self.is_ready

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

    async def end_utterance(self, *, fast_fail: bool = False) -> None:
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
        hard_timeout_ms: int = 5000,
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
        self.hard_timeout_ms = max(self.final_timeout_ms, int(hard_timeout_ms))
        self.on_partial = on_partial
        self.on_final = on_final
        self._ws = None
        self._reader_task: asyncio.Task | None = None
        self._closed = False
        self._final_event = asyncio.Event()
        self._session_ready_event = asyncio.Event()
        self._last_final = ""
        self._partial = ""
        self._final_dispatched = False
        self._dispatch_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._last_connect_attempt = 0.0
        self._reconnect_cooldown_s = 1.0
        self._reconnect_task: asyncio.Task | None = None
        self._utterance_pcm = bytearray()
        self._pre_roll_pcm = bytearray()
        self._pre_roll_bytes = 16000 * 2 * 300 // 1000
        self._max_replay_bytes = 16000 * 2 * 20
        self._speech_capture_active = False
        self._buffer_only_until_commit = False

    async def _dispatch_final(self, text: str, lang: str) -> None:
        async with self._dispatch_lock:
            if self._final_dispatched:
                return
            self._final_dispatched = True
            self._last_final = text
            self._partial = ""
            self._utterance_pcm.clear()
            self._pre_roll_pcm.clear()
            self._speech_capture_active = False
            self._buffer_only_until_commit = False
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
            open_timeout=8.0,
            ping_interval=20.0,
            ping_timeout=20.0,
        )
        self._session_ready_event.clear()
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
        try:
            await asyncio.wait_for(self._session_ready_event.wait(), timeout=3.0)
        except Exception:
            await self._drop_connection(ws)
            raise

    async def _detach_connection(self, ws) -> None:
        """Stop routing events from a stale socket without waiting for close."""
        if self._ws is ws:
            self._ws = None
            self._session_ready_event.clear()
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

    async def _drop_connection(self, ws) -> None:
        await self._detach_connection(ws)
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
        self._schedule_reconnect(force=True)
        # Connection warm-up runs behind the greeting instead of delaying it.
        await asyncio.sleep(0)

    @property
    def is_ready(self) -> bool:
        return (
            not self._closed
            and self._ws is not None
            and self._session_ready_event.is_set()
            and self._reader_task is not None
            and not self._reader_task.done()
        )

    async def wait_until_ready(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while not self.is_ready and not self._closed:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._schedule_reconnect(force=True)
            task = self._reconnect_task
            if task is None:
                await asyncio.sleep(min(0.2, remaining))
                continue
            try:
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=min(remaining, 8.5),
                )
            except asyncio.TimeoutError:
                pass
            if not self.is_ready:
                await asyncio.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        return self.is_ready

    def _schedule_reconnect(self, *, force: bool = False) -> None:
        if self._closed:
            return
        if self._reconnect_task and not self._reconnect_task.done():
            return

        async def run() -> None:
            connected = await self._reconnect(force=force)
            if not connected and not self._closed:
                logger.warning(
                    "Qwen ASR connection unavailable; "
                    "buffering locally until utterance end"
                )

        task = asyncio.create_task(run())
        self._reconnect_task = task

        def clear(done: asyncio.Task) -> None:
            if self._reconnect_task is done:
                self._reconnect_task = None

        task.add_done_callback(clear)

    def mark_speech_start(self) -> None:
        """Start a replayable utterance with 300ms of local pre-roll."""
        if self._speech_capture_active:
            return
        self._utterance_pcm = bytearray(self._pre_roll_pcm)
        self._pre_roll_pcm.clear()
        self._speech_capture_active = True

    def mark_speech_end(self) -> None:
        self._speech_capture_active = False

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
                if event == "session.updated":
                    self._session_ready_event.set()
                elif event == "conversation.item.input_audio_transcription.delta":
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
                self._session_ready_event.clear()
            if self._reader_task is asyncio.current_task():
                self._reader_task = None

    async def push_pcm16(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        if self._speech_capture_active:
            self._utterance_pcm.extend(pcm)
            if len(self._utterance_pcm) > self._max_replay_bytes:
                del self._utterance_pcm[:-self._max_replay_bytes]
        else:
            self._pre_roll_pcm.extend(pcm)
            if len(self._pre_roll_pcm) > self._pre_roll_bytes:
                del self._pre_roll_pcm[:-self._pre_roll_bytes]

        ws = self._ws
        if ws is None:
            # Never hold up local VAD while a cloud handshake is pending.
            self._buffer_only_until_commit = True
            self._schedule_reconnect()
            return
        if self._buffer_only_until_commit:
            # This socket connected after part of the utterance was missed.
            # Keep it clean and submit the complete local buffer at EoU.
            return

        payload = json.dumps(
            {
                "event_id": self._event_id(),
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )
        try:
            await ws.send(payload)
            return
        except Exception as exc:
            logger.warning(
                "Qwen ASR audio send failed; buffering until EoU: %r",
                exc,
            )
        self._buffer_only_until_commit = True
        await self._detach_connection(ws)
        self._schedule_reconnect(force=True)

    async def _finish_failed_utterance(self) -> None:
        fallback = (self._partial or "").strip()
        await self._dispatch_final(fallback, self.language or "yue")

    def _buffered_timeout_s(self, audio: bytes) -> float:
        duration_s = len(audio) / 2 / 16000
        # Replayed audio is uploaded faster than realtime, but long utterances
        # still need more processing time than the normal streaming final.
        return max(
            self.final_timeout_ms / 1000.0,
            min(10.0, 2.0 + duration_s * 0.6),
        )

    async def _wait_for_connection(self) -> bool:
        task = self._reconnect_task
        if task and not task.done():
            try:
                await task
            except asyncio.CancelledError:
                return False
        if self._ws is not None:
            return True
        return await self._reconnect(force=True)

    async def _submit_buffered_utterance(
        self,
        *,
        failed_ws=None,
        reconnect: bool,
        label: str,
    ) -> bool:
        """Submit the locally bounded speech buffer and wait for a final."""
        audio = bytes(self._utterance_pcm)
        if not audio or self._closed:
            return False
        logger.warning(
            "Qwen ASR submitting buffered utterance label=%s bytes=%d duration_ms=%.1f",
            label,
            len(audio),
            len(audio) / 2 / 16000 * 1000.0,
        )
        if reconnect:
            await self._detach_connection(failed_ws)
            if failed_ws:
                try:
                    await asyncio.wait_for(failed_ws.close(), timeout=0.25)
                except Exception:
                    pass
            if not await self._reconnect(force=True):
                return False
        elif not await self._wait_for_connection():
            return False

        submit_ws = self._ws
        if submit_ws is None:
            return False
        self._final_event.clear()
        self._final_dispatched = False
        timeout_s = self._buffered_timeout_s(audio)
        try:
            # Moderate chunks avoid one very large JSON/WebSocket frame.
            chunk_bytes = 3200
            for offset in range(0, len(audio), chunk_bytes):
                chunk = audio[offset : offset + chunk_bytes]
                await submit_ws.send(
                    json.dumps(
                        {
                            "event_id": self._event_id(),
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        }
                    )
                )
            await submit_ws.send(
                json.dumps(
                    {
                        "event_id": self._event_id(),
                        "type": "input_audio_buffer.commit",
                    }
                )
            )
            await asyncio.wait_for(
                self._final_event.wait(),
                timeout=timeout_s,
            )
            logger.info(
                "Qwen ASR buffered utterance succeeded label=%s timeout_s=%.1f",
                label,
                timeout_s,
            )
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "Qwen ASR buffered utterance timed out label=%s timeout_s=%.1f",
                label,
                timeout_s,
            )
            return False
        except Exception as exc:
            logger.warning(
                "Qwen ASR buffered utterance failed label=%s: %r",
                label,
                exc,
            )
            return False

    async def _recover_failed_utterance(self, ws, *, reason: str) -> None:
        """
        Finalize with the last partial, then replace the connection.

        A timed-out DashScope session must not be reused: a late completed
        event from it can otherwise be mistaken for the next utterance.
        """
        fallback = (self._partial or "").strip()
        logger.warning(
            "Qwen ASR utterance failed (%s); dropping connection and "
            "dispatching fallback chars=%d",
            reason,
            len(fallback),
        )
        # Detach first so a late completion cannot leak into the next turn.
        # Dispatch the fallback before waiting for the WebSocket close
        # handshake; close may otherwise add seconds of visible ENDPOINTING.
        await self._detach_connection(ws)
        await self._dispatch_final(fallback, self.language or "yue")
        await self._drop_connection(ws)
        if not self._closed:
            # Pass the failed socket so a concurrent audio-triggered reconnect
            # is preserved instead of being replaced again.
            await self._reconnect(ws, force=True)

    async def _end_utterance_impl(self, *, fast_fail: bool = False) -> None:
        """Commit one utterance and provide a timeout fallback final callback."""
        if self._closed:
            return
        self._final_event.clear()
        self._final_dispatched = False
        ws = self._ws
        if self._buffer_only_until_commit:
            if await self._submit_buffered_utterance(
                failed_ws=ws,
                reconnect=False,
                label="connection_gap",
            ):
                return
            await self._recover_failed_utterance(
                self._ws or ws,
                reason="buffered_submit_failed",
            )
            return
        if ws is None:
            if await self._submit_buffered_utterance(
                reconnect=False,
                label="disconnected",
            ):
                return
            await self._finish_failed_utterance()
            self._schedule_reconnect(force=True)
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
            if await self._submit_buffered_utterance(
                failed_ws=ws,
                reconnect=True,
                label="commit_retry",
            ):
                return
            await self._recover_failed_utterance(
                self._ws or ws,
                reason="commit_failed",
            )
            return
        # Avoid endless ENDPOINTING if cloud ASR never sends completed.
        try:
            await asyncio.wait_for(
                self._final_event.wait(),
                timeout=(
                    min(self.final_timeout_ms, 1200)
                    if fast_fail
                    else self.final_timeout_ms
                )
                / 1000.0,
            )
        except asyncio.TimeoutError:
            if fast_fail and not self._partial:
                await self._recover_failed_utterance(
                    ws,
                    reason="short_utterance_timeout",
                )
                return
            # If DashScope produced no partial at all, replay the locally
            # buffered utterance once on a fresh connection before giving up.
            if not self._partial:
                if await self._submit_buffered_utterance(
                    failed_ws=ws,
                    reconnect=True,
                    label="final_retry",
                ):
                    return
            await self._recover_failed_utterance(
                self._ws or ws,
                reason="final_timeout",
            )

    async def end_utterance(self, *, fast_fail: bool = False) -> None:
        try:
            await asyncio.wait_for(
                self._end_utterance_impl(fast_fail=fast_fail),
                timeout=self.hard_timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Qwen ASR hard timeout after %dms; returning empty final",
                self.hard_timeout_ms,
            )
            if self._final_dispatched:
                return
            failed_ws = self._ws
            await self._detach_connection(failed_ws)
            await self._dispatch_final("", self.language or "yue")
            if failed_ws:
                try:
                    await asyncio.wait_for(failed_ws.close(), timeout=0.25)
                except Exception:
                    pass
            self._schedule_reconnect(force=True)

    async def close(self) -> None:
        self._closed = True
        reconnect_task = self._reconnect_task
        if reconnect_task and not reconnect_task.done():
            reconnect_task.cancel()
            try:
                await reconnect_task
            except asyncio.CancelledError:
                pass
        self._reconnect_task = None
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
