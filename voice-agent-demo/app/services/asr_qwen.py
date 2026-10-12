from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from collections import deque
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
        stagnation_ms: int = 1500,
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
        self.stagnation_ms = max(300, int(stagnation_ms))
        self.on_partial = on_partial
        self.on_final = on_final
        self._ws = None
        self._reader_task: asyncio.Task | None = None
        self._closed = False
        self._final_event = asyncio.Event()
        self._session_ready_event = asyncio.Event()
        self._last_final = ""
        self._partial = ""
        self._partial_updated_at = 0.0
        self._final_dispatched = False
        self._dispatch_lock = asyncio.Lock()
        self._end_utterance_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._last_connect_attempt = 0.0
        self._reconnect_cooldown_s = 1.0
        self._reconnect_task: asyncio.Task | None = None
        self._reconnect_after_current = False
        self._standby_ws = None
        self._standby_task: asyncio.Task | None = None
        self._retire_tasks: set[asyncio.Task] = set()
        self._utterance_pcm = bytearray()
        self._pre_roll_pcm = bytearray()
        self._pre_roll_bytes = 16000 * 2 * 300 // 1000
        self._max_replay_bytes = 16000 * 2 * 20
        self._speech_capture_active = False
        self._buffer_only_until_commit = False
        self._turn_sequence = 0
        self._active_turn_id = 0
        self._capture_turn_id: int | None = None
        self._expired_turn_ids: set[int] = set()
        self._pending_commit_turns: deque[int] = deque()
        self._item_turns: dict[str, int] = {}

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

    def _session_update_payload(self) -> str:
        return json.dumps(
            {
                "event_id": self._event_id(),
                "type": "session.update",
                "session": {
                    "input_audio_format": "pcm",
                    "sample_rate": 16000,
                    "input_audio_transcription": (
                        {"language": self.language}
                        if self.language
                        else {}
                    ),
                    "turn_detection": None,
                },
            }
        )

    async def _recv_one(self, ws):
        receiver = getattr(ws, "recv", None)
        if callable(receiver):
            return await receiver()
        return await ws.__anext__()

    async def _create_ready_socket(self):
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
        try:
            await ws.send(self._session_update_payload())
            deadline = time.monotonic() + 3.0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                raw = await asyncio.wait_for(
                    self._recv_one(ws),
                    timeout=remaining,
                )
                msg = json.loads(raw)
                event = msg.get("type") or ""
                if event == "session.updated":
                    return ws
                if event == "error":
                    raise ConnectionError(f"Qwen ASR setup failed: {msg}")
        except BaseException:
            try:
                await asyncio.wait_for(ws.close(), timeout=0.25)
            except BaseException:
                pass
            raise

    def _activate_socket(self, ws, *, source: str) -> None:
        self._ws = ws
        self._session_ready_event.set()
        self._reader_task = asyncio.create_task(self._read_loop(ws))
        self._pending_commit_turns.clear()
        self._item_turns.clear()
        logger.info("Qwen ASR %s connection activated", source)

    def _activate_standby_if_ready(self) -> bool:
        ws = self._standby_ws
        if ws is None:
            return False
        self._standby_ws = None
        self._activate_socket(ws, source="standby")
        return True

    def prepare_standby(self) -> None:
        """Warm one idle socket so an expired ASR turn can rotate instantly."""
        if (
            self._closed
            or not self.is_ready
            or self._standby_ws is not None
            or (
                self._standby_task is not None
                and not self._standby_task.done()
            )
        ):
            return

        async def run() -> None:
            try:
                ws = await self._create_ready_socket()
                if self._closed:
                    await ws.close()
                    return
                if self._ws is None:
                    self._activate_socket(ws, source="standby-recovery")
                elif self._standby_ws is None:
                    self._standby_ws = ws
                    logger.info("Qwen ASR standby connection ready")
                else:
                    await ws.close()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._closed:
                    logger.warning(
                        "Qwen ASR standby connection failed: %r",
                        exc,
                    )
                    if self._ws is None:
                        self._schedule_reconnect(
                            force=True,
                            after_current=True,
                        )

        task = asyncio.create_task(run())
        self._standby_task = task

        def clear(done: asyncio.Task) -> None:
            try:
                done.exception()
            except asyncio.CancelledError:
                pass
            if self._standby_task is done:
                self._standby_task = None

        task.add_done_callback(clear)

    def _retire_socket(self, ws, reader: asyncio.Task | None) -> None:
        if ws is None:
            return

        async def retire() -> None:
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
            try:
                await asyncio.wait_for(ws.close(), timeout=0.25)
            except Exception:
                pass

        task = asyncio.create_task(retire())
        self._retire_tasks.add(task)
        task.add_done_callback(self._retire_tasks.discard)

    def _rotate_expired_connection(self) -> bool:
        old_ws = self._ws
        old_reader = self._reader_task
        self._ws = None
        self._reader_task = None
        self._session_ready_event.clear()
        self._pending_commit_turns.clear()
        self._item_turns.clear()
        promoted = self._activate_standby_if_ready()
        self._retire_socket(old_ws, old_reader)
        if (
            not promoted
            and (
                self._standby_task is None
                or self._standby_task.done()
            )
        ):
            self._schedule_reconnect(force=True)
        return promoted

    async def _send_commit(
        self,
        ws,
        *,
        turn_id: int,
        deadline: float,
    ) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError
        self._pending_commit_turns.append(turn_id)
        try:
            await asyncio.wait_for(
                ws.send(
                    json.dumps(
                        {
                            "event_id": self._event_id(),
                            "type": "input_audio_buffer.commit",
                        }
                    )
                ),
                timeout=remaining,
            )
        except BaseException:
            try:
                self._pending_commit_turns.remove(turn_id)
            except ValueError:
                pass
            raise

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
        deadline = time.monotonic() + 3.0
        try:
            while not self._session_ready_event.is_set():
                if self._ws is not ws:
                    raise ConnectionError(
                        "Qwen ASR connection was replaced during handshake"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                try:
                    await asyncio.wait_for(
                        self._session_ready_event.wait(),
                        timeout=min(0.05, remaining),
                    )
                except asyncio.TimeoutError:
                    continue
            if self._ws is not ws:
                raise ConnectionError(
                    "Qwen ASR connection was replaced during handshake"
                )
        except Exception:
            await self._drop_connection(ws)
            raise

    async def _detach_connection(self, ws) -> None:
        """Stop routing events from a stale socket without waiting for close."""
        if self._ws is ws:
            self._ws = None
            self._session_ready_event.clear()
            self._pending_commit_turns.clear()
            self._item_turns.clear()
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
                if not self.is_ready:
                    logger.warning(
                        "Qwen ASR connection became stale during handshake"
                    )
                    return False
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
            if self._activate_standby_if_ready():
                continue
            standby_task = self._standby_task
            if standby_task is not None and not standby_task.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(standby_task),
                        timeout=min(remaining, 8.5),
                    )
                except asyncio.TimeoutError:
                    pass
                continue
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

    def _schedule_reconnect(
        self,
        *,
        force: bool = False,
        after_current: bool = False,
    ) -> None:
        if self._closed:
            return
        if self._reconnect_task and not self._reconnect_task.done():
            if after_current:
                self._reconnect_after_current = True
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
            try:
                done.exception()
            except asyncio.CancelledError:
                pass
            if self._reconnect_task is done:
                self._reconnect_task = None
            if self._reconnect_after_current and not self._closed:
                self._reconnect_after_current = False
                self._schedule_reconnect(force=True)

        task.add_done_callback(clear)

    def mark_speech_start(self) -> None:
        """Start a replayable utterance with 300ms of local pre-roll."""
        if self._speech_capture_active:
            return
        self._turn_sequence += 1
        self._active_turn_id = self._turn_sequence
        self._capture_turn_id = self._active_turn_id
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
                if self._ws is not ws:
                    logger.info(
                        "Qwen ASR ignored event from retired connection "
                        "event=%s",
                        event,
                    )
                    continue
                if event == "session.updated":
                    self._session_ready_event.set()
                elif event == "input_audio_buffer.committed":
                    turn_id = (
                        self._pending_commit_turns.popleft()
                        if self._pending_commit_turns
                        else self._active_turn_id
                    )
                    item_id = str(msg.get("item_id") or "")
                    if item_id:
                        self._item_turns[item_id] = turn_id
                    logger.info(
                        "Qwen ASR audio committed turn=%d item_id=%s",
                        turn_id,
                        item_id or "-",
                    )
                elif event in {
                    "conversation.item.input_audio_transcription.delta",
                    "conversation.item.input_audio_transcription.text",
                }:
                    item_id = str(msg.get("item_id") or "")
                    if item_id and item_id not in self._item_turns:
                        logger.info(
                            "Qwen ASR ignored unmapped partial item_id=%s",
                            item_id,
                        )
                        continue
                    turn_id = self._item_turns.get(
                        item_id,
                        self._active_turn_id,
                    )
                    if (
                        turn_id in self._expired_turn_ids
                        or turn_id != self._active_turn_id
                    ):
                        logger.info(
                            "Qwen ASR ignored stale partial turn=%d "
                            "active_turn=%d item_id=%s",
                            turn_id,
                            self._active_turn_id,
                            item_id or "-",
                        )
                        continue
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
                        self._partial_updated_at = time.monotonic()
                        if self.on_partial:
                            await self.on_partial(self._partial)
                elif event == "conversation.item.input_audio_transcription.completed":
                    item_id = str(msg.get("item_id") or "")
                    if item_id and item_id not in self._item_turns:
                        logger.info(
                            "Qwen ASR ignored unmapped final item_id=%s",
                            item_id,
                        )
                        continue
                    turn_id = self._item_turns.get(
                        item_id,
                        self._active_turn_id,
                    )
                    if (
                        turn_id in self._expired_turn_ids
                        or turn_id != self._active_turn_id
                    ):
                        logger.info(
                            "Qwen ASR ignored stale final turn=%d "
                            "active_turn=%d item_id=%s",
                            turn_id,
                            self._active_turn_id,
                            item_id or "-",
                        )
                        if item_id:
                            self._item_turns.pop(item_id, None)
                        continue
                    text = (
                        msg.get("transcript")
                        or msg.get("text")
                        or self._partial
                        or ""
                    ).strip()
                    lang = (msg.get("language") or self.language or "")
                    logger.info(
                        "Qwen ASR transcription completed turn=%d "
                        "item_id=%s chars=%d",
                        turn_id,
                        item_id or "-",
                        len(text),
                    )
                    if item_id:
                        self._item_turns.pop(item_id, None)
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
            disconnected_current = self._ws is ws
            if self._ws is ws:
                self._ws = None
                self._session_ready_event.clear()
                self._pending_commit_turns.clear()
                self._item_turns.clear()
            if self._reader_task is asyncio.current_task():
                self._reader_task = None
            if disconnected_current and not self._closed:
                if not self._activate_standby_if_ready() and (
                    self._standby_task is None
                    or self._standby_task.done()
                ):
                    self._schedule_reconnect(force=True, after_current=True)

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
        for _ in range(2):
            if self.is_ready:
                return True
            task = self._reconnect_task
            if task is not None and task.done():
                if self._reconnect_task is task:
                    self._reconnect_task = None
                task = None
            if task is None:
                self._schedule_reconnect(force=True)
                task = self._reconnect_task
            if task is not None:
                # A caller's utterance timeout must not cancel the shared
                # WebSocket handshake and leak an internal parser Future.
                await asyncio.shield(task)
        return self.is_ready

    async def _submit_buffered_utterance(
        self,
        *,
        turn_id: int,
        deadline: float,
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
            if not await self._wait_for_connection():
                return False
        elif not await self._wait_for_connection():
            return False

        submit_ws = self._ws
        if submit_ws is None:
            return False
        self._final_event.clear()
        self._final_dispatched = False
        timeout_s = self._buffered_timeout_s(audio)
        commit_sent = False
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
            await self._send_commit(
                submit_ws,
                turn_id=turn_id,
                deadline=deadline,
            )
            commit_sent = True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            await asyncio.wait_for(
                self._final_event.wait(),
                timeout=min(timeout_s, remaining),
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
            if commit_sent:
                await self._expire_utterance(
                    turn_id,
                    reason=f"{label}_timeout",
                )
                return True
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
        Finalize with the last partial after a real transport failure, then
        replace the unusable connection.
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
        if ws:
            try:
                await asyncio.wait_for(ws.close(), timeout=0.25)
            except Exception:
                pass
        if not self._closed:
            # Reconnect independently. The caller has already received a final
            # fallback and must not remain blocked behind a slow handshake.
            self._schedule_reconnect(force=True, after_current=True)

    async def _expire_utterance(self, turn_id: int, *, reason: str) -> None:
        """End one turn and rotate away from its backlogged socket."""
        if turn_id != self._active_turn_id:
            return
        self._expired_turn_ids.add(turn_id)
        if len(self._expired_turn_ids) > 32:
            oldest = sorted(self._expired_turn_ids)[:-32]
            self._expired_turn_ids.difference_update(oldest)
            for item_id, mapped_turn in list(self._item_turns.items()):
                if mapped_turn in oldest:
                    self._item_turns.pop(item_id, None)
        fallback = (self._partial or "").strip()
        promoted = self._rotate_expired_connection()
        logger.warning(
            "Qwen ASR utterance expired (%s); rotated connection and "
            "dispatching fallback turn=%d chars=%d standby_promoted=%s",
            reason,
            turn_id,
            len(fallback),
            promoted,
        )
        await self._dispatch_final(fallback, self.language or "yue")

    async def _end_utterance_impl(
        self,
        *,
        turn_id: int,
        deadline: float,
        fast_fail: bool = False,
    ) -> None:
        """Commit one utterance and provide a timeout fallback final callback."""
        if self._closed:
            return
        self._final_event.clear()
        self._final_dispatched = False
        ws = self._ws
        if self._buffer_only_until_commit:
            if await self._submit_buffered_utterance(
                turn_id=turn_id,
                deadline=deadline,
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
                turn_id=turn_id,
                deadline=deadline,
                reconnect=False,
                label="disconnected",
            ):
                return
            await self._finish_failed_utterance()
            self._schedule_reconnect(force=True)
            return
        try:
            await self._send_commit(
                ws,
                turn_id=turn_id,
                deadline=deadline,
            )
        except Exception as exc:
            logger.warning("Qwen ASR commit failed: %r", exc)
            await self._recover_failed_utterance(
                ws,
                reason="commit_failed",
            )
            return
        # Customer-facing wait only. Do not sync-reconnect/replay here: that
        # path races the hard timeout and can kill a fresh handshake.
        # Short turns: final_timeout_ms (default 2.5s) then reprompt.
        # Long turns: finalize early once the partial stops growing, so a lost
        # final event doesn't strand the caller for the full hard timeout.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            await self._expire_utterance(
                turn_id,
                reason=(
                    "short_utterance_timeout"
                    if fast_fail
                    else "final_timeout"
                ),
            )
            return

        stagnated = False
        stagnation_s = self.stagnation_ms / 1000.0
        while True:
            if self._final_dispatched or self._final_event.is_set():
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if (
                not fast_fail
                and self._partial
                and (time.monotonic() - self._partial_updated_at) >= stagnation_s
            ):
                stagnated = True
                break
            try:
                await asyncio.wait_for(
                    self._final_event.wait(),
                    timeout=min(0.1, remaining),
                )
                return
            except asyncio.TimeoutError:
                continue

        # A completed event can race the timeout boundary by a few
        # milliseconds. Never tear down a connection after its final was
        # already accepted and dispatched.
        if self._final_dispatched or self._final_event.is_set():
            return
        await self._expire_utterance(
            turn_id,
            reason=(
                "short_utterance_timeout"
                if fast_fail
                else ("partial_stagnation" if stagnated else "final_timeout")
            ),
        )

    async def end_utterance(self, *, fast_fail: bool = False) -> None:
        async with self._end_utterance_lock:
            turn_id = self._capture_turn_id
            self._capture_turn_id = None
            if turn_id is None:
                self._turn_sequence += 1
                turn_id = self._turn_sequence
                self._active_turn_id = turn_id
            wait_ms = (
                self.final_timeout_ms if fast_fail else self.hard_timeout_ms
            )
            deadline = time.monotonic() + wait_ms / 1000.0
            # Small headroom is only for callback dispatch and cleanup.
            safety_s = (wait_ms / 1000.0) + 0.5
            try:
                await asyncio.wait_for(
                    self._end_utterance_impl(
                        turn_id=turn_id,
                        deadline=deadline,
                        fast_fail=fast_fail,
                    ),
                    timeout=safety_s,
                )
            except asyncio.TimeoutError:
                if self._final_dispatched or self._final_event.is_set():
                    return
                logger.warning(
                    "Qwen ASR hard timeout after %dms; returning empty final",
                    wait_ms,
                )
                await self._expire_utterance(
                    turn_id,
                    reason="end_utterance_safety_timeout",
                )
                if not self.is_ready and not self._closed:
                    self._schedule_reconnect(force=True, after_current=True)

    async def close(self) -> None:
        self._closed = True
        standby_task = self._standby_task
        if standby_task and not standby_task.done():
            standby_task.cancel()
            try:
                await standby_task
            except asyncio.CancelledError:
                pass
        self._standby_task = None
        standby_ws = self._standby_ws
        self._standby_ws = None
        if standby_ws:
            try:
                await asyncio.wait_for(standby_ws.close(), timeout=0.25)
            except Exception:
                pass
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
        retire_tasks = list(self._retire_tasks)
        if retire_tasks:
            await asyncio.gather(*retire_tasks, return_exceptions=True)
