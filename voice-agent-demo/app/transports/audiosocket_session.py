"""Glue AudioSocket (8 kHz) ↔ CallSession (16 kHz) with resample.

```text
收 0x10(8k) → upsample 16k → CallSession.on_pcm16
CallSession 出 16k → downsample 8k → 320-byte 0x10 frames
Hangup / session.completed → close
```
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable

from app.config import Settings, get_settings
from app.models import ServerEvent
from app.orchestrator import CallSession
from app.transports.audiosocket import (
    KIND_AUDIO_SLIN,
    AudioSocketConnection,
    chunk_pcm,
)
from app.transports.resample import (
    AUDIOSOCKET_FRAME_BYTES_8K,
    AUDIOSOCKET_FRAME_MS,
    make_downsampler,
    make_upsampler,
)

logger = logging.getLogger(__name__)

EventHook = Callable[[ServerEvent], Awaitable[None]]


class AudioSocketCallBridge:
    """One Asterisk AudioSocket TCP connection bound to one CallSession."""

    def __init__(
        self,
        conn: AudioSocketConnection,
        settings: Settings | None = None,
        *,
        pace_audio: bool | None = None,
        on_event: EventHook | None = None,
    ) -> None:
        self.conn = conn
        self.settings = settings or get_settings()
        self.pace_audio = (
            self.settings.audiosocket_pace_audio
            if pace_audio is None
            else bool(pace_audio)
        )
        self.on_event = on_event
        self.up = make_upsampler()
        self.down = make_downsampler()
        self._out_buf = bytearray()
        self._session: CallSession | None = None
        self._closed = False
        self._write_lock = asyncio.Lock()

    @property
    def session(self) -> CallSession | None:
        return self._session

    async def run(self) -> None:
        try:
            async for frame in self.conn.read_frames():
                if self._closed:
                    break
                if frame.is_uuid:
                    if self._session is None:
                        await self._start_session(
                            self.conn.call_uuid or str(uuid.uuid4())
                        )
                    continue
                if frame.is_audio:
                    if frame.kind != KIND_AUDIO_SLIN:
                        logger.warning(
                            "AudioSocket unsupported audio kind=0x%02x "
                            "uuid=%s (expected 0x10 / 8k); dropping",
                            frame.kind,
                            self.conn.call_uuid or "-",
                        )
                        continue
                    if self._session is None:
                        await self._start_session(
                            self.conn.call_uuid or str(uuid.uuid4())
                        )
                    pcm16 = self.up.feed(frame.payload)
                    if pcm16 and self._session is not None:
                        await self._session.on_pcm16(pcm16)
                    continue
                if frame.is_dtmf:
                    digit = (
                        frame.payload.decode("ascii", errors="replace")
                        if frame.payload
                        else "?"
                    )
                    logger.info(
                        "AudioSocket DTMF digit=%s uuid=%s",
                        digit,
                        self.conn.call_uuid or "-",
                    )
                    continue
                if frame.is_error:
                    code = frame.payload[0] if frame.payload else None
                    logger.warning(
                        "AudioSocket error code=%s uuid=%s",
                        code,
                        self.conn.call_uuid or "-",
                    )
                    continue
                if frame.is_hangup:
                    logger.info(
                        "AudioSocket hangup uuid=%s",
                        self.conn.call_uuid or "-",
                    )
                    break
        finally:
            await self._shutdown()

    async def _start_session(self, call_id: str) -> None:
        if self._session is not None:
            return

        async def emit(event: ServerEvent) -> None:
            if event.type in {
                "state",
                "asr.final",
                "turn.done",
                "session.completed",
                "error",
            }:
                logger.info(
                    "sip_event type=%s call=%s data=%s",
                    event.type,
                    event.call_id,
                    event.data,
                )
            if self.on_event is not None:
                await self.on_event(event)
            if event.type == "session.completed" and not self._closed:
                # Order saved / call finished → hang up the AudioSocket leg.
                await self.conn.hangup()

        async def on_audio_out(pcm16k: bytes) -> None:
            await self._write_pcm16(pcm16k)

        async def on_audio_flush() -> None:
            await self._flush_out(pad=True)

        self._session = CallSession(
            emit=emit,
            settings=self.settings,
            call_id=call_id,
            on_audio_out=on_audio_out,
            on_audio_flush=on_audio_flush,
            auto_complete_playback=True,
        )
        logger.info(
            "AudioSocket CallSession started uuid=%s mode=%s",
            call_id,
            self.settings.agent_mode,
        )
        await self._session.start()

    async def _write_pcm16(self, pcm16k: bytes) -> None:
        if self._closed or not pcm16k:
            return
        pcm8k = self.down.feed(pcm16k)
        if not pcm8k:
            return
        async with self._write_lock:
            if self._closed:
                return
            self._out_buf.extend(pcm8k)
            frame_bytes = AUDIOSOCKET_FRAME_BYTES_8K
            while len(self._out_buf) >= frame_bytes:
                if self._closed:
                    return
                chunk = bytes(self._out_buf[:frame_bytes])
                del self._out_buf[:frame_bytes]
                await self.conn.send_audio(chunk)
                if self.pace_audio:
                    await asyncio.sleep(AUDIOSOCKET_FRAME_MS / 1000.0)

    async def _flush_out(self, *, pad: bool) -> None:
        async with self._write_lock:
            if self._closed or not self._out_buf:
                return
            pending = bytes(self._out_buf)
            self._out_buf.clear()
            frames = (
                chunk_pcm(pending, frame_bytes=AUDIOSOCKET_FRAME_BYTES_8K)
                if pad
                else (
                    [pending]
                    if len(pending) == AUDIOSOCKET_FRAME_BYTES_8K
                    else []
                )
            )
            for i, chunk in enumerate(frames):
                if self._closed:
                    return
                await self.conn.send_audio(chunk)
                if self.pace_audio and i + 1 < len(frames):
                    await asyncio.sleep(AUDIOSOCKET_FRAME_MS / 1000.0)

    async def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._flush_out(pad=True)
        except Exception:
            logger.exception(
                "AudioSocket flush failed uuid=%s",
                self.conn.call_uuid or "-",
            )
        session = self._session
        self._session = None
        if session is not None:
            try:
                await session.close()
            except Exception:
                logger.exception(
                    "CallSession close failed uuid=%s",
                    self.conn.call_uuid or "-",
                )
        await self.conn.close()


async def handle_audiosocket_call(
    conn: AudioSocketConnection,
    settings: Settings | None = None,
    *,
    pace_audio: bool | None = None,
    on_event: EventHook | None = None,
) -> None:
    """AudioSocketServer handler entrypoint."""
    bridge = AudioSocketCallBridge(
        conn,
        settings=settings,
        pace_audio=pace_audio,
        on_event=on_event,
    )
    await bridge.run()
