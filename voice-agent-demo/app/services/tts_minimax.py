from __future__ import annotations

import asyncio
import json
import math
import struct
from collections.abc import AsyncIterator, Awaitable, Callable

from app.services.text_norm import language_boost_for

FirstAudioCallback = Callable[[], Awaitable[None]]
AudioChunkCallback = Callable[[bytes], Awaitable[None]]


def _sine_pcm16(text: str, *, sample_rate: int = 16000) -> bytes:
    duration = max(0.16, min(2.2, len(text) * 0.04))
    n = int(sample_rate * duration)
    amp = 0.16
    freq = 440.0
    out = bytearray()
    for i in range(n):
        t = i / sample_rate
        env = min(1.0, t * 20) * min(1.0, (duration - t) * 20)
        sample = int(32767 * amp * env * math.sin(2 * math.pi * freq * t))
        out.extend(struct.pack("<h", sample))
    return bytes(out)


class MiniMaxStreamingTTS:
    """
    MiniMax speech-2.8-turbo WebSocket T2A.

    Keeps a warmed socket (connect + task_start) so first clause is not cold.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "speech-2.8-turbo",
        voice: str = "male-qn-qingse",
        language_boost: str = "Chinese,Yue",
        ws_url: str = "wss://api.minimaxi.com/ws/v1/t2a_v2",
        sample_rate: int = 16000,
        audio_format: str = "pcm",
        mock: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.language_boost = language_boost
        self.ws_url = ws_url
        self.sample_rate = sample_rate
        self.audio_format = audio_format
        self.mock = mock or not api_key
        self._ws = None
        self._ready = False
        self._lock = asyncio.Lock()
        self._warm_lang = "yue"

    async def warm(self, reply_lang: str = "yue") -> None:
        if self.mock:
            return
        try:
            await self._ensure_ready(reply_lang)
        except Exception:
            await self._drop()

    async def _drop(self) -> None:
        self._ready = False
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    async def _ensure_ready(self, reply_lang: str) -> None:
        import websockets

        async with self._lock:
            if self._ws is not None and self._ready and self._warm_lang == reply_lang:
                return
            await self._drop()
            boost = language_boost_for(reply_lang) or self.language_boost
            headers = {"Authorization": f"Bearer {self.api_key}"}
            ws = await websockets.connect(
                self.ws_url,
                additional_headers=headers,
                max_size=8 * 1024 * 1024,
                open_timeout=8.0,
            )
            raw = await asyncio.wait_for(ws.recv(), timeout=8.0)
            msg = json.loads(raw)
            if msg.get("event") == "task_failed":
                await ws.close()
                raise RuntimeError(f"MiniMax connect failed: {msg}")

            await ws.send(
                json.dumps(
                    {
                        "event": "task_start",
                        "model": self.model,
                        "language_boost": boost,
                        "voice_setting": {
                            "voice_id": self.voice,
                            "speed": 1.08,
                            "vol": 1,
                            "pitch": 0,
                        },
                        "audio_setting": {
                            "sample_rate": self.sample_rate,
                            "bitrate": 128000,
                            "format": self.audio_format,
                            "channel": 1,
                        },
                    }
                )
            )
            started = json.loads(await asyncio.wait_for(ws.recv(), timeout=8.0))
            if started.get("event") != "task_started":
                await ws.close()
                raise RuntimeError(f"MiniMax task_start failed: {started}")
            self._ws = ws
            self._ready = True
            self._warm_lang = reply_lang

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        *,
        reply_lang: str = "yue",
        on_first_audio: FirstAudioCallback | None = None,
        on_audio_chunk: AudioChunkCallback | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> str:
        if self.mock:
            return await self._mock_synthesize(
                text_stream,
                on_first_audio=on_first_audio,
                on_audio_chunk=on_audio_chunk,
                cancel_event=cancel_event,
            )
        return await self._ws_synthesize(
            text_stream,
            reply_lang=reply_lang,
            on_first_audio=on_first_audio,
            on_audio_chunk=on_audio_chunk,
            cancel_event=cancel_event,
        )

    async def _mock_synthesize(
        self,
        text_stream: AsyncIterator[str],
        *,
        on_first_audio: FirstAudioCallback | None,
        on_audio_chunk: AudioChunkCallback | None,
        cancel_event: asyncio.Event | None,
    ) -> str:
        buf = ""
        full = ""
        first = True
        async for token in text_stream:
            if cancel_event and cancel_event.is_set():
                break
            buf += token
            full += token
            if any(p in buf for p in "，。！？,!.?、；;") or len(buf) >= 8:
                pcm = _sine_pcm16(buf, sample_rate=self.sample_rate)
                if first:
                    first = False
                    if on_first_audio:
                        await on_first_audio()
                if on_audio_chunk:
                    frame = int(self.sample_rate * 0.04) * 2
                    for i in range(0, len(pcm), frame):
                        if cancel_event and cancel_event.is_set():
                            return full
                        await on_audio_chunk(pcm[i : i + frame])
                        await asyncio.sleep(0.008)
                buf = ""
        if buf and not (cancel_event and cancel_event.is_set()):
            pcm = _sine_pcm16(buf, sample_rate=self.sample_rate)
            if first and on_first_audio:
                await on_first_audio()
            if on_audio_chunk:
                await on_audio_chunk(pcm)
        return full

    async def _ws_synthesize(
        self,
        text_stream: AsyncIterator[str],
        *,
        reply_lang: str,
        on_first_audio: FirstAudioCallback | None,
        on_audio_chunk: AudioChunkCallback | None,
        cancel_event: asyncio.Event | None,
    ) -> str:
        # Warm socket in parallel with collecting the first speakable clause.
        ready_task = asyncio.create_task(self._ensure_ready(reply_lang))
        full = ""
        first = True
        pending = ""

        async def drain_until_final(ws) -> None:
            nonlocal first
            while True:
                if cancel_event and cancel_event.is_set():
                    return
                raw_msg = await asyncio.wait_for(ws.recv(), timeout=15.0)
                obj = json.loads(raw_msg)
                status = (obj.get("base_resp") or {}).get("status_code")
                if isinstance(status, int) and status != 0:
                    raise RuntimeError(
                        f"MiniMax error {status}: {(obj.get('base_resp') or {}).get('status_msg')}"
                    )
                if obj.get("event") == "task_failed":
                    raise RuntimeError(f"MiniMax task_failed: {obj}")
                hex_audio = (obj.get("data") or {}).get("audio") or ""
                if hex_audio:
                    pcm = bytes.fromhex(hex_audio)
                    if first:
                        first = False
                        if on_first_audio:
                            await on_first_audio()
                    if on_audio_chunk and pcm:
                        await on_audio_chunk(pcm)
                if obj.get("is_final") is True or obj.get("event") == "task_finished":
                    return

        try:
            async for token in text_stream:
                if cancel_event and cancel_event.is_set():
                    break
                pending += token
                full += token
                # Flush early: punctuation or first 8 chars (FAQ short answers).
                if any(p in pending for p in "，。！？,!.?、；;") or (
                    first and len(pending) >= 8
                ):
                    clause = pending.strip()
                    pending = ""
                    if not clause:
                        continue
                    await ready_task
                    ws = self._ws
                    assert ws is not None
                    await ws.send(
                        json.dumps({"event": "task_continue", "text": clause})
                    )
                    await drain_until_final(ws)

            if pending.strip() and not (cancel_event and cancel_event.is_set()):
                await ready_task
                ws = self._ws
                assert ws is not None
                await ws.send(
                    json.dumps({"event": "task_continue", "text": pending.strip()})
                )
                await drain_until_final(ws)

            if self._ws is not None:
                try:
                    await self._ws.send(json.dumps({"event": "task_finish"}))
                    await asyncio.wait_for(self._ws.recv(), timeout=3.0)
                except Exception:
                    pass
        except Exception:
            await self._drop()
            raise
        finally:
            # Re-warm next turn in background.
            await self._drop()
            asyncio.create_task(self.warm(reply_lang))

        return full
