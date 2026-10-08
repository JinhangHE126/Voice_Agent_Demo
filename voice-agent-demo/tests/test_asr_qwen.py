from __future__ import annotations

import asyncio
import json

import pytest
import websockets

from app.services.asr_qwen import QwenRealtimeASR


class FakeWebSocket:
    def __init__(self, *, fail_event: str | None = None) -> None:
        self.fail_event = fail_event
        self.sent: list[dict] = []
        self._closed = asyncio.Event()

    async def send(self, raw: str) -> None:
        payload = json.loads(raw)
        if payload.get("type") == self.fail_event:
            raise ConnectionError(f"failed {self.fail_event}")
        self.sent.append(payload)

    async def close(self) -> None:
        self._closed.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self._closed.wait()
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_qwen_asr_initial_timeout_does_not_abort_session(monkeypatch) -> None:
    async def failing_connect(*args, **kwargs):
        raise TimeoutError("opening handshake timed out")

    monkeypatch.setattr(websockets, "connect", failing_connect)

    asr = QwenRealtimeASR(api_key="test", language="yue")
    await asr.start()

    assert asr._ws is None
    assert asr._closed is False

    await asr.close()


@pytest.mark.asyncio
async def test_qwen_asr_reconnects_and_retries_audio_send(monkeypatch) -> None:
    first = FakeWebSocket(fail_event="input_audio_buffer.append")
    second = FakeWebSocket()
    sockets = iter((first, second))
    connect_calls = 0

    async def fake_connect(*args, **kwargs):
        nonlocal connect_calls
        connect_calls += 1
        return next(sockets)

    monkeypatch.setattr(websockets, "connect", fake_connect)

    asr = QwenRealtimeASR(api_key="test", language="yue")
    await asr.start()
    await asr.push_pcm16(b"\x01\x02")

    assert connect_calls == 2
    assert any(
        item.get("type") == "input_audio_buffer.append"
        for item in second.sent
    )

    await asr.close()


@pytest.mark.asyncio
async def test_qwen_asr_commit_failure_dispatches_empty_final(monkeypatch) -> None:
    ws = FakeWebSocket(fail_event="input_audio_buffer.commit")
    finals: list[tuple[str, str]] = []

    async def fake_connect(*args, **kwargs):
        return ws

    async def on_final(text: str, lang: str) -> None:
        finals.append((text, lang))

    monkeypatch.setattr(websockets, "connect", fake_connect)

    asr = QwenRealtimeASR(
        api_key="test",
        language="yue",
        on_final=on_final,
    )
    await asr.start()
    await asr.end_utterance()

    assert finals == [("", "yue")]

    await asr.close()
