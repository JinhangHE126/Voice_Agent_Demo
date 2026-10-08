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


class CompletingWebSocket(FakeWebSocket):
    def __init__(self, transcript: str) -> None:
        super().__init__()
        self.transcript = transcript
        self._messages: asyncio.Queue[str | None] = asyncio.Queue()

    async def send(self, raw: str) -> None:
        await super().send(raw)
        payload = json.loads(raw)
        if payload.get("type") == "input_audio_buffer.commit":
            await self._messages.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": self.transcript,
                        "language": "zh",
                    }
                )
            )

    async def close(self) -> None:
        await self._messages.put(None)
        await super().close()

    async def __anext__(self):
        item = await self._messages.get()
        if item is None:
            raise StopAsyncIteration
        return item


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
async def test_qwen_asr_send_failure_buffers_and_submits_after_reconnect(
    monkeypatch,
) -> None:
    first = FakeWebSocket(fail_event="input_audio_buffer.append")
    second = CompletingWebSocket("重连成功")
    sockets = iter((first, second))
    connect_calls = 0

    async def fake_connect(*args, **kwargs):
        nonlocal connect_calls
        connect_calls += 1
        return next(sockets)

    monkeypatch.setattr(websockets, "connect", fake_connect)

    finals: list[tuple[str, str]] = []

    async def on_final(text: str, lang: str) -> None:
        finals.append((text, lang))

    asr = QwenRealtimeASR(
        api_key="test",
        language="yue",
        on_final=on_final,
    )
    await asr.start()
    asr.mark_speech_start()
    await asr.push_pcm16(b"\x01\x02")
    asr.mark_speech_end()
    await asr.end_utterance()

    assert connect_calls == 2
    assert finals == [("重连成功", "zh")]
    assert any(
        item.get("type") == "input_audio_buffer.append"
        for item in second.sent
    )

    await asr.close()


@pytest.mark.asyncio
async def test_qwen_asr_commit_failure_dispatches_empty_final(monkeypatch) -> None:
    first = FakeWebSocket(fail_event="input_audio_buffer.commit")
    second = FakeWebSocket()
    sockets = iter((first, second))
    finals: list[tuple[str, str]] = []

    async def fake_connect(*args, **kwargs):
        return next(sockets)

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
    assert asr._ws is second

    await asr.close()


@pytest.mark.asyncio
async def test_qwen_asr_final_timeout_replaces_connection(monkeypatch) -> None:
    first = FakeWebSocket()
    second = FakeWebSocket()
    sockets = iter((first, second))
    connect_calls = 0
    finals: list[tuple[str, str]] = []

    async def fake_connect(*args, **kwargs):
        nonlocal connect_calls
        connect_calls += 1
        return next(sockets)

    async def on_final(text: str, lang: str) -> None:
        finals.append((text, lang))

    monkeypatch.setattr(websockets, "connect", fake_connect)

    asr = QwenRealtimeASR(
        api_key="test",
        language="yue",
        final_timeout_ms=500,
        on_final=on_final,
    )
    await asr.start()
    await asr.end_utterance()

    assert finals == [("", "yue")]
    assert connect_calls == 2
    assert first._closed.is_set()
    assert asr._ws is second

    await asr.close()


@pytest.mark.asyncio
async def test_timeout_dispatches_fallback_before_slow_socket_close(
    monkeypatch,
) -> None:
    class SlowCloseWebSocket(FakeWebSocket):
        def __init__(self) -> None:
            super().__init__()
            self.close_started = asyncio.Event()
            self.allow_close = asyncio.Event()

        async def close(self) -> None:
            self.close_started.set()
            await self.allow_close.wait()
            await super().close()

    first = SlowCloseWebSocket()
    second = FakeWebSocket()
    sockets = iter((first, second))
    final_called = asyncio.Event()

    async def fake_connect(*args, **kwargs):
        return next(sockets)

    async def on_final(text: str, lang: str) -> None:
        final_called.set()

    monkeypatch.setattr(websockets, "connect", fake_connect)

    asr = QwenRealtimeASR(
        api_key="test",
        language="yue",
        final_timeout_ms=500,
        on_final=on_final,
    )
    await asr.start()
    end_task = asyncio.create_task(asr.end_utterance())

    await asyncio.wait_for(final_called.wait(), timeout=0.7)
    assert not end_task.done()
    assert asr._ws is None

    first.allow_close.set()
    await end_task
    assert asr._ws is second
    await asr.close()


@pytest.mark.asyncio
async def test_empty_timeout_replays_buffer_on_fresh_connection(monkeypatch) -> None:
    first = FakeWebSocket()
    second = CompletingWebSocket("电脑主机进水")
    sockets = iter((first, second))
    finals: list[tuple[str, str]] = []

    async def fake_connect(*args, **kwargs):
        return next(sockets)

    async def on_final(text: str, lang: str) -> None:
        finals.append((text, lang))

    monkeypatch.setattr(websockets, "connect", fake_connect)

    asr = QwenRealtimeASR(
        api_key="test",
        language="",
        final_timeout_ms=500,
        on_final=on_final,
    )
    await asr.start()
    pcm = b"\x01\x02" * 1600
    asr.mark_speech_start()
    await asr.push_pcm16(pcm)
    asr.mark_speech_end()
    await asr.end_utterance()

    assert finals == [("电脑主机进水", "zh")]
    assert first._closed.is_set()
    assert any(
        item.get("type") == "input_audio_buffer.append"
        for item in second.sent
    )
    assert asr._utterance_pcm == bytearray()
    await asr.close()


@pytest.mark.asyncio
async def test_pending_reconnect_does_not_block_audio_push(monkeypatch) -> None:
    allow_connect = asyncio.Event()
    ws = CompletingWebSocket("缓冲成功")
    finals: list[tuple[str, str]] = []

    async def delayed_connect(*args, **kwargs):
        await allow_connect.wait()
        return ws

    async def on_final(text: str, lang: str) -> None:
        finals.append((text, lang))

    monkeypatch.setattr(websockets, "connect", delayed_connect)
    asr = QwenRealtimeASR(
        api_key="test",
        language="",
        on_final=on_final,
    )
    await asr.start()
    asr.mark_speech_start()

    await asyncio.wait_for(asr.push_pcm16(b"\x01\x02" * 320), timeout=0.1)
    assert asr._buffer_only_until_commit is True

    asr.mark_speech_end()
    end_task = asyncio.create_task(asr.end_utterance())
    allow_connect.set()
    await asyncio.wait_for(end_task, timeout=1.0)

    assert finals == [("缓冲成功", "zh")]
    await asr.close()


@pytest.mark.asyncio
async def test_replay_buffer_keeps_only_preroll_and_speech(monkeypatch) -> None:
    ws = FakeWebSocket()

    async def fake_connect(*args, **kwargs):
        return ws

    monkeypatch.setattr(websockets, "connect", fake_connect)
    asr = QwenRealtimeASR(api_key="test", language="")
    await asr.start()

    one_second = b"\x01\x02" * 16000
    await asr.push_pcm16(one_second)
    asr.mark_speech_start()
    await asr.push_pcm16(one_second)
    asr.mark_speech_end()

    # 300ms pre-roll + 1s speech, not the full preceding idle second.
    assert len(asr._utterance_pcm) == (16000 * 2 * 13 // 10)
    await asr.close()
