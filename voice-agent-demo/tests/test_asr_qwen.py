from __future__ import annotations

import asyncio
import json
import time

import pytest
import websockets

from app.services.asr_qwen import QwenRealtimeASR


class FakeWebSocket:
    def __init__(self, *, fail_event: str | None = None) -> None:
        self.fail_event = fail_event
        self.sent: list[dict] = []
        self._closed = asyncio.Event()
        self._messages: asyncio.Queue[str | None] = asyncio.Queue()

    async def send(self, raw: str) -> None:
        payload = json.loads(raw)
        if payload.get("type") == self.fail_event:
            raise ConnectionError(f"failed {self.fail_event}")
        self.sent.append(payload)
        if payload.get("type") == "session.update":
            await self._messages.put(json.dumps({"type": "session.updated"}))

    async def close(self) -> None:
        await self._messages.put(None)
        self._closed.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self._messages.get()
        if item is None:
            raise StopAsyncIteration
        return item


class CompletingWebSocket(FakeWebSocket):
    def __init__(self, transcript: str) -> None:
        super().__init__()
        self.transcript = transcript

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
    assert await asr.wait_until_ready(0.5)
    assert asr._ws is second

    await asr.close()


@pytest.mark.asyncio
async def test_qwen_asr_final_timeout_rotates_connection(monkeypatch) -> None:
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
        hard_timeout_ms=500,
        on_final=on_final,
    )
    await asr.start()
    await asr.end_utterance()

    assert finals == [("", "yue")]
    assert await asr.wait_until_ready(0.5)
    assert connect_calls == 2
    assert asr._ws is second

    await asr.close()


class StagnatingWebSocket(FakeWebSocket):
    """Emits partials then goes silent, never sending a completed event."""

    async def send(self, raw: str) -> None:
        await super().send(raw)
        payload = json.loads(raw)
        if payload.get("type") == "input_audio_buffer.commit":
            await self._messages.put(
                json.dumps(
                    {
                        "type": "input_audio_buffer.committed",
                        "item_id": "i1",
                    }
                )
            )
            await self._messages.put(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "item_id": "i1",
                        "delta": "我电脑主机坏了",
                    }
                )
            )


@pytest.mark.asyncio
async def test_partial_stagnation_finalizes_before_hard_timeout(monkeypatch) -> None:
    sock = StagnatingWebSocket()
    finals: list[tuple[str, str]] = []

    async def fake_connect(*args, **kwargs):
        return sock

    async def on_final(text: str, lang: str) -> None:
        finals.append((text, lang))

    monkeypatch.setattr(websockets, "connect", fake_connect)

    asr = QwenRealtimeASR(
        api_key="test",
        language="yue",
        final_timeout_ms=500,
        hard_timeout_ms=5000,
        stagnation_ms=400,
        on_final=on_final,
    )
    await asr.start()

    started = time.monotonic()
    await asr.end_utterance()
    elapsed = time.monotonic() - started

    # Partial was used as the final, well before the 5s hard timeout.
    assert finals == [("我电脑主机坏了", "yue")]
    assert elapsed < 2.0

    await asr.close()


class CumulativeDeltaWebSocket(FakeWebSocket):
    """Some Qwen variants put the cumulative transcript in `delta` events."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks = [
            "不对，是我电脑主机坏了，它昨天晚上",
            "不对，是我电脑主机坏了，它昨天晚上进了",
            "不对，是我电脑主机坏了，它昨天晚上进了水",
        ]

    async def send(self, raw: str) -> None:
        await super().send(raw)
        payload = json.loads(raw)
        if payload.get("type") == "input_audio_buffer.commit":
            await self._messages.put(
                json.dumps(
                    {"type": "input_audio_buffer.committed", "item_id": "i1"}
                )
            )
            for chunk in self._chunks:
                await self._messages.put(
                    json.dumps(
                        {
                            "type": "conversation.item.input_audio_transcription.delta",
                            "item_id": "i1",
                            "delta": chunk,
                        }
                    )
                )


@pytest.mark.asyncio
async def test_cumulative_delta_is_not_duplicated(monkeypatch) -> None:
    sock = CumulativeDeltaWebSocket()
    finals: list[tuple[str, str]] = []

    async def fake_connect(*args, **kwargs):
        return sock

    async def on_final(text: str, lang: str) -> None:
        finals.append((text, lang))

    monkeypatch.setattr(websockets, "connect", fake_connect)

    asr = QwenRealtimeASR(
        api_key="test",
        language="yue",
        final_timeout_ms=500,
        hard_timeout_ms=5000,
        stagnation_ms=400,
        on_final=on_final,
    )
    await asr.start()
    await asr.end_utterance()

    assert finals == [("不对，是我电脑主机坏了，它昨天晚上进了水", "yue")]

    await asr.close()


@pytest.mark.asyncio
async def test_timeout_dispatches_fallback_before_old_socket_closes(
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
        hard_timeout_ms=500,
        on_final=on_final,
    )
    await asr.start()
    end_task = asyncio.create_task(asr.end_utterance())

    await asyncio.wait_for(final_called.wait(), timeout=0.7)
    await end_task
    assert await asr.wait_until_ready(0.5)
    for _ in range(20):
        if first.close_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert first.close_started.is_set()
    assert asr._ws is second
    first.allow_close.set()
    await asr.close()


@pytest.mark.asyncio
async def test_long_utterance_timeout_skips_sync_replay(monkeypatch) -> None:
    """Long turns wait on the original socket; no sync final_retry replay."""
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
        language="yue",
        final_timeout_ms=500,
        hard_timeout_ms=700,
        on_final=on_final,
    )
    await asr.start()
    assert await asr.wait_until_ready(0.5)
    pcm = b"\x01\x02" * 1600
    asr.mark_speech_start()
    await asr.push_pcm16(pcm)
    asr.mark_speech_end()
    await asr.end_utterance()

    assert finals == [("", "yue")]
    assert await asr.wait_until_ready(0.5)
    assert asr._ws is second
    # Must not block the turn on a buffered replay to the second socket.
    assert not any(
        item.get("type") == "input_audio_buffer.append"
        for item in second.sent
    )
    await asr.close()


@pytest.mark.asyncio
async def test_short_utterance_timeout_returns_empty_without_replay(
    monkeypatch,
) -> None:
    first = FakeWebSocket()
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
        final_timeout_ms=500,
        hard_timeout_ms=2000,
        on_final=on_final,
    )
    await asr.start()
    assert await asr.wait_until_ready(0.5)
    asr.mark_speech_start()
    await asr.push_pcm16(b"\x01\x02" * 320)
    asr.mark_speech_end()

    await asr.end_utterance(fast_fail=True)

    assert finals == [("", "yue")]
    assert await asr.wait_until_ready(0.5)
    assert asr._ws is second
    assert not any(
        item.get("type") == "input_audio_buffer.append"
        for item in second.sent
    )
    await asr.close()


@pytest.mark.asyncio
async def test_short_utterance_final_near_timeout_keeps_connection(
    monkeypatch,
) -> None:
    class DelayedCompletingWebSocket(FakeWebSocket):
        def __init__(self) -> None:
            super().__init__()
            self.delivery_tasks: list[asyncio.Task] = []

        async def send(self, raw: str) -> None:
            await super().send(raw)
            payload = json.loads(raw)
            if payload.get("type") == "input_audio_buffer.commit":
                async def deliver() -> None:
                    await asyncio.sleep(1.5)
                    await self._messages.put(
                        json.dumps(
                            {
                                "type": (
                                    "conversation.item."
                                    "input_audio_transcription.completed"
                                ),
                                "transcript": "就是",
                                "language": "zh",
                            }
                        )
                    )

                self.delivery_tasks.append(asyncio.create_task(deliver()))

    ws = DelayedCompletingWebSocket()
    connect_calls = 0
    finals: list[tuple[str, str]] = []

    async def fake_connect(*args, **kwargs):
        nonlocal connect_calls
        connect_calls += 1
        return ws

    async def on_final(text: str, lang: str) -> None:
        finals.append((text, lang))

    monkeypatch.setattr(websockets, "connect", fake_connect)
    asr = QwenRealtimeASR(
        api_key="test",
        language="",
        final_timeout_ms=2500,
        on_final=on_final,
    )
    await asr.start()
    assert await asr.wait_until_ready(0.5)

    await asr.end_utterance(fast_fail=True)

    assert finals == [("就是", "zh")]
    assert connect_calls == 1
    assert asr._ws is ws
    assert not ws._closed.is_set()
    await asr.close()


@pytest.mark.asyncio
async def test_end_utterance_calls_are_serialized(monkeypatch) -> None:
    asr = QwenRealtimeASR(api_key="test", language="")
    active = 0
    max_active = 0

    async def fake_end_impl(
        *,
        turn_id: int,
        deadline: float,
        fast_fail: bool = False,
    ) -> None:
        nonlocal active, max_active
        assert turn_id > 0
        assert deadline > 0
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1

    monkeypatch.setattr(asr, "_end_utterance_impl", fake_end_impl)

    await asyncio.gather(
        asr.end_utterance(),
        asr.end_utterance(fast_fail=True),
    )

    assert max_active == 1


@pytest.mark.asyncio
async def test_hard_timeout_returns_empty_while_reconnect_is_stalled(
    monkeypatch,
) -> None:
    """Commit-retry path may reconnect; customer still gets an empty final."""
    first = FakeWebSocket(fail_event="input_audio_buffer.commit")
    stall = asyncio.Event()
    connect_count = 0
    final_called = asyncio.Event()

    async def fake_connect(*args, **kwargs):
        nonlocal connect_count
        connect_count += 1
        if connect_count == 1:
            return first
        await stall.wait()
        return FakeWebSocket()

    async def on_final(text: str, lang: str) -> None:
        assert text == ""
        final_called.set()

    monkeypatch.setattr(websockets, "connect", fake_connect)
    asr = QwenRealtimeASR(
        api_key="test",
        language="yue",
        final_timeout_ms=500,
        hard_timeout_ms=700,
        on_final=on_final,
    )
    await asr.start()
    assert await asr.wait_until_ready(0.5)
    asr.mark_speech_start()
    await asr.push_pcm16(b"\x01\x02" * 1600)
    asr.mark_speech_end()

    await asyncio.wait_for(asr.end_utterance(), timeout=2.0)

    assert final_called.is_set()
    for _ in range(10):
        if connect_count >= 2:
            break
        await asyncio.sleep(0)
    assert connect_count >= 2
    await asr.close()
    stall.set()


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


@pytest.mark.asyncio
async def test_late_final_is_ignored_after_standby_promotion(monkeypatch) -> None:
    class DelayedWebSocket(FakeWebSocket):
        def __init__(self) -> None:
            super().__init__()
            self.delivery_tasks: list[asyncio.Task] = []

        async def send(self, raw: str) -> None:
            await super().send(raw)
            payload = json.loads(raw)
            if payload.get("type") != "input_audio_buffer.commit":
                return

            async def deliver() -> None:
                await asyncio.sleep(0.65)
                await self._messages.put(
                    json.dumps(
                        {
                            "type": (
                                "conversation.item."
                                "input_audio_transcription.completed"
                            ),
                            "transcript": "迟到旧结果",
                            "language": "zh",
                        }
                    )
                )

            self.delivery_tasks.append(asyncio.create_task(deliver()))

    first = DelayedWebSocket()
    second = CompletingWebSocket("当前结果")
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
        final_timeout_ms=500,
        hard_timeout_ms=1000,
        on_final=on_final,
    )
    await asr.start()
    assert await asr.wait_until_ready(0.5)
    asr.prepare_standby()
    for _ in range(50):
        if asr._standby_ws is second:
            break
        await asyncio.sleep(0.01)
    assert asr._standby_ws is second

    await asr.end_utterance(fast_fail=True)
    await asr.end_utterance()

    assert finals == [("", "yue"), ("当前结果", "zh")]
    assert asr._ws is second
    await asr.close()
    await asyncio.gather(*first.delivery_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_timeout_budget_includes_slow_commit_send(monkeypatch) -> None:
    class SlowCommitWebSocket(FakeWebSocket):
        async def send(self, raw: str) -> None:
            payload = json.loads(raw)
            if payload.get("type") == "input_audio_buffer.commit":
                await asyncio.sleep(0.4)
            await super().send(raw)

    first = SlowCommitWebSocket()
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
        final_timeout_ms=500,
        on_final=on_final,
    )
    await asr.start()
    assert await asr.wait_until_ready(0.5)

    started = asyncio.get_running_loop().time()
    await asr.end_utterance(fast_fail=True)
    elapsed = asyncio.get_running_loop().time() - started

    assert finals == [("", "yue")]
    assert 0.45 <= elapsed < 0.7
    assert await asr.wait_until_ready(0.5)
    assert asr._ws is second
    await asr.close()


@pytest.mark.asyncio
async def test_read_disconnect_reconnects_in_background(monkeypatch) -> None:
    first = FakeWebSocket()
    second = FakeWebSocket()
    sockets = iter((first, second))
    connect_count = 0

    async def fake_connect(*args, **kwargs):
        nonlocal connect_count
        connect_count += 1
        return next(sockets)

    monkeypatch.setattr(websockets, "connect", fake_connect)
    asr = QwenRealtimeASR(api_key="test", language="yue")
    await asr.start()
    assert await asr.wait_until_ready(0.5)

    await first._messages.put(None)
    for _ in range(20):
        if connect_count >= 2:
            break
        await asyncio.sleep(0.01)
    assert await asr.wait_until_ready(0.5)

    assert connect_count == 2
    assert asr._ws is second
    await asr.close()
