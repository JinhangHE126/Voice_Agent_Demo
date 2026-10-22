from __future__ import annotations

import asyncio

import pytest

from app.config import Settings
from app.models import ServerEvent, SessionState
from app.orchestrator import CallSession
from app.services.llm_router import FAQMatcher, GradedLLMRouter
from app.services.text_norm import detect_reply_lang, split_for_tts
from app.vad import EnergyVAD
from repair_order.domain.repair_order import OrderState


def test_vad_detects_speech_and_end() -> None:
    vad = EnergyVAD(
        silence_ms=60,
        min_speech_ms=40,
        frame_ms=20,
        energy_threshold=100,
        webrtc_mode=None,
    )
    # 320 PCM16 samples = 20ms at 16kHz.
    loud = b"\x00\x10" * 320
    quiet = b"\x00\x00" * 320
    events = []
    for _ in range(5):
        events.append(vad.accept(loud)["event"])
    for _ in range(5):
        events.append(vad.accept(quiet)["event"])
    assert "speech_start" in events
    assert "speech_end" in events


def test_vad_confirmation_override_accepts_short_reply() -> None:
    vad = EnergyVAD(
        silence_ms=60,
        min_speech_ms=200,
        frame_ms=20,
        energy_threshold=100,
        webrtc_mode=None,
    )
    loud = b"\x00\x10" * 320
    quiet = b"\x00\x00" * 320

    events = [
        vad.accept(
            loud,
            min_speech_ms=120,
            enable_window_endpoint=False,
            report_short_candidate=True,
        )["event"]
        for _ in range(6)
    ]
    events.extend(
        vad.accept(
            quiet,
            min_speech_ms=120,
            enable_window_endpoint=False,
            report_short_candidate=True,
        )["event"]
        for _ in range(4)
    )

    assert "speech_start" in events
    assert "speech_end" in events


def test_faq_match_cantonese_hours(tmp_path) -> None:
    path = tmp_path / "faq.json"
    path.write_text(
        '[{"id":"h","keywords":["星期日","关门"],"reply_yue":"八点关门。","reply_zh":"八点。","reply_en":"8 PM."}]',
        encoding="utf-8",
    )
    faq = FAQMatcher(path)
    assert faq.match("我想知星期日几点关门", "yue") == "八点关门。"


def test_detect_reply_lang() -> None:
    assert detect_reply_lang("星期日关门", "yue") == "yue"
    assert detect_reply_lang("What time do you close?", "en") == "en"


def test_split_for_tts() -> None:
    parts = split_for_tts("星期日我哋夜晚八点关门。欢迎再来。")
    assert len(parts) >= 2


@pytest.mark.asyncio
async def test_mock_session_first_audio_under_1s() -> None:
    events: list[ServerEvent] = []

    async def emit(ev: ServerEvent) -> None:
        events.append(ev)

    settings = Settings(
        agent_mode="mock",
        business_mode="repair_order",
        input_denoise_enabled=False,
        repair_order_use_llm_extractor=False,
        vad_silence_ms=40,
        min_speech_ms=40,
        vad_webrtc_mode=None,
    )
    session = CallSession(emit=emit, settings=settings, call_id="test")
    await session.start()

    loud = b"\x00\x20" * 160
    quiet = b"\x00\x00" * 160
    for _ in range(20):
        await session.on_pcm16(loud)
    for _ in range(10):
        await session.on_pcm16(quiet)

    for _ in range(60):
        if any(e.type == "tts.first_audio" for e in events):
            break
        await asyncio.sleep(0.05)

    first = next((e for e in events if e.type == "tts.first_audio"), None)
    assert first is not None, f"no first audio; events={[e.type for e in events]}"
    latency = first.data.get("latency_ms")
    assert latency is not None and latency <= 1000, latency

    await session.close()


@pytest.mark.asyncio
async def test_greeting_stays_uninterruptible_until_browser_playback_done() -> None:
    events: list[ServerEvent] = []

    async def emit(ev: ServerEvent) -> None:
        events.append(ev)

    settings = Settings(
        agent_mode="mock",
        business_mode="repair_order",
        input_denoise_enabled=False,
        repair_order_use_llm_extractor=False,
        playback_done_grace_ms=0,
    )
    session = CallSession(emit=emit, settings=settings, call_id="greeting-lock")
    await session.start()

    for _ in range(400):
        greeting_done = any(
            e.type == "turn.done" and e.data.get("route") == "repair_greeting"
            for e in events
        )
        if greeting_done:
            break
        await asyncio.sleep(0.02)

    assert greeting_done, f"greeting did not finish streaming; events={[e.type for e in events]}"
    assert session._greeting_uninterruptible is True
    assert sum(e.type == "tts.audio" for e in events) == 1

    event_count = len(events)
    await session.on_pcm16(b"\x00\x20" * 320)
    assert len(events) == event_count
    assert session._greeting_uninterruptible is True

    await session.on_playback_done(session.generation_id)
    for _ in range(100):
        ask_name_done = any(
            e.type == "turn.done" and e.data.get("route") == "repair_ask_name"
            for e in events
        )
        if ask_name_done:
            break
        await asyncio.sleep(0.01)

    assert ask_name_done
    assert session._greeting_uninterruptible is True
    assert sum(e.type == "tts.audio" for e in events) == 2

    await session.on_playback_done(session.generation_id)
    assert session._greeting_uninterruptible is False
    assert session.state.value == "listening"

    await session.close()


@pytest.mark.asyncio
async def test_auto_complete_playback_advances_without_browser_done() -> None:
    """SIP-style sink: on_audio_out + auto_complete_playback, no playback.done."""
    events: list[ServerEvent] = []
    audio_chunks: list[bytes] = []

    async def emit(ev: ServerEvent) -> None:
        events.append(ev)

    async def on_audio_out(pcm: bytes) -> None:
        audio_chunks.append(pcm)

    settings = Settings(
        agent_mode="mock",
        business_mode="repair_order",
        input_denoise_enabled=False,
        repair_order_use_llm_extractor=False,
        playback_done_grace_ms=0,
    )
    session = CallSession(
        emit=emit,
        settings=settings,
        call_id="auto-playback",
        on_audio_out=on_audio_out,
        auto_complete_playback=True,
    )
    # Short local prompts so the auto-complete wait stays sub-second in CI.
    short_pcm = b"\x00\x10" * (settings.sample_rate // 5)  # 200ms
    session.prompts.load_pcm16 = (  # type: ignore[method-assign]
        lambda prompt_id, lang="yue": short_pcm
    )

    await session.start()

    for _ in range(200):
        if (
            session.state == SessionState.LISTENING
            and session._greeting_uninterruptible is False
        ):
            break
        await asyncio.sleep(0.02)

    assert session.state == SessionState.LISTENING
    assert session._greeting_uninterruptible is False
    assert audio_chunks, "on_audio_out should receive greeting/ask_name PCM"
    assert sum(len(c) for c in audio_chunks) >= len(short_pcm)
    assert any(
        e.type == "turn.done" and e.data.get("route") == "repair_greeting"
        for e in events
    )
    assert any(
        e.type == "turn.done" and e.data.get("route") == "repair_ask_name"
        for e in events
    )
    # Browser path still emits tts.audio for observability.
    assert any(e.type == "tts.audio" for e in events)

    await session.close()


@pytest.mark.asyncio
async def test_greeting_waits_for_asr_ready_before_listening() -> None:
    events: list[ServerEvent] = []

    async def emit(ev: ServerEvent) -> None:
        events.append(ev)

    session = CallSession(
        emit=emit,
        settings=Settings(
            agent_mode="mock",
            business_mode="repair_order",
            input_denoise_enabled=False,
            playback_done_grace_ms=0,
        ),
        call_id="asr-ready-gate",
    )

    class DelayedASR:
        def __init__(self) -> None:
            self.ready = False
            self.release = asyncio.Event()

        @property
        def is_ready(self) -> bool:
            return self.ready

        async def wait_until_ready(self, timeout_s: float) -> bool:
            await asyncio.wait_for(self.release.wait(), timeout=timeout_s)
            self.ready = True
            return True

        async def close(self) -> None:
            return

    asr = DelayedASR()
    session._asr = asr
    session.generation_id = 1
    session._awaiting_playback_done = True
    session._greeting_uninterruptible = True
    session._startup_phase = "greeting"

    await session.on_playback_done(1)
    assert session.state == SessionState.CONNECTING_ASR
    assert session._greeting_uninterruptible is True
    assert any(
        event.type == "prompt.audio"
        and event.data.get("prompt_id") == "connecting_asr"
        for event in events
    )

    asr.release.set()
    for _ in range(100):
        ask_name_done = any(
            event.type == "turn.done"
            and event.data.get("route") == "repair_ask_name"
            for event in events
        )
        if ask_name_done:
            break
        await asyncio.sleep(0.01)

    assert ask_name_done
    assert session.state == SessionState.SPEAKING
    assert session._greeting_uninterruptible is True
    assert any(event.type == "asr.ready" for event in events)

    await session.on_playback_done(session.generation_id)
    assert session.state == SessionState.LISTENING
    assert session._greeting_uninterruptible is False
    await session.close()


@pytest.mark.asyncio
async def test_startup_asr_timeout_plays_unavailable_prompt() -> None:
    events: list[ServerEvent] = []

    async def emit(ev: ServerEvent) -> None:
        events.append(ev)

    session = CallSession(
        emit=emit,
        settings=Settings(
            agent_mode="mock",
            business_mode="repair_order",
            input_denoise_enabled=False,
            playback_done_grace_ms=0,
            asr_startup_timeout_ms=20,
        ),
        call_id="asr-startup-timeout",
    )

    class NeverReadyASR:
        @property
        def is_ready(self) -> bool:
            return False

        async def wait_until_ready(self, timeout_s: float) -> bool:
            await asyncio.sleep(timeout_s)
            return False

        async def close(self) -> None:
            return

    session._asr = NeverReadyASR()
    session.generation_id = 1
    session._awaiting_playback_done = True
    session._greeting_uninterruptible = True
    session._startup_phase = "greeting"

    await session.on_playback_done(1)
    await asyncio.sleep(0.08)

    assert session._startup_phase == "failed"
    assert session.state == SessionState.CONNECTING_ASR
    assert any(
        event.type == "prompt.audio"
        and event.data.get("prompt_id") == "asr_unavailable"
        for event in events
    )
    assert any(event.type == "error" for event in events)
    await session.close()


@pytest.mark.asyncio
async def test_endpointing_plays_local_wait_prompt_after_delay() -> None:
    events: list[ServerEvent] = []

    async def emit(ev: ServerEvent) -> None:
        events.append(ev)

    session = CallSession(
        emit=emit,
        settings=Settings(
            agent_mode="mock",
            business_mode="repair_order",
            input_denoise_enabled=False,
            asr_wait_prompt_delay_ms=10,
        ),
        call_id="asr-wait-prompt",
    )
    session.state = SessionState.ENDPOINTING
    session._start_asr_wait_prompt()
    await asyncio.sleep(0.05)

    assert any(
        event.type == "prompt.audio"
        and event.data.get("prompt_id") == "please_wait_asr"
        for event in events
    )
    assert any(event.type == "tts.audio" for event in events)
    await session.close()


@pytest.mark.asyncio
async def test_empty_asr_final_reprompts_current_repair_field() -> None:
    events: list[ServerEvent] = []

    async def emit(ev: ServerEvent) -> None:
        events.append(ev)

    session = CallSession(
        emit=emit,
        settings=Settings(
            agent_mode="mock",
            business_mode="repair_order",
            input_denoise_enabled=False,
        ),
        call_id="asr-empty-reprompt",
    )
    session.dialog.greeting(session.draft)

    await session._on_asr_final("", lang="yue")
    for _ in range(20):
        if any(event.type == "prompt.audio" for event in events):
            break
        await asyncio.sleep(0.01)

    assert any(
        event.type == "prompt.audio"
        and event.data.get("prompt_id") == "ask_name"
        for event in events
    )
    await session.close()


@pytest.mark.asyncio
async def test_wait_filler_does_not_mark_playback_awaiting() -> None:
    events: list[ServerEvent] = []

    async def emit(ev: ServerEvent) -> None:
        events.append(ev)

    session = CallSession(
        emit=emit,
        settings=Settings(
            agent_mode="mock",
            business_mode="repair_order",
            input_denoise_enabled=False,
            repair_order_use_llm_extractor=False,
            enable_wait_filler=True,
            filler_min_user_chars=1,
        ),
        call_id="filler-no-await",
    )
    session.draft.awaiting_field = "repair_description"
    session.state = SessionState.THINKING
    session.generation_id = 7
    session._turn_clock = 0.0

    played: list[tuple] = []

    async def fake_play_prompt_id(
        prompt_id: str,
        *,
        generation_id: int,
        route: str,
        mark_latency: bool = True,
    ) -> bool:
        played.append((prompt_id, route, mark_latency))
        return True

    session._play_prompt_id = fake_play_prompt_id  # type: ignore[method-assign]

    # Dynamic confirm → late filler covers MiniMax wait (not early, no LLM).
    from repair_order.services.dialog_manager import ResponsePlan, TurnResult

    async def fake_process(_call_id, _draft, _text):
        return TurnResult(
            plan=ResponsePlan(
                text="我确认一下维修内容。",
                prompt_id="confirm_repair",
                dynamic=True,
            ),
            updated_fields=["repair_description"],
        )

    session.dialog.process_user_text_async = fake_process  # type: ignore[method-assign]

    await session._run_reply(
        "我电脑主机坏了，昨天进水了，现在打不开",
        generation_id=7,
    )

    assert sum(1 for p in played if p[1] == "repair_filler") == 1
    assert any(e.type == "filler.started" for e in events)
    await session.close()


@pytest.mark.asyncio
async def test_local_wav_reply_does_not_start_filler() -> None:
    events: list[ServerEvent] = []

    async def emit(ev: ServerEvent) -> None:
        events.append(ev)

    session = CallSession(
        emit=emit,
        settings=Settings(
            agent_mode="mock",
            business_mode="repair_order",
            input_denoise_enabled=False,
            repair_order_use_llm_extractor=True,
            enable_wait_filler=True,
            filler_min_user_chars=1,
        ),
        call_id="no-filler-local",
    )
    session.state = SessionState.THINKING
    session.generation_id = 4

    played: list[tuple] = []

    async def fake_play_prompt_id(
        prompt_id: str,
        *,
        generation_id: int,
        route: str,
        mark_latency: bool = True,
    ) -> bool:
        played.append((prompt_id, route, mark_latency))
        return True

    session._play_prompt_id = fake_play_prompt_id  # type: ignore[method-assign]

    from repair_order.services.dialog_manager import ResponsePlan, TurnResult

    async def fake_process(_call_id, _draft, _text):
        return TurnResult(
            plan=ResponsePlan(
                text="请问要维修什么？",
                prompt_id="ask_repair",
                dynamic=False,
            ),
            updated_fields=["customer_name"],
        )

    session.dialog.process_user_text_async = fake_process  # type: ignore[method-assign]

    # 11 chars: previously forced filler and overlapped ask_repair.
    await session._run_reply("哎，你叫我何先生就行。", generation_id=4)

    assert not any(p[1] == "repair_filler" for p in played)
    assert any(p[0] == "ask_repair" for p in played)
    assert not any(e.type == "filler.started" for e in events)
    await session.close()


@pytest.mark.asyncio
async def test_filler_plays_only_once_even_if_reply_is_slow() -> None:
    stop = asyncio.Event()
    played: list[str] = []

    events: list[ServerEvent] = []

    async def emit(ev: ServerEvent) -> None:
        events.append(ev)

    session = CallSession(
        emit=emit,
        settings=Settings(
            agent_mode="mock",
            business_mode="repair_order",
            input_denoise_enabled=False,
            enable_wait_filler=True,
        ),
        call_id="filler-once",
    )
    session.generation_id = 3

    async def fake_play_prompt_id(
        prompt_id: str,
        *,
        generation_id: int,
        route: str,
        mark_latency: bool = True,
    ) -> bool:
        played.append(prompt_id)
        return True

    session._play_prompt_id = fake_play_prompt_id  # type: ignore[method-assign]

    task = asyncio.create_task(
        session._play_filler_loop("ack_checking", stop, generation_id=3)
    )
    await asyncio.sleep(0.05)
    assert played == ["ack_checking"]
    # Still waiting for real reply: must not replay.
    await asyncio.sleep(0.15)
    assert played == ["ack_checking"]
    stop.set()
    await asyncio.wait_for(task, timeout=1.0)
    assert played == ["ack_checking"]
    await session.close()


async def test_empty_asr_final_does_not_replay_dynamic_confirmation() -> None:
    events: list[ServerEvent] = []

    async def emit(ev: ServerEvent) -> None:
        events.append(ev)

    session = CallSession(
        emit=emit,
        settings=Settings(
            agent_mode="mock",
            business_mode="repair_order",
            input_denoise_enabled=False,
        ),
        call_id="empty-confirmation",
    )
    session.draft.state = OrderState.CONFIRMING_REPAIR
    session.draft.repair_description = "电脑主机进水，无法开机"

    await session._on_asr_final("", lang="yue")
    for _ in range(20):
        if any(event.type == "prompt.audio" for event in events):
            break
        await asyncio.sleep(0.01)

    prompt_ids = [
        event.data.get("prompt_id")
        for event in events
        if event.type == "prompt.audio"
    ]
    assert "ask_confirm_yesno" in prompt_ids
    assert "confirm_repair" not in prompt_ids
    assert session._asr_empty_count == 1
    await session.close()


@pytest.mark.asyncio
async def test_saved_order_stops_listening_and_ignores_later_speech() -> None:
    events: list[ServerEvent] = []

    async def emit(ev: ServerEvent) -> None:
        events.append(ev)

    session = CallSession(
        emit=emit,
        settings=Settings(
            agent_mode="mock",
            business_mode="repair_order",
            input_denoise_enabled=False,
            playback_done_grace_ms=0,
        ),
        call_id="order-completed",
    )
    session.state = SessionState.SPEAKING
    session.generation_id = 3
    session._awaiting_playback_done = True
    session._call_completed = True
    session._last_tts_audio_at = 0.0

    await session.on_playback_done(3)
    assert session.state == SessionState.COMPLETED
    assert any(event.type == "session.completed" for event in events)

    event_count = len(events)
    await session.on_pcm16(b"\x00\x20" * 320)
    assert len(events) == event_count

    await session._on_asr_final("好，拜拜。", lang="zh")
    assert not any(
        event.type == "asr.final" and event.data.get("text") == "好，拜拜。"
        for event in events
    )
    assert session.state == SessionState.COMPLETED
    await session.close()


@pytest.mark.asyncio
async def test_asr_watchdog_forces_end_when_no_partial_progress() -> None:
    events: list[ServerEvent] = []

    async def emit(ev: ServerEvent) -> None:
        events.append(ev)

    settings = Settings(
        agent_mode="mock",
        business_mode="faq",
        input_denoise_enabled=False,
        asr_no_progress_timeout_ms=120,
        vad_webrtc_mode=None,
    )
    session = CallSession(emit=emit, settings=settings, call_id="watchdog")
    await session.start()

    class FakeASR:
        def __init__(self) -> None:
            self.end_called = asyncio.Event()

        async def push_pcm16(self, pcm: bytes) -> None:
            return

        async def end_utterance(self, *, fast_fail: bool = False) -> None:
            self.end_called.set()

        async def close(self) -> None:
            return

    fake_asr = FakeASR()
    session._asr = fake_asr

    seen = {"start": False}

    def fake_accept(_pcm: bytes) -> dict:
        if not seen["start"]:
            seen["start"] = True
            return {"event": "speech_start", "energy": 320.0, "threshold": 200.0}
        return {"event": "none", "energy": 0.0, "threshold": 200.0}

    session.vad.accept = fake_accept  # type: ignore[assignment]

    await session.on_pcm16(b"\x00\x20" * 320)
    await asyncio.wait_for(fake_asr.end_called.wait(), timeout=1.0)

    assert any(e.type == "speech.ended" and e.data.get("manual") for e in events)
    assert session.state.value == "endpointing"

    await session.close()


@pytest.mark.asyncio
async def test_wait_prompt_prepares_asr_standby() -> None:
    async def emit(_ev: ServerEvent) -> None:
        return

    settings = Settings(
        agent_mode="mock",
        business_mode="repair_order",
        asr_wait_prompt_delay_ms=0,
        input_denoise_enabled=False,
    )
    session = CallSession(
        emit=emit,
        settings=settings,
        call_id="standby-warmup",
    )

    class FakeASR:
        def __init__(self) -> None:
            self.prepared = False

        def prepare_standby(self) -> None:
            self.prepared = True

    fake_asr = FakeASR()
    session._asr = fake_asr
    session.state = SessionState.ENDPOINTING

    await session._run_asr_wait_prompt(session.generation_id)

    assert fake_asr.prepared is True


@pytest.mark.asyncio
async def test_empty_final_waits_for_asr_before_reprompt() -> None:
    events: list[ServerEvent] = []

    async def emit(ev: ServerEvent) -> None:
        events.append(ev)

    settings = Settings(
        agent_mode="mock",
        business_mode="repair_order",
        input_denoise_enabled=False,
    )
    session = CallSession(
        emit=emit,
        settings=settings,
        call_id="reprompt-after-recovery",
    )
    session.dialog.greeting(session.draft)

    class RecoveringASR:
        def __init__(self) -> None:
            self.is_ready = False

        async def wait_until_ready(self, _timeout_s: float) -> bool:
            await asyncio.sleep(0)
            self.is_ready = True
            return True

        async def close(self) -> None:
            return

    session._asr = RecoveringASR()
    await session._on_asr_final("", lang="yue")

    for _ in range(50):
        if any(
            event.type == "prompt.audio"
            and event.data.get("prompt_id") == "ask_name"
            for event in events
        ):
            break
        await asyncio.sleep(0.01)

    assert any(event.type == "asr.connecting" for event in events)
    assert any(event.type == "asr.ready" for event in events)
    assert any(
        event.type == "prompt.audio"
        and event.data.get("prompt_id") == "ask_name"
        for event in events
    )
    assert session._pending_asr_reprompt is None
    await session.close()


@pytest.mark.asyncio
async def test_speaking_ignores_user_audio_no_barge_in() -> None:
    events: list[ServerEvent] = []

    async def emit(ev: ServerEvent) -> None:
        events.append(ev)

    settings = Settings(
        agent_mode="mock",
        business_mode="faq",
        input_denoise_enabled=False,
        vad_silence_ms=40,
        min_speech_ms=40,
        vad_energy_threshold=100,
        vad_webrtc_mode=None,
        barge_in_min_ms=0,
    )
    session = CallSession(emit=emit, settings=settings, call_id="no-barge")

    class FakeASR:
        def __init__(self) -> None:
            self.pushed = bytearray()
            self.end_called = asyncio.Event()

        async def push_pcm16(self, pcm: bytes) -> None:
            self.pushed.extend(pcm)

        async def end_utterance(self, *, fast_fail: bool = False) -> None:
            self.end_called.set()

        async def close(self) -> None:
            return

    fake_asr = FakeASR()
    session._asr = fake_asr
    session.state = SessionState.SPEAKING
    session._speaking_started_at = 0.0

    loud = b"\x00\x20" * 320
    await session.on_pcm16(loud)
    await session.on_pcm16(loud)

    # SPEAKING 期间客户语音被忽略：状态保持不变，音频不推给 ASR。
    assert session.state == SessionState.SPEAKING
    assert not fake_asr.pushed
    assert not any(
        event.type == "speech.started" and event.data.get("barge_in")
        for event in events
    )
    await session.close()


@pytest.mark.asyncio
async def test_graded_router_faq_route(tmp_path) -> None:
    path = tmp_path / "faq.json"
    path.write_text(
        '[{"id":"hello","keywords":["你好"],"reply_yue":"你好。","reply_zh":"你好。","reply_en":"Hi."}]',
        encoding="utf-8",
    )
    router = GradedLLMRouter(
        faq_path=path,
        system_prompt="test",
        fast_base_url="http://localhost",
        fast_api_key="",
        fast_model="x",
        mock=True,
    )
    route: dict[str, str] = {}
    chunks = []
    async for c in router.stream_reply("你好啊", asr_lang="yue", route_out=route):
        chunks.append(c)
    assert route["route"] == "L0_faq"
    assert "".join(chunks) == "你好。"
