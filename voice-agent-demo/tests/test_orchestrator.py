from __future__ import annotations

import asyncio

import pytest

from app.config import Settings
from app.models import ServerEvent
from app.orchestrator import CallSession
from app.services.llm_router import FAQMatcher, GradedLLMRouter
from app.services.text_norm import detect_reply_lang, split_for_tts
from app.vad import EnergyVAD


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
        repair_order_use_llm_extractor=False,
        playback_done_grace_ms=0,
    )
    session = CallSession(emit=emit, settings=settings, call_id="greeting-lock")
    await session.start()

    for _ in range(100):
        greeting_done = any(
            e.type == "turn.done" and e.data.get("route") == "repair_greeting"
            for e in events
        )
        if greeting_done:
            break
        await asyncio.sleep(0.02)

    assert greeting_done, f"greeting did not finish streaming; events={[e.type for e in events]}"
    assert session._greeting_uninterruptible is True

    event_count = len(events)
    await session.on_pcm16(b"\x00\x20" * 320)
    assert len(events) == event_count
    assert session._greeting_uninterruptible is True

    await session.on_playback_done(session.generation_id)
    assert session._greeting_uninterruptible is False
    assert session.state.value == "listening"

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
