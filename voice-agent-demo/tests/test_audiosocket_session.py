from __future__ import annotations

import asyncio
import uuid

import pytest

from app.config import Settings
from app.transports.audiosocket import (
    KIND_AUDIO_SLIN,
    AudioSocketServer,
    encode_audio_slin8,
    encode_hangup,
    encode_uuid,
    read_frame,
)
from app.transports.audiosocket_session import handle_audiosocket_call
from app.transports.resample import AUDIOSOCKET_FRAME_BYTES_8K, RATE_8K


def _loud_pcm8(duration_s: float = 0.3) -> bytes:
    samples = int(RATE_8K * duration_s)
    return b"\x00\x40" * samples


def _pad_frames(pcm: bytes) -> list[bytes]:
    frames = []
    for offset in range(0, len(pcm), AUDIOSOCKET_FRAME_BYTES_8K):
        chunk = pcm[offset : offset + AUDIOSOCKET_FRAME_BYTES_8K]
        if len(chunk) < AUDIOSOCKET_FRAME_BYTES_8K:
            chunk = chunk + b"\x00" * (AUDIOSOCKET_FRAME_BYTES_8K - len(chunk))
        frames.append(chunk)
    return frames


@pytest.fixture
def short_prompts(monkeypatch):
    short_pcm16 = b"\x00\x10" * 1600  # 100ms @ 16k
    from app.services.prompt_audio import PromptAudioBank

    monkeypatch.setattr(
        PromptAudioBank,
        "load_pcm16",
        lambda self, prompt_id, lang="yue": short_pcm16,
    )
    return short_pcm16


@pytest.mark.asyncio
async def test_bridge_plays_greeting_then_reaches_listening(short_prompts) -> None:
    events: list[str] = []
    settings = Settings(
        agent_mode="mock",
        business_mode="repair_order",
        input_denoise_enabled=False,
        repair_order_use_llm_extractor=False,
        enable_wait_filler=False,
        playback_done_grace_ms=0,
        audiosocket_pace_audio=False,
        vad_silence_ms=40,
        min_speech_ms=40,
        vad_webrtc_mode=None,
    )

    async def on_event(ev) -> None:
        events.append(ev.type)

    async def handler(conn) -> None:
        await handle_audiosocket_call(
            conn,
            settings,
            pace_audio=False,
            on_event=on_event,
        )

    server = AudioSocketServer("127.0.0.1", 0, handler)
    await server.start()
    audio_frames = 0

    try:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", server.bound_port
        )
        writer.write(encode_uuid(uuid.uuid4()))
        await writer.drain()

        deadline = asyncio.get_running_loop().time() + 5.0
        while asyncio.get_running_loop().time() < deadline:
            try:
                frame = await asyncio.wait_for(read_frame(reader), timeout=0.4)
            except asyncio.TimeoutError:
                if events.count("turn.done") >= 2 and audio_frames >= 1:
                    break
                continue
            if frame.kind == KIND_AUDIO_SLIN:
                audio_frames += 1
                assert len(frame.payload) == AUDIOSOCKET_FRAME_BYTES_8K
            if events.count("turn.done") >= 2 and audio_frames >= 1:
                try:
                    while True:
                        frame = await asyncio.wait_for(
                            read_frame(reader), timeout=0.1
                        )
                        if frame.kind == KIND_AUDIO_SLIN:
                            audio_frames += 1
                except asyncio.TimeoutError:
                    break
                break

        assert audio_frames >= 1, f"frames={audio_frames} events={events}"
        assert events.count("turn.done") >= 2
        assert "session.started" in events

        writer.write(encode_hangup())
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_bridge_user_speech_triggers_reply_audio(short_prompts) -> None:
    turn_routes: list[str] = []
    reply_audio = 0

    settings = Settings(
        agent_mode="mock",
        business_mode="repair_order",
        input_denoise_enabled=False,
        repair_order_use_llm_extractor=False,
        enable_wait_filler=False,
        playback_done_grace_ms=0,
        audiosocket_pace_audio=False,
        vad_silence_ms=40,
        min_speech_ms=40,
        vad_webrtc_mode=None,
        vad_energy_threshold=100,
    )

    async def on_event(ev) -> None:
        if ev.type == "turn.done":
            turn_routes.append(str(ev.data.get("route") or ""))

    async def handler(conn) -> None:
        await handle_audiosocket_call(
            conn,
            settings,
            pace_audio=False,
            on_event=on_event,
        )

    server = AudioSocketServer("127.0.0.1", 0, handler)
    await server.start()

    try:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", server.bound_port
        )
        writer.write(encode_uuid(uuid.uuid4()))
        await writer.drain()

        deadline = asyncio.get_running_loop().time() + 5.0
        while asyncio.get_running_loop().time() < deadline:
            try:
                frame = await asyncio.wait_for(read_frame(reader), timeout=0.3)
                if frame.kind == KIND_AUDIO_SLIN:
                    reply_audio += 1
            except asyncio.TimeoutError:
                pass
            if "repair_ask_name" in turn_routes:
                try:
                    while True:
                        frame = await asyncio.wait_for(
                            read_frame(reader), timeout=0.1
                        )
                        if frame.kind == KIND_AUDIO_SLIN:
                            reply_audio += 1
                except asyncio.TimeoutError:
                    break
                break

        startup_frames = reply_audio
        assert "repair_greeting" in turn_routes
        assert "repair_ask_name" in turn_routes
        assert startup_frames >= 1

        speech = _loud_pcm8(0.35)
        quiet = b"\x00\x00" * int(RATE_8K * 0.25)
        for chunk in _pad_frames(speech + quiet):
            writer.write(encode_audio_slin8(chunk))
        await writer.drain()

        got_repair_turn = False
        deadline = asyncio.get_running_loop().time() + 5.0
        while asyncio.get_running_loop().time() < deadline:
            try:
                frame = await asyncio.wait_for(read_frame(reader), timeout=0.4)
            except asyncio.TimeoutError:
                if "repair_order" in turn_routes:
                    got_repair_turn = True
                    break
                continue
            if frame.kind == KIND_AUDIO_SLIN:
                reply_audio += 1
            if "repair_order" in turn_routes:
                got_repair_turn = True
                break

        assert got_repair_turn, f"routes={turn_routes} frames={reply_audio}"
        assert reply_audio > startup_frames

        writer.write(encode_hangup())
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    finally:
        await server.close()
