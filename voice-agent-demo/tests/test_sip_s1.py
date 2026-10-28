"""S1 SIP skeleton tests (no live FreeSWITCH / pjsua2 required)."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from app.config import Settings
from app.transports import sip_pjsua
from app.transports.sip_session import (
    SipCallBridge,
    build_s1_startup_wav_8k,
    ensure_s1_playback_wav_8k,
    resolve_s1_prompt_wav,
)  # noqa: F401 — SipCallBridge used below


def test_resolve_s1_prompt_wav_greeting() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = Settings(
        prompt_audio_dir=root / "audio" / "yue",
        sip_s1_prompt_id="greeting",
    )
    path = resolve_s1_prompt_wav(settings)
    assert path.is_file()
    assert path.name == "greeting.wav"


def test_ensure_s1_playback_wav_8k(tmp_path: Path) -> None:
    import wave

    root = Path(__file__).resolve().parents[1]
    src = root / "audio" / "yue" / "greeting.wav"
    out = ensure_s1_playback_wav_8k(src, cache_dir=tmp_path)
    assert out.is_file()
    with wave.open(str(out), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 8000


def test_build_s1_startup_wav_includes_ask_name(tmp_path: Path) -> None:
    import wave

    root = Path(__file__).resolve().parents[1]
    settings = Settings(
        prompt_audio_dir=root / "audio" / "yue",
        sip_s1_prompt_id="greeting",
    )
    out = build_s1_startup_wav_8k(settings, cache_dir=tmp_path, gap_ms=200)
    assert out.is_file()
    with wave.open(str(out), "rb") as wf:
        assert wf.getframerate() == 8000
        # greeting(~4.2s) + gap + ask_name(~3.9s) ⇒ clearly longer than greeting alone
        assert wf.getnframes() / 8000 > 7.5


def test_resolve_s1_prompt_wav_missing() -> None:
    settings = Settings(
        prompt_audio_dir=Path("/nonexistent/prompt/root"),
        sip_s1_prompt_id="greeting",
    )
    with pytest.raises(FileNotFoundError, match="S1 prompt wav not found"):
        resolve_s1_prompt_wav(settings)


def test_sip_call_bridge_requires_loop() -> None:
    with pytest.raises(TypeError):
        SipCallBridge(call_id="test-call")  # type: ignore[call-arg]


def test_ensure_pjsua2_message_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pjsua2" or name.startswith("pjsua2."):
            raise ImportError("simulated missing pjsua2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="pjsua2 is not importable"):
        sip_pjsua.ensure_pjsua2()


def test_sip_settings_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.sip_user == "2000"
    assert s.sip_registrar == "172.17.7.53"
    assert "PCMU" in s.sip_codecs.upper()
    assert s.sip_s1_prompt_id == "greeting"
    assert s.sip_s1_on_eof == "keep"
    assert s.sip_public_addr == ""
    assert s.sip_media_mode == "session"


def test_sip_call_bridge_tx_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from app.transports import sip_session as sip_session_mod

    monkeypatch.setattr(sip_session_mod, "SIP_PLAYBACK_CACHE_DIR", tmp_path)
    loop = asyncio.new_event_loop()
    bridge = SipCallBridge(call_id="t1", settings=Settings(_env_file=None), loop=loop)

    def drain_duration() -> float:
        total = 0.0
        while not bridge._play_q.empty():
            item = bridge._play_q.get_nowait()
            assert item.path.is_file()
            total += item.duration_s
        return total

    # A small streaming chunk below the single-chunk threshold only buffers.
    bridge._tx_buf.clear()
    bridge._enqueue_tx_pcm16(b"\x00\x10" * 3200, force=False)  # 200 ms
    assert bridge._play_q.empty()

    # force flushes the buffered partial as one item.
    bridge._enqueue_tx_pcm16(b"", force=True)
    assert drain_duration() == pytest.approx(0.2, abs=0.01)

    # A whole local wav (>= 500 ms) is written as ONE item immediately.
    bridge._tx_buf.clear()
    bridge._enqueue_tx_pcm16(b"\x00\x10" * 9600, force=False)  # 600 ms
    assert drain_duration() == pytest.approx(0.6, abs=0.02)

    # Small streaming chunks accumulate WHOLE and flush as ONE item, so a
    # long dynamic TTS confirmation has no inter-chunk conference gaps.
    bridge._tx_buf.clear()
    for _ in range(4):
        bridge._enqueue_tx_pcm16(b"\x00\x10" * 4800, force=False)  # 300 ms each
    # Nothing is enqueued until the turn flushes.
    assert bridge._play_q.empty()
    bridge._enqueue_tx_pcm16(b"", force=True)
    assert drain_duration() == pytest.approx(1.2, abs=0.02)
    loop.close()


def test_sip_bridge_async_close_never_touches_native_media() -> None:
    import asyncio

    class NativeObject:
        def __getattr__(self, name):
            raise AssertionError(f"async close touched native media: {name}")

    async def run() -> None:
        loop = asyncio.get_running_loop()
        bridge = SipCallBridge(
            call_id="t-close",
            settings=Settings(_env_file=None),
            loop=loop,
        )
        bridge._aud = NativeObject()
        bridge._player = NativeObject()
        bridge._recorder = NativeObject()
        await bridge.close()

    asyncio.run(run())


def test_sip_start_disables_native_workers_and_polls_on_owner_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    settings = Settings(
        prompt_audio_dir=root / "audio" / "yue",
        sip_password="test-secret",
        sip_port=0,
        sip_public_addr="172.17.2.91",
        sip_media_mode="wav",
    )
    captured: dict[str, object] = {}

    class FakeAudioDevManager:
        def setNullDev(self) -> None:
            captured["null_audio"] = True

    class FakeEndpoint:
        def libCreate(self) -> None:
            pass

        def libInit(self, cfg) -> None:
            captured["cfg"] = cfg

        def transportCreate(self, transport, cfg) -> None:
            captured["transport_cfg"] = cfg

        def libStart(self) -> None:
            pass

        def audDevManager(self):
            return FakeAudioDevManager()

        def codecEnum2(self):
            return []

        def libHandleEvents(self, timeout_ms: int) -> int:
            captured["timeout_ms"] = timeout_ms
            return 3

        def libDestroy(self) -> None:
            captured["destroyed"] = True

    class FakeAccount:
        def create(self, cfg) -> None:
            captured["account_cfg"] = cfg

        def shutdown(self) -> None:
            captured["shutdown"] = True

    class FakeAccountConfig:
        def __init__(self) -> None:
            self.idUri = ""
            self.regConfig = SimpleNamespace(registrarUri="", registerOnAdd=False)
            self.sipConfig = SimpleNamespace(authCreds=[])

    class FakeEpConfig:
        def __init__(self) -> None:
            self.logConfig = SimpleNamespace(level=0, consoleLevel=0)
            self.uaConfig = SimpleNamespace(threadCnt=1, mainThreadOnly=False)
            self.medConfig = SimpleNamespace(channelCount=2)

    fake_pj = SimpleNamespace(
        Endpoint=FakeEndpoint,
        EpConfig=FakeEpConfig,
        TransportConfig=lambda: SimpleNamespace(port=None, publicAddress=""),
        AccountConfig=FakeAccountConfig,
        Account=FakeAccount,
        AuthCredInfo=lambda *args: args,
        PJSIP_TRANSPORT_UDP=1,
    )
    monkeypatch.setattr(sip_pjsua, "ensure_pjsua2", lambda: fake_pj)

    app = sip_pjsua.SipUaApp(settings)
    assert "8k" in app.wav_path.name
    app.start()
    cfg = captured["cfg"]
    assert cfg.uaConfig.threadCnt == 0
    assert cfg.uaConfig.mainThreadOnly is True
    assert captured["null_audio"] is True
    assert captured["transport_cfg"].publicAddress == "172.17.2.91"
    assert app.handle_events(25) == 3
    assert captured["timeout_ms"] == 25
    assert not hasattr(app, "_thread")
    app.stop()
    assert captured["destroyed"] is True
