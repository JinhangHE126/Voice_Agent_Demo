"""S1 SIP skeleton tests (no live FreeSWITCH / pjsua2 required)."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from app.config import Settings
from app.transports import sip_pjsua
from app.transports.sip_session import SipCallBridge, resolve_s1_prompt_wav


def test_resolve_s1_prompt_wav_greeting() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = Settings(
        prompt_audio_dir=root / "audio" / "yue",
        sip_s1_prompt_id="greeting",
    )
    path = resolve_s1_prompt_wav(settings)
    assert path.is_file()
    assert path.name == "greeting.wav"


def test_resolve_s1_prompt_wav_missing() -> None:
    settings = Settings(
        prompt_audio_dir=Path("/nonexistent/prompt/root"),
        sip_s1_prompt_id="greeting",
    )
    with pytest.raises(FileNotFoundError, match="S1 prompt wav not found"):
        resolve_s1_prompt_wav(settings)


def test_sip_call_bridge_s2_stub() -> None:
    import asyncio

    bridge = SipCallBridge(call_id="test-call")

    async def _run() -> None:
        with pytest.raises(NotImplementedError, match="S2"):
            await bridge.run()

    asyncio.run(_run())


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


def test_sip_start_disables_native_workers_and_polls_on_owner_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    settings = Settings(
        prompt_audio_dir=root / "audio" / "yue",
        sip_password="test-secret",
        sip_port=0,
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
            pass

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
        TransportConfig=lambda: SimpleNamespace(port=None),
        AccountConfig=FakeAccountConfig,
        Account=FakeAccount,
        AuthCredInfo=lambda *args: args,
        PJSIP_TRANSPORT_UDP=1,
    )
    monkeypatch.setattr(sip_pjsua, "ensure_pjsua2", lambda: fake_pj)

    app = sip_pjsua.SipUaApp(settings)
    app.start()
    cfg = captured["cfg"]
    assert cfg.uaConfig.threadCnt == 0
    assert cfg.uaConfig.mainThreadOnly is True
    assert captured["null_audio"] is True
    assert app.handle_events(25) == 3
    assert captured["timeout_ms"] == 25
    assert not hasattr(app, "_thread")
    app.stop()
    assert captured["destroyed"] is True
