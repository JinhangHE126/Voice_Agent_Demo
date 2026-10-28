"""FreeSWITCH SIP User Agent via pjsua2.

Modes (``SIP_MEDIA_MODE``):
  - ``session`` (default, S2): RTP ↔ CallSession (ASR → DialogManager → TTS)
  - ``wav`` (S1): Answer + play greeting/ask_name file only (no listen)

pjsua2 is a native binding and is **not** installed via requirements.txt.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.transports.sip_session import (
    SipCallBridge,
    build_s1_startup_wav_8k,
)

logger = logging.getLogger(__name__)

PJSUA2_INSTALL_HINT = (
    "pjsua2 is not importable. Build/install PJSIP Python bindings first, e.g.\n"
    "  https://docs.pjsip.org/en/latest/pjsua2/using.html\n"
    "On Debian/Ubuntu you may try: sudo apt install python3-pjproject\n"
    "Then verify: python -c \"import pjsua2\""
)


def ensure_pjsua2() -> Any:
    """Import pjsua2 or raise RuntimeError with install hints."""
    try:
        import pjsua2 as pj  # type: ignore
    except ImportError as exc:
        raise RuntimeError(PJSUA2_INSTALL_HINT) from exc
    return pj


@dataclass(frozen=True)
class SipUaStatus:
    registered: bool
    reg_status: int
    reg_status_text: str
    active_calls: int


class SipUaApp:
    """Long-lived SIP UA: register, auto-answer, wav or CallSession media."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        wav_path: Path | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.media_mode = (self.settings.sip_media_mode or "session").strip().lower()
        if self.media_mode not in {"session", "wav"}:
            raise ValueError(
                f"SIP_MEDIA_MODE must be session|wav (got {self.media_mode!r})"
            )
        self.wav_path = (
            Path(wav_path)
            if wav_path
            else build_s1_startup_wav_8k(self.settings)
        )
        self.on_status = on_status
        self._pj: Any | None = None
        self._ep: Any | None = None
        self._acc: Any | None = None
        self._calls: dict[int, Any] = {}
        self._players: dict[int, Any] = {}
        self._recorders: dict[int, Any] = {}
        self._bridges: dict[int, SipCallBridge] = {}
        self._hangup_ids: list[int] = []
        self._pending_media: list[Any] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._owner_thread_id: int | None = None
        self._registered = False
        self._reg_status = 0
        self._reg_status_text = "init"
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_thread: threading.Thread | None = None

    def _emit(self, msg: str) -> None:
        logger.info("%s", msg)
        if self.on_status is not None:
            self.on_status(msg)

    def _ensure_async_loop(self) -> asyncio.AbstractEventLoop:
        if self._async_loop is not None:
            return self._async_loop

        loop_holder: dict[str, asyncio.AbstractEventLoop] = {}
        ready = threading.Event()

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop_holder["loop"] = loop
            ready.set()
            loop.run_forever()
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

        self._async_thread = threading.Thread(
            target=_runner,
            name="sip-asyncio",
            daemon=True,
        )
        self._async_thread.start()
        if not ready.wait(timeout=5.0):
            raise RuntimeError("failed to start SIP asyncio loop")
        self._async_loop = loop_holder["loop"]
        return self._async_loop

    def start(self) -> None:
        """Create endpoint, transport, account; poll events on this thread."""
        if self._ep is not None:
            return
        if self.media_mode == "wav" and not self.wav_path.is_file():
            raise FileNotFoundError(f"S1 wav missing: {self.wav_path}")
        if not (self.settings.sip_password or "").strip():
            logger.warning(
                "SIP_PASSWORD is empty — REGISTER will likely fail until PBX "
                "provides the extension secret"
            )
        if self.media_mode == "session":
            self._ensure_async_loop()

        pj = ensure_pjsua2()
        self._pj = pj
        self._owner_thread_id = threading.get_ident()

        ep = pj.Endpoint()
        ep.libCreate()

        ep_cfg = pj.EpConfig()
        ep_cfg.logConfig.level = int(self.settings.sip_log_level)
        ep_cfg.logConfig.consoleLevel = int(self.settings.sip_log_level)
        # Python callbacks must not run on PJSIP worker threads.
        ep_cfg.uaConfig.threadCnt = 0
        ep_cfg.uaConfig.mainThreadOnly = True
        ep_cfg.medConfig.channelCount = 1
        ep.libInit(ep_cfg)

        tcfg = pj.TransportConfig()
        tcfg.port = int(self.settings.sip_port)
        public_addr = (self.settings.sip_public_addr or "").strip()
        if public_addr:
            tcfg.publicAddress = public_addr
        transport = self.settings.sip_transport.strip().lower() or "udp"
        if transport != "udp":
            raise ValueError(
                f"S1 only supports SIP_TRANSPORT=udp for now (got {transport!r})"
            )
        ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, tcfg)
        ep.libStart()
        # Null device: conference clock without a Windows sound card.
        ep.audDevManager().setNullDev()
        self._configure_codecs(ep, pj)

        self._ep = ep
        self._acc = self._create_account(pj)
        self._emit(
            f"SIP UA started user={self.settings.sip_user} "
            f"registrar={self.settings.sip_registrar} "
            f"mode={self.media_mode} "
            f"wav={self.wav_path.name}"
        )
        self._stop.clear()

    def _configure_codecs(self, ep: Any, pj: Any) -> None:
        wanted = {
            c.strip().upper()
            for c in (self.settings.sip_codecs or "PCMU,PCMA").split(",")
            if c.strip()
        }
        try:
            codecs = ep.codecEnum2()
        except Exception:
            codecs = []
        for c in codecs:
            try:
                ep.codecSetPriority(c.codecId, 0)
            except Exception:
                pass
        priority = 255
        for c in codecs:
            cid = str(getattr(c, "codecId", "") or "")
            base = cid.split("/", 1)[0].upper()
            if base in wanted or cid.upper() in wanted:
                try:
                    ep.codecSetPriority(cid, priority)
                    logger.info("SIP codec priority %s -> %s", cid, priority)
                    priority = max(1, priority - 1)
                except Exception:
                    logger.exception("codecSetPriority failed for %s", cid)

    def _create_account(self, pj: Any) -> Any:
        user = self.settings.sip_user.strip()
        domain = self.settings.sip_domain.strip()
        registrar = self.settings.sip_registrar.strip() or domain
        password = self.settings.sip_password

        acc_cfg = pj.AccountConfig()
        acc_cfg.idUri = f"sip:{user}@{domain}"
        acc_cfg.regConfig.registrarUri = f"sip:{registrar}"
        acc_cfg.regConfig.registerOnAdd = True
        cred = pj.AuthCredInfo("digest", "*", user, 0, password)
        acc_cfg.sipConfig.authCreds.append(cred)

        app = self

        class AgentAccount(pj.Account):
            def onRegState(self, prm) -> None:  # noqa: N802
                app._on_reg_state(self)

            def onIncomingCall(self, prm) -> None:  # noqa: N802
                app._on_incoming_call(self, prm)

        acc = AgentAccount()
        acc.create(acc_cfg)
        return acc

    def _on_reg_state(self, acc: Any) -> None:
        try:
            info = acc.getInfo()
            self._reg_status = int(info.regStatus)
            self._reg_status_text = str(info.regStatusText or "")
            self._registered = bool(info.regIsActive) and 200 <= self._reg_status < 300
            self._emit(
                f"SIP reg status={self._reg_status} "
                f"text={self._reg_status_text!r} active={info.regIsActive}"
            )
        except Exception:
            logger.exception("onRegState failed")

    def _on_incoming_call(self, acc: Any, prm: Any) -> None:
        pj = self._pj
        assert pj is not None
        call_id = int(prm.callId)
        self._emit(f"SIP incoming call id={call_id}")

        app = self

        class AgentCall(pj.Call):
            def __init__(self) -> None:
                pj.Call.__init__(self, acc, call_id)
                self._player: Any | None = None
                self._port: Any | None = None
                self._bridge: SipCallBridge | None = None
                self._played = False
                self._media_ready = False

            def onCallState(self, prm) -> None:  # noqa: N802
                app._on_call_state(self)

            def onCallMediaState(self, prm) -> None:  # noqa: N802
                app._on_call_media_state(self)

        call = AgentCall()
        with self._lock:
            self._calls[call_id] = call

        try:
            op = pj.CallOpParam()
            op.statusCode = 200
            call.answer(op)
            self._emit(f"SIP answered call id={call_id}")
        except Exception:
            logger.exception("answer failed call id=%s", call_id)
            with self._lock:
                self._calls.pop(call_id, None)

    def _on_call_state(self, call: Any) -> None:
        pj = self._pj
        assert pj is not None
        try:
            ci = call.getInfo()
        except Exception:
            logger.exception("getInfo failed")
            return
        self._emit(
            f"SIP call id={ci.id} state={ci.stateText} last={ci.lastStatusCode}"
        )
        if ci.state == pj.PJSIP_INV_STATE_CONFIRMED:
            # Defer media wiring until after this PJSIP callback returns.
            self._queue_start_media(call)
        if ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            self._cleanup_call(call, ci.id)

    def _on_call_media_state(self, call: Any) -> None:
        pj = self._pj
        assert pj is not None
        try:
            ci = call.getInfo()
        except Exception:
            logger.exception("getInfo (media) failed")
            return

        media_active = False
        for mi in ci.media:
            if (
                mi.type == pj.PJMEDIA_TYPE_AUDIO
                and mi.status == pj.PJSUA_CALL_MEDIA_ACTIVE
            ):
                media_active = True
                break
        call._media_ready = media_active
        self._emit(
            f"SIP media call={ci.id} active={media_active} state={ci.stateText}"
        )
        if media_active and ci.state == pj.PJSIP_INV_STATE_CONFIRMED:
            self._queue_start_media(call)

    def _queue_start_media(self, call: Any) -> None:
        """Schedule media start outside the PJSIP callback stack."""
        if getattr(call, "_played", False) or getattr(call, "_media_queued", False):
            return
        if not getattr(call, "_media_ready", False):
            return
        call._media_queued = True
        with self._lock:
            self._pending_media.append(call)
        self._emit("SIP media start queued")

    def _process_pending_media(self) -> None:
        with self._lock:
            pending = list(self._pending_media)
            self._pending_media.clear()
        for call in pending:
            if getattr(call, "_played", False):
                continue
            if not getattr(call, "_media_ready", False):
                call._media_queued = False
                continue
            try:
                if self.media_mode == "session":
                    self._start_session_media(call)
                else:
                    self._start_wav_media(call)
            except Exception:
                logger.exception("SIP deferred media start failed")
                call._media_queued = False

    def _get_call_audio(self, call: Any) -> Any | None:
        pj = self._pj
        assert pj is not None
        try:
            ci = call.getInfo()
        except Exception:
            logger.exception("getInfo (audio) failed")
            return None
        if ci.state != pj.PJSIP_INV_STATE_CONFIRMED:
            return None
        for mi in ci.media:
            if mi.type != pj.PJMEDIA_TYPE_AUDIO:
                continue
            if mi.status != pj.PJSUA_CALL_MEDIA_ACTIVE:
                continue
            try:
                return call.getAudioMedia(mi.index)
            except Exception:
                logger.exception("getAudioMedia failed")
        return None

    def _start_wav_media(self, call: Any) -> None:
        pj = self._pj
        assert pj is not None
        aud = self._get_call_audio(call)
        if aud is None:
            return
        try:
            ci = call.getInfo()
            player = pj.AudioMediaPlayer()
            no_loop = getattr(pj, "PJMEDIA_FILE_NO_LOOP", 1)
            player.createPlayer(str(self.wav_path), no_loop)
            player.startTransmit(aud)
            call._player = player
            call._played = True
            with self._lock:
                self._players[int(ci.id)] = player
            self._emit(f"SIP S1 playing wav={self.wav_path.name} call={ci.id}")
        except Exception:
            logger.exception("S1 playback failed")

    def _start_session_media(self, call: Any) -> None:
        pj = self._pj
        assert pj is not None
        loop = self._async_loop
        if loop is None:
            logger.error("SIP session mode requires asyncio loop")
            return
        aud = self._get_call_audio(call)
        if aud is None:
            call._media_queued = False
            self._emit("SIP session media deferred: audio not ready")
            return
        try:
            ci = call.getInfo()
            call_id = int(ci.id)
        except Exception:
            logger.exception("getInfo failed before session media")
            call._media_queued = False
            return

        def request_hangup() -> None:
            with self._lock:
                if call_id not in self._hangup_ids:
                    self._hangup_ids.append(call_id)

        bridge = SipCallBridge(
            call_id=f"sip-{call_id}",
            settings=self.settings,
            loop=loop,
            request_hangup=request_hangup,
        )
        player_holder: dict[str, Any] = {}
        recorder = pj.AudioMediaRecorder()
        self._emit(f"SIP attaching recorder/player bridge call={call_id}")
        try:
            bridge.attach_pjsua_media(
                pj,
                aud,
                recorder=recorder,
                player_holder=player_holder,
            )
        except Exception:
            logger.exception("SIP recorder attach failed call=%s", call_id)
            call._media_queued = False
            self._emit("SIP falling back to wav player after recorder failure")
            self._start_wav_media(call)
            return

        call._bridge = bridge
        call._recorder = recorder
        call._player_holder = player_holder
        call._played = True
        with self._lock:
            self._recorders[call_id] = recorder
            self._bridges[call_id] = bridge
            # Keep a slot for the active chunk player (replaced over time).
            self._players[call_id] = player_holder

        def _on_started(fut: Any) -> None:
            try:
                fut.result()
                self._emit(f"SIP S2 CallSession started call={call_id}")
            except Exception:
                logger.exception("SIP CallSession start failed call=%s", call_id)

        fut = asyncio.run_coroutine_threadsafe(bridge.start(), loop)
        fut.add_done_callback(_on_started)
        self._emit(f"SIP S2 CallSession start scheduled call={call_id}")

    def _cleanup_call(self, call: Any, call_id: int) -> None:
        with self._lock:
            player_or_holder = self._players.pop(int(call_id), None)
            recorder = self._recorders.pop(int(call_id), None)
            bridge = self._bridges.pop(int(call_id), None)
            self._calls.pop(int(call_id), None)
            self._hangup_ids = [i for i in self._hangup_ids if i != int(call_id)]

        if isinstance(player_or_holder, dict):
            player = player_or_holder.get("player")
        else:
            player = player_or_holder

        # Sever asyncio-side references first; all native destruction below is
        # deliberately performed on this PJSIP owner thread.
        if bridge is not None:
            bridge.detach_pjsua_media_owner()
        # Also remove duplicate references held by the Call wrapper before
        # destroying native objects, preventing a later SWIG double-destroy.
        call._player = None
        call._player_holder = None
        call._recorder = None

        if player is not None:
            # Drop the final Python reference on the owner thread. Calling
            # destroy() explicitly and then letting SWIG destruct it produced
            # duplicate conference-port removal.
            player = None

        if recorder is not None:
            recorder = None

        if bridge is not None and self._async_loop is not None:
            call._bridge = None
            fut = asyncio.run_coroutine_threadsafe(
                bridge.close(), self._async_loop
            )

            def _closed(done: Any) -> None:
                try:
                    done.result()
                    self._emit(f"SIP async bridge closed call={call_id}")
                except Exception:
                    logger.debug("bridge close failed", exc_info=True)

            fut.add_done_callback(_closed)

        self._emit(f"SIP call cleaned id={call_id}")

    def hangup_call(self, call: Any) -> None:
        pj = self._pj
        if pj is None:
            return
        try:
            op = pj.CallOpParam()
            call.hangup(op)
        except Exception:
            logger.exception("hangup failed")

    def _process_pending_hangups(self) -> None:
        with self._lock:
            ids = list(self._hangup_ids)
            self._hangup_ids.clear()
            calls = {cid: self._calls.get(cid) for cid in ids}
        for cid, call in calls.items():
            if call is not None:
                self._emit(f"SIP hangup after session.completed call={cid}")
                self.hangup_call(call)

    def handle_events(self, timeout_ms: int = 50) -> int:
        """Poll PJSIP once on the Endpoint creator thread."""
        ep = self._ep
        if ep is None or self._stop.is_set():
            return 0
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError(
                "SipUaApp.handle_events() must run on the thread that called start()"
            )
        try:
            n = int(ep.libHandleEvents(max(0, int(timeout_ms))))
        except Exception:
            logger.exception("libHandleEvents failed")
            n = 0
        self._process_pending_media()
        with self._lock:
            bridges = list(self._bridges.values())
        for bridge in bridges:
            try:
                bridge.pump()
            except Exception:
                logger.exception("SIP bridge pump failed")
        self._process_pending_hangups()
        return n

    def status(self) -> SipUaStatus:
        with self._lock:
            n = len(self._calls)
        return SipUaStatus(
            registered=self._registered,
            reg_status=self._reg_status,
            reg_status_text=self._reg_status_text,
            active_calls=n,
        )

    def wait_registered(self, timeout_s: float = 15.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._registered:
                return True
            self.handle_events(50)
        return self._registered

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                self.handle_events(100)
        except KeyboardInterrupt:
            self._emit("SIP UA interrupted")
        finally:
            self.stop()

    def stop(self) -> None:
        if (
            self._owner_thread_id is not None
            and threading.get_ident() != self._owner_thread_id
        ):
            raise RuntimeError(
                "SipUaApp.stop() must run on the thread that called start()"
            )
        self._stop.set()

        with self._lock:
            calls = list(self._calls.items())
            self._calls.clear()
        for call_id, call in calls:
            self.hangup_call(call)
            self._cleanup_call(call, call_id)

        acc = self._acc
        self._acc = None
        if acc is not None:
            try:
                acc.shutdown()
            except Exception:
                logger.exception("account shutdown failed")

        ep = self._ep
        self._ep = None
        if ep is not None:
            try:
                ep.libDestroy()
            except Exception:
                logger.exception("libDestroy failed")

        loop = self._async_loop
        self._async_loop = None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        thread = self._async_thread
        self._async_thread = None
        if thread is not None:
            thread.join(timeout=3.0)

        self._players.clear()
        self._recorders.clear()
        self._bridges.clear()
        self._owner_thread_id = None
        self._emit("SIP UA stopped")
