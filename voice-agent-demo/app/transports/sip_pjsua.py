"""FreeSWITCH SIP User Agent skeleton (S1) via pjsua2.

S1 behaviour:
  REGISTER as SIP_USER @ SIP_DOMAIN toward SIP_REGISTRAR
  → on INVITE: Answer 200
  → on media active: play local wav once (default greeting)
  → hang up on remote BYE (optionally hang up after wav EOF)

pjsua2 is a native binding and is **not** installed via requirements.txt.
Import fails with a clear message if the library is missing.

S2 will replace file playback with CallSession PCM bridging
(see ``sip_session.SipCallBridge``).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.transports.sip_session import resolve_s1_prompt_wav

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
    """Long-lived SIP UA: register, auto-answer, S1 wav playback."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        wav_path: Path | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.wav_path = Path(wav_path) if wav_path else resolve_s1_prompt_wav(self.settings)
        self.on_status = on_status
        self._pj: Any | None = None
        self._ep: Any | None = None
        self._acc: Any | None = None
        self._calls: dict[int, Any] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._owner_thread_id: int | None = None
        self._registered = False
        self._reg_status = 0
        self._reg_status_text = "init"

    def _emit(self, msg: str) -> None:
        logger.info("%s", msg)
        if self.on_status is not None:
            self.on_status(msg)

    def start(self) -> None:
        """Create endpoint, transport, account; start lib event thread."""
        if self._ep is not None:
            return
        if not self.wav_path.is_file():
            raise FileNotFoundError(f"S1 wav missing: {self.wav_path}")
        if not (self.settings.sip_password or "").strip():
            logger.warning(
                "SIP_PASSWORD is empty — REGISTER will likely fail until PBX "
                "provides the extension secret"
            )

        pj = ensure_pjsua2()
        self._pj = pj
        self._owner_thread_id = threading.get_ident()

        ep = pj.Endpoint()
        ep.libCreate()

        ep_cfg = pj.EpConfig()
        ep_cfg.logConfig.level = int(self.settings.sip_log_level)
        ep_cfg.logConfig.consoleLevel = int(self.settings.sip_log_level)
        # Python callbacks must not run on PJSIP worker threads. Poll all SIP
        # events on the same thread that created the Endpoint instead.
        ep_cfg.uaConfig.threadCnt = 0
        ep_cfg.uaConfig.mainThreadOnly = True
        # Prefer G.711 clock; pjsua2 resamples file/player as needed.
        ep_cfg.medConfig.channelCount = 1
        ep.libInit(ep_cfg)

        tcfg = pj.TransportConfig()
        tcfg.port = int(self.settings.sip_port)
        transport = self.settings.sip_transport.strip().lower() or "udp"
        if transport != "udp":
            raise ValueError(
                f"S1 only supports SIP_TRANSPORT=udp for now (got {transport!r})"
            )
        ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, tcfg)
        ep.libStart()
        # This UA runs as a server-side media endpoint. A physical Windows
        # microphone/speaker may not exist (RDP/headless hosts), but PJSIP's
        # conference bridge still needs a clock to move WAV/PCM to RTP.
        ep.audDevManager().setNullDev()
        self._configure_codecs(ep, pj)

        self._ep = ep
        self._acc = self._create_account(pj)
        self._emit(
            f"SIP UA started user={self.settings.sip_user} "
            f"registrar={self.settings.sip_registrar} "
            f"wav={self.wav_path.name}"
        )

        self._stop.clear()

    def _configure_codecs(self, ep: Any, pj: Any) -> None:
        wanted = {
            c.strip().upper()
            for c in (self.settings.sip_codecs or "PCMU,PCMA").split(",")
            if c.strip()
        }
        # Lower everything, then boost requested G.711 names.
        try:
            codecs = ep.codecEnum2()
        except Exception:
            codecs = []
        for c in codecs:
            try:
                ep.codecSetPriority(c.codecId, 0)
            except Exception:
                pass
        # Common id forms: "PCMU/8000/1", "PCMU"
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
            def onRegState(self, prm) -> None:  # noqa: N802 — pjsua2 API
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
                self._played = False

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
        if ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            self._cleanup_call(call, ci.id)

    def _on_call_media_state(self, call: Any) -> None:
        pj = self._pj
        assert pj is not None
        if getattr(call, "_played", False):
            return
        try:
            ci = call.getInfo()
        except Exception:
            logger.exception("getInfo (media) failed")
            return

        for mi in ci.media:
            if mi.type != pj.PJMEDIA_TYPE_AUDIO:
                continue
            if mi.status != pj.PJSUA_CALL_MEDIA_ACTIVE:
                continue
            try:
                aud = call.getAudioMedia(mi.index)
            except Exception:
                logger.exception("getAudioMedia failed")
                continue
            try:
                player = _PromptPlayer(
                    app=self,
                    call=call,
                    pj=pj,
                    on_eof=self.settings.sip_s1_on_eof,
                )
                # PJMEDIA_FILE_NO_LOOP = 1 when available
                no_loop = getattr(pj, "PJMEDIA_FILE_NO_LOOP", 1)
                player.createPlayer(str(self.wav_path), no_loop)
                player.startTransmit(aud)
                call._player = player
                call._played = True
                self._emit(
                    f"SIP S1 playing wav={self.wav_path.name} call={ci.id}"
                )
            except Exception:
                logger.exception("S1 playback failed call=%s", ci.id)
            return

    def _cleanup_call(self, call: Any, call_id: int) -> None:
        player = getattr(call, "_player", None)
        if player is not None:
            try:
                if hasattr(player, "destroy"):
                    player.destroy()
            except Exception:
                logger.debug("player destroy failed", exc_info=True)
            call._player = None
        with self._lock:
            self._calls.pop(int(call_id), None)
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

    def handle_events(self, timeout_ms: int = 50) -> int:
        """Poll PJSIP once on the Endpoint creator thread.

        PJSUA2's Python binding is not safe when callbacks enter Python from
        native worker threads. ``start()``, this method, and ``stop()`` must be
        called by the same thread.
        """
        ep = self._ep
        if ep is None or self._stop.is_set():
            return 0
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError(
                "SipUaApp.handle_events() must run on the thread that called start()"
            )
        try:
            return int(ep.libHandleEvents(max(0, int(timeout_ms))))
        except Exception:
            logger.exception("libHandleEvents failed")
            return 0

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
        """Block until KeyboardInterrupt / stop()."""
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
            calls = list(self._calls.values())
            self._calls.clear()
        for call in calls:
            self.hangup_call(call)

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
        self._owner_thread_id = None
        self._emit("SIP UA stopped")


class _PromptPlayer:
    """AudioMediaPlayer subclass factory — real base class comes from pjsua2."""

    def __new__(cls, *, app: SipUaApp, call: Any, pj: Any, on_eof: str):
        on_eof_norm = (on_eof or "keep").strip().lower()

        class _Player(pj.AudioMediaPlayer):
            def onEof2(self) -> None:  # noqa: N802
                app._emit("SIP S1 wav EOF")
                if on_eof_norm == "hangup":
                    app.hangup_call(call)

        return _Player()
