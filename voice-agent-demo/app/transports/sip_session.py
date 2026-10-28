"""SIP ↔ CallSession glue (S2).

This Windows ``pjsua2-wheel`` build **native-crashes** when subclassing
``AudioMediaPort`` (SWIG directors). So S2 uses only stable APIs:

```text
RX: call AudioMedia → AudioMediaRecorder (.wav growing file)
    → tail-read PCM16 @ conference clock (16 kHz) → CallSession.on_pcm16

TX: CallSession.on_audio_out (16 kHz)
    → buffer → write short wav → AudioMediaPlayer → call AudioMedia
```

S1 ``SIP_MEDIA_MODE=wav`` still uses a single startup wav player in sip_pjsua.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import ROOT_DIR, Settings, get_settings
from app.models import ServerEvent
from app.orchestrator import CallSession
from app.services.prompt_audio import PromptAudioBank
from app.transports.resample import RATE_8K, RATE_16K, RateConverter

logger = logging.getLogger(__name__)

SIP_PLAYBACK_CACHE_DIR = ROOT_DIR / "output" / "sip"
WAV_HEADER = 44
# Play TX in ~200 ms chunks for lower first-byte latency than whole-utterance.
TX_CHUNK_BYTES_16K = RATE_16K * 2 * 200 // 1000  # 6400 bytes

HangupFn = Callable[[], None]


def _read_pcm16_mono(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"wav must be 16-bit PCM (got sampwidth={width}): {path}")
    pcm = frames
    if channels > 1:
        sample_bytes = width * channels
        left = bytearray()
        for i in range(0, len(pcm) - sample_bytes + 1, sample_bytes):
            left.extend(pcm[i : i + width])
        pcm = bytes(left)
    return pcm, int(rate)


def resolve_s1_prompt_wav(settings: Settings | None = None) -> Path:
    """Resolve ``SIP_S1_PROMPT_ID`` under the prompt audio directory."""
    cfg = settings or get_settings()
    prompt_id = (cfg.sip_s1_prompt_id or "greeting").strip() or "greeting"
    root = cfg.resolved_prompt_audio_dir()
    bank = PromptAudioBank(root, sample_rate=cfg.sample_rate)
    path = bank.resolve(prompt_id, lang="yue")
    if path is None:
        raise FileNotFoundError(
            f"S1 prompt wav not found for id={prompt_id!r} under {root} "
            f"(expected {root / 'yue' / (prompt_id + '.wav')} or "
            f"{root / (prompt_id + '.wav')})"
        )
    return path.resolve()


def ensure_s1_playback_wav_8k(
    source: Path,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Return a mono 16-bit PCM WAV at 8 kHz (legacy S1 helper)."""
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"S1 source wav missing: {source}")

    out_dir = (cache_dir or SIP_PLAYBACK_CACHE_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source.stem}_8k.wav"
    if (
        out_path.is_file()
        and out_path.stat().st_mtime >= source.stat().st_mtime
    ):
        return out_path

    pcm, rate = _read_pcm16_mono(source)
    out_pcm = pcm if rate == RATE_8K else RateConverter(rate, RATE_8K).feed(pcm)

    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE_8K)
        out.writeframes(out_pcm)
    return out_path


def build_s1_startup_wav_8k(
    settings: Settings | None = None,
    *,
    cache_dir: Path | None = None,
    gap_ms: int = 400,
) -> Path:
    """Build greeting + gap + ask_name @ 8 kHz (S1 wav mode)."""
    cfg = settings or get_settings()
    greeting = resolve_s1_prompt_wav(cfg)
    root = cfg.resolved_prompt_audio_dir()
    bank = PromptAudioBank(root, sample_rate=cfg.sample_rate)
    ask = bank.resolve("ask_name", lang="yue")
    if ask is None:
        return ensure_s1_playback_wav_8k(greeting, cache_dir=cache_dir)

    out_dir = (cache_dir or SIP_PLAYBACK_CACHE_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "s1_startup_greeting_ask_name_8k.wav"
    sources = [greeting.resolve(), ask.resolve()]
    if out_path.is_file() and all(
        out_path.stat().st_mtime >= src.stat().st_mtime for src in sources
    ):
        return out_path

    chunks: list[bytes] = []
    for src in sources:
        eight = ensure_s1_playback_wav_8k(src, cache_dir=out_dir)
        pcm, rate = _read_pcm16_mono(eight)
        if rate != RATE_8K:
            pcm = RateConverter(rate, RATE_8K).feed(pcm)
        chunks.append(pcm)
        gap_bytes = RATE_8K * 2 * max(0, int(gap_ms)) // 1000
        chunks.append(b"\x00" * gap_bytes)
    if gap_ms > 0 and chunks:
        chunks.pop()

    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE_8K)
        out.writeframes(b"".join(chunks))
    return out_path


def write_pcm16_wav(path: Path, pcm16: bytes, *, sample_rate: int = RATE_16K) -> float:
    """Write mono PCM16 wav; return duration seconds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return len(pcm16) / (2 * sample_rate)


@dataclass
class _PlayItem:
    path: Path
    duration_s: float


class SipCallBridge:
    """One SIP call bound to one CallSession using recorder/player files."""

    def __init__(
        self,
        *,
        call_id: str,
        settings: Settings | None = None,
        loop: asyncio.AbstractEventLoop,
        request_hangup: HangupFn | None = None,
        conf_rate: int = RATE_16K,
    ) -> None:
        self.call_id = call_id
        self.settings = settings or get_settings()
        self.loop = loop
        self.request_hangup = request_hangup
        self.conf_rate = int(conf_rate)
        self._session: CallSession | None = None
        self._closed = False
        self._rx_path = (SIP_PLAYBACK_CACHE_DIR / f"rx_{call_id}.wav").resolve()
        self._rx_offset = WAV_HEADER
        self._rx_frames = 0
        self._tx_seq = 0
        self._tx_buf = bytearray()
        self._tx_lock = threading.Lock()
        self._play_q: queue.Queue[_PlayItem] = queue.Queue()
        self._player: Any | None = None
        self._play_until = 0.0
        self._aud: Any | None = None
        self._pj: Any | None = None
        self._recorder: Any | None = None

    @property
    def session(self) -> CallSession | None:
        return self._session

    def attach_pjsua_media(
        self,
        pj: Any,
        aud: Any,
        *,
        recorder: Any,
        player_holder: dict[str, Any],
    ) -> None:
        """Bind call audio + recorder; player created per TX chunk."""
        self._pj = pj
        self._aud = aud
        self._recorder = recorder
        self._player_holder = player_holder
        SIP_PLAYBACK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if self._rx_path.exists():
            try:
                self._rx_path.unlink()
            except OSError:
                pass
        # Conference clock on this wheel is 16 kHz with null device.
        options = getattr(pj, "PJMEDIA_FILE_WRITE_PCM", 0)
        recorder.createRecorder(str(self._rx_path), 0, 0, options)
        aud.startTransmit(recorder)
        logger.info("SIP RX recorder → %s", self._rx_path)

    def pump(self) -> None:
        """Called from SipUaApp.handle_events (owner thread)."""
        if self._closed:
            return
        self._pump_rx()
        self._pump_tx_player()

    def detach_pjsua_media_owner(self) -> None:
        """Forget native media references on the PJSIP owner thread.

        Native objects are destroyed by ``SipUaApp``. The asyncio thread must
        never call stopTransmit/destroy after PJSIP has deinitialized a call.
        """
        self._aud = None
        self._recorder = None
        self._player = None
        holder = getattr(self, "_player_holder", None)
        if holder is not None:
            holder.clear()

    def _pump_rx(self) -> None:
        path = self._rx_path
        if not path.is_file() or self._session is None:
            return
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size <= self._rx_offset:
            return
        try:
            with path.open("rb") as f:
                f.seek(self._rx_offset)
                data = f.read()
        except OSError:
            return
        if not data:
            return
        # Keep even length for int16.
        if len(data) % 2:
            data = data[:-1]
        if not data:
            return
        self._rx_offset += len(data)
        self._rx_frames += 1
        if self._rx_frames == 1 or self._rx_frames % 25 == 0:
            logger.info(
                "SIP RX wav tail frames=%s call=%s bytes=%s",
                self._rx_frames,
                self.call_id,
                len(data),
            )
        pcm = data
        if self.conf_rate != RATE_16K:
            pcm = RateConverter(self.conf_rate, RATE_16K).feed(data)
        if pcm:
            asyncio.run_coroutine_threadsafe(self._session.on_pcm16(pcm), self.loop)

    def _enqueue_tx_pcm16(self, pcm16: bytes, *, force: bool = False) -> None:
        if self._closed:
            return
        with self._tx_lock:
            if pcm16:
                self._tx_buf.extend(pcm16)
            while self._tx_buf and (
                force or len(self._tx_buf) >= TX_CHUNK_BYTES_16K
            ):
                take = len(self._tx_buf) if force else TX_CHUNK_BYTES_16K
                chunk = bytes(self._tx_buf[:take])
                del self._tx_buf[:take]
                self._tx_seq += 1
                path = (
                    SIP_PLAYBACK_CACHE_DIR / f"tx_{self.call_id}_{self._tx_seq}.wav"
                ).resolve()
                dur = write_pcm16_wav(path, chunk, sample_rate=RATE_16K)
                try:
                    self._play_q.put_nowait(_PlayItem(path=path, duration_s=dur))
                except queue.Full:
                    logger.warning("SIP TX play queue full call=%s", self.call_id)
                if force:
                    break

    def _pump_tx_player(self) -> None:
        pj = self._pj
        aud = self._aud
        if pj is None or aud is None:
            return
        now = time.monotonic()
        if self._player is not None and now < self._play_until:
            return
        if self._player is not None and now >= self._play_until:
            try:
                self._player.stopTransmit(aud)
            except Exception:
                pass
            # Drop the Python reference on the owner thread and let the SWIG
            # destructor release the file-player port. An explicit destroy()
            # here double-removes the conference port (PJ_EINVAL noise).
            self._player = None
            self._player_holder.pop("player", None)

        try:
            item = self._play_q.get_nowait()
        except queue.Empty:
            return
        try:
            player = pj.AudioMediaPlayer()
            no_loop = getattr(pj, "PJMEDIA_FILE_NO_LOOP", 1)
            player.createPlayer(str(item.path), no_loop)
            player.startTransmit(aud)
            self._player = player
            self._player_holder["player"] = player
            # Small pad so we don't cut the tail.
            self._play_until = time.monotonic() + item.duration_s + 0.05
            logger.info(
                "SIP TX play %s dur=%.2fs call=%s",
                item.path.name,
                item.duration_s,
                self.call_id,
            )
        except Exception:
            logger.exception("SIP TX play failed call=%s", self.call_id)

    async def start(self) -> None:
        if self._session is not None:
            return

        async def emit(event: ServerEvent) -> None:
            if event.type in {
                "state",
                "asr.final",
                "asr.partial",
                "turn.done",
                "session.completed",
                "error",
            }:
                logger.info(
                    "sip_event type=%s call=%s data=%s",
                    event.type,
                    event.call_id,
                    event.data,
                )
            if event.type == "session.completed" and not self._closed:
                if self.request_hangup is not None:
                    self.request_hangup()

        async def on_audio_out(pcm16k: bytes) -> None:
            self._enqueue_tx_pcm16(pcm16k, force=False)

        async def on_audio_flush() -> None:
            # Flush whatever remains as a final play chunk.
            self._enqueue_tx_pcm16(b"\x00\x00", force=True)

        session_settings = self.settings
        if (
            self.settings.input_denoise_enabled
            and not self.settings.sip_input_denoise_enabled
        ):
            # G.711 phone audio is already clean; running RNNoise on the
            # asyncio thread contends with the pjsua2 poll thread for the GIL
            # and stalls every chunk by 100 ms+. Disable it for SIP by default.
            session_settings = self.settings.model_copy(
                update={"input_denoise_enabled": False}
            )
            logger.info(
                "SIP disabling RNNoise input denoise call=%s (sip_input_denoise_enabled=false)",
                self.call_id,
            )

        construct_started = time.perf_counter()
        self._session = CallSession(
            emit=emit,
            settings=session_settings,
            call_id=self.call_id,
            on_audio_out=on_audio_out,
            on_audio_flush=on_audio_flush,
            auto_complete_playback=True,
        )
        logger.info(
            "SIP CallSession constructed call=%s mode=%s elapsed_ms=%.1f",
            self.call_id,
            self.settings.agent_mode,
            (time.perf_counter() - construct_started) * 1000,
        )
        await self._session.start()
        logger.info("SIP CallSession started call=%s", self.call_id)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        session = self._session
        self._session = None
        # No pjsua2 calls here: this coroutine runs on the asyncio thread,
        # while all native media lifecycle is owned by SipUaApp's main thread.
        if session is not None:
            try:
                await session.close()
            except Exception:
                logger.exception("CallSession close failed call=%s", self.call_id)
        logger.info(
            "SIP CallSession closed call=%s rx_frames=%s",
            self.call_id,
            self._rx_frames,
        )


# Back-compat export name used by older imports/tests.
def make_sip_audio_port(pj: Any, bridge: SipCallBridge) -> Any:  # noqa: ARG001
    raise RuntimeError(
        "AudioMediaPort is unsafe on this pjsua2 Windows wheel; "
        "SipCallBridge now uses AudioMediaRecorder/Player instead"
    )
