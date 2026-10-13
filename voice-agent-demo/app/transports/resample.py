"""PCM16 rate conversion for AudioSocket (8 kHz) ↔ CallSession (16 kHz).

Uses stdlib ``audioop.ratecv`` (available on CPython 3.12; on 3.13+ install
``audioop-lts``). Keep a ``RateConverter`` per direction per call so filter
state carries across 20 ms frames.
"""

from __future__ import annotations

try:
    import audioop
except ImportError as exc:  # pragma: no cover - Python 3.13+ without audioop-lts
    raise ImportError(
        "audioop is required for SIP resampling. "
        "On Python 3.13+ install the audioop-lts package."
    ) from exc

WIDTH = 2  # 16-bit samples
CHANNELS = 1
RATE_8K = 8000
RATE_16K = 16000

# AudioSocket default: 20 ms @ 8 kHz mono PCM16 → 160 samples → 320 bytes.
AUDIOSOCKET_FRAME_MS = 20
AUDIOSOCKET_FRAME_BYTES_8K = RATE_8K * WIDTH * AUDIOSOCKET_FRAME_MS // 1000


def align_pcm16(pcm: bytes) -> bytes:
    """Drop a trailing odd byte so length is a whole number of int16 samples."""
    n = len(pcm) - (len(pcm) % WIDTH)
    return pcm if n == len(pcm) else pcm[:n]


class RateConverter:
    """Stateful PCM16 resampler (one direction, one call)."""

    def __init__(
        self,
        from_rate: int,
        to_rate: int,
        *,
        width: int = WIDTH,
        nchannels: int = CHANNELS,
    ) -> None:
        if from_rate <= 0 or to_rate <= 0:
            raise ValueError("sample rates must be positive")
        if width <= 0 or nchannels <= 0:
            raise ValueError("width and nchannels must be positive")
        self.from_rate = int(from_rate)
        self.to_rate = int(to_rate)
        self.width = int(width)
        self.nchannels = int(nchannels)
        self._state = None

    def feed(self, pcm: bytes) -> bytes:
        if not pcm:
            return b""
        if self.width == WIDTH:
            pcm = align_pcm16(pcm)
        if not pcm:
            return b""
        converted, self._state = audioop.ratecv(
            pcm,
            self.width,
            self.nchannels,
            self.from_rate,
            self.to_rate,
            self._state,
        )
        return converted

    def reset(self) -> None:
        self._state = None


def resample_up(pcm8k: bytes) -> bytes:
    """One-shot 8 kHz → 16 kHz (new converter; for tests / complete buffers)."""
    return RateConverter(RATE_8K, RATE_16K).feed(pcm8k)


def resample_down(pcm16k: bytes) -> bytes:
    """One-shot 16 kHz → 8 kHz (new converter; for tests / complete buffers)."""
    return RateConverter(RATE_16K, RATE_8K).feed(pcm16k)


def make_upsampler() -> RateConverter:
    """Streaming 8 kHz → 16 kHz converter for one AudioSocket call."""
    return RateConverter(RATE_8K, RATE_16K)


def make_downsampler() -> RateConverter:
    """Streaming 16 kHz → 8 kHz converter for one AudioSocket call."""
    return RateConverter(RATE_16K, RATE_8K)
