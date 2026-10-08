from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DenoiseConfig:
    enabled: bool = False
    mode: str = "off"
    sample_rate: int = 16000


class RNNoiseDenoiser:
    """Streaming RNNoise wrapper that accepts and returns mono PCM16."""

    def __init__(self, cfg: DenoiseConfig) -> None:
        self.cfg = cfg
        try:
            from pyrnnoise import RNNoise
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "RNNoise is enabled but pyrnnoise is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc
        self._denoiser = RNNoise(sample_rate=cfg.sample_rate)

    def process(self, pcm16: bytes) -> bytes:
        if not pcm16:
            return b""
        usable = pcm16[: len(pcm16) - (len(pcm16) % 2)]
        if not usable:
            return b""
        samples = np.frombuffer(usable, dtype="<i2").copy()
        out = bytearray()
        for _speech_prob, frame in self._denoiser.denoise_chunk(samples):
            pcm = np.asarray(frame).reshape(-1)
            if pcm.dtype != np.int16:
                pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
            out.extend(pcm.astype("<i2", copy=False).tobytes())
        return bytes(out)

    def flush(self) -> bytes:
        # RNNoise works on fixed 10 ms frames. Any sub-frame tail is dropped.
        self._denoiser.reset()
        return b""


class InputDenoiser:
    def __init__(self, cfg: DenoiseConfig) -> None:
        self.cfg = cfg
        mode = (cfg.mode or "off").strip().lower()
        self.mode = mode if cfg.enabled else "off"
        self._impl = RNNoiseDenoiser(cfg) if self.mode == "rnnoise" else None

    @property
    def active(self) -> bool:
        return self._impl is not None

    def process(self, pcm16: bytes) -> bytes:
        if not self._impl:
            return pcm16
        return self._impl.process(pcm16)

    def flush(self) -> bytes:
        if not self._impl:
            return b""
        return self._impl.flush()
