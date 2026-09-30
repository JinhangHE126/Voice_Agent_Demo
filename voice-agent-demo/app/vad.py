from __future__ import annotations

import array
from collections import deque

try:
    import webrtcvad
except ImportError:  # pragma: no cover - energy fallback for minimal installs
    webrtcvad = None


class EnergyVAD:
    """WebRTC speech detector with an adaptive RMS noise gate.

    The class name is retained to avoid changing callers. When WebRTC VAD is
    unavailable it falls back to the adaptive energy detector.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        silence_ms: int = 500,
        min_speech_ms: int = 200,
        energy_threshold: float = 280.0,
        webrtc_mode: int | None = 2,
        noise_multiplier: float = 2.5,
        max_speech_ms: int = 12000,
        window_ms: int = 1200,
        window_min_voiced_ratio: float = 0.35,
    ) -> None:
        if frame_ms not in {10, 20, 30}:
            raise ValueError("WebRTC VAD frame_ms must be 10, 20, or 30")
        if sample_rate not in {8000, 16000, 32000, 48000}:
            raise ValueError("Unsupported WebRTC VAD sample rate")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.energy_threshold = energy_threshold
        self.noise_multiplier = noise_multiplier
        self.max_speech_ms = max_speech_ms
        self.window_ms = max(frame_ms, window_ms)
        self.window_min_voiced_ratio = max(0.0, min(1.0, window_min_voiced_ratio))
        self._frame_bytes = int(sample_rate * frame_ms / 1000) * 2
        self._buffer = bytearray()
        self._noise_floor = max(40.0, energy_threshold / noise_multiplier)
        self._noise_alpha = 0.03
        self._webrtc = None
        if webrtcvad is not None and webrtc_mode is not None:
            self._webrtc = webrtcvad.Vad(max(0, min(3, webrtc_mode)))
        self._in_speech = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._window_voiced = deque(
            maxlen=max(1, int(self.window_ms / self.frame_ms))
        )

    @property
    def backend(self) -> str:
        return "webrtc+adaptive_energy" if self._webrtc else "adaptive_energy"

    def reset(self) -> None:
        self._buffer.clear()
        self._reset_state()

    def _reset_state(self) -> None:
        self._in_speech = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._window_voiced.clear()

    def accept(self, pcm16: bytes) -> dict:
        if not pcm16:
            return {"event": "none", "energy": 0.0}
        self._buffer.extend(pcm16[: len(pcm16) - (len(pcm16) % 2)])
        selected = {"event": "none", "energy": 0.0}

        while len(self._buffer) >= self._frame_bytes:
            frame = bytes(self._buffer[: self._frame_bytes])
            del self._buffer[: self._frame_bytes]
            event = self._accept_frame(frame)
            selected["energy"] = event["energy"]
            selected["threshold"] = event["threshold"]
            # Preserve state transitions if a browser packet contains several
            # 20ms frames. speech_end has highest priority.
            if event["event"] == "speech_end":
                return event
            if event["event"] == "speech_start":
                selected = event
            elif (
                event["event"] == "speech_continue"
                and selected["event"] == "none"
            ):
                selected = event
        return selected

    def _accept_frame(self, frame: bytes) -> dict:
        energy = self._rms(frame)
        web_speech = (
            self._webrtc.is_speech(frame, self.sample_rate)
            if self._webrtc
            else True
        )
        threshold = max(
            self.energy_threshold,
            self._noise_floor * self.noise_multiplier,
        )
        voiced = web_speech and energy >= threshold
        self._window_voiced.append(1 if voiced else 0)

        # Learn ambient noise only from frames WebRTC considers non-speech.
        # A slow EMA follows fans/air-conditioning without chasing syllables.
        if not web_speech and not self._in_speech:
            self._noise_floor = (
                (1.0 - self._noise_alpha) * self._noise_floor
                + self._noise_alpha * energy
            )

        if voiced:
            self._silence_ms = 0.0
            self._speech_ms += self.frame_ms
            if self.max_speech_ms > 0 and self._speech_ms >= self.max_speech_ms:
                speech_ms = round(self._speech_ms)
                self._reset_state()
                return {
                    "event": "speech_end",
                    "energy": energy,
                    "threshold": threshold,
                    "speech_ms": speech_ms,
                    "reason": "max_speech_ms",
                }
            if not self._in_speech and self._speech_ms >= self.min_speech_ms:
                self._in_speech = True
                return {
                    "event": "speech_start",
                    "energy": energy,
                    "threshold": threshold,
                }
            if self._in_speech:
                return {
                    "event": "speech_continue",
                    "energy": energy,
                    "threshold": threshold,
                }
            return {"event": "none", "energy": energy, "threshold": threshold}

        if self._in_speech:
            if len(self._window_voiced) == self._window_voiced.maxlen:
                voiced_ratio = sum(self._window_voiced) / len(self._window_voiced)
                if voiced_ratio < self.window_min_voiced_ratio:
                    speech_ms = round(self._speech_ms)
                    self._reset_state()
                    return {
                        "event": "speech_end",
                        "energy": energy,
                        "threshold": threshold,
                        "speech_ms": speech_ms,
                        "reason": "window_ratio",
                        "voiced_ratio": round(voiced_ratio, 3),
                    }
            self._silence_ms += self.frame_ms
            if self._silence_ms >= self.silence_ms:
                speech_ms = round(self._speech_ms)
                self._reset_state()
                return {
                    "event": "speech_end",
                    "energy": energy,
                    "threshold": threshold,
                    "speech_ms": speech_ms,
                    "reason": "silence",
                }
            return {
                "event": "speech_continue",
                "energy": energy,
                "threshold": threshold,
            }

        self._speech_ms = 0.0
        return {"event": "none", "energy": energy, "threshold": threshold}

    @staticmethod
    def _rms(pcm16: bytes) -> float:
        if len(pcm16) < 2:
            return 0.0
        samples = array.array("h")
        samples.frombytes(pcm16[: len(pcm16) - (len(pcm16) % 2)])
        if not samples:
            return 0.0
        acc = 0.0
        for s in samples:
            acc += float(s) * float(s)
        return (acc / len(samples)) ** 0.5
