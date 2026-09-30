from __future__ import annotations

import wave
from pathlib import Path


class PromptAudioBank:
    """
    Local pre-recorded prompt/filler audio.

    Expected layout:
      audio/yue/ack_ok.wav
      audio/yue/ack_got_it.wav
      audio/yue/ack_checking.wav
      audio/yue/greeting.wav
      ...
    """

    def __init__(self, root: Path, *, sample_rate: int = 16000) -> None:
        self.root = root
        self.sample_rate = sample_rate
        self._cache: dict[tuple[str, str], bytes] = {}
        self._ack_idx = 0
        self._ack_ids = ("ack_checking", "ack_got_it", "ack_ok", "ack_writing")

    def resolve(self, prompt_id: str, *, lang: str = "yue") -> Path | None:
        if not prompt_id:
            return None
        # preferred: root/<lang>/<prompt>.wav, fallback: root/<prompt>.wav
        candidates = [
            self.root / lang / f"{prompt_id}.wav",
            self.root / f"{prompt_id}.wav",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def load_pcm16(self, prompt_id: str, *, lang: str = "yue") -> bytes | None:
        key = (lang, prompt_id)
        if key in self._cache:
            return self._cache[key]
        path = self.resolve(prompt_id, lang=lang)
        if path is None:
            return None
        with wave.open(str(path), "rb") as wf:
            if (
                wf.getnchannels() != 1
                or wf.getframerate() != self.sample_rate
                or wf.getsampwidth() != 2
            ):
                return None
            pcm = wf.readframes(wf.getnframes())
        self._cache[key] = pcm
        return pcm

    def choose_filler(
        self, user_text: str, *, lang: str = "yue", force: bool = False
    ) -> str | None:
        text = (user_text or "").strip()
        if not force and len(text) < 12:
            # Short answers ("陈先生"/"啱") usually don't need filler.
            return None
        # Prefer checking phrasing for long one-shot turns.
        if len(text) >= 24 and self.resolve("ack_checking", lang=lang):
            return "ack_checking"
        available = [pid for pid in self._ack_ids if self.resolve(pid, lang=lang)]
        if not available:
            return None
        prompt_id = available[self._ack_idx % len(available)]
        self._ack_idx += 1
        return prompt_id

    def has_any_filler(self, *, lang: str = "yue") -> bool:
        return any(self.resolve(pid, lang=lang) for pid in self._ack_ids)
