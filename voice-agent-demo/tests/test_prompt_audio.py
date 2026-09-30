from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from app.services.prompt_audio import PromptAudioBank


def _write_tone_wav(path: Path, *, seconds: float = 0.2, sample_rate: int = 16000) -> None:
    n = int(sample_rate * seconds)
    frames = bytearray()
    for i in range(n):
        sample = int(12000 * math.sin(2 * math.pi * 440 * i / sample_rate))
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))


def test_prompt_audio_bank_loads_and_chooses_filler(tmp_path: Path) -> None:
    root = tmp_path / "yue"
    root.mkdir()
    _write_tone_wav(root / "ack_checking.wav")
    _write_tone_wav(root / "ack_ok.wav")
    bank = PromptAudioBank(root)
    assert bank.has_any_filler() is True
    assert (
        bank.choose_filler("屋企冷气唔冻要维修，我叫陈先生电话91234567")
        == "ack_checking"
    )
    assert bank.load_pcm16("ack_ok") is not None
    assert bank.choose_filler("啱") is None
