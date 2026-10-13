from __future__ import annotations

import math
import struct

from app.transports.resample import (
    AUDIOSOCKET_FRAME_BYTES_8K,
    RATE_8K,
    RATE_16K,
    RateConverter,
    align_pcm16,
    make_downsampler,
    make_upsampler,
    resample_down,
    resample_up,
)


def _sine_pcm16(
    freq_hz: float,
    duration_s: float,
    sample_rate: int,
    *,
    amp: float = 0.4,
) -> bytes:
    n = int(sample_rate * duration_s)
    out = bytearray()
    for i in range(n):
        sample = int(
            32767 * amp * math.sin(2 * math.pi * freq_hz * i / sample_rate)
        )
        out.extend(struct.pack("<h", sample))
    return bytes(out)


def _rms(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    samples = struct.unpack("<" + "h" * (len(pcm) // 2), pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0.0
    acc = sum(float(s) * float(s) for s in samples)
    return (acc / len(samples)) ** 0.5


def test_align_pcm16_drops_odd_byte() -> None:
    assert align_pcm16(b"") == b""
    assert align_pcm16(b"\x01\x02\x03") == b"\x01\x02"
    assert align_pcm16(b"\x01\x02\x03\x04") == b"\x01\x02\x03\x04"


def test_empty_passthrough() -> None:
    assert resample_up(b"") == b""
    assert resample_down(b"") == b""


def test_upsample_length_roughly_doubles() -> None:
    pcm8 = _sine_pcm16(440.0, 0.2, RATE_8K)
    pcm16 = resample_up(pcm8)
    assert len(pcm16) % 2 == 0
    # Expect ~2x samples; ratecv filter delay allows a few ms slack.
    expected = len(pcm8) * 2
    assert abs(len(pcm16) - expected) <= RATE_16K * 2 * 4 // 1000  # ≤4ms


def test_downsample_length_roughly_halves() -> None:
    pcm16 = _sine_pcm16(440.0, 0.2, RATE_16K)
    pcm8 = resample_down(pcm16)
    assert len(pcm8) % 2 == 0
    expected = len(pcm16) // 2
    assert abs(len(pcm8) - expected) <= RATE_8K * 2 * 4 // 1000  # ≤4ms


def test_audiosocket_20ms_frame_upsamples_to_about_640_bytes() -> None:
    frame8 = _sine_pcm16(350.0, 0.02, RATE_8K)
    assert len(frame8) == AUDIOSOCKET_FRAME_BYTES_8K
    frame16 = resample_up(frame8)
    assert len(frame16) % 2 == 0
    # 20ms @ 16k = 640 bytes; allow a small ratecv edge difference.
    assert 600 <= len(frame16) <= 680


def test_streaming_upsample_matches_oneshot() -> None:
    pcm8 = _sine_pcm16(500.0, 0.5, RATE_8K)
    oneshot = resample_up(pcm8)

    up = make_upsampler()
    streamed = bytearray()
    step = AUDIOSOCKET_FRAME_BYTES_8K
    for offset in range(0, len(pcm8), step):
        streamed.extend(up.feed(pcm8[offset : offset + step]))

    # Streaming keeps filter state; totals should match within a small slack.
    assert abs(len(streamed) - len(oneshot)) <= RATE_16K * 2 * 6 // 1000
    assert _rms(bytes(streamed)) > 1000
    assert _rms(oneshot) > 1000


def test_roundtrip_keeps_energy() -> None:
    original = _sine_pcm16(400.0, 0.3, RATE_16K)
    down = resample_down(original)
    back = resample_up(down)
    assert _rms(original) > 1000
    assert _rms(back) > 0.5 * _rms(original)


def test_rate_converter_reset_clears_state() -> None:
    conv = RateConverter(RATE_8K, RATE_16K)
    first = conv.feed(_sine_pcm16(300.0, 0.05, RATE_8K))
    conv.reset()
    second = conv.feed(_sine_pcm16(300.0, 0.05, RATE_8K))
    assert len(first) == len(second)


def test_downsampler_factory() -> None:
    pcm16 = _sine_pcm16(600.0, 0.04, RATE_16K)
    down = make_downsampler()
    out = down.feed(pcm16)
    assert len(out) % 2 == 0
    assert len(out) < len(pcm16)
