from __future__ import annotations

from scripts.vad_experiment import _extract_range


def test_extract_range_uses_frame_units_once() -> None:
    frame_bytes = 4
    pcm = bytes(range(40))

    extracted = _extract_range(
        pcm,
        start_frame=2,
        end_frame=5,
        frame_bytes=frame_bytes,
    )

    assert extracted == pcm[8:20]
    assert len(extracted) == 12
