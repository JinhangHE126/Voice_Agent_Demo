from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.services.denoise import DenoiseConfig, InputDenoiser
from app.vad import EnergyVAD


@dataclass
class Segment:
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int
    duration_ms: int
    end_reason: str | None
    speech_ms: int | None
    voiced_ratio: float | None


def _read_wav_mono_16k_pcm16(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        if channels != 1 or sample_rate != 16000 or sample_width != 2:
            raise ValueError(
                f"WAV must be mono/16k/16-bit PCM, got channels={channels}, "
                f"sample_rate={sample_rate}, sample_width={sample_width}"
            )
        return wf.readframes(wf.getnframes()), wf.getframerate()


def _write_wav_mono_16k_pcm16(path: Path, pcm: bytes, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def _split_frames(pcm: bytes, frame_bytes: int) -> list[bytes]:
    if frame_bytes <= 0:
        return []
    usable = len(pcm) - (len(pcm) % frame_bytes)
    return [pcm[i : i + frame_bytes] for i in range(0, usable, frame_bytes)]


def _extract_range(pcm: bytes, *, start_frame: int, end_frame: int, frame_bytes: int) -> bytes:
    start = max(0, start_frame) * frame_bytes
    end = max(start_frame, end_frame) * frame_bytes
    return pcm[start:end]


def _default_profile(settings: Settings) -> dict[str, Any]:
    return {
        "sample_rate": settings.sample_rate,
        "frame_ms": settings.frame_ms,
        "silence_ms": settings.vad_silence_ms,
        "min_speech_ms": settings.min_speech_ms,
        "energy_threshold": settings.vad_energy_threshold,
        "webrtc_mode": settings.vad_webrtc_mode,
        "noise_multiplier": settings.vad_noise_multiplier,
        "max_speech_ms": settings.vad_max_speech_ms,
        "window_ms": settings.vad_window_ms,
        "window_min_voiced_ratio": settings.vad_window_min_voiced_ratio,
    }


def _coerce_value(text: str) -> Any:
    raw = text.strip()
    if raw.lower() in {"none", "null"}:
        return None
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _normalize_key(key: str) -> str:
    mapping = {
        "vad_silence_ms": "silence_ms",
        "vad_energy_threshold": "energy_threshold",
        "vad_webrtc_mode": "webrtc_mode",
        "vad_noise_multiplier": "noise_multiplier",
        "vad_max_speech_ms": "max_speech_ms",
        "vad_window_ms": "window_ms",
        "vad_window_min_voiced_ratio": "window_min_voiced_ratio",
    }
    return mapping.get(key.strip(), key.strip())


def _parse_profile_override(spec: str) -> tuple[str, dict[str, Any]]:
    if ":" not in spec:
        raise ValueError(
            f"Invalid profile override '{spec}'. "
            "Expected format: name:key=value,key=value"
        )
    name, pairs = spec.split(":", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid profile name in '{spec}'")
    updates: dict[str, Any] = {}
    for pair in pairs.split(","):
        piece = pair.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(
                f"Invalid key/value '{piece}' in '{spec}'. Use key=value."
            )
        key, value = piece.split("=", 1)
        updates[_normalize_key(key)] = _coerce_value(value)
    return name, updates


def _load_profiles_json(path: Path) -> dict[str, dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("profiles_json root must be an object")
    out: dict[str, dict[str, Any]] = {}
    for name, updates in obj.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("profile names in profiles_json must be non-empty strings")
        if not isinstance(updates, dict):
            raise ValueError(f"profile '{name}' must map to an object of parameters")
        out[name] = {_normalize_key(k): v for k, v in updates.items()}
    return out


def _run_profile(
    *,
    pcm: bytes,
    profile_name: str,
    profile: dict[str, Any],
    pre_roll_ms: int,
    post_roll_ms: int,
    save_frame_metrics: bool,
) -> dict[str, Any]:
    vad = EnergyVAD(
        sample_rate=int(profile["sample_rate"]),
        frame_ms=int(profile["frame_ms"]),
        silence_ms=int(profile["silence_ms"]),
        min_speech_ms=int(profile["min_speech_ms"]),
        energy_threshold=float(profile["energy_threshold"]),
        webrtc_mode=profile["webrtc_mode"],
        noise_multiplier=float(profile["noise_multiplier"]),
        max_speech_ms=int(profile["max_speech_ms"]),
        window_ms=int(profile["window_ms"]),
        window_min_voiced_ratio=float(profile["window_min_voiced_ratio"]),
    )

    frame_bytes = int(profile["sample_rate"] * profile["frame_ms"] / 1000) * 2
    frames = _split_frames(pcm, frame_bytes)
    pre_roll_frames = max(0, math.ceil(pre_roll_ms / float(profile["frame_ms"])))
    post_roll_frames = max(0, math.ceil(post_roll_ms / float(profile["frame_ms"])))
    tail = pcm[len(frames) * frame_bytes :]

    segments: list[Segment] = []
    metrics: list[dict[str, Any]] = []
    active_start: int | None = None
    pending_end: int | None = None
    pending_meta: dict[str, Any] = {}

    for idx, frame in enumerate(frames):
        event = vad.accept(frame)
        energy = float(event.get("energy") or 0.0)
        threshold = float(event.get("threshold") or 0.0)

        if save_frame_metrics:
            metrics.append(
                {
                    "frame_index": idx,
                    "t_ms": int(idx * profile["frame_ms"]),
                    "event": event.get("event"),
                    "energy": round(energy, 2),
                    "threshold": round(threshold, 2),
                    "in_speech": bool(getattr(vad, "_in_speech", False)),
                    "noise_floor": round(float(getattr(vad, "_noise_floor", 0.0)), 2),
                }
            )

        name = event.get("event")
        if active_start is None and name == "speech_start":
            active_start = max(0, idx - pre_roll_frames)

        if active_start is not None and pending_end is None and name == "speech_end":
            pending_end = min(len(frames), idx + 1 + post_roll_frames)
            pending_meta = {
                "reason": event.get("reason"),
                "speech_ms": event.get("speech_ms"),
                "voiced_ratio": event.get("voiced_ratio"),
            }

        if active_start is not None and pending_end is not None and (idx + 1) >= pending_end:
            start_frame = active_start
            end_frame = pending_end
            start_ms = int(start_frame * profile["frame_ms"])
            end_ms = int(end_frame * profile["frame_ms"])
            segments.append(
                Segment(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=end_ms - start_ms,
                    end_reason=pending_meta.get("reason"),
                    speech_ms=(
                        int(pending_meta["speech_ms"])
                        if pending_meta.get("speech_ms") is not None
                        else None
                    ),
                    voiced_ratio=(
                        float(pending_meta["voiced_ratio"])
                        if pending_meta.get("voiced_ratio") is not None
                        else None
                    ),
                )
            )
            active_start = None
            pending_end = None
            pending_meta = {}

    if active_start is not None:
        start_ms = int(active_start * profile["frame_ms"])
        end_ms = int(len(frames) * profile["frame_ms"])
        segments.append(
            Segment(
                start_frame=active_start,
                end_frame=len(frames),
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=end_ms - start_ms,
                end_reason="eof",
                speech_ms=None,
                voiced_ratio=None,
            )
        )

    segments_pcm = bytearray()
    per_segment_bytes: list[int] = []
    per_segment_audio_start_frames: list[int] = []
    silence_gap_ms = 200
    silence_gap_frames = max(1, int(silence_gap_ms / profile["frame_ms"]))
    gap = b"\x00" * (silence_gap_frames * frame_bytes)

    previous_end_frame = 0
    for i, seg in enumerate(segments):
        # Pre/post-roll can make adjacent segments overlap. Clamp the second
        # segment so concatenated listening output never repeats audio.
        audio_start_frame = max(seg.start_frame, previous_end_frame)
        chunk = _extract_range(
            pcm,
            start_frame=audio_start_frame,
            end_frame=seg.end_frame,
            frame_bytes=frame_bytes,
        )
        previous_end_frame = max(previous_end_frame, seg.end_frame)
        per_segment_audio_start_frames.append(audio_start_frame)
        per_segment_bytes.append(len(chunk))
        segments_pcm.extend(chunk)
        if i != len(segments) - 1:
            segments_pcm.extend(gap)
    segments_pcm.extend(tail)

    return {
        "profile_name": profile_name,
        "profile": profile,
        "frame_bytes": frame_bytes,
        "frame_count": len(frames),
        "tail_bytes": len(tail),
        "segments": [
            {
                "index": i + 1,
                "start_ms": s.start_ms,
                "audio_start_ms": int(
                    per_segment_audio_start_frames[i] * profile["frame_ms"]
                ),
                "end_ms": s.end_ms,
                "duration_ms": s.duration_ms,
                "end_reason": s.end_reason,
                "speech_ms": s.speech_ms,
                "voiced_ratio": s.voiced_ratio,
                "bytes": per_segment_bytes[i],
            }
            for i, s in enumerate(segments)
        ],
        "segments_pcm16": bytes(segments_pcm),
        "frame_metrics": metrics if save_frame_metrics else None,
    }


def _apply_optional_denoise(
    *,
    pcm: bytes,
    sample_rate: int,
    frame_ms: int,
    enabled: bool,
    mode: str,
) -> tuple[bytes, dict[str, Any]]:
    cfg = DenoiseConfig(
        enabled=enabled,
        mode=mode,
        sample_rate=sample_rate,
    )
    denoiser = InputDenoiser(cfg)
    if not denoiser.active:
        return pcm, {"enabled": enabled, "mode": mode, "active": False}

    frame_bytes = int(sample_rate * frame_ms / 1000) * 2
    out = bytearray()
    for i in range(0, len(pcm), frame_bytes):
        out.extend(denoiser.process(pcm[i : i + frame_bytes]))
    out.extend(denoiser.flush())
    return bytes(out), {
        "enabled": enabled,
        "mode": mode,
        "active": denoiser.active,
        "input_bytes": len(pcm),
        "output_bytes": len(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline VAD experiments on one wav with baseline + parameter variants."
        )
    )
    parser.add_argument(
        "--wav",
        required=True,
        help="Absolute path to mono/16k/16-bit PCM wav.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help=(
            "Output directory. Default: voice-agent-demo/output/vad_experiment/"
            "<wav_stem>_<timestamp>"
        ),
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help=(
            "Variant profile override. Format: "
            "name:key=value,key=value. "
            "Example: sensitive:vad_webrtc_mode=1,vad_energy_threshold=220"
        ),
    )
    parser.add_argument(
        "--profiles-json",
        default="",
        help="Optional JSON file of variant profile overrides.",
    )
    parser.add_argument(
        "--pre-roll-ms",
        type=int,
        default=220,
        help="Extra audio before speech_start in segment wav.",
    )
    parser.add_argument(
        "--post-roll-ms",
        type=int,
        default=200,
        help="Extra audio after speech_end in segment wav.",
    )
    parser.add_argument(
        "--save-frame-metrics",
        action="store_true",
        help="Include per-frame energy/threshold metrics in report json.",
    )
    parser.add_argument(
        "--denoise-enabled",
        action="store_true",
        help="Enable server-side denoise before running VAD profiles.",
    )
    parser.add_argument(
        "--denoise-mode",
        default="rnnoise",
        help="Denoise mode. Supported: rnnoise.",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav).expanduser().resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    settings = Settings()
    baseline = _default_profile(settings)

    profiles: dict[str, dict[str, Any]] = {"baseline": baseline.copy()}
    for spec in args.profile:
        name, updates = _parse_profile_override(spec)
        profile = baseline.copy()
        profile.update(updates)
        profiles[name] = profile

    if args.profiles_json:
        json_profiles = _load_profiles_json(Path(args.profiles_json).expanduser().resolve())
        for name, updates in json_profiles.items():
            profile = baseline.copy()
            profile.update(updates)
            profiles[name] = profile

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else ROOT / "output" / "vad_experiment" / f"{wav_path.stem}_{ts}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    pcm, sample_rate = _read_wav_mono_16k_pcm16(wav_path)
    shutil.copy2(wav_path, out_root / "original.wav")
    pcm_for_vad, denoise_meta = _apply_optional_denoise(
        pcm=pcm,
        sample_rate=sample_rate,
        frame_ms=settings.frame_ms,
        enabled=bool(args.denoise_enabled),
        mode=str(args.denoise_mode),
    )
    if denoise_meta.get("active"):
        _write_wav_mono_16k_pcm16(out_root / "denoised.wav", pcm_for_vad, sample_rate)

    summary: dict[str, Any] = {
        "input_wav": str(wav_path),
        "sample_rate": sample_rate,
        "pre_roll_ms": args.pre_roll_ms,
        "post_roll_ms": args.post_roll_ms,
        "denoise": denoise_meta,
        "profiles": [],
    }

    for name, profile in profiles.items():
        result = _run_profile(
            pcm=pcm_for_vad,
            profile_name=name,
            profile=profile,
            pre_roll_ms=args.pre_roll_ms,
            post_roll_ms=args.post_roll_ms,
            save_frame_metrics=args.save_frame_metrics,
        )

        seg_wav = out_root / f"{name}_segments.wav"
        _write_wav_mono_16k_pcm16(seg_wav, result.pop("segments_pcm16"), sample_rate)

        profile_out = out_root / f"{name}_report.json"
        profile_out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        segments = result["segments"]
        total_duration_ms = sum(int(s["duration_ms"]) for s in segments)
        summary["profiles"].append(
            {
                "name": name,
                "segments_wav": str(seg_wav),
                "report_json": str(profile_out),
                "segments": len(segments),
                "total_segment_ms": total_duration_ms,
                "first_start_ms": segments[0]["start_ms"] if segments else None,
                "last_end_ms": segments[-1]["end_ms"] if segments else None,
                "profile": profile,
            }
        )

        print(
            f"[{name}] segments={len(segments)} total_ms={total_duration_ms} "
            f"wav={seg_wav.name} report={profile_out.name}",
            flush=True,
        )

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("VAD experiment done")
    print(f"Output dir : {out_root}")
    print(f"Original   : {out_root / 'original.wav'}")
    if denoise_meta.get("active"):
        print(f"Denoised   : {out_root / 'denoised.wav'}")
    print(f"Summary    : {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
