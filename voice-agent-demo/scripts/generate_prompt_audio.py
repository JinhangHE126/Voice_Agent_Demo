"""Generate fixed Yue prompt WAV files via MiniMax HTTP TTS."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402

# Fixed prompts used by the repair-order flow.
PROMPTS: dict[str, str] = {
    "greeting": "你好，广东话、普通话或者英文都可以。",
    "ask_name": "请问点称呼你？可以讲姓氏加先生或女士。",
    "ask_name_title": "收到，请问应该称呼你做先生、女士，定係其他称呼？",
    "ask_repair": "请简单讲下要维修嘅内容。",
    "ask_phone": "请讲一下你嘅联络电话。",
    "ask_address": (
        "请讲一下维修地址。最好按呢个顺序讲：先讲香港、九龙定新界，"
        "再讲地区，然后街道同门牌，再讲大厦名，最后讲楼层同室号。"
        "例如：香港湾仔坚尼地道105号雅佳大厦12楼25室。"
    ),
    "ask_confirm_yesno": "请讲啱定唔啱；如果唔啱，可以直接讲要改边项资料。",
    "order_saved": "已经帮你登记咗维修订单，我哋会尽快安排同事联络你。",
    "invalid_phone_length": "你啱啱讲嘅电话号码位数唔啱，请逐个数字再讲一次。",
    "ask_correct_field": "请问边项资料唔啱？你可以直接讲正确嘅资料。",
    "please_wait_asr": "收到，等我一下。",
    "connecting_asr": "正在帮你接通，请稍等一下。",
    "asr_unavailable": "唔好意思，暂时未能接通，请稍后再试。",
}


def _read_env_fallback(key: str) -> str:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return ""


def _write_wav(path: Path, pcm: bytes, *, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def _synthesize(text: str, settings) -> bytes:
    api_key = settings.minimax_api_key.strip() or _read_env_fallback("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY is missing")

    urls = [
        settings.minimax_http_url.strip() or "https://api.minimax.io/v1/t2a_v2",
        "https://api.minimaxi.com/v1/t2a_v2",
        "https://api.minimax.io/v1/t2a_v2",
    ]
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    candidates = []
    for url in urls:
        if url and url not in seen:
            seen.add(url)
            candidates.append(url)

    payload = {
        "model": settings.minimax_tts_model,
        "text": text,
        "stream": False,
        "output_format": "hex",
        "language_boost": settings.minimax_language_boost or "Chinese,Yue",
        "voice_setting": {
            "voice_id": settings.minimax_tts_voice,
            "speed": 1.08,
            "vol": 1,
            "pitch": 0,
        },
        "audio_setting": {
            "sample_rate": settings.minimax_sample_rate,
            "bitrate": 128000,
            "format": "pcm",
            "channel": 1,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = ""
    with httpx.Client(timeout=60.0) as client:
        for url in candidates:
            try:
                resp = client.post(url, headers=headers, json=payload)
            except Exception as exc:  # noqa: BLE001
                last_error = f"{url}: {exc!r}"
                continue
            if resp.status_code >= 400:
                last_error = f"{url}: HTTP {resp.status_code} {resp.text[:240]}"
                continue
            data = resp.json()
            status = (data.get("base_resp") or {}).get("status_code", 0)
            if status not in (0, None):
                last_error = (
                    f"{url}: API {status} "
                    f"{(data.get('base_resp') or {}).get('status_msg')}"
                )
                continue
            audio_hex = (data.get("data") or {}).get("audio") or ""
            if not audio_hex:
                last_error = f"{url}: empty audio payload"
                continue
            return bytes.fromhex(audio_hex)
    raise RuntimeError(last_error or "MiniMax TTS failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "audio" / "yue",
        help="Output directory for WAV files",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Optional subset of prompt ids",
    )
    args = parser.parse_args()

    settings = get_settings()
    selected = args.only or list(PROMPTS.keys())
    unknown = [pid for pid in selected if pid not in PROMPTS]
    if unknown:
        raise SystemExit(f"Unknown prompt ids: {unknown}")

    print(
        json.dumps(
            {
                "model": settings.minimax_tts_model,
                "voice": settings.minimax_tts_voice,
                "language_boost": settings.minimax_language_boost,
                "sample_rate": settings.minimax_sample_rate,
                "out_dir": str(args.out_dir),
                "count": len(selected),
            },
            ensure_ascii=False,
        )
    )

    for prompt_id in selected:
        text = PROMPTS[prompt_id]
        print(f"generating {prompt_id}.wav ...")
        pcm = _synthesize(text, settings)
        out_path = args.out_dir / f"{prompt_id}.wav"
        _write_wav(out_path, pcm, sample_rate=settings.minimax_sample_rate)
        duration_ms = round(len(pcm) / 2 / settings.minimax_sample_rate * 1000, 1)
        print(f"  saved {out_path.name} bytes={len(pcm)} duration_ms={duration_ms}")

    print("done")


if __name__ == "__main__":
    main()
