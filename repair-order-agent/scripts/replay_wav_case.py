from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import uuid
import wave
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repair_order.domain.repair_order import RepairOrderDraft
from repair_order.repositories.order_repository import OrderRepository
from repair_order.services.dialog_manager import DialogManager


def _read_env_value(env_path: Path, key: str) -> str:
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


def _event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:12]}"


def _load_expected_case(wav_path: Path) -> dict | None:
    expected_path = ROOT / "data" / "expected" / f"{wav_path.stem}.json"
    if not expected_path.exists():
        return None
    try:
        obj = json.loads(expected_path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _field_match(expected: dict, actual: dict, key: str) -> bool:
    exp = expected.get(key)
    got = actual.get(key)
    if exp is None:
        return got is None
    if got is None:
        return False
    return str(exp).strip() == str(got).strip()


def _load_pcm16_mono_16k(wav_path: Path) -> bytes:
    with wave.open(str(wav_path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getframerate() != 16000 or wf.getsampwidth() != 2:
            raise ValueError("WAV must be mono, 16kHz, 16-bit PCM.")
        return wf.readframes(wf.getnframes())


async def transcribe_wav_with_qwen(
    wav_path: Path,
    *,
    api_key: str,
    model: str = "qwen3-asr-flash-realtime",
    ws_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
    language: str = "",
    frame_ms: int = 40,
    segment_s: float = 25.0,
    send_pace_ms: float = 8.0,
    final_timeout_s: float = 45.0,
) -> tuple[str, str]:
    """
    Long calls are split into segments. Each segment is committed and finalized,
    then transcripts are concatenated. This avoids realtime WS buffer/keepalive
    failures when dumping multi-minute audio at once.
    """
    pcm = _load_pcm16_mono_16k(wav_path)
    sample_rate = 16000
    bytes_per_sample = 2
    samples_per_frame = int(sample_rate * (frame_ms / 1000.0))
    bytes_per_frame = samples_per_frame * bytes_per_sample
    segment_bytes = int(sample_rate * segment_s) * bytes_per_sample

    segments = [pcm[i : i + segment_bytes] for i in range(0, len(pcm), segment_bytes)]
    transcripts: list[str] = []
    detected_lang = ""

    url = f"{ws_url}?model={model}"
    headers = {"Authorization": f"Bearer {api_key}"}

    async with websockets.connect(
        url,
        additional_headers=headers,
        max_size=8 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=60,
        open_timeout=20,
        close_timeout=5,
    ) as ws:
        await ws.send(
            json.dumps(
                {
                    "event_id": _event_id(),
                    "type": "session.update",
                    "session": {
                        "input_audio_format": "pcm",
                        "sample_rate": sample_rate,
                        "input_audio_transcription": (
                            {"language": language} if language else {}
                        ),
                        "turn_detection": None,
                    },
                }
            )
        )

        for index, segment in enumerate(segments, start=1):
            print(
                f"[ASR] segment {index}/{len(segments)} "
                f"({len(segment) / (sample_rate * bytes_per_sample):.1f}s)...",
                flush=True,
            )
            partial = ""
            final_text = ""
            final_event = asyncio.Event()
            stop_reader = asyncio.Event()

            async def reader() -> None:
                nonlocal final_text, detected_lang, partial
                while not stop_reader.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except TimeoutError:
                        continue
                    except Exception:
                        return
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    event = msg.get("type") or ""
                    if event == "conversation.item.input_audio_transcription.delta":
                        chunk = (
                            msg.get("delta")
                            or msg.get("text")
                            or msg.get("transcript")
                            or ""
                        )
                        if chunk:
                            partial = (
                                chunk
                                if len(chunk) > len(partial)
                                else (partial + chunk)
                            )
                            if msg.get("transcript"):
                                partial = msg["transcript"]
                    elif event == "conversation.item.input_audio_transcription.completed":
                        final_text = (
                            msg.get("transcript")
                            or msg.get("text")
                            or partial
                            or ""
                        ).strip()
                        detected_lang = (
                            msg.get("language") or language or detected_lang or ""
                        ).strip()
                        final_event.set()
                    elif event == "error":
                        err = msg.get("error") or msg
                        raise RuntimeError(f"ASR error: {err}")

            read_task = asyncio.create_task(reader())
            try:
                for offset in range(0, len(segment), bytes_per_frame):
                    chunk = segment[offset : offset + bytes_per_frame]
                    if not chunk:
                        continue
                    await ws.send(
                        json.dumps(
                            {
                                "event_id": _event_id(),
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(chunk).decode("ascii"),
                            }
                        )
                    )
                    if send_pace_ms > 0:
                        await asyncio.sleep(send_pace_ms / 1000.0)

                await ws.send(
                    json.dumps(
                        {
                            "event_id": _event_id(),
                            "type": "input_audio_buffer.commit",
                        }
                    )
                )
                try:
                    await asyncio.wait_for(final_event.wait(), timeout=final_timeout_s)
                except TimeoutError:
                    final_text = final_text or partial
            finally:
                stop_reader.set()
                read_task.cancel()
                try:
                    await read_task
                except asyncio.CancelledError:
                    pass

            text = (final_text or partial or "").strip()
            if text:
                transcripts.append(text)
                print(f"[ASR] segment {index} text: {text[:80]}...", flush=True)
            else:
                print(f"[ASR] segment {index} empty", flush=True)

        try:
            await ws.send(
                json.dumps({"event_id": _event_id(), "type": "session.finish"})
            )
        except Exception:
            pass

    return " ".join(transcripts).strip(), detected_lang


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Replay one wav case into repair order flow."
    )
    parser.add_argument("--wav", required=True, help="Absolute path to mono 16k PCM wav.")
    parser.add_argument(
        "--call-id", default=f"call-{int(time.time())}", help="Synthetic call id."
    )
    parser.add_argument(
        "--language", default="", help="ASR language. Empty enables auto detect."
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional path to write JSON result.",
    )
    parser.add_argument(
        "--use-llm-extractor",
        action="store_true",
        help="Enable LLM structured extraction (with rule fallback).",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav).expanduser().resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        fallback_env = (
            Path(__file__).resolve().parents[2] / "voice-agent-demo" / ".env"
        )
        api_key = _read_env_value(fallback_env, "DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY not found in env or voice-agent-demo/.env"
        )

    if args.use_llm_extractor:
        os.environ["REPAIR_ORDER_USE_LLM_EXTRACTOR"] = "1"

    transcript, detected_lang = await transcribe_wav_with_qwen(
        wav_path=wav_path,
        api_key=api_key,
        language=args.language,
    )

    repo = OrderRepository()
    manager = DialogManager(repository=repo)
    draft = RepairOrderDraft()
    turn = manager.process_user_text(args.call_id, draft, transcript)

    result = {
        "wav": str(wav_path),
        "asr_language": detected_lang or "",
        "llm_extractor_enabled": args.use_llm_extractor,
        "asr_text": transcript,
        "draft": {
            "customer_name": draft.customer_name,
            "customer_phone": draft.customer_phone,
            "repair_description": draft.repair_description,
            "state": draft.state.value,
            "confirmed": draft.confirmed,
        },
        "next_response": {
            "prompt_id": turn.plan.prompt_id,
            "dynamic": turn.plan.dynamic,
            "text": turn.plan.text,
        },
        "saved_orders": len(repo.saved_orders),
    }

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else ROOT / "output" / f"replay_{wav_path.stem}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    expected = _load_expected_case(wav_path)
    compare = None
    if expected:
        compare = {
            "customer_name_match": _field_match(expected, result["draft"], "customer_name"),
            "customer_phone_match": _field_match(expected, result["draft"], "customer_phone"),
            "repair_description_present": bool(result["draft"].get("repair_description")),
        }
        result["expected_compare"] = compare
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print("=" * 60)
    print("WAV:", wav_path)
    print("ASR language:", detected_lang or "(unknown)")
    print("ASR text:", transcript or "(empty)")
    print("-" * 60)
    print("Draft:")
    print("  customer_name       =", draft.customer_name)
    print("  customer_phone      =", draft.customer_phone)
    print("  repair_description  =", draft.repair_description)
    print("  state               =", draft.state.value)
    print("  confirmed           =", draft.confirmed)
    print("-" * 60)
    print("Next response:")
    print("  prompt_id =", turn.plan.prompt_id)
    print("  dynamic   =", turn.plan.dynamic)
    print("  text      =", turn.plan.text)
    print("-" * 60)
    print("Saved orders:", len(repo.saved_orders))
    if compare:
        print("-" * 60)
        print("Expected compare:")
        print("  customer_name_match      =", compare["customer_name_match"])
        print("  customer_phone_match     =", compare["customer_phone_match"])
        print("  repair_description_found =", compare["repair_description_present"])
    print("Result JSON:", out_path)
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
