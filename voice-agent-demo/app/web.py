from __future__ import annotations

import base64
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.models import ServerEvent
from app.orchestrator import CallSession

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"

app = FastAPI(title="Voice Agent Demo", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {
        "ok": True,
        "mode": settings.agent_mode,
        "stack": "Qwen ASR → DialogManager/LLM → MiniMax TTS",
        "business_mode": settings.business_mode,
        "target": "EoU-to-First-Audio ≤ 1000ms",
        "has_asr": settings.has_asr_key,
        "has_tts": settings.has_minimax_key,
        "has_fast_llm": settings.has_fast_llm,
        "has_bonsai": settings.has_bonsai,
    }


@app.websocket("/ws")
async def ws_session(websocket: WebSocket) -> None:
    await websocket.accept()
    settings = get_settings()

    async def emit(event: ServerEvent) -> None:
        try:
            await websocket.send_text(event.to_json())
        except Exception:
            pass

    session = CallSession(emit=emit, settings=settings)
    await session.start()

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"] is not None:
                await session.on_pcm16(message["bytes"])
                continue

            text = message.get("text")
            if not text:
                continue

            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                await emit(ServerEvent(type="error", data={"message": "invalid json"}))
                continue

            event_type = payload.get("type")
            data = payload.get("data") or {}

            if event_type == "audio.pcm16":
                raw = base64.b64decode(data.get("b64", ""))
                await session.on_pcm16(raw)
            elif event_type == "utterance.end":
                await session.force_end_utterance()
            elif event_type == "playback.done":
                await session.on_playback_done(
                    int(data.get("generation_id") or 0)
                )
            elif event_type == "ping":
                await emit(ServerEvent(type="pong", call_id=session.call_id))
            elif event_type == "session.close":
                break
            else:
                await emit(
                    ServerEvent(
                        type="error",
                        call_id=session.call_id,
                        data={"message": f"unknown event: {event_type}"},
                    )
                )
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()
