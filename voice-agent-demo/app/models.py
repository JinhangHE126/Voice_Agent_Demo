from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SessionState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    ENDPOINTING = "endpointing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    BARGE_IN = "barge_in"


class LatencyMetrics(BaseModel):
    eou_ms: float | None = None
    asr_final_ms: float | None = None
    llm_first_token_ms: float | None = None
    tts_first_audio_ms: float | None = None
    route: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump()


class ServerEvent(BaseModel):
    type: str
    call_id: str = ""
    turn_id: int = 0
    generation_id: int = 0
    data: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json()
