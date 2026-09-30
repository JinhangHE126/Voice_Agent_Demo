#!/usr/bin/env python
"""Start the voice-agent demo web server (port 7862)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    print(f"mode={settings.agent_mode} business={settings.business_mode}  →  http://{settings.host}:{settings.port}/")
    print("stack: Qwen ASR → DialogManager(repair_order) / graded LLM → MiniMax TTS")
    print("target: EoU → first audio ≤ 1000ms")
    uvicorn.run(
        "app.web:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
