#!/usr/bin/env python
"""Start the AudioSocket TCP bridge for Asterisk (default port 9092).

Asterisk dialplan example (8 kHz default application):

    exten => _X.,1,Answer()
     same => n,Set(UUID=${UNIQUEID})
     same => n,AudioSocket(${UUID},127.0.0.1:9092)
     same => n,Hangup()

Use AGENT_MODE=mock first, then cloud after the wiring works.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.transports.audiosocket import AudioSocketServer
from app.transports.audiosocket_session import handle_audiosocket_call


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    host = settings.audiosocket_host
    port = settings.audiosocket_port
    print(
        f"AudioSocket bridge  mode={settings.agent_mode} "
        f"business={settings.business_mode}  →  tcp://{host}:{port}"
    )
    print("stack: AudioSocket 8k ↔ resample 16k → CallSession → TTS → 8k")
    print("Asterisk should connect OUT to this host:port (AudioSocket app).")

    async def handler(conn):
        await handle_audiosocket_call(conn, settings)

    async def run() -> None:
        server = AudioSocketServer(host, port, handler)
        try:
            await server.serve_forever()
        except KeyboardInterrupt:
            await server.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
