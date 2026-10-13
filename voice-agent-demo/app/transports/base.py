"""Transport interface for non-browser media paths (SIP / AudioSocket).

Stage 0 only defines the contract. Stage 1+ implement resample + AudioSocket.
Browser Demo keeps using CallSession.emit + playback.done and does not need
this ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable


class AudioTransport(ABC):
    """Bidirectional PCM bridge between a media link and CallSession.

    CallSession side is always 16-bit LE mono at the session sample rate
    (currently 16 kHz). The transport may resample for the wire format
    (e.g. AudioSocket 8 kHz).
    """

    @abstractmethod
    async def start(self) -> None:
        """Open the underlying connection / server accept loop."""

    @abstractmethod
    async def close(self) -> None:
        """Tear down the connection and release resources."""

    @abstractmethod
    def on_pcm_from_caller(self) -> Callable[[bytes], Awaitable[None]]:
        """Return a callback: wire/PCM-in → CallSession.on_pcm16 (16 kHz)."""

    @abstractmethod
    def on_pcm_to_caller(self) -> Callable[[bytes], Awaitable[None]]:
        """Return a callback suitable for CallSession.on_audio_out (16 kHz)."""
