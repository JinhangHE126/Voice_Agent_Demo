"""Asterisk AudioSocket protocol codec + asyncio TCP server.

Asterisk dialplan ``AudioSocket(uuid, host:port)`` connects *out* to this
server. Default audio is kind ``0x10`` (8 kHz slin, typically 320 bytes / 20 ms).

Stage 2: protocol + accept loop only. Stage 3 wires CallSession + resample.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass

from app.transports.resample import AUDIOSOCKET_FRAME_BYTES_8K

logger = logging.getLogger(__name__)

KIND_HANGUP = 0x00
KIND_UUID = 0x01
KIND_DTMF = 0x03
KIND_AUDIO_SLIN = 0x10  # 8 kHz
KIND_AUDIO_SLIN12 = 0x11
KIND_AUDIO_SLIN16 = 0x12
KIND_AUDIO_SLIN24 = 0x13
KIND_AUDIO_SLIN32 = 0x14
KIND_AUDIO_SLIN44 = 0x15
KIND_AUDIO_SLIN48 = 0x16
KIND_AUDIO_SLIN96 = 0x17
KIND_AUDIO_SLIN192 = 0x18
KIND_ERROR = 0xFF

HEADER_SIZE = 3
UUID_PAYLOAD_BYTES = 16

AUDIO_KINDS = frozenset(
    {
        KIND_AUDIO_SLIN,
        KIND_AUDIO_SLIN12,
        KIND_AUDIO_SLIN16,
        KIND_AUDIO_SLIN24,
        KIND_AUDIO_SLIN32,
        KIND_AUDIO_SLIN44,
        KIND_AUDIO_SLIN48,
        KIND_AUDIO_SLIN96,
        KIND_AUDIO_SLIN192,
    }
)

ConnectionHandler = Callable[["AudioSocketConnection"], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class Frame:
    kind: int
    payload: bytes

    @property
    def is_hangup(self) -> bool:
        return self.kind == KIND_HANGUP

    @property
    def is_audio(self) -> bool:
        return self.kind in AUDIO_KINDS

    @property
    def is_uuid(self) -> bool:
        return self.kind == KIND_UUID

    @property
    def is_dtmf(self) -> bool:
        return self.kind == KIND_DTMF

    @property
    def is_error(self) -> bool:
        return self.kind == KIND_ERROR


def encode_frame(kind: int, payload: bytes = b"") -> bytes:
    if not (0 <= int(kind) <= 0xFF):
        raise ValueError(f"invalid AudioSocket kind: {kind!r}")
    if len(payload) > 0xFFFF:
        raise ValueError(
            f"AudioSocket payload too large: {len(payload)} > 65535"
        )
    return bytes([int(kind) & 0xFF]) + struct.pack(">H", len(payload)) + payload


def encode_hangup() -> bytes:
    return encode_frame(KIND_HANGUP, b"")


def encode_uuid(call_uuid: uuid.UUID | str) -> bytes:
    u = call_uuid if isinstance(call_uuid, uuid.UUID) else uuid.UUID(str(call_uuid))
    return encode_frame(KIND_UUID, u.bytes)


def encode_audio_slin8(pcm: bytes) -> bytes:
    """Encode 8 kHz PCM16 LE mono (kind 0x10)."""
    return encode_frame(KIND_AUDIO_SLIN, pcm)


def encode_dtmf(digit: str) -> bytes:
    if len(digit) != 1:
        raise ValueError("DTMF payload must be a single ASCII character")
    return encode_frame(KIND_DTMF, digit.encode("ascii"))


def encode_error(code: int | None = None) -> bytes:
    if code is None:
        return encode_frame(KIND_ERROR, b"")
    if not (0 <= int(code) <= 0xFF):
        raise ValueError(f"invalid error code: {code!r}")
    return encode_frame(KIND_ERROR, bytes([int(code) & 0xFF]))


def decode_header(header: bytes) -> tuple[int, int]:
    if len(header) != HEADER_SIZE:
        raise ValueError(f"header must be {HEADER_SIZE} bytes, got {len(header)}")
    kind = header[0]
    length = struct.unpack(">H", header[1:3])[0]
    return kind, length


def parse_uuid_payload(payload: bytes) -> str:
    if len(payload) != UUID_PAYLOAD_BYTES:
        raise ValueError(
            f"UUID payload must be {UUID_PAYLOAD_BYTES} bytes, got {len(payload)}"
        )
    return str(uuid.UUID(bytes=payload))


def chunk_pcm(
    pcm: bytes,
    frame_bytes: int = AUDIOSOCKET_FRAME_BYTES_8K,
) -> list[bytes]:
    """Split PCM into fixed-size frames (pad the last frame with zeros)."""
    if frame_bytes <= 0:
        raise ValueError("frame_bytes must be positive")
    if not pcm:
        return []
    # Keep sample alignment for PCM16.
    aligned = pcm[: len(pcm) - (len(pcm) % 2)]
    if not aligned:
        return []
    frames: list[bytes] = []
    for offset in range(0, len(aligned), frame_bytes):
        chunk = aligned[offset : offset + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
        frames.append(chunk)
    return frames


async def read_frame(reader: asyncio.StreamReader) -> Frame:
    header = await reader.readexactly(HEADER_SIZE)
    kind, length = decode_header(header)
    payload = await reader.readexactly(length) if length else b""
    return Frame(kind=kind, payload=payload)


class AudioSocketConnection:
    """One accepted TCP connection from Asterisk (or a test client)."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        peer: str | None = None,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.peer = peer or _peer_name(writer)
        self.call_uuid: str | None = None
        self._closed = False
        self._write_lock = asyncio.Lock()

    async def read_frames(self) -> AsyncIterator[Frame]:
        try:
            while not self._closed:
                try:
                    frame = await read_frame(self.reader)
                except asyncio.IncompleteReadError:
                    logger.info(
                        "AudioSocket peer closed without hangup peer=%s uuid=%s",
                        self.peer,
                        self.call_uuid or "-",
                    )
                    return
                except (ConnectionError, OSError) as exc:
                    logger.warning(
                        "AudioSocket read failed peer=%s uuid=%s: %r",
                        self.peer,
                        self.call_uuid or "-",
                        exc,
                    )
                    return

                if frame.is_uuid:
                    try:
                        self.call_uuid = parse_uuid_payload(frame.payload)
                    except ValueError as exc:
                        logger.warning(
                            "AudioSocket invalid UUID peer=%s: %s",
                            self.peer,
                            exc,
                        )
                        self.call_uuid = frame.payload.hex()

                yield frame

                if frame.is_hangup:
                    return
        except asyncio.CancelledError:
            raise

    async def send_raw(self, data: bytes) -> None:
        if self._closed or not data:
            return
        async with self._write_lock:
            if self._closed:
                return
            try:
                self.writer.write(data)
                await self.writer.drain()
            except (ConnectionError, OSError) as exc:
                logger.warning(
                    "AudioSocket write failed peer=%s uuid=%s: %r",
                    self.peer,
                    self.call_uuid or "-",
                    exc,
                )
                await self.close()

    async def send_frame(self, kind: int, payload: bytes = b"") -> None:
        await self.send_raw(encode_frame(kind, payload))

    async def send_audio(self, pcm8k: bytes) -> None:
        """Send one 8 kHz PCM payload as kind 0x10 (may be any length)."""
        if pcm8k:
            await self.send_frame(KIND_AUDIO_SLIN, pcm8k)

    async def send_audio_frames(
        self,
        pcm8k: bytes,
        *,
        frame_bytes: int = AUDIOSOCKET_FRAME_BYTES_8K,
        pace: bool = False,
        frame_ms: int = 20,
    ) -> None:
        """Chunk PCM into AudioSocket audio frames; optionally pace in realtime."""
        frames = chunk_pcm(pcm8k, frame_bytes=frame_bytes)
        for i, chunk in enumerate(frames):
            if self._closed:
                return
            await self.send_audio(chunk)
            if pace and i + 1 < len(frames):
                await asyncio.sleep(max(0.0, frame_ms / 1000.0))

    async def hangup(self) -> None:
        if not self._closed:
            await self.send_raw(encode_hangup())
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception:
            pass


class AudioSocketServer:
    """TCP server that Asterisk (or tests) dial into."""

    def __init__(
        self,
        host: str,
        port: int,
        handler: ConnectionHandler,
        *,
        backlog: int = 100,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.handler = handler
        self.backlog = backlog
        self._server: asyncio.Server | None = None

    @property
    def sockets(self) -> tuple:
        if self._server is None:
            return ()
        return tuple(self._server.sockets or ())

    @property
    def bound_port(self) -> int:
        socks = self.sockets
        if not socks:
            raise RuntimeError("AudioSocket server is not started")
        return int(socks[0].getsockname()[1])

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await asyncio.start_server(
            self._on_client,
            host=self.host,
            port=self.port,
            backlog=self.backlog,
        )
        logger.info(
            "AudioSocket server listening on %s:%s",
            self.host,
            self.bound_port,
        )

    async def serve_forever(self) -> None:
        await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        server = self._server
        self._server = None
        if server is None:
            return
        server.close()
        await server.wait_closed()

    async def _on_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        conn = AudioSocketConnection(reader, writer)
        logger.info("AudioSocket accept peer=%s", conn.peer)
        try:
            await self.handler(conn)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "AudioSocket handler failed peer=%s uuid=%s",
                conn.peer,
                conn.call_uuid or "-",
            )
        finally:
            await conn.close()
            logger.info(
                "AudioSocket closed peer=%s uuid=%s",
                conn.peer,
                conn.call_uuid or "-",
            )


async def echo_handler(conn: AudioSocketConnection) -> None:
    """Stage-2 loopback: echo every 0x10 audio frame until hangup."""
    async for frame in conn.read_frames():
        if frame.is_audio and frame.kind == KIND_AUDIO_SLIN:
            await conn.send_audio(frame.payload)
        elif frame.is_hangup:
            return


def _peer_name(writer: asyncio.StreamWriter) -> str:
    try:
        addr = writer.get_extra_info("peername")
        if isinstance(addr, tuple) and len(addr) >= 2:
            return f"{addr[0]}:{addr[1]}"
        return str(addr or "-")
    except Exception:
        return "-"


def frames_from_bytes(data: bytes) -> list[Frame]:
    """Parse a contiguous buffer of frames (tests / offline tools)."""
    frames: list[Frame] = []
    offset = 0
    while offset + HEADER_SIZE <= len(data):
        kind, length = decode_header(data[offset : offset + HEADER_SIZE])
        offset += HEADER_SIZE
        end = offset + length
        if end > len(data):
            raise ValueError("truncated AudioSocket frame payload")
        frames.append(Frame(kind=kind, payload=data[offset:end]))
        offset = end
    if offset != len(data):
        raise ValueError("trailing bytes after AudioSocket frames")
    return frames


def encode_frames(frames: Iterable[Frame]) -> bytes:
    return b"".join(encode_frame(f.kind, f.payload) for f in frames)
