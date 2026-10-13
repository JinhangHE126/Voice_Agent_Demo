from __future__ import annotations

import asyncio
import uuid

import pytest

from app.transports.audiosocket import (
    KIND_AUDIO_SLIN,
    KIND_DTMF,
    KIND_ERROR,
    KIND_HANGUP,
    KIND_UUID,
    AudioSocketServer,
    chunk_pcm,
    decode_header,
    echo_handler,
    encode_audio_slin8,
    encode_dtmf,
    encode_error,
    encode_frame,
    encode_hangup,
    encode_uuid,
    frames_from_bytes,
    parse_uuid_payload,
    read_frame,
)
from app.transports.resample import AUDIOSOCKET_FRAME_BYTES_8K


def test_encode_hangup_is_three_zero_bytes() -> None:
    assert encode_hangup() == b"\x00\x00\x00"


def test_encode_decode_uuid() -> None:
    u = uuid.UUID("40325ec2-5efd-4bd3-805f-53576e581d13")
    raw = encode_uuid(u)
    assert raw[0] == KIND_UUID
    kind, length = decode_header(raw[:3])
    assert kind == KIND_UUID
    assert length == 16
    assert parse_uuid_payload(raw[3:]) == str(u)


def test_encode_audio_and_dtmf_and_error() -> None:
    pcm = b"\x01\x00" * 160
    audio = encode_audio_slin8(pcm)
    assert audio[0] == KIND_AUDIO_SLIN
    assert int.from_bytes(audio[1:3], "big") == len(pcm)
    assert audio[3:] == pcm

    dtmf = encode_dtmf("1")
    assert dtmf == bytes([KIND_DTMF, 0x00, 0x01, ord("1")])

    err = encode_error(0x01)
    assert err == bytes([KIND_ERROR, 0x00, 0x01, 0x01])


def test_chunk_pcm_pads_last_frame() -> None:
    pcm = b"\x11\x22" * 10  # 20 bytes
    frames = chunk_pcm(pcm, frame_bytes=320)
    assert len(frames) == 1
    assert len(frames[0]) == 320
    assert frames[0][:20] == pcm
    assert frames[0][20:] == b"\x00" * 300


def test_frames_from_bytes_roundtrip() -> None:
    u = uuid.uuid4()
    blob = (
        encode_uuid(u)
        + encode_audio_slin8(b"\x00\x10" * 160)
        + encode_hangup()
    )
    frames = frames_from_bytes(blob)
    assert [f.kind for f in frames] == [KIND_UUID, KIND_AUDIO_SLIN, KIND_HANGUP]
    assert parse_uuid_payload(frames[0].payload) == str(u)
    assert len(frames[1].payload) == AUDIOSOCKET_FRAME_BYTES_8K


@pytest.mark.asyncio
async def test_read_frame_from_stream() -> None:
    reader = asyncio.StreamReader()
    pcm = b"\x00\x20" * 160
    reader.feed_data(encode_audio_slin8(pcm))
    reader.feed_eof()
    frame = await read_frame(reader)
    assert frame.kind == KIND_AUDIO_SLIN
    assert frame.payload == pcm


@pytest.mark.asyncio
async def test_echo_server_loopback() -> None:
    server = AudioSocketServer("127.0.0.1", 0, echo_handler)
    await server.start()
    port = server.bound_port
    call_id = uuid.uuid4()
    pcm = b"\x00\x30" * 160

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(encode_uuid(call_id))
        writer.write(encode_audio_slin8(pcm))
        writer.write(encode_audio_slin8(pcm))
        await writer.drain()

        echo1 = await read_frame(reader)
        echo2 = await read_frame(reader)
        assert echo1.kind == KIND_AUDIO_SLIN
        assert echo1.payload == pcm
        assert echo2.payload == pcm

        writer.write(encode_hangup())
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_connection_captures_uuid_and_send_audio_frames() -> None:
    got_uuid = asyncio.Event()
    captured: dict[str, str | None] = {"uuid": None}

    async def handler(conn) -> None:
        async for frame in conn.read_frames():
            if frame.is_uuid:
                captured["uuid"] = conn.call_uuid
                got_uuid.set()
            if frame.is_audio:
                await conn.send_audio_frames(frame.payload, pace=False)
            if frame.is_hangup:
                return

    server = AudioSocketServer("127.0.0.1", 0, handler)
    await server.start()
    call_id = uuid.UUID("11111111-2222-3333-4444-555555555555")
    # 1.5 frames worth → two padded 320-byte outbound frames
    pcm = b"\x01\x00" * 240  # 480 bytes

    try:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", server.bound_port
        )
        writer.write(encode_uuid(call_id))
        writer.write(encode_audio_slin8(pcm))
        await writer.drain()

        await asyncio.wait_for(got_uuid.wait(), timeout=1.0)
        assert captured["uuid"] == str(call_id)

        out1 = await asyncio.wait_for(read_frame(reader), timeout=1.0)
        out2 = await asyncio.wait_for(read_frame(reader), timeout=1.0)
        assert out1.kind == KIND_AUDIO_SLIN
        assert len(out1.payload) == AUDIOSOCKET_FRAME_BYTES_8K
        assert len(out2.payload) == AUDIOSOCKET_FRAME_BYTES_8K
        assert out1.payload.startswith(pcm[:320])
        assert out2.payload.startswith(pcm[320:])

        writer.write(encode_hangup())
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_peer_close_without_hangup_ends_cleanly() -> None:
    finished = asyncio.Event()

    async def handler(conn) -> None:
        async for _frame in conn.read_frames():
            pass
        finished.set()

    server = AudioSocketServer("127.0.0.1", 0, handler)
    await server.start()
    try:
        _reader, writer = await asyncio.open_connection(
            "127.0.0.1", server.bound_port
        )
        writer.write(encode_uuid(uuid.uuid4()))
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        await asyncio.wait_for(finished.wait(), timeout=1.0)
    finally:
        await server.close()
