"""Media transports for CallSession (browser WebSocket, AudioSocket, …)."""

from app.transports.audiosocket import (
    AudioSocketConnection,
    AudioSocketServer,
    Frame,
    echo_handler,
    encode_audio_slin8,
    encode_hangup,
    encode_uuid,
)
from app.transports.audiosocket_session import (
    AudioSocketCallBridge,
    handle_audiosocket_call,
)
from app.transports.base import AudioTransport
from app.transports.resample import (
    AUDIOSOCKET_FRAME_BYTES_8K,
    RateConverter,
    make_downsampler,
    make_upsampler,
    resample_down,
    resample_up,
)
from app.transports.sip_session import (
    SipCallBridge,
    build_s1_startup_wav_8k,
    ensure_s1_playback_wav_8k,
    resolve_s1_prompt_wav,
)

__all__ = [
    "AUDIOSOCKET_FRAME_BYTES_8K",
    "AudioSocketCallBridge",
    "AudioSocketConnection",
    "AudioSocketServer",
    "AudioTransport",
    "Frame",
    "RateConverter",
    "SipCallBridge",
    "echo_handler",
    "encode_audio_slin8",
    "encode_hangup",
    "encode_uuid",
    "build_s1_startup_wav_8k",
    "ensure_s1_playback_wav_8k",
    "handle_audiosocket_call",
    "make_downsampler",
    "make_upsampler",
    "resample_down",
    "resample_up",
    "resolve_s1_prompt_wav",
]
