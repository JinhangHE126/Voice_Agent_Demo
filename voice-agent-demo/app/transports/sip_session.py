"""SIP ↔ CallSession glue (S2+).

S1 (`sip_pjsua.py`) only REGISTER + Answer + play a fixed local wav via pjsua2.
S2 will mirror `AudioSocketCallBridge`:

```text
RTP (G.711 @ 8k via pjsua2)
  ↔ PCM16 @ 16k → CallSession.on_pcm16
CallSession.on_audio_out (16k) → feed pjsua2 / RTP
auto_complete_playback=True  (no browser playback.done)
```

This module holds shared path helpers and a stub bridge so the S2 wiring site
is obvious without pulling CallSession into S1.
"""

from __future__ import annotations

from pathlib import Path

from app.config import Settings, get_settings
from app.services.prompt_audio import PromptAudioBank


def resolve_s1_prompt_wav(settings: Settings | None = None) -> Path:
    """Resolve ``SIP_S1_PROMPT_ID`` under the prompt audio directory."""
    cfg = settings or get_settings()
    prompt_id = (cfg.sip_s1_prompt_id or "greeting").strip() or "greeting"
    root = cfg.resolved_prompt_audio_dir()
    bank = PromptAudioBank(root, sample_rate=cfg.sample_rate)
    path = bank.resolve(prompt_id, lang="yue")
    if path is None:
        raise FileNotFoundError(
            f"S1 prompt wav not found for id={prompt_id!r} under {root} "
            f"(expected {root / 'yue' / (prompt_id + '.wav')} or "
            f"{root / (prompt_id + '.wav')})"
        )
    return path.resolve()


class SipCallBridge:
    """Placeholder for S2: one SIP call bound to one CallSession.

    S1 does not instantiate this; keep the name stable for the next stage.
    """

    def __init__(self, *, call_id: str, settings: Settings | None = None) -> None:
        self.call_id = call_id
        self.settings = settings or get_settings()

    async def run(self) -> None:
        raise NotImplementedError(
            "SipCallBridge is S2 — wire RTP PCM ↔ CallSession "
            "(see AudioSocketCallBridge)."
        )
