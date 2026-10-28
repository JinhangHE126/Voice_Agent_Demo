#!/usr/bin/env python
"""FreeSWITCH SIP UA (pjsua2): REGISTER + auto-answer + AI session (S2).

Prerequisites:
  - pjsua2 importable (native PJSIP binding; not in requirements.txt)
  - PBX extension + SIP_PASSWORD in .env
  - Network reachability to SIP_REGISTRAR:5060 and bidirectional RTP

Usage:

    cd voice-agent-demo
    python scripts/run_sip.py --check
    python scripts/run_sip.py

SIP_MEDIA_MODE=session (default): RTP ↔ CallSession (ASR → Dialog → TTS)
SIP_MEDIA_MODE=wav: S1 file player only (no listen / no reply)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.transports.sip_pjsua import SipUaApp, ensure_pjsua2
from app.transports.sip_session import (
    build_s1_startup_wav_8k,
    resolve_s1_prompt_wav,
)


def _check(settings) -> int:
    mode = (settings.sip_media_mode or "session").strip().lower()
    print("SIP UA configuration check")
    print(f"  media mode = {mode}")
    print(f"  user       = {settings.sip_user}")
    print(f"  domain     = {settings.sip_domain}")
    print(f"  registrar  = {settings.sip_registrar}")
    print(f"  transport  = {settings.sip_transport}")
    print(f"  local port = {settings.sip_port}")
    public = (settings.sip_public_addr or "").strip()
    print(f"  public addr= {public or '(auto / rport rewrite)'}")
    print(f"  codecs     = {settings.sip_codecs}")
    print(f"  agent mode = {settings.agent_mode}")
    pwd = (settings.sip_password or "").strip()
    print(f"  password   = {'(set)' if pwd else '(EMPTY — REGISTER will fail)'}")

    try:
        src = resolve_s1_prompt_wav(settings)
        wav = build_s1_startup_wav_8k(settings)
        print(f"  wav src    = {src} (+ ask_name)")
        print(f"  wav 8k     = {wav}  OK")
    except (FileNotFoundError, ValueError) as exc:
        print(f"  wav        = MISSING/BAD: {exc}")
        return 1

    try:
        ensure_pjsua2()
        print("  pjsua2     = import OK")
    except RuntimeError as exc:
        print(f"  pjsua2     = MISSING\n{exc}")
        return 2

    if mode == "session":
        print("  note       = S2 session: will listen + reply via CallSession")
    else:
        print("  note       = S1 wav: play only (no ASR / no reply)")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Agent SIP UA")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate config + wav + pjsua2 import, then exit",
    )
    parser.add_argument(
        "--wait-reg",
        type=float,
        default=15.0,
        help="Seconds to wait for REGISTER success after start (0=skip)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()

    if args.check:
        raise SystemExit(_check(settings))

    code = _check(settings)
    if code != 0:
        raise SystemExit(code)

    app = SipUaApp(settings)
    app.start()
    if args.wait_reg > 0:
        ok = app.wait_registered(timeout_s=args.wait_reg)
        st = app.status()
        if ok:
            print(
                f"REGISTERED status={st.reg_status} {st.reg_status_text!r} "
                f"— dial {settings.sip_user} from Softphone"
            )
        else:
            print(
                f"REGISTER not confirmed within {args.wait_reg}s "
                f"(status={st.reg_status} {st.reg_status_text!r}). "
                "Process keeps running; check password / firewall / domain."
            )
    print("SIP UA running — Ctrl+C to stop")
    try:
        while True:
            # pjsua2's SWIG call may hold the GIL while polling. Long/tight
            # 100 ms polls starve the asyncio thread that builds CallSession.
            app.handle_events(10)
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("stopping…")
    finally:
        app.stop()


if __name__ == "__main__":
    main()
