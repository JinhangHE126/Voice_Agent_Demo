#!/usr/bin/env python
"""S1 FreeSWITCH SIP UA (pjsua2): REGISTER + auto-answer + play greeting wav.

Prerequisites:
  - pjsua2 importable (native PJSIP binding; not in requirements.txt)
  - PBX extension (default 2000) + SIP_PASSWORD in .env
  - Network reachability to SIP_REGISTRAR:5060 and bidirectional RTP

Usage:

    cd voice-agent-demo
    # optional: copy SIP_* from .env.example into .env
    python scripts/run_sip.py --check
    python scripts/run_sip.py

Then from MicroSIP (ext 1000): dial 2000. You should hear greeting.wav.
S2 will replace wav playback with CallSession (ASR → DialogManager → TTS).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.transports.sip_pjsua import SipUaApp, ensure_pjsua2
from app.transports.sip_session import resolve_s1_prompt_wav


def _check(settings) -> int:
    print("SIP S1 configuration check")
    print(f"  user       = {settings.sip_user}")
    print(f"  domain     = {settings.sip_domain}")
    print(f"  registrar  = {settings.sip_registrar}")
    print(f"  transport  = {settings.sip_transport}")
    print(f"  local port = {settings.sip_port}")
    print(f"  codecs     = {settings.sip_codecs}")
    print(f"  s1 prompt  = {settings.sip_s1_prompt_id}")
    print(f"  s1 on eof  = {settings.sip_s1_on_eof}")
    pwd = (settings.sip_password or "").strip()
    print(f"  password   = {'(set)' if pwd else '(EMPTY — REGISTER will fail)'}")

    try:
        wav = resolve_s1_prompt_wav(settings)
        print(f"  wav        = {wav}  OK")
    except FileNotFoundError as exc:
        print(f"  wav        = MISSING: {exc}")
        return 1

    try:
        ensure_pjsua2()
        print("  pjsua2     = import OK")
    except RuntimeError as exc:
        print(f"  pjsua2     = MISSING\n{exc}")
        return 2

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Agent SIP UA (S1)")
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
    if code == 1:
        raise SystemExit(code)
    if code == 2:
        raise SystemExit(code)

    app = SipUaApp(settings)
    app.start()
    if args.wait_reg > 0:
        ok = app.wait_registered(timeout_s=args.wait_reg)
        st = app.status()
        if ok:
            print(
                f"REGISTERED status={st.reg_status} {st.reg_status_text!r} "
                f"— dial {settings.sip_user} from Softphone to test S1 audio"
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
            app.handle_events(100)
    except KeyboardInterrupt:
        print("stopping…")
    finally:
        app.stop()


if __name__ == "__main__":
    main()
