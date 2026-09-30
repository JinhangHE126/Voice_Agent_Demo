from __future__ import annotations

import asyncio
import base64
import os
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from app.config import Settings, get_settings
from app.models import LatencyMetrics, ServerEvent, SessionState
from app.services.asr_qwen import MockStreamingASR, QwenRealtimeASR
from app.services.llm_router import GradedLLMRouter
from app.services.prompt_audio import PromptAudioBank
from app.services.text_norm import detect_reply_lang
from app.services.tts_minimax import MiniMaxStreamingTTS
from app.vad import EnergyVAD

# Import sibling package without colliding with this project's `app`.
_REPAIR_ROOT = Path(__file__).resolve().parents[2] / "repair-order-agent"
if str(_REPAIR_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPAIR_ROOT))

from repair_order.domain.repair_order import RepairOrderDraft  # noqa: E402
from repair_order.repositories.order_repository import OrderRepository  # noqa: E402
from repair_order.services.dialog_manager import DialogManager  # noqa: E402

EmitFn = Callable[[ServerEvent], Awaitable[None]]


def _b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


class CallSession:
    """
    Qwen ASR → repair-order DialogManager (or graded LLM) → MiniMax TTS.

    Target: EoU → first playable audio ≤ 1000ms (cloud path; mock proves wiring).
    """

    def __init__(
        self,
        emit: EmitFn,
        settings: Settings | None = None,
        call_id: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.call_id = call_id or uuid.uuid4().hex[:12]
        self.emit = emit

        self.state = SessionState.IDLE
        self.turn_id = 0
        self.generation_id = 0

        self.vad = EnergyVAD(
            sample_rate=self.settings.sample_rate,
            frame_ms=self.settings.frame_ms,
            silence_ms=self.settings.vad_silence_ms,
            min_speech_ms=self.settings.min_speech_ms,
            energy_threshold=self.settings.vad_energy_threshold,
            webrtc_mode=self.settings.vad_webrtc_mode,
            noise_multiplier=self.settings.vad_noise_multiplier,
            max_speech_ms=self.settings.vad_max_speech_ms,
            window_ms=self.settings.vad_window_ms,
            window_min_voiced_ratio=self.settings.vad_window_min_voiced_ratio,
        )

        self.history: list[dict[str, str]] = []
        self._asr = None
        self._cancel = asyncio.Event()
        self._reply_task: asyncio.Task | None = None
        self._turn_clock: float | None = None
        self._latency = LatencyMetrics()
        self._partial_text = ""
        self._speaking_started_at: float | None = None
        self._closed = False
        self._reply_lang = "yue"
        self._asr_lang = self.settings.dashscope_asr_language
        self._first_audio_marked = False
        self._handling_final = False
        self._awaiting_playback_done = False
        self._barge_buffer = bytearray()
        self._barge_started_at: float | None = None
        self._last_tts_audio_at: float | None = None
        self._greeting_uninterruptible = False

        self.business_mode = (
            self.settings.business_mode.strip().lower() or "repair_order"
        )
        self.draft = RepairOrderDraft()
        self.order_repo = OrderRepository(
            sqlite_path=self.settings.repair_order_sqlite_path
            if self.business_mode == "repair_order"
            else None
        )
        self.dialog = DialogManager(repository=self.order_repo)
        os.environ["REPAIR_ORDER_USE_LLM_EXTRACTOR"] = (
            "1" if self.settings.repair_order_use_llm_extractor else "0"
        )
        os.environ["REPAIR_ORDER_EXTRACT_TIMEOUT_S"] = str(
            self.settings.repair_order_extract_timeout_s
        )
        self.prompts = PromptAudioBank(
            self.settings.resolved_prompt_audio_dir(),
            sample_rate=self.settings.sample_rate,
        )
        self._filler_rot = 0

        mock = not self.settings.use_cloud
        self.llm = GradedLLMRouter(
            faq_path=self.settings.faq_path,
            system_prompt=self.settings.llm_system_prompt,
            fast_base_url=self.settings.fast_llm_base_url,
            fast_api_key=self.settings.resolved_fast_llm_key(),
            fast_model=self.settings.fast_llm_model,
            fast_max_tokens=self.settings.fast_llm_max_tokens,
            fast_temperature=self.settings.fast_llm_temperature,
            fast_timeout_s=self.settings.fast_llm_timeout_s,
            bonsai_base_url=self.settings.bonsai_base_url,
            bonsai_api_key=self.settings.bonsai_api_key,
            bonsai_model=self.settings.bonsai_model,
            bonsai_max_tokens=self.settings.bonsai_max_tokens,
            bonsai_temperature=self.settings.bonsai_temperature,
            bonsai_disable_thinking=self.settings.bonsai_disable_thinking,
            mock=mock or not self.settings.has_fast_llm,
        )
        self.tts = MiniMaxStreamingTTS(
            api_key=self.settings.minimax_api_key,
            model=self.settings.minimax_tts_model,
            voice=self.settings.minimax_tts_voice,
            language_boost=self.settings.minimax_language_boost,
            ws_url=self.settings.minimax_ws_url,
            sample_rate=self.settings.minimax_sample_rate,
            audio_format=self.settings.minimax_audio_format,
            mock=mock or not self.settings.has_minimax_key,
        )

    def _prompt_lang(self) -> str:
        lang = (self.draft.session_language or "").strip().lower()
        if lang in {"cmn", "zh", "zh-cn", "zh-hans"}:
            return "zh"
        if lang == "en":
            return "en"
        return "yue"

    def _reply_lang_for_tts(self) -> str:
        lang = (self.draft.session_language or "").strip().lower()
        if lang in {"cmn", "zh", "zh-cn", "zh-hans"}:
            return "zh"
        if lang == "en":
            return "en"
        return "yue"

    async def start(self) -> None:
        self.state = SessionState.LISTENING
        await self._emit(
            "session.started",
            {
                "mode": self.settings.agent_mode,
                "business_mode": self.business_mode,
                "asr": (
                    "qwen"
                    if self.settings.use_cloud and self.settings.has_asr_key
                    else "mock"
                ),
                "tts": (
                    "minimax"
                    if self.settings.use_cloud and self.settings.has_minimax_key
                    else "mock"
                ),
                "llm": (
                    "repair_order"
                    if self.business_mode == "repair_order"
                    else ("graded" if self.settings.use_cloud else "mock")
                ),
                "vad": self.vad.backend,
                "vad_max_speech_ms": self.settings.vad_max_speech_ms,
                "vad_window_ms": self.settings.vad_window_ms,
                "vad_window_min_voiced_ratio": self.settings.vad_window_min_voiced_ratio,
                "barge_in_min_ms": self.settings.barge_in_min_ms,
                "barge_in_thinking_min_ms": self.settings.barge_in_thinking_min_ms,
                "barge_in_energy_ratio": self.settings.barge_in_energy_ratio,
                "playback_done_grace_ms": self.settings.playback_done_grace_ms,
                "prompt_audio_dir": str(self.settings.resolved_prompt_audio_dir()),
                "has_filler_audio": self.prompts.has_any_filler(
                    lang=self._prompt_lang()
                ),
            },
        )
        await self._reset_asr()
        if self.business_mode != "repair_order":
            asyncio.create_task(self.llm.warm())
        asyncio.create_task(self.tts.warm(self._reply_lang))
        if self.business_mode == "repair_order":
            # From session start until greeting playback completes, ignore
            # microphone input so callers cannot preempt the first line.
            self._greeting_uninterruptible = True
            asyncio.create_task(self._speak_greeting())

    async def _speak_greeting(self) -> None:
        await asyncio.sleep(0.15)
        if self._closed:
            self._greeting_uninterruptible = False
            return
        plan = self.dialog.greeting(self.draft)
        self.turn_id += 1
        self.generation_id += 1
        gen = self.generation_id
        self._turn_clock = time.perf_counter()
        self._latency = LatencyMetrics(eou_ms=0.0)
        self._first_audio_marked = False
        self._awaiting_playback_done = False
        self._cancel = asyncio.Event()
        self.state = SessionState.THINKING
        await self._emit("state", {"state": self.state.value})
        await self._emit(
            "order.draft",
            {
                "customer_name": self.draft.customer_name,
                "customer_phone": self.draft.customer_phone,
                "customer_address": self.draft.customer_address,
                "repair_description": self.draft.repair_description,
                "state": self.draft.state.value,
                "awaiting_field": self.draft.awaiting_field,
                "session_language": self.draft.session_language,
            },
        )
        self._reply_task = asyncio.create_task(
            self._speak_text(
                plan.text,
                generation_id=gen,
                route="repair_greeting",
                prompt_id=plan.prompt_id,
            )
        )

    async def close(self) -> None:
        self._closed = True
        self._greeting_uninterruptible = False
        await self._cancel_generation()
        if self._asr:
            await self._asr.close()

    async def on_pcm16(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        if self._greeting_uninterruptible:
            return

        if self.state in {SessionState.THINKING, SessionState.SPEAKING}:
            # During THINKING, avoid false barge-in from room noise or speech
            # tail, otherwise the pending reply gets canceled repeatedly.
            if self.state == SessionState.THINKING:
                return
            self._barge_buffer.extend(pcm)
            max_bytes = self.settings.sample_rate * 2
            if len(self._barge_buffer) > max_bytes:
                del self._barge_buffer[:-max_bytes]
            ev = self.vad.accept(pcm)
            now = time.perf_counter()
            if ev["event"] == "speech_start":
                self._barge_started_at = now
            elif ev["event"] not in {"speech_continue"}:
                self._barge_started_at = None

            if self._barge_started_at is not None:
                candidate_ms = (now - self._barge_started_at) * 1000.0
                min_ms = float(self.settings.barge_in_min_ms)
                speaking_for = 0.0
                if self._speaking_started_at is not None:
                    speaking_for = (
                        time.perf_counter() - self._speaking_started_at
                    ) * 1000
                energy = float(ev.get("energy") or 0.0)
                threshold = float(ev.get("threshold") or 0.0)
                energy_ok = (
                    threshold <= 0.0
                    or energy >= threshold * float(self.settings.barge_in_energy_ratio)
                )
                if (
                    candidate_ms >= min_ms
                    and speaking_for >= min_ms
                    and energy_ok
                ):
                    buffered = bytes(self._barge_buffer)
                    await self._handle_barge_in()
                    if self._asr and buffered:
                        await self._asr.push_pcm16(buffered)
            return

        if self.state not in {SessionState.LISTENING}:
            return

        if self._asr:
            await self._asr.push_pcm16(pcm)

        ev = self.vad.accept(pcm)
        if ev["event"] == "speech_start":
            await self._emit(
                "speech.started",
                {"energy": round(float(ev.get("energy") or 0), 1)},
            )
        elif ev["event"] == "speech_end":
            if self._turn_clock is None:
                self._turn_clock = time.perf_counter()
                self._latency = LatencyMetrics(eou_ms=0.0)
            self.state = SessionState.ENDPOINTING
            await self._emit(
                "speech.ended",
                {
                    "speech_ms": ev.get("speech_ms"),
                    "reason": ev.get("reason"),
                    "voiced_ratio": ev.get("voiced_ratio"),
                },
            )
            await self._emit("state", {"state": self.state.value})
            if self._asr:
                asyncio.create_task(self._asr.end_utterance())

    async def force_end_utterance(self) -> None:
        if self.state != SessionState.LISTENING:
            return
        if self._turn_clock is None:
            self._turn_clock = time.perf_counter()
            self._latency = LatencyMetrics(eou_ms=0.0)
        self.state = SessionState.ENDPOINTING
        await self._emit("speech.ended", {"manual": True})
        await self._emit("state", {"state": self.state.value})
        if self._asr:
            asyncio.create_task(self._asr.end_utterance())

    async def _reset_asr(self) -> None:
        if self._asr:
            await self._asr.close()

        async def on_partial(text: str) -> None:
            self._partial_text = text
            await self._emit("asr.partial", {"text": text})

        async def on_final(text: str, lang: str = "yue") -> None:
            await self._on_asr_final(text, lang=lang)

        use_qwen = self.settings.use_cloud and self.settings.has_asr_key
        if use_qwen:
            self._asr = QwenRealtimeASR(
                api_key=self.settings.dashscope_api_key,
                model=self.settings.dashscope_asr_model,
                language=self.settings.dashscope_asr_language,
                ws_url=self.settings.dashscope_asr_ws_url,
                silence_ms=self.settings.dashscope_asr_silence_ms,
                final_timeout_ms=self.settings.dashscope_asr_final_timeout_ms,
                on_partial=on_partial,
                on_final=on_final,
            )
        else:
            self._asr = MockStreamingASR(
                on_partial=on_partial,
                on_final=on_final,
                language=self.settings.dashscope_asr_language,
            )
        await self._asr.start()

    async def _on_asr_final(self, text: str, *, lang: str = "yue") -> None:
        text = (text or "").strip()
        if self._handling_final:
            return
        if not text:
            self.state = SessionState.LISTENING
            await self._emit("asr.final", {"text": "", "empty": True})
            await self._emit("state", {"state": self.state.value})
            self.vad.reset()
            return

        if self.state in {SessionState.THINKING, SessionState.SPEAKING}:
            await self._handle_barge_in()

        self._handling_final = True
        try:
            if self._turn_clock is None:
                self._turn_clock = time.perf_counter()
                self._latency = LatencyMetrics(eou_ms=0.0)

            self._latency.asr_final_ms = round(
                (time.perf_counter() - self._turn_clock) * 1000, 1
            )
            self._asr_lang = lang or self._asr_lang
            self._reply_lang = detect_reply_lang(text, self._asr_lang)
            self._first_audio_marked = False
            self._awaiting_playback_done = False
            self._barge_buffer.clear()

            self.turn_id += 1
            self.generation_id += 1
            gen = self.generation_id
            await self._emit("asr.final", {"text": text, "lang": self._asr_lang})
            self.state = SessionState.THINKING
            await self._emit("state", {"state": self.state.value})

            self._cancel = asyncio.Event()
            self._reply_task = asyncio.create_task(
                self._run_reply(text, generation_id=gen)
            )
        finally:
            self._handling_final = False

    async def _mark_first_audio(self, *, generation_id: int) -> None:
        if generation_id != self.generation_id or self._first_audio_marked:
            return
        self._first_audio_marked = True
        self._awaiting_playback_done = True
        if self._turn_clock is not None:
            self._latency.tts_first_audio_ms = round(
                (time.perf_counter() - self._turn_clock) * 1000, 1
            )
        self.state = SessionState.SPEAKING
        self._speaking_started_at = time.perf_counter()
        await self._emit(
            "tts.first_audio",
            {
                "latency_ms": self._latency.tts_first_audio_ms,
                "ok_under_1s": (
                    self._latency.tts_first_audio_ms is not None
                    and self._latency.tts_first_audio_ms <= 1000
                ),
                "route": self._latency.route,
            },
        )
        await self._emit("state", {"state": self.state.value})

    async def _play_pcm_chunks(
        self,
        pcm: bytes,
        *,
        generation_id: int,
        cancel_event: asyncio.Event | None = None,
        on_first_audio=None,
        frame_ms: int = 40,
    ) -> None:
        if not pcm:
            return
        first = True
        frame = int(self.settings.sample_rate * (frame_ms / 1000.0)) * 2
        for i in range(0, len(pcm), frame):
            if cancel_event and cancel_event.is_set():
                return
            if generation_id != self.generation_id:
                return
            chunk = pcm[i : i + frame]
            if first:
                first = False
                if on_first_audio:
                    await on_first_audio()
            await self._emit(
                "tts.audio",
                {
                    "pcm16_b64": _b64(chunk),
                    "sample_rate": self.settings.sample_rate,
                },
            )
            self._last_tts_audio_at = time.perf_counter()
            # Pace roughly realtime so barge-in remains usable.
            await asyncio.sleep(frame_ms / 1000.0)

    async def _play_prompt_id(
        self,
        prompt_id: str,
        *,
        generation_id: int,
        route: str,
        mark_latency: bool = True,
    ) -> bool:
        pcm = self.prompts.load_pcm16(prompt_id, lang=self._prompt_lang())
        if not pcm:
            return False

        async def on_first_audio() -> None:
            if mark_latency:
                await self._mark_first_audio(generation_id=generation_id)

        if mark_latency and self._turn_clock is not None and self._latency.llm_first_token_ms is None:
            self._latency.llm_first_token_ms = round(
                (time.perf_counter() - self._turn_clock) * 1000, 1
            )
            self._latency.route = route
            await self._emit(
                "llm.first_token",
                {
                    "latency_ms": self._latency.llm_first_token_ms,
                    "route": route,
                    "prompt_id": prompt_id,
                    "source": "local_wav",
                },
            )

        await self._emit(
            "prompt.audio",
            {"prompt_id": prompt_id, "source": "local_wav", "bytes": len(pcm)},
        )
        await self._play_pcm_chunks(
            pcm,
            generation_id=generation_id,
            cancel_event=self._cancel,
            on_first_audio=on_first_audio,
        )
        return True

    def _choose_wait_filler(self, user_text: str) -> str | None:
        if not self.settings.enable_wait_filler:
            return None
        min_chars = max(0, int(self.settings.filler_min_user_chars))
        force = len((user_text or "").strip()) >= min_chars
        # Long turns / LLM extractor path benefit most from filler.
        if self.settings.repair_order_use_llm_extractor:
            force = force or len((user_text or "").strip()) >= 8
        return self.prompts.choose_filler(
            user_text,
            lang=self._prompt_lang(),
            force=force,
        )

    async def _speak_text(
        self,
        reply_text: str,
        *,
        generation_id: int,
        route: str,
        prompt_id: str | None = None,
        user_text: str = "",
        saved: bool = False,
        prefer_local: bool = True,
        emit_turn_done: bool = True,
    ) -> None:
        full_reply = reply_text
        try:
            used_local = False
            if prefer_local and prompt_id:
                used_local = await self._play_prompt_id(
                    prompt_id,
                    generation_id=generation_id,
                    route=route,
                )

            if not used_local:

                async def on_first_token() -> None:
                    if self._turn_clock is not None:
                        self._latency.llm_first_token_ms = round(
                            (time.perf_counter() - self._turn_clock) * 1000, 1
                        )
                    self._latency.route = route
                    await self._emit(
                        "llm.first_token",
                        {
                            "latency_ms": self._latency.llm_first_token_ms,
                            "route": self._latency.route,
                            "prompt_id": prompt_id,
                        },
                    )

                async def on_first_audio() -> None:
                    await self._mark_first_audio(generation_id=generation_id)

                async def on_audio_chunk(pcm: bytes) -> None:
                    if generation_id != self.generation_id:
                        return
                    await self._emit(
                        "tts.audio",
                        {
                            "pcm16_b64": _b64(pcm),
                            "sample_rate": self.settings.sample_rate,
                        },
                    )
                    self._last_tts_audio_at = time.perf_counter()

                async def text_gen():
                    await on_first_token()
                    await self._emit("llm.delta", {"text": full_reply})
                    yield full_reply

                await self.tts.synthesize_stream(
                    text_gen(),
                    reply_lang=self._reply_lang,
                    on_first_audio=on_first_audio,
                    on_audio_chunk=on_audio_chunk,
                    cancel_event=self._cancel,
                )
            else:
                await self._emit("llm.delta", {"text": full_reply})

            if generation_id != self.generation_id:
                return

            if full_reply.strip():
                if user_text:
                    self.history.append({"role": "user", "content": user_text})
                self.history.append(
                    {"role": "assistant", "content": full_reply.strip()}
                )

            if emit_turn_done:
                await self._emit(
                    "turn.done",
                    {
                        "user": user_text,
                        "reply": full_reply.strip(),
                        "latency": self._latency.as_dict(),
                        "route": route,
                        "prompt_id": prompt_id,
                        "saved": saved,
                        "draft": {
                            "customer_name": self.draft.customer_name,
                            "customer_phone": self.draft.customer_phone,
                            "customer_address": self.draft.customer_address,
                            "repair_description": self.draft.repair_description,
                            "state": self.draft.state.value,
                            "awaiting_field": self.draft.awaiting_field,
                            "session_language": self.draft.session_language,
                        },
                    },
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            await self._emit("error", {"message": str(exc)})
        finally:
            # Keep the greeting locked after all audio has been sent: the
            # browser may still have queued audio left to play. The normal
            # success path unlocks only when playback.done arrives.
            if (
                route == "repair_greeting"
                and generation_id == self.generation_id
                and not self._awaiting_playback_done
            ):
                self._greeting_uninterruptible = False
            if emit_turn_done and generation_id == self.generation_id:
                if not self._awaiting_playback_done:
                    self._finish_listening()
                    await self._emit("state", {"state": self.state.value})

    async def _run_reply(self, user_text: str, *, generation_id: int) -> None:
        if self.business_mode == "repair_order":
            filler_id = self._choose_wait_filler(user_text)
            filler_task: asyncio.Task | None = None
            if filler_id:
                # Start filler immediately while LLM extraction runs.
                filler_task = asyncio.create_task(
                    self._play_prompt_id(
                        filler_id,
                        generation_id=generation_id,
                        route="repair_filler",
                        mark_latency=True,
                    )
                )
                await self._emit(
                    "filler.started",
                    {"prompt_id": filler_id, "user_chars": len(user_text)},
                )

            turn_task = asyncio.create_task(
                self.dialog.process_user_text_async(
                    self.call_id, self.draft, user_text
                )
            )
            turn = await turn_task
            self._reply_lang = self._reply_lang_for_tts()

            if filler_task is not None:
                try:
                    await filler_task
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass

            if generation_id != self.generation_id:
                return

            await self._emit(
                "order.draft",
                {
                    "customer_name": self.draft.customer_name,
                    "customer_phone": self.draft.customer_phone,
                    "customer_address": self.draft.customer_address,
                    "repair_description": self.draft.repair_description,
                    "state": self.draft.state.value,
                    "awaiting_field": self.draft.awaiting_field,
                    "session_language": self.draft.session_language,
                    "language_source": self.draft.language_source,
                    "language_confidence": self.draft.language_confidence,
                    "phone_candidate": getattr(turn, "phone_candidate", None),
                    "phone_error": getattr(turn, "phone_error", None),
                    "updated_fields": turn.updated_fields or [],
                    "saved": turn.saved,
                },
            )
            await self._speak_text(
                turn.plan.text,
                generation_id=generation_id,
                route="repair_order",
                prompt_id=turn.plan.prompt_id,
                user_text=user_text,
                saved=turn.saved,
                prefer_local=not turn.plan.dynamic,
            )
            return

        route_out: dict[str, str] = {}
        full_reply = ""
        try:

            async def on_first_token() -> None:
                if self._turn_clock is not None:
                    self._latency.llm_first_token_ms = round(
                        (time.perf_counter() - self._turn_clock) * 1000, 1
                    )
                self._latency.route = route_out.get("route")
                await self._emit(
                    "llm.first_token",
                    {
                        "latency_ms": self._latency.llm_first_token_ms,
                        "route": self._latency.route,
                    },
                )

            async def on_first_audio() -> None:
                await self._mark_first_audio(generation_id=generation_id)

            async def on_audio_chunk(pcm: bytes) -> None:
                if generation_id != self.generation_id:
                    return
                await self._emit(
                    "tts.audio",
                    {
                        "pcm16_b64": _b64(pcm),
                        "sample_rate": self.settings.sample_rate,
                    },
                )
                self._last_tts_audio_at = time.perf_counter()

            async def text_gen():
                nonlocal full_reply
                async for chunk in self.llm.stream_reply(
                    user_text,
                    asr_lang=self._asr_lang,
                    history=self.history,
                    on_first_token=on_first_token,
                    cancel_event=self._cancel,
                    route_out=route_out,
                ):
                    if generation_id != self.generation_id:
                        return
                    full_reply += chunk
                    await self._emit("llm.delta", {"text": chunk})
                    yield chunk

            await self.tts.synthesize_stream(
                text_gen(),
                reply_lang=self._reply_lang,
                on_first_audio=on_first_audio,
                on_audio_chunk=on_audio_chunk,
                cancel_event=self._cancel,
            )

            if generation_id != self.generation_id:
                return

            if full_reply.strip():
                self.history.append({"role": "user", "content": user_text})
                self.history.append(
                    {"role": "assistant", "content": full_reply.strip()}
                )

            await self._emit(
                "turn.done",
                {
                    "user": user_text,
                    "reply": full_reply.strip(),
                    "latency": self._latency.as_dict(),
                    "route": route_out.get("route"),
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            await self._emit("error", {"message": str(exc)})
        finally:
            if generation_id == self.generation_id:
                if not self._awaiting_playback_done:
                    self._finish_listening()
                    await self._emit("state", {"state": self.state.value})

    async def on_playback_done(self, generation_id: int) -> None:
        if (
            generation_id != self.generation_id
            or not self._awaiting_playback_done
        ):
            return
        if self._last_tts_audio_at is not None:
            quiet_ms = (time.perf_counter() - self._last_tts_audio_at) * 1000.0
            if quiet_ms < float(self.settings.playback_done_grace_ms):
                # Browser may report done too early when streamed chunks arrive
                # with jitter. Ignore premature completion and wait.
                return
        self._awaiting_playback_done = False
        self._greeting_uninterruptible = False
        self._finish_listening()
        await self._emit("state", {"state": self.state.value})

    def _finish_listening(self) -> None:
        self.state = SessionState.LISTENING
        self.vad.reset()
        self._barge_buffer.clear()
        self._barge_started_at = None
        self._turn_clock = None
        self._speaking_started_at = None

    async def _handle_barge_in(self) -> None:
        self.state = SessionState.BARGE_IN
        self._awaiting_playback_done = False
        await self._emit("barge_in", {})
        await self._cancel_generation()
        self._finish_listening()
        await self._emit("state", {"state": self.state.value})

    async def _cancel_generation(self) -> None:
        self.generation_id += 1
        self._cancel.set()
        if self._reply_task and not self._reply_task.done():
            self._reply_task.cancel()
            try:
                await self._reply_task
            except (asyncio.CancelledError, Exception):
                pass
        self._reply_task = None
        await self._emit("tts.cancel", {})

    async def _emit(self, event_type: str, data: dict) -> None:
        await self.emit(
            ServerEvent(
                type=event_type,
                call_id=self.call_id,
                turn_id=self.turn_id,
                generation_id=self.generation_id,
                data=data,
            )
        )
