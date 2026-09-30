from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # mock | cloud
    agent_mode: str = "mock"
    # faq | repair_order
    business_mode: str = "repair_order"
    repair_order_sqlite_path: Path = ROOT_DIR / "output" / "repair_orders.sqlite"
    repair_order_use_llm_extractor: bool = True
    # Local pre-recorded Yue prompts/fillers (mono 16k PCM wav).
    prompt_audio_dir: Path = ROOT_DIR / "audio" / "yue"
    enable_wait_filler: bool = True
    filler_min_user_chars: int = 12

    dashscope_api_key: str = ""
    dashscope_asr_model: str = "qwen3-asr-flash-realtime"
    # Empty = automatic Cantonese/Mandarin/English detection.
    dashscope_asr_language: str = ""
    dashscope_asr_ws_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
    dashscope_asr_silence_ms: int = 300
    dashscope_asr_final_timeout_ms: int = 2500

    minimax_api_key: str = ""
    minimax_tts_model: str = "speech-2.8-turbo"
    minimax_tts_voice: str = "male-qn-qingse"
    minimax_language_boost: str = "Chinese,Yue"
    minimax_ws_url: str = "wss://api.minimaxi.com/ws/v1/t2a_v2"
    minimax_http_url: str = "https://api.minimax.io/v1/t2a_v2"
    minimax_sample_rate: int = 16000
    minimax_audio_format: str = "pcm"

    fast_llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    fast_llm_api_key: str = ""
    fast_llm_model: str = "qwen-plus"
    fast_llm_max_tokens: int = 60
    fast_llm_temperature: float = 0.4
    fast_llm_timeout_s: float = 3.0

    bonsai_base_url: str = "http://172.17.7.90:8080/v1"
    bonsai_api_key: str = ""
    bonsai_model: str = "bonsai-2-27b"
    bonsai_max_tokens: int = 80
    bonsai_temperature: float = 0.5
    bonsai_disable_thinking: bool = True

    llm_system_prompt: str = (
        "你是电话语音助手。用一两句短口语回答，总字数不超过40字。"
        "用户说粤语就用粤语，说普通话就用普通话，说英文就用英文。"
        "第一句直接给答案，不要解释、不要列表。不要说你是AI。"
    )

    faq_path: Path = ROOT_DIR / "faq.json"

    host: str = "127.0.0.1"
    port: int = 7862
    sample_rate: int = 16000
    frame_ms: int = 20
    # Long enough to keep clauses separated by a natural comma pause together.
    vad_silence_ms: int = 650
    vad_energy_threshold: float = 280.0
    vad_webrtc_mode: int | None = 2
    vad_noise_multiplier: float = 2.5
    vad_max_speech_ms: int = 12000
    vad_window_ms: int = 1200
    vad_window_min_voiced_ratio: float = 0.35
    min_speech_ms: int = 200
    barge_in_min_ms: int = 180
    barge_in_thinking_min_ms: int = 500
    barge_in_energy_ratio: float = 1.2
    playback_done_grace_ms: int = 450

    output_dir: Path = ROOT_DIR / "output"

    @property
    def use_cloud(self) -> bool:
        return self.agent_mode.strip().lower() == "cloud"

    @property
    def has_asr_key(self) -> bool:
        return bool(self.dashscope_api_key.strip())

    @property
    def has_minimax_key(self) -> bool:
        return bool(self.minimax_api_key.strip())

    @property
    def has_fast_llm(self) -> bool:
        return bool(self.fast_llm_api_key.strip() or self.dashscope_api_key.strip())

    @property
    def has_bonsai(self) -> bool:
        return bool(self.bonsai_api_key.strip())

    def resolved_fast_llm_key(self) -> str:
        return self.fast_llm_api_key.strip() or self.dashscope_api_key.strip()

    def resolved_prompt_audio_dir(self) -> Path:
        path = self.prompt_audio_dir
        if not path.is_absolute():
            return (ROOT_DIR / path).resolve()
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
