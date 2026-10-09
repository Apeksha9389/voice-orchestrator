from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="VO_", extra="ignore")

    llm_provider: str = "openrouter"
    llm_model: str = "openai/gpt-4o-mini"
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None

    stt_model: str = "large-v3-turbo"
    stt_device: str = "auto"
    stt_compute_type: str = "int8"
    stt_language: str | None = "en"

    tts_model_path: str = "models/kokoro-v1.0.onnx"
    tts_voices_path: str = "models/voices-v1.0.bin"
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0

    denoiser: str = "rnnoise"  # rnnoise | none

    vad_activation: float = 0.5
    vad_min_silence: float = 0.4

    system_prompt: str = (
        "You are a concise, friendly voice assistant. Reply in one or two short spoken "
        "sentences. No markdown, lists, or emojis."
    )


@lru_cache
def settings() -> Settings:
    return Settings()
