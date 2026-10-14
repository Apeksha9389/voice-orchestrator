from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="VO_", extra="ignore")

    llm_provider: str = "openrouter"  # primary: openrouter | openai
    llm_model: str = "openai/gpt-4o-mini"
    llm_fallback_model: str = "gpt-4o-mini"  # used on the other provider when its key is set
    llm_attempt_timeout: float = 4.0  # per-provider TTFT budget before failing over
    llm_temperature: float = 0.4
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None

    stt_model: str = "large-v3-turbo"
    stt_device: str = "auto"
    stt_compute_type: str = "int8"
    stt_language: str | None = "en"  # default / starting language
    languages: list[str] = ["en", "hi"]  # >1 enables per-utterance language routing
    hindi_stt: str = "indic"  # indic (IndicConformer) | whisper
    indic_model: str = "ai4bharat/indic-conformer-600m-multilingual"
    indic_decoding: str = "ctc"  # ctc (faster) | rnnt (more accurate)
    lid_switch_prob: float = 0.7
    lid_min_seconds: float = 1.0

    tts_model_path: str = "models/kokoro-v1.0.onnx"
    tts_voices_path: str = "models/voices-v1.0.bin"
    tts_voice: str = "af_heart"
    tts_hindi_voice: str = "hf_alpha"
    tts_speed: float = 1.0

    denoiser: str = "rnnoise"  # rnnoise | none

    vad_activation: float = 0.5
    vad_min_silence: float = 0.4

    turn_detector: str = "multilingual"  # multilingual | vad
    endpointing_min_delay: float = 0.4
    endpointing_max_delay: float = 2.5
    interrupt_min_duration: float = 0.4  # user speech (s) before agent audio is paused
    false_interruption_timeout: float = 1.0  # silence (s) after a pause before resuming
    backchannel_max_words: int = 3

    # latency / observability
    preemptive_generation: bool = True  # start the LLM on the final transcript, before EOU
    otel_endpoint: str | None = None  # OTLP/HTTP base URL, e.g. http://localhost:4318

    # telephony
    agent_name: str = ""  # set for explicit dispatch (e.g. "vo-agent" in the SIP dispatch rule)
    sip_vad_activation: float = 0.35  # narrowband G.711 speech scores lower in Silero
    sip_interrupt_min_duration: float = 0.5  # PSTN echo/line noise: require a bit more speech
    sip_endpointing_min_delay: float = 0.5  # PSTN adds jitter-buffer delay to pauses
    user_away_timeout: float = 12.0  # silence (s) before asking "are you still there?"
    transfer_number: str | None = None  # E.164, e.g. +9180xxxxxxxx; unset disables transfers
    orders_path: str = "data/orders.json"

    system_prompt: str = (
        "You are a concise, friendly voice assistant. Reply in one or two short spoken "
        "sentences. No markdown, lists, or emojis. Reply in the language the user speaks: "
        "English in English; Hindi or Hinglish in Hindi written in Devanagari, keeping common "
        "English words (order, payment, account) in Latin script. Write numbers as digits. "
        "You can look up orders, transfer to a human, and end the call. Before ending the call, "
        "say a short goodbye; before transferring, tell the caller you are connecting them."
    )


@lru_cache
def settings() -> Settings:
    return Settings()
