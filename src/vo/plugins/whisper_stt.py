import asyncio

import numpy as np
from faster_whisper import WhisperModel
from livekit.agents import APIConnectOptions, stt, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer

TARGET_SR = 16000


def keep_segment(no_speech_prob: float, avg_logprob: float) -> bool:
    """Whisper's own silence heuristic: drop segments it believes are non-speech and decoded with
    low confidence. This removes the classic noise hallucinations ("Thank you.", "you")."""
    return not (no_speech_prob > 0.6 and avg_logprob < -1.0)


def to_mono_16k(buffer: AudioBuffer) -> np.ndarray:
    frame = utils.combine_frames(buffer)
    pcm = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
    if frame.num_channels > 1:
        pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1)
    if frame.sample_rate != TARGET_SR:
        n = int(len(pcm) * TARGET_SR / frame.sample_rate)
        pcm = np.interp(np.linspace(0, len(pcm), n, endpoint=False), np.arange(len(pcm)), pcm)
    return pcm.astype(np.float32)


class WhisperSTT(stt.STT):
    """Batch faster-whisper STT; pair with stt.StreamAdapter + VAD for streaming."""

    def __init__(self, *, model: str, device: str = "auto", compute_type: str = "int8",
                 language: str | None = None, beam_size: int = 1) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self._language = language
        self._beam_size = beam_size

    @property
    def model(self) -> str:
        return "faster-whisper"

    def detect_language(self, pcm: np.ndarray) -> tuple[str, float]:
        """Language ID from Whisper's first 30 s window (one encoder pass, no decoding)."""
        lang, prob, _ = self._model.detect_language(audio=pcm, language_detection_segments=1)
        return lang, prob

    def _transcribe(self, pcm: np.ndarray, language: str | None) -> tuple[str, str, float]:
        segments, info = self._model.transcribe(
            pcm, language=language, beam_size=self._beam_size,
            vad_filter=False, condition_on_previous_text=False,
        )
        text = " ".join(
            s.text.strip() for s in segments if keep_segment(s.no_speech_prob, s.avg_logprob)
        ).strip()
        return text, info.language, info.language_probability

    async def _recognize_impl(self, buffer: AudioBuffer, *, language: NotGivenOr[str] = NOT_GIVEN,
                              conn_options: APIConnectOptions) -> stt.SpeechEvent:
        lang = language if language is not NOT_GIVEN else self._language
        text, detected, prob = await asyncio.to_thread(self._transcribe, to_mono_16k(buffer), lang)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=detected, text=text, confidence=prob)],
        )
