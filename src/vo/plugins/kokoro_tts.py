import uuid
from collections.abc import AsyncIterable, AsyncIterator

import numpy as np
from kokoro_onnx import Kokoro
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, tts

from vo.lang.script import script_runs

SAMPLE_RATE = 24000
ESPEAK_LANG = {"en": "en-us", "hi": "hi"}


class KokoroTTS(tts.TTS):
    """Kokoro-82M with per-script phonemization for English, Hindi and code-mixed text.

    Text is split into Latin and Devanagari runs, and each run is phonemized with its own espeak
    language. Feeding mixed text to one espeak voice makes it insert "(en)…(hi)" switch markers,
    which Kokoro would then read aloud. Any Devanagari in a reply selects the Hindi voice for the
    whole reply, so the timbre stays consistent across runs.
    """

    def __init__(self, *, model_path: str, voices_path: str, voice: str = "af_heart",
                 hindi_voice: str = "hf_alpha", speed: float = 1.0) -> None:
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False),
                         sample_rate=SAMPLE_RATE, num_channels=1)
        self._engine = Kokoro(model_path, voices_path)
        self._voice, self._hindi_voice, self._speed = voice, hindi_voice, speed

    @property
    def model(self) -> str:
        return "kokoro-82m"

    def plan(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """(voice, [(espeak_lang, chunk), ...]) for a piece of text."""
        runs = script_runs(text)
        voice = self._hindi_voice if any(lang == "hi" for lang, _ in runs) else self._voice
        return voice, [(ESPEAK_LANG[lang], chunk) for lang, chunk in runs]

    async def pcm(self, text: str) -> AsyncIterator[np.ndarray]:
        voice, runs = self.plan(text)
        for lang, chunk in runs:
            async for samples, _ in self._engine.create_stream(
                chunk, voice=voice, speed=self._speed, lang=lang
            ):
                yield (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)

    def synthesize(self, text: str, *,
                   conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> "_Stream":
        return _Stream(tts=self, input_text=text, conn_options=conn_options)


class _Stream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        t: KokoroTTS = self._tts  # type: ignore[assignment]
        output_emitter.initialize(request_id=uuid.uuid4().hex, sample_rate=SAMPLE_RATE,
                                  num_channels=1, mime_type="audio/pcm")
        async for pcm in t.pcm(self.input_text):
            output_emitter.push(pcm.tobytes())
        output_emitter.flush()


async def split_on_danda(text: AsyncIterable[str]) -> AsyncIterator[str]:
    """TTS text transform: LiveKit's sentence tokenizer does not split on the Hindi danda (।),
    so a Hindi reply would be synthesized as one chunk and delay first audio. Map it to a period."""
    async for chunk in text:
        yield chunk.replace("।", ". ").replace("॥", ". ")
