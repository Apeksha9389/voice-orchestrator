import uuid

import numpy as np
from kokoro_onnx import Kokoro
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, tts

SAMPLE_RATE = 24000


class KokoroTTS(tts.TTS):
    def __init__(self, *, model_path: str, voices_path: str, voice: str = "af_heart",
                 speed: float = 1.0, lang: str = "en-us") -> None:
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False),
                         sample_rate=SAMPLE_RATE, num_channels=1)
        self._engine = Kokoro(model_path, voices_path)
        self._voice, self._speed, self._lang = voice, speed, lang

    @property
    def model(self) -> str:
        return "kokoro-82m"

    def synthesize(self, text: str, *,
                   conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> "_Stream":
        return _Stream(tts=self, input_text=text, conn_options=conn_options)


class _Stream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        t: KokoroTTS = self._tts  # type: ignore[assignment]
        output_emitter.initialize(request_id=uuid.uuid4().hex, sample_rate=SAMPLE_RATE,
                                  num_channels=1, mime_type="audio/pcm")
        async for samples, _ in t._engine.create_stream(
            self.input_text, voice=t._voice, speed=t._speed, lang=t._lang
        ):
            pcm = np.clip(samples, -1.0, 1.0)
            output_emitter.push((pcm * 32767).astype(np.int16).tobytes())
        output_emitter.flush()
