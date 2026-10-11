"""AI4Bharat IndicConformer (600M, 22 Indian languages) as a batch LiveKit STT.

Needs the `indic` extra and a Hugging Face token with access to the gated model:
    pip install -e ".[indic]" && huggingface-cli login
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import numpy as np
from livekit.agents import APIConnectOptions, stt
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer

from vo.plugins.whisper_stt import to_mono_16k

DEFAULT_MODEL = "ai4bharat/indic-conformer-600m-multilingual"


class IndicConformerSTT(stt.STT):
    def __init__(self, *, model_id: str = DEFAULT_MODEL, language: str = "hi",
                 decoding: Literal["ctc", "rnnt"] = "ctc", device: str = "cpu",
                 model: Any = None) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._language, self._decoding, self._device = language, decoding, device
        self._model = model if model is not None else self._load(model_id, device)

    @staticmethod
    def _load(model_id: str, device: str) -> Any:
        from transformers import AutoModel

        return AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

    @property
    def model(self) -> str:
        return f"indic-conformer-{self._decoding}"

    def _transcribe(self, pcm: np.ndarray, language: str) -> str:
        import torch

        with torch.inference_mode():
            wav = torch.from_numpy(pcm).unsqueeze(0).to(self._device)
            out = self._model(wav, language, self._decoding)
        return (out[0] if isinstance(out, list | tuple) else out).strip()

    async def _recognize_impl(self, buffer: AudioBuffer, *, language: NotGivenOr[str] = NOT_GIVEN,
                              conn_options: APIConnectOptions) -> stt.SpeechEvent:
        lang = language if language is not NOT_GIVEN else self._language
        text = await asyncio.to_thread(self._transcribe, to_mono_16k(buffer), lang)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=lang, text=text, confidence=1.0)],
        )
