"""Per-utterance STT routing by spoken language, with session-level hysteresis."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping

import numpy as np
from livekit.agents import APIConnectOptions, stt
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer

from vo.plugins.whisper_stt import TARGET_SR, to_mono_16k

log = logging.getLogger("vo.lang")

# Whisper often labels Hindi speech as Urdu (same spoken language, different script)
LANG_ALIASES = {"ur": "hi"}

Detector = Callable[[np.ndarray], tuple[str, float]]


class LanguageRouter:
    """Sticky language choice. Short utterances ("haan", "ok") give noisy language ID, so the
    session language only changes on a confident detection over enough audio."""

    def __init__(self, supported: set[str], default: str, *, switch_prob: float = 0.7,
                 min_seconds: float = 1.0) -> None:
        if default not in supported:
            raise ValueError(f"default language {default!r} not in {supported}")
        self.supported, self.current = supported, default
        self._switch_prob, self._min_seconds = switch_prob, min_seconds

    def update(self, lang: str, prob: float, seconds: float) -> str:
        lang = LANG_ALIASES.get(lang, lang)
        if (seconds >= self._min_seconds and prob >= self._switch_prob
                and lang in self.supported and lang != self.current):
            log.info("language switch %s -> %s (p=%.2f, %.1fs)", self.current, lang, prob, seconds)
            self.current = lang
        return self.current


class RoutedSTT(stt.STT):
    """Detects the language of each utterance, then transcribes it with that language's engine."""

    def __init__(self, *, detector: Detector, engines: Mapping[str, stt.STT],
                 router: LanguageRouter) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        missing = router.supported - engines.keys()
        if missing:
            raise ValueError(f"no STT engine for {missing}")
        self._detect, self._engines, self._router = detector, dict(engines), router

    @property
    def model(self) -> str:
        return "routed:" + ",".join(f"{k}={v.model}" for k, v in self._engines.items())

    @property
    def language(self) -> str:
        return self._router.current

    async def _recognize_impl(self, buffer: AudioBuffer, *, language: NotGivenOr[str] = NOT_GIVEN,
                              conn_options: APIConnectOptions) -> stt.SpeechEvent:
        if language is NOT_GIVEN or language not in self._engines:
            pcm = to_mono_16k(buffer)
            detected, prob = await asyncio.to_thread(self._detect, pcm)
            language = self._router.update(detected, prob, len(pcm) / TARGET_SR)
        return await self._engines[language].recognize(
            buffer, language=language, conn_options=conn_options
        )
