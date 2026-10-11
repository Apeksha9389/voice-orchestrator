from pathlib import Path

import numpy as np
import pytest
from livekit import rtc
from livekit.agents import APIConnectOptions, stt

from vo.lang.router import LanguageRouter, RoutedSTT
from vo.lang.script import script_runs
from vo.plugins.kokoro_tts import split_on_danda


def test_script_runs_code_mixed():
    assert script_runs("आपका order कल आएगा।") == [
        ("hi", "आपका "), ("en", "order "), ("hi", "कल आएगा।"),
    ]
    assert script_runs("123, hello!") == [("en", "123, hello!")]
    assert script_runs(" ... ") == []
    assert script_runs("42") == [("en", "42")]


def test_router_hysteresis():
    r = LanguageRouter({"en", "hi"}, "en", switch_prob=0.7, min_seconds=1.0)
    assert r.update("hi", 0.95, 0.4) == "en"  # too short to trust
    assert r.update("hi", 0.5, 2.0) == "en"  # not confident
    assert r.update("ur", 0.9, 2.0) == "hi"  # Urdu label on Hindi speech
    assert r.update("en", 0.99, 0.6) == "hi"  # short "okay" keeps Hindi
    assert r.update("fr", 0.99, 3.0) == "hi"  # unsupported
    assert r.update("en", 0.9, 1.5) == "en"
    with pytest.raises(ValueError):
        LanguageRouter({"en"}, "hi")


class FakeSTT(stt.STT):
    def __init__(self, name: str):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self.name = name

    async def _recognize_impl(self, buffer, *, language, conn_options: APIConnectOptions):
        return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                               alternatives=[stt.SpeechData(language=language, text=self.name)])


def audio(seconds: float) -> rtc.AudioFrame:
    n = int(16000 * seconds)
    return rtc.AudioFrame(np.zeros(n, np.int16).tobytes(), 16000, 1, n)


async def test_routed_stt_picks_engine_per_language():
    detections = iter([("hi", 0.9), ("en", 0.95), ("en", 0.95)])
    router = LanguageRouter({"en", "hi"}, "en", min_seconds=1.0)
    routed = RoutedSTT(detector=lambda pcm: next(detections),
                       engines={"en": FakeSTT("whisper"), "hi": FakeSTT("indic")}, router=router)

    async def run(seconds, **kw):
        ev = await routed.recognize(audio(seconds), **kw)
        return ev.alternatives[0].text, ev.alternatives[0].language

    assert await run(2.0) == ("indic", "hi")
    assert await run(0.5) == ("indic", "hi")  # "okay" mid-Hindi call stays on Hindi engine
    assert await run(2.0) == ("whisper", "en")
    assert await run(2.0, language="hi") == ("indic", "hi")  # explicit language skips LID
    assert routed.language == "en"


def test_routed_stt_requires_engine_for_each_language():
    with pytest.raises(ValueError):
        RoutedSTT(detector=lambda pcm: ("en", 1.0), engines={"en": FakeSTT("w")},
                  router=LanguageRouter({"en", "hi"}, "en"))


async def test_indic_stt_calls_model_with_16k_mono():
    pytest.importorskip("torch")
    from vo.plugins.indic_stt import IndicConformerSTT

    calls = []

    def fake_model(wav, lang, decoding):
        calls.append((tuple(wav.shape), lang, decoding))
        return "नमस्ते "

    s = IndicConformerSTT(model=fake_model, decoding="rnnt")
    frame = rtc.AudioFrame(np.zeros(48000, np.int16).tobytes(), 48000, 1, 48000)
    ev = await s.recognize(frame)
    assert ev.alternatives[0].text == "नमस्ते" and ev.alternatives[0].language == "hi"
    assert calls == [((1, 16000), "hi", "rnnt")]


async def test_split_on_danda():
    async def gen():
        for c in ["नमस्ते। आपका ", "ऑर्डर आएगा।"]:
            yield c

    assert "".join([c async for c in split_on_danda(gen())]) == "नमस्ते.  आपका ऑर्डर आएगा. "


MODELS = Path("models")
needs_kokoro = pytest.mark.skipif(not (MODELS / "kokoro-v1.0.onnx").exists(),
                                  reason="kokoro model files missing")


@needs_kokoro
async def test_kokoro_bilingual_plan_and_synthesis():
    from vo.plugins.kokoro_tts import SAMPLE_RATE, KokoroTTS

    tts = KokoroTTS(model_path=str(MODELS / "kokoro-v1.0.onnx"),
                    voices_path=str(MODELS / "voices-v1.0.bin"))
    assert tts.plan("Your order ships today.") == ("af_heart", [("en-us", "Your order ships today.")])
    voice, runs = tts.plan("आपका order कल आएगा।")
    assert voice == "hf_alpha" and [lang for lang, _ in runs] == ["hi", "en-us", "hi"]

    pcm = np.concatenate([c async for c in tts.pcm("आपका order कल आएगा।")])
    assert len(pcm) > SAMPLE_RATE * 0.8 and np.abs(pcm).max() > 1000
