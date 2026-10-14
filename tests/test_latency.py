import time
from pathlib import Path

import pytest
from livekit.agents import llm, metrics

from vo.config import Settings
from vo.lang.script import clauses
from vo.observability import LatencyTracker


def eou(sid, delay, stt=0.2):
    return metrics.EOUMetrics(timestamp=time.time(), end_of_utterance_delay=delay,
                              transcription_delay=stt, on_user_turn_completed_delay=0.0,
                              speech_id=sid)


def llm_m(sid, ttft, cancelled=False):
    return metrics.LLMMetrics(label="llm", request_id="r", timestamp=time.time(), duration=1.0,
                              ttft=ttft, cancelled=cancelled, completion_tokens=10,
                              prompt_tokens=100, prompt_cached_tokens=0, total_tokens=110,
                              tokens_per_second=50.0, speech_id=sid)


def tts_m(sid, ttfb):
    return metrics.TTSMetrics(label="tts", request_id="r", timestamp=time.time(), ttfb=ttfb,
                              duration=1.0, audio_duration=2.0, cancelled=False,
                              characters_count=40, streamed=True, speech_id=sid)


def test_latency_tracker_combines_stages_per_turn():
    lt = LatencyTracker()
    assert lt.collect(eou("s1", 0.5)) is None
    assert lt.collect(llm_m("s1", 0.9, cancelled=True)) is None  # discarded preemptive attempt
    assert lt.collect(llm_m("s1", 0.4)) is None
    done = lt.collect(tts_m("s1", 0.3))
    assert done is not None and done.response == pytest.approx(1.2)
    assert lt.collect(tts_m("s1", 0.8)) is None  # later sentences don't change first audio

    for i, r in enumerate([0.2, 0.3, 0.4]):
        sid = f"t{i}"
        lt.collect(eou(sid, r))
        lt.collect(llm_m(sid, 0.5))
        lt.collect(tts_m(sid, 0.2))
    s = lt.summary()
    assert s["response"]["n"] == 4
    assert s["response"]["p50"] == pytest.approx(1.05)  # responses 1.2, 0.9, 1.0, 1.1
    assert s["eou"]["p95"] <= 0.5


def test_metrics_without_speech_id_are_ignored():
    lt = LatencyTracker()
    assert lt.collect(eou(None, 0.5)) is None and lt.summary() == {}


def test_clauses_split_after_min_chars():
    text = "Sure, I checked your order A1234, it shipped yesterday and arrives Wednesday."
    assert clauses(text) == ["Sure, I checked your order A1234,",
                             " it shipped yesterday and arrives Wednesday."]
    assert clauses("Okay.") == ["Okay."]
    assert "".join(clauses(text)) == text


def test_llm_chain():
    from vo.agent import build_llm

    both = build_llm(Settings(openrouter_api_key="a", openai_api_key="b"))
    assert isinstance(both, llm.FallbackAdapter)
    only = build_llm(Settings(openrouter_api_key="a", openai_api_key=None))
    assert not isinstance(only, llm.FallbackAdapter)
    with pytest.raises(RuntimeError):
        build_llm(Settings(openrouter_api_key=None, openai_api_key=None, _env_file=None))


def test_preemptive_generation_is_configured():
    from vo.agent import turn_handling

    th = turn_handling(Settings(), "vad")
    assert th["preemptive_generation"]["enabled"] is True


MODELS = Path("models")


@pytest.mark.skipif(not (MODELS / "kokoro-v1.0.onnx").exists(), reason="kokoro model files missing")
async def test_clause_chunking_starts_audio_before_full_sentence():
    from vo.plugins.kokoro_tts import KokoroTTS

    tts = KokoroTTS(model_path=str(MODELS / "kokoro-v1.0.onnx"),
                    voices_path=str(MODELS / "voices-v1.0.bin"))
    tts.warmup()
    text = ("Sure, I checked your order A1234, it was shipped yesterday with Delhivery and "
            "should reach you by Wednesday evening.")
    t0 = time.perf_counter()
    first, chunks = None, 0
    async for _ in tts.pcm(text):
        first = first or time.perf_counter() - t0
        chunks += 1
    total = time.perf_counter() - t0
    assert chunks >= 2 and first < total * 0.6
