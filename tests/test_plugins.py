from pathlib import Path

import numpy as np
import pytest
from livekit import rtc

from vo.plugins.whisper_stt import to_mono_16k

MODELS = Path("models")
has_kokoro = (MODELS / "kokoro-v1.0.onnx").exists()


def test_resample_stereo_48k_to_mono_16k():
    pcm = (np.sin(np.linspace(0, 200, 48000)) * 8000).astype(np.int16)
    frame = rtc.AudioFrame(np.repeat(pcm, 2).tobytes(), 48000, 2, 48000)
    out = to_mono_16k([frame])
    assert out.dtype == np.float32 and len(out) == 16000 and np.abs(out).max() <= 1.0


@pytest.mark.skipif(not has_kokoro, reason="kokoro model files missing")
async def test_kokoro_emits_pcm():
    from vo.plugins.kokoro_tts import SAMPLE_RATE, KokoroTTS

    tts = KokoroTTS(model_path=str(MODELS / "kokoro-v1.0.onnx"),
                    voices_path=str(MODELS / "voices-v1.0.bin"))
    frames = [ev.frame async for ev in tts.synthesize("Hello, this is a test.")]
    total = sum(f.samples_per_channel for f in frames)
    assert frames[0].sample_rate == SAMPLE_RATE and total > SAMPLE_RATE * 0.5
    await tts.aclose()
