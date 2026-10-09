from pathlib import Path

import numpy as np
import pytest
from livekit import rtc

from vo.audio.denoise import BlockDenoiser, SnrEstimator, build_denoiser


class Identity:
    sample_rate, block_size, delay = 48000, 480, 0

    def process_block(self, block):
        return block.copy(), None

    def close(self):
        pass


def frames(pcm: np.ndarray, sr: int, ms: int) -> list[rtc.AudioFrame]:
    n = sr * ms // 1000
    return [rtc.AudioFrame(pcm[i : i + n].tobytes(), sr, 1, n) for i in range(0, len(pcm) - n + 1, n)]


@pytest.mark.parametrize("sr,ms", [(48000, 50), (48000, 10), (24000, 50), (16000, 20)])
def test_output_length_matches_input(sr, ms):
    pcm = (np.random.default_rng(0).standard_normal(sr) * 3000).astype(np.int16)
    bd = BlockDenoiser(Identity())
    for f in frames(pcm, sr, ms):
        out = bd.process(f)
        assert out.samples_per_channel == f.samples_per_channel and out.sample_rate == sr


def test_native_rate_is_exact_delay_line():
    pcm = (np.random.default_rng(1).standard_normal(48000) * 3000).astype(np.int16)
    bd = BlockDenoiser(Identity())
    out = np.concatenate([np.frombuffer(bd.process(f).data, np.int16) for f in frames(pcm, 48000, 50)])
    assert bd.latency == 480
    np.testing.assert_array_equal(out[bd.latency :], pcm[: len(out) - bd.latency])


def test_snr_estimator_tracks_noise_floor():
    rng = np.random.default_rng(2)
    est = SnrEstimator()
    noise = lambda: (rng.standard_normal(480) * 100).astype(np.int16)
    for _ in range(300):
        quiet = est.update(noise(), speech_prob=0.0)
    loud = est.update((rng.standard_normal(480) * 10000).astype(np.int16), speech_prob=0.9)
    assert abs(quiet) < 3 and loud > 30


def test_build_denoiser():
    assert build_denoiser("none") is None
    with pytest.raises(ValueError):
        build_denoiser("bogus")


MODELS = Path("models")
needs_assets = pytest.mark.skipif(
    not (MODELS / "kokoro-v1.0.onnx").exists(), reason="kokoro model files missing"
)


@needs_assets
async def test_rnnoise_improves_noisy_speech():
    pytest.importorskip("pyrnnoise")
    from kokoro_onnx import Kokoro
    from livekit.plugins import silero

    from vo.audio.denoise import RNNoiseDenoiser
    from vo.eval.noise import (
        colored_noise,
        denoise,
        mix_at_snr,
        noise_reduction_db,
        score,
        snr_db,
        to_frames,
        vad_probabilities,
    )

    tts = Kokoro(str(MODELS / "kokoro-v1.0.onnx"), str(MODELS / "voices-v1.0.bin"))
    sp, _ = tts.create("I would like to check the status of my order from last week.", voice="af_heart")
    sp = np.interp(np.arange(0, len(sp), 0.5), np.arange(len(sp)), sp).astype(np.float32)
    pad = np.zeros(48000, np.float32)
    clean = np.concatenate([pad, sp, pad])
    speech = np.zeros(len(clean), bool)
    speech[len(pad) : len(pad) + len(sp)] = True

    noisy = mix_at_snr(clean, colored_noise(len(clean), "white", np.random.default_rng(0)), 0)
    enhanced = denoise(noisy, 48000, RNNoiseDenoiser())

    assert noise_reduction_db(noisy, enhanced, ~speech) > 20
    assert snr_db(clean, enhanced, speech) - snr_db(clean, noisy, speech) > 5

    vad = silero.VAD.load()
    ref = await vad_probabilities(vad, to_frames(clean, 48000)) >= 0.5
    raw = score(await vad_probabilities(vad, to_frames(noisy, 48000)), ref)
    den = score(await vad_probabilities(vad, to_frames(enhanced, 48000)), ref)
    assert den.false_positive_rate <= raw.false_positive_rate
    assert den.recall >= raw.recall - 0.05


async def test_denoised_audio_input_wraps_source():
    from livekit.agents.voice.io import AudioInput

    from vo.audio.denoise import DenoisedAudioInput

    pcm = (np.random.default_rng(3).standard_normal(48000) * 3000).astype(np.int16)
    src_frames = frames(pcm, 48000, 50)

    class Source(AudioInput):
        def __init__(self):
            super().__init__(label="fake")
            self._it = iter(src_frames)
            self.attached = False

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration from None

        def on_attached(self):
            self.attached = True

    src = Source()
    wrapped = DenoisedAudioInput(src, Identity())
    wrapped.on_attached()
    out = [f async for f in wrapped]
    assert src.attached and len(out) == len(src_frames) and wrapped.stats.frames == len(out)
