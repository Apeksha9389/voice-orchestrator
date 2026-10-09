"""Streaming noise suppression in front of VAD/STT.

RNNoise runs on fixed 10 ms blocks at 48 kHz. `BlockDenoiser` adapts arbitrary frame sizes and
sample rates to that block size and returns exactly as many samples as it receives, so it can sit
inline in a real-time stream. Added latency: one 10 ms buffering block plus
the model's own delay (RNNoise: 20 ms), + resampler delay if the stream is not 48 kHz.
"""

from __future__ import annotations

import logging
import math
import weakref
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from livekit import rtc
from livekit.agents.voice.io import AudioInput

log = logging.getLogger("vo.denoise")


class Denoiser(Protocol):
    sample_rate: int
    block_size: int
    delay: int  # algorithmic delay in samples at `sample_rate`

    def process_block(self, block: np.ndarray) -> tuple[np.ndarray, float | None]:
        """int16 block of `block_size` samples -> (denoised int16 block, speech prob or None)."""

    def close(self) -> None: ...


class RNNoiseDenoiser:
    sample_rate = 48000
    block_size = 480
    delay = 960  # measured by cross-correlation against the clean input

    def __init__(self) -> None:
        from pyrnnoise import rnnoise

        self._process = rnnoise.process_mono_frame
        self._state = rnnoise.create()
        self._finalizer = weakref.finalize(self, rnnoise.destroy, self._state)

    def process_block(self, block: np.ndarray) -> tuple[np.ndarray, float | None]:
        out, prob = self._process(self._state, block)
        return out, float(prob)

    def close(self) -> None:
        self._finalizer()


class SnrEstimator:
    """Frame SNR against a noise floor tracked during non-speech (fast down, slow up)."""

    def __init__(self, floor_db: float = -70.0, rise: float = 0.02, fall: float = 0.3) -> None:
        self.noise_db = floor_db
        self._rise, self._fall = rise, fall

    def update(self, block: np.ndarray, speech_prob: float | None) -> float:
        rms = np.sqrt(np.mean((block.astype(np.float32) / 32768.0) ** 2)) + 1e-9
        level_db = 20 * math.log10(rms)
        if speech_prob is None or speech_prob < 0.2:
            rate = self._fall if level_db < self.noise_db else self._rise
            self.noise_db += rate * (level_db - self.noise_db)
        return level_db - self.noise_db


@dataclass
class DenoiseStats:
    speech_prob: float | None = None
    snr_db: float = 0.0
    frames: int = 0


class BlockDenoiser:
    """Frame-size/rate adapter around a block `Denoiser`. One instance per audio stream."""

    def __init__(self, denoiser: Denoiser) -> None:
        self._dn = denoiser
        self._snr = SnrEstimator()
        self._rate: int | None = None
        self.latency = 0  # samples at stream rate (steady state, excluding resampler delay)
        self._up: rtc.AudioResampler | None = None
        self._down: rtc.AudioResampler | None = None
        self._in = np.zeros(0, np.int16)
        self._out = np.zeros(0, np.int16)
        self.stats = DenoiseStats()

    def _configure(self, rate: int) -> None:
        self._rate = rate
        if rate != self._dn.sample_rate:
            self._up = rtc.AudioResampler(rate, self._dn.sample_rate)
            self._down = rtc.AudioResampler(self._dn.sample_rate, rate)
        # prime with one block of silence so output length always matches input length
        prime = math.ceil(self._dn.block_size * rate / self._dn.sample_rate)
        self._out = np.zeros(prime, np.int16)
        self.latency = prime + round(self._dn.delay * rate / self._dn.sample_rate)

    @staticmethod
    def _cat(frames: list[rtc.AudioFrame]) -> np.ndarray:
        if not frames:
            return np.zeros(0, np.int16)
        return np.concatenate([np.frombuffer(f.data, np.int16) for f in frames])

    def process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if frame.num_channels != 1:
            raise ValueError("denoiser expects mono audio")
        if frame.sample_rate != self._rate:
            self._configure(frame.sample_rate)

        pcm = np.frombuffer(frame.data, np.int16)
        self._in = np.concatenate([self._in, self._cat(self._up.push(frame)) if self._up else pcm])

        bs, done = self._dn.block_size, []
        while len(self._in) >= bs:
            block, self._in = self._in[:bs], self._in[bs:]
            out, prob = self._dn.process_block(block)
            self.stats.speech_prob = prob
            self.stats.snr_db = self._snr.update(block, prob)
            done.append(out)

        if done:
            clean = np.concatenate(done)
            if self._down:
                n = len(clean)
                clean = self._cat(self._down.push(rtc.AudioFrame(clean.tobytes(), self._dn.sample_rate, 1, n)))
            self._out = np.concatenate([self._out, clean])

        n = frame.samples_per_channel
        if len(self._out) < n:  # resampler warm-up only
            self._out = np.concatenate([np.zeros(n - len(self._out), np.int16), self._out])
        out, self._out = self._out[:n], self._out[n:]
        self.stats.frames += 1
        return rtc.AudioFrame(out.tobytes(), frame.sample_rate, 1, n)

    def close(self) -> None:
        self._dn.close()


class DenoisedAudioInput(AudioInput):
    """Wraps the session's audio input; works for both room and console IO."""

    def __init__(self, source: AudioInput, denoiser: Denoiser, *, log_every: int = 500) -> None:
        super().__init__(label="denoised", source=source)
        self._bd = BlockDenoiser(denoiser)
        self._log_every = log_every

    @property
    def stats(self) -> DenoiseStats:
        return self._bd.stats

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await self.source.__anext__()  # type: ignore[union-attr]
        out = self._bd.process(frame)
        s = self._bd.stats
        if self._log_every and s.frames % self._log_every == 0:
            log.debug("denoise snr=%.1fdB speech_prob=%s", s.snr_db, s.speech_prob)
        return out


def build_denoiser(name: str) -> Denoiser | None:
    if name in ("", "none", "off"):
        return None
    if name == "rnnoise":
        try:
            return RNNoiseDenoiser()
        except (ImportError, OSError) as e:
            log.warning("rnnoise unavailable (%s); running without noise suppression", e)
            return None
    raise ValueError(f"unknown denoiser: {name!r}")
