"""Offline helpers to measure how noise and denoising affect VAD decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from livekit import rtc
from livekit.agents import vad as agents_vad

from vo.audio.denoise import BlockDenoiser, Denoiser


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """float32 [-1, 1] arrays; noise is scaled so speech/noise power equals `snr_db`."""
    noise = np.resize(noise, len(speech))
    ps = np.mean(speech[np.abs(speech) > 1e-4] ** 2)
    pn = np.mean(noise**2) + 1e-12
    mixed = speech + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    return np.clip(mixed, -1.0, 1.0)


def colored_noise(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n).astype(np.float32)
    if kind == "white":
        return white
    if kind == "impulsive":  # keyboard/knock-like transients: decaying noise bursts
        out = np.zeros(n, np.float32)
        burst = int(0.06 * 48000)
        env = np.exp(-np.linspace(0, 8, burst)).astype(np.float32)
        for start in rng.integers(0, max(n - burst, 1), size=max(n // 12000, 1)):
            out[start : start + burst] += white[start : start + burst] * env
        return out / (np.abs(out).max() + 1e-9)
    spec = np.fft.rfft(white)
    f = np.maximum(np.fft.rfftfreq(n), 1.0 / n)
    spec /= f if kind == "brown" else np.sqrt(f)  # pink: 1/f power
    out = np.fft.irfft(spec, n).astype(np.float32)
    return out / (np.abs(out).max() + 1e-9)


def to_frames(pcm: np.ndarray, sr: int, frame_ms: int = 10) -> list[rtc.AudioFrame]:
    n = sr * frame_ms // 1000
    i16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
    return [rtc.AudioFrame(i16[i : i + n].tobytes(), sr, 1, n) for i in range(0, len(i16) - n + 1, n)]


def denoise(pcm: np.ndarray, sr: int, denoiser: Denoiser, frame_ms: int = 10) -> np.ndarray:
    """Run the streaming denoiser over a float32 signal; output is re-aligned (latency removed)."""
    bd = BlockDenoiser(denoiser)
    out = np.concatenate([np.frombuffer(bd.process(f).data, np.int16)
                          for f in to_frames(pcm, sr, frame_ms)]).astype(np.float32) / 32768.0
    bd.close()
    out = np.concatenate([out[bd.latency :], np.zeros(bd.latency, np.float32)])
    return np.pad(out, (0, max(len(pcm) - len(out), 0)))[: len(pcm)]


def noise_reduction_db(noisy: np.ndarray, enhanced: np.ndarray, noise_only: np.ndarray) -> float:
    """Energy drop in noise-only samples (higher = more noise removed)."""
    e_in = np.mean(noisy[noise_only] ** 2) + 1e-12
    e_out = np.mean(enhanced[noise_only] ** 2) + 1e-12
    return float(10 * np.log10(e_in / e_out))


def snr_db(clean: np.ndarray, estimate: np.ndarray, region: np.ndarray) -> float:
    """SNR of `estimate` against `clean` inside `region` (speech samples)."""
    err = np.mean((estimate[region] - clean[region]) ** 2) + 1e-12
    return float(10 * np.log10(np.mean(clean[region] ** 2) / err))


async def vad_probabilities(vad: agents_vad.VAD, frames: list[rtc.AudioFrame]) -> np.ndarray:
    """Per-inference-window speech probability, in stream order."""
    stream = vad.stream()
    for f in frames:
        stream.push_frame(f)
    stream.end_input()
    probs = [ev.probability async for ev in stream
             if ev.type == agents_vad.VADEventType.INFERENCE_DONE]
    await stream.aclose()
    return np.asarray(probs, np.float32)


@dataclass
class VadScore:
    false_positive_rate: float  # fraction of noise-only windows flagged as speech
    recall: float  # fraction of reference-speech windows flagged as speech


def score(probs: np.ndarray, reference: np.ndarray, threshold: float = 0.5) -> VadScore:
    n = min(len(probs), len(reference))
    pred, ref = probs[:n] >= threshold, reference[:n]
    return VadScore(
        false_positive_rate=float(pred[~ref].mean()) if (~ref).any() else 0.0,
        recall=float(pred[ref].mean()) if ref.any() else 0.0,
    )
