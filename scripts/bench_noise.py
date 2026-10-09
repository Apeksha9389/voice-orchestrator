"""Benchmark the noise-suppression front-end against synthetic noise.

Speech comes from Kokoro TTS so the run is self-contained. Prints a markdown table of noise
reduction, speech SNR, and Silero VAD false-positive rate / recall before vs after denoising.

    python scripts/bench_noise.py --snr 0 5 10 --noise white pink impulsive
"""

import argparse
import asyncio

import numpy as np
from kokoro_onnx import Kokoro
from livekit.plugins import silero

from vo.audio.denoise import build_denoiser
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

SR = 48000
TEXT = ("Hello, I would like to check the status of my order placed last week. "
        "Can you help me with that please?")


async def main(args: argparse.Namespace) -> None:
    tts = Kokoro(args.model, args.voices)
    sp, tts_sr = tts.create(TEXT, voice="af_heart")
    sp = np.interp(np.arange(0, len(sp), tts_sr / SR), np.arange(len(sp)), sp).astype(np.float32)
    pad = np.zeros(SR * 2, np.float32)
    clean = np.concatenate([pad, sp, pad])
    speech = np.zeros(len(clean), bool)
    speech[len(pad) : len(pad) + len(sp)] = True

    vad = silero.VAD.load()
    ref = await vad_probabilities(vad, to_frames(clean, SR)) >= 0.5
    rng = np.random.default_rng(args.seed)

    print(f"denoiser: {args.denoiser}\n")
    print("| noise | input SNR | noise reduction | speech SNR in→out | VAD FPR in→out | VAD recall in→out |")
    print("|---|---|---|---|---|---|")
    for kind in args.noise:
        for target in args.snr:
            noisy = mix_at_snr(clean, colored_noise(len(clean), kind, rng), target)
            enhanced = denoise(noisy, SR, build_denoiser(args.denoiser))
            a = score(await vad_probabilities(vad, to_frames(noisy, SR)), ref)
            b = score(await vad_probabilities(vad, to_frames(enhanced, SR)), ref)
            print(f"| {kind} | {target} dB | {noise_reduction_db(noisy, enhanced, ~speech):.1f} dB "
                  f"| {snr_db(clean, noisy, speech):.1f} → {snr_db(clean, enhanced, speech):.1f} dB "
                  f"| {a.false_positive_rate:.3f} → {b.false_positive_rate:.3f} "
                  f"| {a.recall:.3f} → {b.recall:.3f} |")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--denoiser", default="rnnoise")
    p.add_argument("--snr", type=float, nargs="+", default=[0, 5, 10])
    p.add_argument("--noise", nargs="+", default=["white", "pink", "impulsive"])
    p.add_argument("--model", default="models/kokoro-v1.0.onnx")
    p.add_argument("--voices", default="models/voices-v1.0.bin")
    p.add_argument("--seed", type=int, default=0)
    asyncio.run(main(p.parse_args()))
