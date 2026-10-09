# Noise suppression

## Where it sits

```
mic / SIP ─► LiveKit room input (48 kHz mono) ─► DenoisedAudioInput ─► Silero VAD ─► STT
                                                     │                    │
                                                     └─ SNR / speech prob └─ barge-in detection
```

`DenoisedAudioInput` (`src/vo/audio/denoise.py`) wraps `session.input.audio` after the session
starts, so the same path covers room and console IO. Everything downstream (VAD, interruption
detection, STT, turn detection) sees the cleaned signal.

`BlockDenoiser` adapts any frame size and sample rate to the model's fixed block. It returns exactly as
many samples as it receives, so it can stay inline in a real-time stream:

- The room input is requested at 48 kHz, RNNoise's native rate, so no resampling happens in the hot path.
- At other rates it resamples with LiveKit's soxr-based `rtc.AudioResampler` on the way in and
  back out.
- The output is primed with one block of silence, so after warm-up the output stream never runs short.

## Backend: RNNoise

| property | value |
|---|---|
| model | RNNoise (GRU + DSP, BSD-3), via `pyrnnoise` |
| block | 480 samples (10 ms) at 48 kHz |
| added latency | 30 ms (10 ms buffering + 20 ms algorithmic, measured by cross-correlation) |
| CPU | ~3 ms per 50 ms frame on one core (RTF ≈ 0.06) |
| by-products | per-block speech probability, frame SNR vs. a tracked noise floor |

Select it with `VO_DENOISER=rnnoise` (the default) or turn it off with `VO_DENOISER=none`. If the native
library fails to load, the agent logs a warning and runs without suppression rather than failing
the call.

### Why not DeepFilterNet3 in the live path

DeepFilterNet3 gives better quality, but its pip package runs the PyTorch model on whole
utterances. The GRU state is not carried across calls, so streaming it means chunking with
latency equal to the chunk and artifacts at chunk boundaries. Its real-time implementation is the
Rust/tract `libDF` runtime, which has no Python binding for the model. DeepFilterNet3 is therefore
used only as an offline quality reference in the Day 7 evaluation harness. The `Denoiser` protocol
is small (`process_block`, `block_size`, `sample_rate`, `delay`), so adding it later through a
native binding or ONNX streaming export is a single class.

## Measured effect

`python scripts/bench_noise.py` produces this table. Kokoro TTS generates the speech, padded with
2 s of noise-only audio on each side, then mixed with synthetic noise at the given SNR. VAD metrics
use Silero windows, with the clean signal's VAD decisions as reference.

| noise | input SNR | noise reduction | speech SNR in→out | VAD FPR in→out | VAD recall in→out |
|---|---|---|---|---|---|
| white | 0 dB | 65.8 dB | -0.2 → 11.5 dB | 0.007 → 0.000 | 0.940 → 0.960 |
| white | 5 dB | 41.7 dB | 4.8 → 13.3 dB | 0.007 → 0.000 | 0.966 → 0.960 |
| white | 10 dB | 40.9 dB | 9.8 → 14.9 dB | 0.007 → 0.000 | 0.980 → 0.993 |
| pink | 0 dB | 58.2 dB | -0.1 → 9.3 dB | 0.007 → 0.007 | 0.980 → 0.940 |
| pink | 5 dB | 46.7 dB | 4.8 → 12.9 dB | 0.036 → 0.007 | 0.993 → 0.993 |
| pink | 10 dB | 40.4 dB | 9.5 → 14.3 dB | 0.021 → 0.007 | 0.973 → 0.993 |
| impulsive | 0 dB | 40.7 dB | -0.6 → 14.0 dB | 0.007 → 0.000 | 0.980 → 0.987 |
| impulsive | 5 dB | 52.6 dB | 5.8 → 16.4 dB | 0.000 → 0.000 | 1.000 → 1.000 |
| impulsive | 10 dB | 64.4 dB | 9.2 → 16.6 dB | 0.007 → 0.000 | 0.993 → 0.993 |

What this shows:

- **Speech SNR improves by 5–14 dB**, and noise-only regions drop by 40–65 dB.
- **False VAD triggers on noise go to ~0.** The barge-in path depends on this, because a false
  VAD trigger while the agent speaks is a false interruption.
- **Silero VAD is already robust to stationary synthetic noise**, so the false-positive baseline is
  low. The gain will be larger on real non-stationary noise (traffic, fans, keyboards).
- **One regression:** recall drops 4 points at 0 dB pink noise, where RNNoise eats some low-energy
  speech onsets.

Limits: these noises are synthetic. RNNoise does not remove competing speech (TV, other talkers),
which needs background voice cancellation; it is listed under future work. Day 7 re-runs this on
recorded noise and adds STT WER.
