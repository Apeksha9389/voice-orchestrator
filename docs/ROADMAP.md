# Roadmap

One milestone per day. Each milestone ships with tests and a docs update, then is pushed to `main`.
Scheduled build runs pick the first unchecked milestone.

- [x] **Day 1 — Oct 8 · Core cascade pipeline.** LiveKit Agents worker; Silero VAD; faster-whisper
  STT (VAD-segmented via `StreamAdapter`); OpenRouter/OpenAI LLM; Kokoro-82M ONNX TTS with streamed
  PCM; per-turn metrics; local LiveKit server via docker compose.
- [x] **Day 2 — Oct 9 · Noise suppression front-end.** `DenoisedAudioInput` wraps the session
  input with streaming RNNoise (48 kHz, 10 ms blocks, 30 ms total latency) through a
  rate/frame-size adapter, and tracks per-frame SNR and speech probability. Selected with
  `VO_DENOISER` and degrades to pass-through if the library is missing. `scripts/bench_noise.py`
  measures noise reduction, speech SNR, and VAD FPR/recall under white, pink and impulsive noise
  ([NOISE.md](NOISE.md)). DeepFilterNet3 was moved to the Day 7 offline reference because its pip
  package cannot stream.
- [x] **Day 3 — Oct 10 · Interruption & turn-taking.** The agent pauses on overlapping speech
  and resumes on a false interruption. `VoiceAgent.stt_node` filters backchannels (English,
  Hinglish, Devanagari lexicon; overlap-aware, with stop-word override). Whisper hallucinations
  are filtered. An interrupted reply gets a context-repair note naming what the user heard. The
  multilingual end-of-utterance model runs with a VAD fallback, and a structured `turn_event` log
  records each event ([TURN_TAKING.md](TURN_TAKING.md)). LiveKit's `min_words` was dropped: batch
  STT has no interim words, so it would delay every barge-in.
- [ ] **Day 4 — Oct 11 · Hindi + code-mixed speech.** AI4Bharat IndicConformer (hi) STT plugin;
  language-ID router (whisper LID on first 1.5 s) selecting en/hi model; Kokoro `hf_alpha` Hindi
  voice; Hinglish prompt; tests on per-language routing.
- [ ] **Day 5 — Oct 12 · Telephony ingest & tools.** LiveKit SIP inbound trunk + dispatch rule
  config; 8 kHz path (VAD at 8k, upsample for STT); function tools (end call, transfer, lookup);
  graceful call teardown.
- [ ] **Day 6 — Oct 13 · Latency & robustness.** Preemptive generation; sentence-level TTS
  streaming; LLM fallback chain (OpenRouter → OpenAI); OpenTelemetry traces per turn
  (VAD→EOU→STT→LLM TTFT→TTS TTFB); retry/backoff on providers.
- [ ] **Day 7 — Oct 14 · Evaluation harness & docs.** Offline eval: scripted conversations mixed
  with MUSAN-style noise at 0/5/10 dB SNR (RNNoise vs. offline DeepFilterNet3 reference); metrics WER, turn latency p50/p95, interruption
  precision/recall; CI (ruff + pytest); final architecture and benchmark docs.
