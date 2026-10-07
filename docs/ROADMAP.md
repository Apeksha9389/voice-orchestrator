# Roadmap

One milestone per day. Each milestone ships with tests and a docs update, then is pushed to `main`.
Scheduled build runs pick the first unchecked milestone.

- [x] **Day 1 — Oct 8 · Core cascade pipeline.** LiveKit Agents worker; Silero VAD; faster-whisper
  STT (VAD-segmented via `StreamAdapter`); OpenRouter/OpenAI LLM; Kokoro-82M ONNX TTS with streamed
  PCM; per-turn metrics; local LiveKit server via docker compose.
- [ ] **Day 2 — Oct 9 · Noise suppression front-end.** `DenoisedAudioInput` chained in front of
  VAD/STT using DeepFilterNet3 (48 kHz, ~10 ms hop) with RNNoise fallback; per-frame SNR estimate;
  config switch; unit test on synthetic speech + babble/white noise showing VAD false-trigger drop.
- [ ] **Day 3 — Oct 10 · Interruption & turn-taking.** LiveKit multilingual EOU turn detector;
  false-interruption recovery (resume after `false_interruption_timeout`); backchannel filter
  ("haan", "okay", "hmm") that does not cut agent speech; min-words gate; interruption event log
  with agent-speech truncation point written into chat context.
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
  with MUSAN-style noise at 0/5/10 dB SNR; metrics WER, turn latency p50/p95, interruption
  precision/recall; CI (ruff + pytest); final architecture and benchmark docs.
