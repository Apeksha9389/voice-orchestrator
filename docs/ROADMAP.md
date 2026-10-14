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
- [x] **Day 4 — Oct 11 · Hindi + code-mixed speech.** `RoutedSTT` identifies each utterance's
  language with Whisper's encoder and routes it through a sticky `LanguageRouter` (p ≥ 0.7 and
  ≥ 1 s to switch, `ur`→`hi`). The `IndicConformerSTT` plugin (600M, CTC/RNNT, optional `[indic]`
  extra) handles Hindi, falling back to Whisper `hi`. Kokoro TTS is script-aware, and
  `split_on_danda` sentence-chunks Hindi replies. The Hinglish reply prompt is in
  ([MULTILINGUAL.md](MULTILINGUAL.md)).
- [x] **Day 5 — Oct 12 · Telephony ingest & tools.** SIP stack: `docker-compose.sip.yml`
  (livekit-sip + Redis) plus inbound trunk and dispatch-rule JSON. `CallInfo` reads the SIP
  attributes, and the session is built after the caller joins, so phone calls get a SIP profile:
  VAD 0.35 (measured on a simulated G.711 channel), 0.5 s barge-in, 0.5 s endpointing. Tools:
  `end_call` (hangs up after the goodbye plays), `transfer_to_human` (SIP REFER) and `lookup_order`
  (spoken-ID normalisation). An `AwayHandler` silence policy asks once, then hangs up
  ([TELEPHONY.md](TELEPHONY.md)). There is no separate 8 kHz path: the bridge decodes G.711 and
  the agent consumes 48 kHz band-limited audio.
- [x] **Day 6 — Oct 13 · Latency & robustness.** Preemptive generation; sentence plus clause
  chunked Kokoro synthesis (first audio 3.6 s → 1.5 s on a long sentence); Whisper and Kokoro
  warmup in `prewarm`; `FallbackAdapter` across OpenRouter and OpenAI with a 4 s TTFT budget;
  fail-fast retries for local models. `LatencyTracker` logs per-turn response latency
  (EOU + LLM TTFT + TTS TTFB) with p50/p95, and LiveKit's OpenTelemetry spans export over OTLP
  ([LATENCY.md](LATENCY.md)).
- [ ] **Day 7 — Oct 14 · Evaluation harness & docs.** Offline eval: scripted conversations mixed
  with MUSAN-style noise at 0/5/10 dB SNR (RNNoise vs. offline DeepFilterNet3 reference); metrics WER, turn latency p50/p95, interruption
  precision/recall; CI (ruff + pytest); final architecture and benchmark docs.
