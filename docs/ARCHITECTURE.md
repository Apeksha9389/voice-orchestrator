# Architecture

## Pipeline

```
caller (WebRTC | PSTN ─► livekit-sip) ──► LiveKit room ──► Agent worker (profile: web | sip)
                                             │
   audio in ─► RNNoise (48 kHz) ─────► Silero VAD ─┬─► language ID ─► STT (Whisper en | IndicConformer hi)
                                                  │          │
                                                  │   backchannel filter (stt_node)
                                                  │          │
                                                  │   turn detector (multilingual EOU)
                                                  │          ▼
                                pause / resume / cancel ◄── LLM: FallbackAdapter[OpenRouter → OpenAI], preemptive
                                                             │  tools: lookup_order · transfer_to_human · end_call
                                                             │
                                                  │          ▼
   audio out ◄──────────────────────────── Kokoro TTS (script-aware en/hi, sentence + clause chunked)
```

## Why a cascade on LiveKit Agents

- **Orchestration.** LiveKit Agents (Apache-2.0) owns the hard real-time parts: WebRTC/SIP
  transport, jitter buffering, the `AgentSession` turn state machine, speech scheduling, and
  cancellation of in-flight LLM/TTS on barge-in. Alternatives considered: Pipecat (frame-pipeline
  model, strong but transport-agnostic so more glue for SIP), Vocode (less maintained), and
  speech-to-speech models (lower latency but no control over STT/TTS, poor Hindi, no tool
  determinism). A cascade keeps every stage swappable and observable.
- **Noise suppression.** RNNoise wraps the session audio input (`DenoisedAudioInput`) before VAD,
  STT and barge-in detection. It adds 30 ms of latency at ~6% of one core, and the room input is
  requested at 48 kHz so the hot path never resamples. Benchmarks and the reasoning on
  DeepFilterNet3 are in [NOISE.md](NOISE.md).
- **VAD.** Silero VAD (ONNX, CPU, <1 ms/frame) gates STT and drives interruption detection.
- **STT.** faster-whisper `large-v3-turbo` int8 (CTranslate2). Non-streaming, so it is wrapped in
  `StreamAdapter`, which segments utterances with the VAD and transcribes each on speech end.
  Each utterance is language-identified with Whisper's encoder and routed by a sticky
  `LanguageRouter`: English goes to Whisper, Hindi and Hinglish to AI4Bharat IndicConformer 600M,
  falling back to Whisper `hi` if IndicConformer is unavailable. See [MULTILINGUAL.md](MULTILINGUAL.md).
- **LLM.** OpenAI-compatible client; OpenRouter by default for model choice and fallbacks.
- **TTS.** Kokoro-82M via ONNX Runtime: Apache-2.0, real-time on CPU, 24 kHz. Audio is streamed per
  sentence chunk so time-to-first-audio is bounded by the first sentence, not the full reply.
  Code-mixed replies are split into Latin and Devanagari runs, and each run is phonemized with its
  own espeak language. A reply containing Devanagari uses the Hindi voice throughout.

## Turn-taking and interruptions

Barge-in works as pause first, decide on the transcript:

1. **Pause.** Overlapping user speech of 0.4 s or more (VAD) *pauses* agent audio without
   cancelling the generation.
2. **Backchannel.** `VoiceAgent.stt_node` drops a backchannel ("haan", "okay", "mm-hmm", in
   English, Hinglish or Devanagari) that started while the agent was speaking. No turn is created,
   and the false-interruption timer resumes the agent.
3. **Real interruption.** Anything else cancels the paused speech. The chat context keeps only the
   played text, marked `interrupted`, and a one-turn system note tells the LLM what the user heard.
4. **End of turn.** The multilingual end-of-utterance model decides when the user's turn ends,
   falling back to VAD endpointing if its files are missing.
5. **Hallucination filter.** Whisper segments with high `no_speech_prob` and low `avg_logprob` are
   discarded.

Every backchannel, interruption and resume is written to a structured `turn_event` log. Details and
tuning are in [TURN_TAKING.md](TURN_TAKING.md).

## Latency and robustness

Each latency stage is attacked separately:

- **End-of-turn wait.** The end-of-utterance model keeps the minimum silence wait at 0.4 s.
- **LLM first token.** Preemptive generation overlaps the LLM's time to first token with the
  end-of-turn decision.
- **TTS first audio.** Kokoro synthesizes each sentence clause by clause, so first audio waits only
  for the first clause. Measured: 3.6 s → 1.5 s on a long sentence, on 2 vCPUs.
- **Cold start.** Whisper and Kokoro are warmed in `prewarm`, which removes about 1.3 s from the
  first turn.

**Failover.** The LLM sits behind `FallbackAdapter`: OpenRouter first, OpenAI if it produces no
first token within 4 s. Local models fail fast instead of retrying.

**Observability.** `LatencyTracker` logs per-turn response latency and its parts, plus p50/p95 at
call end. LiveKit's OpenTelemetry spans export over OTLP when `VO_OTEL_ENDPOINT` is set.

Details are in [LATENCY.md](LATENCY.md).

## Telephony and tools

Phone calls arrive through LiveKit's SIP bridge as ordinary room participants. The session is
built once the caller is known:

- **SIP profile.** SIP callers get a lower VAD threshold (0.35, measured on simulated G.711
  speech), a longer barge-in minimum (0.5 s, for line echo) and a longer endpointing minimum.
- **Tools.** The LLM can look up orders, transfer the caller with a SIP REFER, and end the call
  with a hang-up that waits for the goodbye to finish playing.
- **Silence policy.** A silent caller is asked once whether they are still there, then the agent
  says goodbye and hangs up.

Setup, the profile measurements and the tool contracts are in [TELEPHONY.md](TELEPHONY.md).

## Configuration

All settings are environment variables prefixed `VO_` (see `.env.example`, `src/vo/config.py`).
