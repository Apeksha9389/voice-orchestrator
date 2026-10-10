# Architecture

## Pipeline

```
caller (WebRTC / SIP) ──► LiveKit room ──► Agent worker
                                             │
   audio in ─► RNNoise (48 kHz) ─────► Silero VAD ─┬─► STT (faster-whisper / IndicConformer)
                                                  │          │
                                                  │   backchannel filter (stt_node)
                                                  │          │
                                                  │   turn detector (multilingual EOU)
                                                  │          ▼
                                pause / resume / cancel ◄── LLM (OpenRouter / OpenAI, streamed)
                                                             │
                                                  │          ▼
   audio out ◄──────────────────────────── Kokoro TTS (sentence-chunked PCM)
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
  IndicConformer is added for Hindi on Day 4.
- **LLM.** OpenAI-compatible client; OpenRouter by default for model choice and fallbacks.
- **TTS.** Kokoro-82M via ONNX Runtime: Apache-2.0, real-time on CPU, 24 kHz. Audio is streamed per
  sentence chunk so time-to-first-audio is bounded by the first sentence, not the full reply.

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

## Configuration

All settings are environment variables prefixed `VO_` (see `.env.example`, `src/vo/config.py`).
