# Architecture

## Pipeline

```
caller (WebRTC / SIP) ──► LiveKit room ──► Agent worker
                                             │
   audio in ─► [noise suppression] ─► Silero VAD ─┬─► STT (faster-whisper / IndicConformer)
                                                  │          │
                                                  │     turn detector (EOU)
                                                  │          ▼
                                       interruption ◄── LLM (OpenRouter / OpenAI, streamed)
                                          control            │
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
- **VAD.** Silero VAD (ONNX, CPU, <1 ms/frame) gates STT and drives interruption detection.
- **STT.** faster-whisper `large-v3-turbo` int8 (CTranslate2). Non-streaming, so it is wrapped in
  `StreamAdapter`, which segments utterances with the VAD and transcribes each on speech end.
  IndicConformer is added for Hindi on Day 4.
- **LLM.** OpenAI-compatible client; OpenRouter by default for model choice and fallbacks.
- **TTS.** Kokoro-82M via ONNX Runtime: Apache-2.0, real-time on CPU, 24 kHz. Audio is streamed per
  sentence chunk so time-to-first-audio is bounded by the first sentence, not the full reply.

## Turn-taking and interruptions

The session runs with VAD-based interruption (`min_duration` 0.4 s): user speech while the agent is
speaking cancels playback and the pending generation. Day 3 layers on semantic end-of-utterance
detection, false-interruption resume, and backchannel suppression.

## Configuration

All settings are environment variables prefixed `VO_` (see `.env.example`, `src/vo/config.py`).
