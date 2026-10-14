# Latency and robustness

## Where the time goes

The caller-perceived delay is **user stops speaking → first agent audio**:

```
response = endpointing/EOU wait + LLM time-to-first-token + TTS time-to-first-audio
```

The STT runs during the endpointing wait, because the batch decode starts on VAD end of speech.
Each stage is attacked separately.

| stage | technique | where |
|---|---|---|
| EOU wait | Multilingual end-of-utterance model: 0.4 s minimum, extended only when the sentence looks unfinished | Day 3 |
| LLM TTFT | **Preemptive generation**: the LLM starts on the final transcript while the EOU decision is pending, and the result is discarded if the transcript changes | `turn_handling.preemptive_generation` |
| TTS first audio | **Sentence and clause chunking**: LiveKit's `StreamAdapter` splits the LLM stream into sentences, and `KokoroTTS` synthesizes each sentence clause by clause | `vo.plugins.kokoro_tts`, `vo.lang.script.clauses` |
| cold start | **Warmup** in `prewarm`: one Whisper decode and one Kokoro synthesis before the first call | `vo.agent.prewarm` |

## TTS measurements

Measured on the 2-vCPU build machine, Kokoro-82M ONNX on CPU (RTF ≈ 0.5). This is time to first
audio frame, warm, averaged over 3 runs.

| reply | whole reply as one chunk | sentence-chunked | + clause-chunked |
|---|---|---|---|
| 3 sentences, about 10 s of audio | 4.5 s | 2.1 s | 2.1 s (first sentence has no clause break) |
| 1 long sentence with commas, about 7 s of audio | 3.6 s | 3.6 s | **1.5 s** |

| warmup | first call | later calls |
|---|---|---|
| Kokoro, same 3-sentence reply | 5.9 s | 4.5 s |

**Why clause chunking matters.** Kokoro synthesizes a whole phoneme batch, up to about 510
phonemes, before returning any audio. Time to first audio therefore grows linearly with chunk
length, and sentence chunking alone does not help a long first sentence. Splitting at `, ; : —`,
once a piece has at least 15 characters, matches the pause Kokoro already inserts there, so the
prosody holds. On a GPU or a modern 8-core CPU the absolute numbers shrink by roughly 4–10×, and
the ratios still hold.

**Prompting for short openers.** The system prompt asks for one or two short sentences. A short
first clause is the cheapest latency win available.

## Robustness

| failure | handling |
|---|---|
| OpenRouter slow or down | `llm.FallbackAdapter([OpenRouter, OpenAI])`. If a provider produces no first token within 4 s (`VO_LLM_ATTEMPT_TIMEOUT`) it is marked unavailable and the next one is used; it is probed in the background and restored when healthy. One retry per provider. The fallback only exists when both keys are set |
| No LLM key | Fails at session start with a clear error, not mid-call |
| Local STT/TTS error | `max_retry=0`: a local model that throws will throw again, so the turn fails fast instead of stalling the caller through retries |
| IndicConformer / turn detector / RNNoise unavailable | Each degrades to its fallback (Whisper `hi` / VAD endpointing / pass-through) with a warning (Days 2–4) |

## Observability

**Per-turn latency.** `LatencyTracker` joins LiveKit's `EOUMetrics`, `LLMMetrics` and
`TTSMetrics` on `speech_id`. It logs each turn and writes p50/p95 at call end:

```
turn SP_3a… response=1180ms (eou=520 llm=410 tts=250)
call end … latency={'response': {'p50': 1.05, 'p95': 1.42, 'n': 12}, 'eou': {…}, …}
```

Cancelled LLM attempts, such as a discarded preemptive generation, are ignored. Only the first TTS
segment counts, because that is what the caller waits for. With preemptive generation on, the sum
is an upper bound, since the LLM already started during the EOU wait.

**Traces.** Set `VO_OTEL_ENDPOINT=http://localhost:4318` to export LiveKit Agents' built-in
OpenTelemetry spans over OTLP/HTTP. The spans cover user turn, STT, EOU, `on_user_turn_completed`,
LLM, tool calls and TTS. Any OTLP backend works (Jaeger, Grafana Tempo, Langfuse):

```bash
docker run -d -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one:latest
VO_OTEL_ENDPOINT=http://localhost:4318 python -m vo.agent dev
```
