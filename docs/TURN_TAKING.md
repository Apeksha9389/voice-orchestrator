# Turn-taking and interruptions

## Goals

| situation | desired behaviour |
|---|---|
| User says "haan" / "okay" / "mm-hmm" while the agent talks | Agent keeps talking; no user turn |
| User says "wait, how much is it?" while the agent talks | Agent stops within ~0.4 s and answers the new question |
| Cough, door slam or short noise while the agent talks | Agent pauses briefly, then resumes where it stopped |
| User pauses mid-sentence ("I want to book a… flight to Delhi") | Agent waits instead of jumping in |
| Whisper hallucinates "Thank you." from noise | Ignored |

## Mechanism

```
user speech ─► VAD ≥ 0.4 s while agent speaks ─► PAUSE agent audio (not cancel)
                                                   │
             VAD end of speech ─► STT final transcript
                                                   │
       ┌──────────────── stt_node filter ──────────┴───────────────┐
       │ overlap && backchannel(text)            otherwise          │
       ▼                                         ▼                  │
  drop transcript                          forward transcript        │
  (log "backchannel")                      → cancel paused speech    │
       │                                   → chat ctx keeps only the │
  no transcript, no turn                     played text, marked     │
       ▼                                     interrupted             │
  false-interruption timer (1.0 s)         → system note to LLM:     │
  → RESUME agent audio                       "they heard only: …"    │
    (log "false_interruption")             → turn detector → reply   │
```

### 1. Pause first, decide later

LiveKit's `resume_false_interruption` pauses playout on VAD speech instead of cancelling the
generation. The first 400 ms of overlap decides the barge-in latency (`VO_INTERRUPT_MIN_DURATION`).
Anything shorter, like a click or a breath, never touches the agent. Noise suppression (Day 2)
keeps the VAD from firing on background noise in the first place.

### 2. Backchannel filter on the final transcript

`VoiceAgent.stt_node` (`src/vo/turns/agent.py`) wraps the default STT node:

- `OverlapTracker` notes whether each utterance *started* while the agent was speaking. A 0.5 s
  grace window covers the race with the VAD pause, which flips the agent to `listening`.
- When an utterance overlaps and its transcript is a backchannel (`src/vo/turns/backchannel.py`),
  the transcript is dropped. LiveKit then sees speech with no words, so no turn is committed, and
  its false-interruption timer resumes the paused audio.
- The lexicon covers English, romanised Hindi and Devanagari, at up to 3 words (`VO_BACKCHANNEL_MAX_WORDS`).
  Text is normalised for case, punctuation, repeated letters (`hmmm`, `okayyy`) and filler
  spellings (`mhm`, `mm`, `hmm`).
- Explicit stop words ("wait", "no", "stop", "ruko", "nahi", "रुको") always count as an
  interruption, even inside a short phrase.
- The same word *without* overlap ("okay" in answer to a question) is a normal turn.

**Why not LiveKit's `min_words`?** The STT is batch: faster-whisper runs on each VAD segment, so
there are no interim words while the user is still speaking. A word-count gate would hold every
barge-in until the user stops talking, and the agent would talk over them. Filtering on the final
transcript keeps the fast VAD pause and still rejects backchannels.

**Why not `interruption.mode="adaptive"`?** LiveKit's adaptive model runs on LiveKit Cloud
inference. This project stays self-hostable.

### 3. Truncation and context repair

When a real interruption cancels the agent, LiveKit stores only the played portion of the reply
in the chat context and sets `interrupted=True`. Before the next reply,
`VoiceAgent.on_user_turn_completed` adds a one-turn system note naming what the user actually
heard. The LLM then neither assumes the user heard the rest nor repeats it.

### 4. End of utterance

`turn_detection` uses LiveKit's open-weights multilingual end-of-utterance model (local ONNX,
covers Hindi). It scores the transcript in conversation context and stretches the silence wait
from 0.4 s up to 2.5 s when the sentence looks unfinished. If the model files are missing
(`python -m vo.agent download-files`), the agent falls back to VAD endpointing and logs a warning.
The plugin is deprecated in favour of LiveKit's hosted `inference.TurnDetector`, so it stays
pinned to the 1.8.5 local model.

### 5. Whisper hallucinations

Segments where Whisper's own `no_speech_prob > 0.6` and `avg_logprob < -1.0` are discarded. This
removes the "Thank you." / "you" outputs it produces on noise or echo.

## Observability

`InterruptionLog` writes one structured log line per event:

```
turn_event {"kind": "backchannel", "text": "haan ji", "resumed": null, "at": 1760000000.1}
turn_event {"kind": "interruption", "text": "We have three plans. The basic", ...}
turn_event {"kind": "false_interruption", "resumed": true, ...}
```

The counts are logged at session end next to the usage summary. Day 7 turns them into
interruption precision/recall on scripted conversations.

## Tuning

| env | default | effect |
|---|---|---|
| `VO_INTERRUPT_MIN_DURATION` | 0.4 | Speech needed to pause the agent. Lower is snappier and more noise-sensitive. |
| `VO_FALSE_INTERRUPTION_TIMEOUT` | 1.0 | Silence before resuming after a pause. Keep it above STT p95 latency, or a real barge-in can briefly resume before its transcript lands. Use ~1.5 on CPU-only Whisper. |
| `VO_BACKCHANNEL_MAX_WORDS` | 3 | Longest utterance that can count as a backchannel. |
| `VO_TURN_DETECTOR` | multilingual | `vad` to disable the end-of-utterance model. |
| `VO_ENDPOINTING_MIN_DELAY` / `MAX_DELAY` | 0.4 / 2.5 | Silence bounds before the end of the user's turn. |

## Limits

- The backchannel decision is lexical. Prosody-aware classification (pitch, duration) would also
  catch untranscribed or ambiguous acknowledgements. That is the job LiveKit's adaptive model does in
  the cloud.
- Resume latency for a backchannel is `false_interruption_timeout` after the user stops, so the
  agent goes briefly silent. A streaming STT with interim results would allow deciding before the
  pause.
