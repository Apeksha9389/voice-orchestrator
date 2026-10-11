# Hindi, English and Hinglish

## Speech-to-text: per-utterance language routing

```
VAD segment ─► Whisper language ID (encoder only) ─► LanguageRouter (sticky)
                                                        │
                       ┌────────────────────────────────┴──────────────┐
                       ▼ en                                             ▼ hi
          faster-whisper large-v3-turbo                    IndicConformer 600M (CTC)
          (language forced, no re-detection)               → Devanagari transcript
```

- **Language ID.** `WhisperSTT.detect_language` runs one Whisper encoder pass on the utterance.
  Whisper is already loaded, so this needs no extra model.
- **Hysteresis.** `LanguageRouter` (`src/vo/lang/router.py`) changes the session language only on
  a detection with p ≥ 0.7 over at least 1 s of audio. Language ID on "haan" or "ok" is close to
  random. Without this rule a Hindi caller saying "okay" would be routed to the English model, and
  a short English acknowledgement would flip the session.
- **Aliases.** Whisper labels Hindi speech as Urdu (`ur`) often enough to matter. Both are the same
  spoken language in different scripts, so `ur` routes to Hindi.
- **Hindi engine.** AI4Bharat IndicConformer 600M multilingual is trained on Indian-language
  speech, including code-mixed speech, and outputs Devanagari, with English words transliterated.
  It is MIT-licensed but gated on Hugging Face:

  ```bash
  pip install -e ".[indic]"     # torch, torchaudio, transformers
  huggingface-cli login         # after accepting the model terms on its HF page
  ```

  If the extra or the access token is missing, the agent logs a warning and uses Whisper with
  `language="hi"` for Hindi instead. `VO_INDIC_DECODING=rnnt` is more accurate than the default
  `ctc` and slower.
- **Downstream.** The transcript carries its language, so the multilingual end-of-utterance model
  and the Devanagari backchannel lexicon (Day 3) both apply to Hindi turns.

Turn routing off with `VO_LANGUAGES='["en"]'`.

## LLM: reply in the caller's language

The default system prompt asks for Hindi replies in Devanagari, with common English nouns
(order, payment, account) left in Latin script. That matches how Hinglish speakers write and gives
the TTS a deterministic script boundary.

## Text-to-speech: script-aware Kokoro

Kokoro-82M ships Hindi voices (`hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`) that phonemize
through espeak-ng.

- **Script runs.** Mixed text is split into Latin and Devanagari runs (`src/vo/lang/script.py`), and
  each run is phonemized with its own espeak language. With one espeak voice on mixed text, espeak
  inserts language-switch markers such as `(en)ˈɔːdə(hi)`, and Kokoro would speak the marker
  letters aloud.
- **One voice per reply.** Any Devanagari in a reply selects the Hindi voice for every run, so the
  timbre stays the same mid-sentence. Pure English replies use the English voice.
- **Danda.** LiveKit's sentence tokenizer does not split on `।`, so a Hindi reply would be
  synthesized in one chunk and delay first audio. A TTS text transform (`split_on_danda`) maps it to
  a period. The visible transcript therefore shows `.` where the LLM wrote `।`.

## Configuration

| env | default | |
|---|---|---|
| `VO_LANGUAGES` | `["en","hi"]` | Languages to route between; a single entry disables routing |
| `VO_STT_LANGUAGE` | `en` | Starting session language |
| `VO_HINDI_STT` | `indic` | `whisper` to skip IndicConformer |
| `VO_INDIC_DECODING` | `ctc` | `rnnt` for accuracy |
| `VO_LID_SWITCH_PROB` / `VO_LID_MIN_SECONDS` | 0.7 / 1.0 | Router hysteresis |
| `VO_TTS_HINDI_VOICE` | `hf_alpha` | Kokoro Hindi voice |

## Status

- **Verified here:** routing, hysteresis, the IndicConformer call contract (with a stub model, in
  a torch environment), script segmentation, and real Kokoro Hindi and Hinglish synthesis.
- **Not run here:** Whisper language ID and IndicConformer accuracy, because Hugging Face is
  unreachable from the build environment. The Day 7 eval adds WER on Hindi and English test
  utterances.
