"""Lexical backchannel classifier for English, Hindi and Hinglish.

A backchannel ("haan", "okay", "mm-hmm") said while the agent is talking signals that the user is
listening. It should not cut the agent off or become a user turn. Explicit stop words always count
as a real interruption.
"""

from __future__ import annotations

import re
import unicodedata

BACKCHANNEL_PHRASES = frozenset({
    # English
    "yeah", "yes", "yep", "yup", "ya", "ok", "okay", "right", "sure", "alright", "all right",
    "uh huh", "mhm", "mm hm", "hm", "mm", "um", "uh", "i see", "got it", "cool", "nice", "great",
    "oh", "ah", "oh okay", "ah okay", "yeah yeah", "okay okay", "right right", "makes sense",
    # Hindi / Hinglish (romanised)
    "haan", "han", "ha", "haanji", "haan ji", "ji", "ji haan", "accha", "acha", "achha", "achcha",
    "theek", "thik", "theek hai", "thik hai", "sahi", "sahi hai", "hmm ji", "acha ji", "accha ji",
    # Hindi (Devanagari)
    "हाँ", "हां", "हा", "जी", "जी हाँ", "हाँ जी", "अच्छा", "ठीक", "ठीक है", "सही", "सही है", "हम्म",
})

INTERRUPT_WORDS = frozenset({
    "stop", "wait", "hold", "no", "nope", "sorry", "excuse", "but", "actually", "hello",
    "ruko", "ruk", "ruka", "nahi", "nahin", "mat", "suno", "रुको", "नहीं", "सुनो", "रुकिए",
})

_REPEATS = re.compile(r"(\w)\1{2,}", re.UNICODE)  # "hmmm" -> "hm", "okayyy" -> "okay"
_FILLER = re.compile(r"^(?:h?m+h?m*|u+h+|u+m+)$")


def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower().replace("-", " ")
    # strip punctuation/symbols by Unicode category; \w would also strip Devanagari vowel signs
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    text = _REPEATS.sub(r"\1", text)
    tokens = []
    for tok in text.split():
        tokens.append("hm" if _FILLER.match(tok) and tok.startswith(("h", "m")) else tok)
    return tokens


def is_backchannel(text: str, *, max_words: int = 3) -> bool:
    tokens = normalize(text)
    if not tokens or len(tokens) > max_words or INTERRUPT_WORDS.intersection(tokens):
        return False
    if " ".join(tokens) in BACKCHANNEL_PHRASES:
        return True
    return all(t in BACKCHANNEL_PHRASES for t in tokens)
