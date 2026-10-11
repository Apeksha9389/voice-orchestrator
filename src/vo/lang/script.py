"""Script-level language segmentation for TTS."""

from __future__ import annotations

import unicodedata


def char_lang(c: str) -> str | None:
    """'hi' for Devanagari, 'en' for Latin letters, None for neutral characters."""
    if "ऀ" <= c <= "ॿ" or "꣠" <= c <= "ꣿ":
        return "hi"
    if c.isalpha() and unicodedata.name(c, "").startswith("LATIN"):
        return "en"
    return None


def script_runs(text: str, default: str = "en") -> list[tuple[str, str]]:
    """Split text into (lang, chunk) runs by script. Neutral characters (digits, spaces,
    punctuation) join the run they follow, or the next run when they lead the text."""
    runs: list[list[str]] = []
    pending = ""
    for c in text:
        lang = char_lang(c)
        if lang is None:
            if runs:
                runs[-1][1] += c
            else:
                pending += c
        elif runs and runs[-1][0] == lang:
            runs[-1][1] += c
        else:
            runs.append([lang, pending + c])
            pending = ""
    if not runs:
        return [(default, pending)] if any(c.isalnum() for c in pending) else []
    return [(lang, chunk) for lang, chunk in runs if chunk.strip()]

