"""Overlap tracking and the interruption event log."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

log = logging.getLogger("vo.turns")

EventKind = Literal["interruption", "backchannel", "false_interruption"]


class OverlapTracker:
    """Answers "did this user utterance start while the agent was talking?".

    The VAD pauses agent audio (agent_state -> listening) only after `min_duration` of user speech,
    so a short grace window covers the race between that pause and the utterance-start event.
    """

    def __init__(self, grace: float = 0.5, clock=time.monotonic) -> None:
        self._grace = grace
        self._clock = clock
        self._speaking = False
        self._stopped_at = float("-inf")
        self.overlapping = False

    def on_agent_state(self, new_state: str) -> None:
        if self._speaking and new_state != "speaking":
            self._stopped_at = self._clock()
        self._speaking = new_state == "speaking"

    def begin_utterance(self) -> None:
        self.overlapping = self._speaking or self._clock() - self._stopped_at <= self._grace


@dataclass
class TurnEvent:
    kind: EventKind
    text: str = ""
    resumed: bool | None = None
    at: float = field(default_factory=time.time)


class InterruptionLog:
    """Structured record of barge-ins, filtered backchannels and false-interruption resumes."""

    def __init__(self, maxlen: int = 1000) -> None:
        self.events: list[TurnEvent] = []
        self._maxlen = maxlen

    def record(self, kind: EventKind, text: str = "", resumed: bool | None = None) -> TurnEvent:
        ev = TurnEvent(kind=kind, text=text, resumed=resumed)
        self.events.append(ev)
        del self.events[: -self._maxlen]
        log.info("turn_event %s", json.dumps(asdict(ev), ensure_ascii=False))
        return ev

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for ev in self.events:
            out[ev.kind] = out.get(ev.kind, 0) + 1
        return out
