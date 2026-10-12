"""Agent with overlap-aware turn handling.

Barge-in flow (LiveKit with `resume_false_interruption`):
1. VAD detects user speech of at least `min_duration` while the agent talks, and agent audio is
   *paused*, not cancelled.
2a. The transcript is a backchannel: `stt_node` drops it. With no transcript there is no user turn,
    so the false-interruption timer resumes the agent where it paused.
2b. Anything else: the final transcript cancels the paused speech. LiveKit truncates the assistant
    message to what was actually played and marks it `interrupted`. `on_user_turn_completed` then
    tells the LLM where it was cut off.
"""

from __future__ import annotations

from collections.abc import AsyncIterable

from livekit import rtc
from livekit.agents import Agent, ModelSettings, llm, stt

from vo.turns.backchannel import is_backchannel
from vo.turns.tracking import InterruptionLog, OverlapTracker


def last_interrupted_reply(ctx: llm.ChatContext) -> llm.ChatMessage | None:
    for item in reversed(ctx.items):
        if isinstance(item, llm.ChatMessage) and item.role == "assistant":
            return item if item.interrupted else None
    return None


class VoiceAgent(Agent):
    def __init__(self, *, instructions: str, tracker: OverlapTracker, events: InterruptionLog,
                 backchannel_max_words: int = 3, tools: list[llm.Tool] | None = None) -> None:
        super().__init__(instructions=instructions, tools=tools)
        self._tracker = tracker
        self._events = events
        self._bc_max_words = backchannel_max_words

    def filter_transcript(self, ev: stt.SpeechEvent) -> bool:
        """True to forward the event to the session, False to drop it."""
        if ev.type == stt.SpeechEventType.START_OF_SPEECH:
            self._tracker.begin_utterance()
            return True
        if ev.type != stt.SpeechEventType.FINAL_TRANSCRIPT or not ev.alternatives:
            return True
        text = ev.alternatives[0].text
        if self._tracker.overlapping and is_backchannel(text, max_words=self._bc_max_words):
            self._events.record("backchannel", text)
            return False
        return True

    async def stt_node(self, audio: AsyncIterable[rtc.AudioFrame],
                       model_settings: ModelSettings) -> AsyncIterable[stt.SpeechEvent]:
        async for ev in Agent.default.stt_node(self, audio, model_settings):
            if self.filter_transcript(ev):
                yield ev

    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext,
                                     new_message: llm.ChatMessage) -> None:
        cut = last_interrupted_reply(turn_ctx)
        if cut is None:
            return
        heard = (cut.text_content or "").strip()
        self._events.record("interruption", heard)
        # applies to this generation only; the persistent context keeps the truncated reply
        turn_ctx.add_message(
            role="system",
            content=(
                "The user interrupted your previous reply. They heard only: "
                f"\"{heard or '(nothing)'}\". Respond to what they just said; repeat earlier "
                "content only if it is still relevant."
            ),
        )
