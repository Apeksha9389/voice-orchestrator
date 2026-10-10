import pytest
from livekit.agents import Agent, llm, stt

from vo.plugins.whisper_stt import keep_segment
from vo.turns.agent import VoiceAgent, last_interrupted_reply
from vo.turns.backchannel import is_backchannel, normalize
from vo.turns.tracking import InterruptionLog, OverlapTracker


@pytest.mark.parametrize("text", [
    "Okay.", "okay okay", "Mm-hmm", "Hmmm...", "uh huh", "Yeah, right.", "I see",
    "haan", "Haan ji", "achha", "theek hai", "हाँ जी", "अच्छा", "ठीक है",
])
def test_backchannels(text):
    assert is_backchannel(text)


@pytest.mark.parametrize("text", [
    "", "wait", "no stop", "okay but what about my refund", "haan lekin price kya hai",
    "ruko", "sorry", "रुको", "what is the status of my order", "yeah yeah yeah yeah",
])
def test_not_backchannels(text):
    assert not is_backchannel(text)


def test_normalize_collapses_fillers_and_repeats():
    assert normalize("Hmmmm, okayyy!") == ["hm", "okay"]


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_overlap_tracker_grace_window():
    clock = Clock()
    tr = OverlapTracker(grace=0.5, clock=clock)
    tr.begin_utterance()
    assert not tr.overlapping
    tr.on_agent_state("speaking")
    tr.begin_utterance()
    assert tr.overlapping
    clock.t = 10.0
    tr.on_agent_state("listening")  # VAD paused the agent
    clock.t = 10.3
    tr.begin_utterance()
    assert tr.overlapping
    clock.t = 11.0
    tr.begin_utterance()
    assert not tr.overlapping


def final(text: str) -> stt.SpeechEvent:
    return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                           alternatives=[stt.SpeechData(language="en", text=text)])


def start() -> stt.SpeechEvent:
    return stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def agent(clock):
    tracker, events = OverlapTracker(clock=clock), InterruptionLog()
    return VoiceAgent(instructions="x", tracker=tracker, events=events), tracker, events


async def test_stt_node_drops_backchannel_only_during_overlap(agent, clock, monkeypatch):
    va, tracker, events = agent

    async def fake_default_stt_node(self, audio, model_settings):
        tracker.on_agent_state("speaking")
        yield start()
        yield final("haan ji")  # listening noise while the agent talks: dropped
        yield start()
        yield final("wait, cancel that")  # real barge-in: forwarded
        tracker.on_agent_state("listening")
        clock.t = 5.0
        yield start()
        yield final("okay")  # answer after the agent finished: forwarded

    monkeypatch.setattr(Agent.default, "stt_node", fake_default_stt_node)
    out = [ev async for ev in va.stt_node(None, None)]
    texts = [ev.alternatives[0].text for ev in out
             if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert texts == ["wait, cancel that", "okay"]
    assert events.counts() == {"backchannel": 1}


async def test_interrupted_reply_adds_context_note(agent):
    va, _, events = agent
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="tell me about plans")
    ctx.add_message(role="assistant", content="We have three plans. The basic", interrupted=True)
    assert last_interrupted_reply(ctx) is not None

    await va.on_user_turn_completed(ctx, llm.ChatMessage(role="user", content=["just the price"]))
    note = ctx.items[-1]
    assert note.role == "system" and "The basic" in note.text_content
    assert events.counts() == {"interruption": 1}


async def test_completed_reply_adds_nothing(agent):
    va, _, events = agent
    ctx = llm.ChatContext()
    ctx.add_message(role="assistant", content="Done.")
    n = len(ctx.items)
    await va.on_user_turn_completed(ctx, llm.ChatMessage(role="user", content=["thanks"]))
    assert len(ctx.items) == n and events.counts() == {}


def test_whisper_hallucination_filter():
    assert keep_segment(no_speech_prob=0.1, avg_logprob=-0.3)
    assert keep_segment(no_speech_prob=0.9, avg_logprob=-0.4)  # confident decode: keep
    assert not keep_segment(no_speech_prob=0.9, avg_logprob=-1.4)
