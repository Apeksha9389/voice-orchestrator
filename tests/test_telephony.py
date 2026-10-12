import json
from pathlib import Path

import numpy as np
import pytest
from livekit import rtc
from livekit.agents import ToolError

from vo.telephony.call import CallInfo
from vo.telephony.tools import AwayHandler, OrderStore, make_call_tools, normalize_order_id


@pytest.mark.parametrize("raw,expected", [
    ("A1234", "A1234"), ("order # a-12 34", "A1234"), ("A one two three four", "A1234"),
    ("order number b nine oh one two", "B9012"),
])
def test_normalize_order_id(raw, expected):
    assert normalize_order_id(raw) == expected


class FakeControl:
    def __init__(self, fail_transfer: bool = False):
        self.calls: list[tuple] = []
        self._fail = fail_transfer

    async def hangup(self):
        self.calls.append(("hangup",))

    async def transfer(self, to):
        if self._fail:
            raise RuntimeError("transfer is only possible on phone (SIP) calls")
        self.calls.append(("transfer", to))


class FakeContext:
    def __init__(self, log):
        self._log = log

    async def wait_for_playout(self):
        self._log.append(("playout",))


def tools_by_name(control, transfer_to=None):
    orders = OrderStore.load(Path("data/orders.json"))
    return {t.info.name: t for t in make_call_tools(control, orders, transfer_to)}


async def test_end_call_waits_for_goodbye_then_hangs_up():
    control = FakeControl()
    await tools_by_name(control)["end_call"](FakeContext(control.calls))
    assert control.calls == [("playout",), ("hangup",)]


async def test_transfer_disabled_without_number():
    with pytest.raises(ToolError):
        await tools_by_name(FakeControl())["transfer_to_human"](FakeContext([]), reason="refund")


async def test_transfer_to_configured_number():
    control = FakeControl()
    out = await tools_by_name(control, "+918000000000")["transfer_to_human"](
        FakeContext(control.calls), reason="wants a human")
    assert out == "Transfer started."
    assert control.calls == [("playout",), ("transfer", "+918000000000")]


async def test_transfer_on_webrtc_call_becomes_tool_error():
    control = FakeControl(fail_transfer=True)
    with pytest.raises(ToolError):
        await tools_by_name(control, "+918000000000")["transfer_to_human"](
            FakeContext([]), reason="x")


async def test_lookup_order():
    lookup = tools_by_name(FakeControl())["lookup_order"]
    assert json.loads(await lookup("a one two three four"))["status"] == "shipped"
    with pytest.raises(ToolError):
        await lookup("Z9999")


def test_away_handler_prompts_once_then_hangs_up():
    h = AwayHandler(max_prompts=1)
    assert h.on_user_state("away") == "prompt"
    assert h.on_user_state("listening") is None
    assert h.on_user_state("away") == "hangup"
    h.on_user_state("speaking")  # user came back: reset
    assert h.on_user_state("away") == "prompt"


class FakeParticipant:
    def __init__(self, kind, attributes):
        self.identity, self.kind, self.attributes = "sip_+919800000000", kind, attributes


def test_call_info_from_sip_participant():
    p = FakeParticipant(rtc.ParticipantKind.PARTICIPANT_KIND_SIP, {
        "sip.phoneNumber": "+919800000000", "sip.trunkPhoneNumber": "+918000000000",
        "sip.callID": "SCL_abc"})
    info = CallInfo.from_participant(p)
    assert info.is_sip and info.caller_number == "+919800000000" and info.call_id == "SCL_abc"
    web = CallInfo.from_participant(FakeParticipant(rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD, {}))
    assert not web.is_sip and web.caller_number is None


def test_turn_handling_telephony_profile():
    from vo.agent import turn_handling
    from vo.config import Settings

    s = Settings()
    web, sip = turn_handling(s, "vad"), turn_handling(s, "vad", sip=True)
    assert sip["interruption"]["min_duration"] > web["interruption"]["min_duration"]
    assert sip["endpointing"]["min_delay"] >= web["endpointing"]["min_delay"]


def test_mulaw_roundtrip_error_is_small():
    from vo.eval.noise import mulaw_roundtrip

    x = np.linspace(-1, 1, 2001)
    assert np.max(np.abs(mulaw_roundtrip(x) - x)) < 0.05


MODELS = Path("models")


@pytest.mark.skipif(not (MODELS / "kokoro-v1.0.onnx").exists(), reason="kokoro model files missing")
async def test_vad_on_pstn_channel_needs_lower_threshold():
    """Narrowband G.711 speech lowers Silero's speech probability; the SIP profile compensates
    with a lower activation threshold (VO_SIP_VAD_ACTIVATION) without adding false triggers."""
    from kokoro_onnx import Kokoro
    from livekit.plugins import silero

    from vo.audio.denoise import RNNoiseDenoiser
    from vo.config import Settings
    from vo.eval.noise import denoise, score, simulate_telephony, to_frames, vad_probabilities

    tts = Kokoro(str(MODELS / "kokoro-v1.0.onnx"), str(MODELS / "voices-v1.0.bin"))
    sp, _ = tts.create("Hi, I am calling about my order from last week.", voice="am_adam")
    sp = np.interp(np.arange(0, len(sp), 0.5), np.arange(len(sp)), sp).astype(np.float32)
    clean = np.concatenate([np.zeros(24000, np.float32), sp, np.zeros(24000, np.float32)])
    phone = simulate_telephony(clean, 48000)

    vad = silero.VAD.load()
    ref = await vad_probabilities(vad, to_frames(clean, 48000)) >= 0.5
    sip_threshold = Settings().sip_vad_activation
    for signal in (phone, denoise(phone, 48000, RNNoiseDenoiser())):
        probs = await vad_probabilities(vad, to_frames(signal, 48000))
        default, tuned = score(probs, ref, 0.5), score(probs, ref, sip_threshold)
        assert tuned.recall >= default.recall and tuned.recall > 0.85
        assert tuned.false_positive_rate < 0.05
