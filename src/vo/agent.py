import asyncio
import logging

from livekit.agents import (
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    stt,
)
from livekit.agents.voice import room_io
from livekit.plugins import openai, silero

from vo.audio.denoise import DenoisedAudioInput, build_denoiser
from vo.config import settings
from vo.lang.router import LanguageRouter, RoutedSTT
from vo.plugins.kokoro_tts import KokoroTTS, split_on_danda
from vo.plugins.whisper_stt import WhisperSTT
from vo.telephony.call import CallInfo, LiveKitCallControl
from vo.telephony.tools import AwayHandler, OrderStore, make_call_tools
from vo.turns.agent import VoiceAgent
from vo.turns.tracking import InterruptionLog, OverlapTracker

log = logging.getLogger("vo")


def build_llm(s):
    if s.llm_provider == "openrouter":
        return openai.LLM.with_openrouter(model=s.llm_model, api_key=s.openrouter_api_key,
                                          app_name="voice-orchestrator")
    return openai.LLM(model=s.llm_model, api_key=s.openai_api_key)


def build_turn_detector(s):
    if s.turn_detector == "vad":
        return "vad"
    try:
        from livekit.plugins.turn_detector.multilingual import MultilingualModel

        return MultilingualModel()
    except (ImportError, OSError, ValueError) as e:  # files missing: run `download-files`
        log.warning("turn detector unavailable (%s); using VAD endpointing", e)
        return "vad"


def turn_handling(s, turn_detection, *, sip: bool = False) -> dict:
    min_delay = s.sip_endpointing_min_delay if sip else s.endpointing_min_delay
    return {
        "turn_detection": turn_detection,
        "endpointing": {"min_delay": min_delay, "max_delay": s.endpointing_max_delay},
        "interruption": {
            "mode": "vad",  # "adaptive" needs LiveKit Cloud inference
            "min_duration": s.sip_interrupt_min_duration if sip else s.interrupt_min_duration,
            # batch STT has no interim words, so a word gate would delay every barge-in until the
            # user stops; backchannels are filtered on the final transcript instead
            "min_words": 0,
            "resume_false_interruption": True,
            "false_interruption_timeout": s.false_interruption_timeout,
        },
    }


def load_indic(s):
    if "hi" not in s.languages or s.hindi_stt != "indic":
        return None
    try:
        from vo.plugins.indic_stt import IndicConformerSTT

        return IndicConformerSTT(model_id=s.indic_model, decoding=s.indic_decoding,
                                 device="cuda" if s.stt_device == "cuda" else "cpu")
    except (ImportError, OSError, ValueError, RuntimeError) as e:  # extra missing / gated model
        log.warning("IndicConformer unavailable (%s); Hindi falls back to Whisper", e)
        return None


def build_stt(s, whisper: WhisperSTT, indic) -> stt.STT:
    """Per-session STT; the router holds the session's sticky language."""
    if len(s.languages) < 2:
        return whisper
    engines = {lang: whisper for lang in s.languages}
    if indic is not None:
        engines["hi"] = indic
    router = LanguageRouter(set(s.languages), s.stt_language or s.languages[0],
                            switch_prob=s.lid_switch_prob, min_seconds=s.lid_min_seconds)
    return RoutedSTT(detector=whisper.detect_language, engines=engines, router=router)


def prewarm(proc: JobProcess) -> None:
    s = settings()
    proc.userdata["vad"] = silero.VAD.load(activation_threshold=s.vad_activation,
                                           min_silence_duration=s.vad_min_silence)
    proc.userdata["vad_sip"] = silero.VAD.load(activation_threshold=s.sip_vad_activation,
                                               min_silence_duration=s.vad_min_silence)
    proc.userdata["whisper"] = WhisperSTT(model=s.stt_model, device=s.stt_device,
                                          compute_type=s.stt_compute_type,
                                          language=s.stt_language)
    proc.userdata["indic"] = load_indic(s)
    proc.userdata["tts"] = KokoroTTS(model_path=s.tts_model_path, voices_path=s.tts_voices_path,
                                     voice=s.tts_voice, hindi_voice=s.tts_hindi_voice,
                                     speed=s.tts_speed)


async def entrypoint(ctx: JobContext) -> None:
    s, ud = settings(), ctx.proc.userdata
    await ctx.connect()
    call = CallInfo.from_participant(await ctx.wait_for_participant())
    log.info("call start identity=%s sip=%s caller=%s", call.identity, call.is_sip,
             call.caller_number)
    control = LiveKitCallControl(ctx.api, ctx.room.name, call)

    vad = ud["vad_sip" if call.is_sip else "vad"]
    session = AgentSession(
        vad=vad,
        stt=stt.StreamAdapter(stt=build_stt(s, ud["whisper"], ud["indic"]), vad=vad),
        llm=build_llm(s),
        tts=ud["tts"],
        tts_text_transforms=["filter_markdown", "filter_emoji", split_on_danda],
        turn_handling=turn_handling(s, build_turn_detector(s), sip=call.is_sip),
        user_away_timeout=s.user_away_timeout,
    )
    usage = metrics.UsageCollector()
    tracker, events, away = OverlapTracker(), InterruptionLog(), AwayHandler()
    background: set[asyncio.Task] = set()
    session.on("agent_state_changed", lambda ev: tracker.on_agent_state(ev.new_state))
    session.on("agent_false_interruption",
               lambda ev: events.record("false_interruption", resumed=ev.resumed))

    @session.on("user_state_changed")
    def _on_user_state(ev):
        action = away.on_user_state(ev.new_state)
        if action == "prompt":
            session.generate_reply(instructions="Briefly ask if the caller is still there.")
        elif action == "hangup":
            task = asyncio.create_task(_hangup_after_goodbye())
            background.add(task)
            task.add_done_callback(background.discard)

    async def _hangup_after_goodbye() -> None:
        await session.say("Ending the call now. Goodbye!", allow_interruptions=False)
        await control.hangup()

    @session.on("metrics_collected")
    def _on_metrics(ev):
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    async def _report() -> None:
        log.info("call end identity=%s usage=%s turn_events=%s", call.identity,
                 usage.get_summary(), events.counts())

    ctx.add_shutdown_callback(_report)
    tools = make_call_tools(control, OrderStore.load(s.orders_path), s.transfer_number)
    await session.start(
        agent=VoiceAgent(instructions=s.system_prompt, tracker=tracker, events=events,
                         backchannel_max_words=s.backchannel_max_words, tools=tools),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            participant_identity=call.identity,
            # 48 kHz matches RNNoise's native rate, so no resampling before denoising.
            # SIP audio is decoded from G.711 by LiveKit; it arrives band-limited at this rate.
            audio_input=room_io.AudioInputOptions(sample_rate=48000),
        ),
    )
    denoiser = build_denoiser(s.denoiser)
    if denoiser and session.input.audio:
        session.input.audio = DenoisedAudioInput(session.input.audio, denoiser)
    await session.generate_reply(instructions="Greet the caller briefly and ask how you can help.")


def main() -> None:
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm,
                              agent_name=settings().agent_name))


if __name__ == "__main__":
    main()
