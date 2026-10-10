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
from vo.plugins.kokoro_tts import KokoroTTS
from vo.plugins.whisper_stt import WhisperSTT
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


def turn_handling(s, turn_detection) -> dict:
    return {
        "turn_detection": turn_detection,
        "endpointing": {"min_delay": s.endpointing_min_delay, "max_delay": s.endpointing_max_delay},
        "interruption": {
            "mode": "vad",  # "adaptive" needs LiveKit Cloud inference
            "min_duration": s.interrupt_min_duration,
            # batch STT has no interim words, so a word gate would delay every barge-in until the
            # user stops; backchannels are filtered on the final transcript instead
            "min_words": 0,
            "resume_false_interruption": True,
            "false_interruption_timeout": s.false_interruption_timeout,
        },
    }


def prewarm(proc: JobProcess) -> None:
    s = settings()
    proc.userdata["vad"] = silero.VAD.load(activation_threshold=s.vad_activation,
                                           min_silence_duration=s.vad_min_silence)
    proc.userdata["stt"] = WhisperSTT(model=s.stt_model, device=s.stt_device,
                                      compute_type=s.stt_compute_type, language=s.stt_language)
    proc.userdata["tts"] = KokoroTTS(model_path=s.tts_model_path, voices_path=s.tts_voices_path,
                                     voice=s.tts_voice, speed=s.tts_speed)


async def entrypoint(ctx: JobContext) -> None:
    s, ud = settings(), ctx.proc.userdata
    vad = ud["vad"]
    session = AgentSession(
        vad=vad,
        stt=stt.StreamAdapter(stt=ud["stt"], vad=vad),
        llm=build_llm(s),
        tts=ud["tts"],
        turn_handling=turn_handling(s, build_turn_detector(s)),
    )
    usage = metrics.UsageCollector()
    tracker, events = OverlapTracker(), InterruptionLog()
    session.on("agent_state_changed", lambda ev: tracker.on_agent_state(ev.new_state))
    session.on("agent_false_interruption",
               lambda ev: events.record("false_interruption", resumed=ev.resumed))

    @session.on("metrics_collected")
    def _on_metrics(ev):
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    async def _report() -> None:
        log.info("usage: %s turn_events: %s", usage.get_summary(), events.counts())

    ctx.add_shutdown_callback(_report)
    await session.start(
        agent=VoiceAgent(instructions=s.system_prompt, tracker=tracker, events=events,
                         backchannel_max_words=s.backchannel_max_words),
        room=ctx.room,
        # 48 kHz matches RNNoise's native rate, so no resampling before denoising
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(sample_rate=48000)
        ),
    )
    denoiser = build_denoiser(s.denoiser)
    if denoiser and session.input.audio:
        session.input.audio = DenoisedAudioInput(session.input.audio, denoiser)
    await session.generate_reply(instructions="Greet the user briefly.")


def main() -> None:
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))


if __name__ == "__main__":
    main()
