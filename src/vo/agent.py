import logging

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    stt,
)
from livekit.plugins import openai, silero

from vo.config import settings
from vo.plugins.kokoro_tts import KokoroTTS
from vo.plugins.whisper_stt import WhisperSTT

log = logging.getLogger("vo")


def build_llm(s):
    if s.llm_provider == "openrouter":
        return openai.LLM.with_openrouter(model=s.llm_model, api_key=s.openrouter_api_key,
                                          app_name="voice-orchestrator")
    return openai.LLM(model=s.llm_model, api_key=s.openai_api_key)


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
        turn_handling={"interruption": {"mode": "vad", "min_duration": 0.4}},
    )
    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev):
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    ctx.add_shutdown_callback(lambda: log.info("usage: %s", usage.get_summary()))
    await session.start(agent=Agent(instructions=s.system_prompt), room=ctx.room)
    await session.generate_reply(instructions="Greet the user briefly.")


def main() -> None:
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))


if __name__ == "__main__":
    main()
