"""Per-turn latency accounting and optional OpenTelemetry export."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from livekit.agents import metrics

log = logging.getLogger("vo.latency")


@dataclass
class TurnTiming:
    eou: float | None = None  # end of user speech -> turn committed (endpointing + EOU model)
    stt: float | None = None  # end of user speech -> final transcript
    llm_ttft: float | None = None
    tts_ttfb: float | None = None

    @property
    def complete(self) -> bool:
        return None not in (self.eou, self.llm_ttft, self.tts_ttfb)

    @property
    def response(self) -> float:
        """User stops speaking -> first agent audio. Upper bound: with preemptive generation the
        LLM starts before the turn is committed, so part of `llm_ttft` overlaps `eou`."""
        return self.eou + self.llm_ttft + self.tts_ttfb  # type: ignore[operator]


@dataclass
class LatencyTracker:
    turns: dict[str, TurnTiming] = field(default_factory=dict)

    def collect(self, m: metrics.AgentMetrics) -> TurnTiming | None:
        sid = getattr(m, "speech_id", None)
        if not sid:
            return None
        t = self.turns.setdefault(sid, TurnTiming())
        was_complete = t.complete
        if isinstance(m, metrics.EOUMetrics):
            t.eou, t.stt = m.end_of_utterance_delay, m.transcription_delay
        elif isinstance(m, metrics.LLMMetrics) and t.llm_ttft is None and not m.cancelled:
            t.llm_ttft = m.ttft
        elif isinstance(m, metrics.TTSMetrics) and t.tts_ttfb is None and not m.cancelled:
            t.tts_ttfb = m.ttfb  # first sentence of the reply is what the caller waits for
        if t.complete and not was_complete:
            log.info("turn %s response=%.0fms (eou=%.0f llm=%.0f tts=%.0f)", sid,
                     t.response * 1e3, t.eou * 1e3, t.llm_ttft * 1e3, t.tts_ttfb * 1e3)
            return t
        return None

    def summary(self) -> dict[str, dict[str, float]]:
        done = [t for t in self.turns.values() if t.complete]
        if not done:
            return {}
        cols = {"response": [t.response for t in done], "eou": [t.eou for t in done],
                "llm_ttft": [t.llm_ttft for t in done], "tts_ttfb": [t.tts_ttfb for t in done]}
        return {k: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
                    "n": len(v)} for k, v in cols.items()}


def setup_tracing(endpoint: str, service_name: str = "voice-orchestrator"):
    """Export LiveKit Agents' built-in spans (user turn, STT, EOU, LLM, TTS, tool calls) over
    OTLP/HTTP, e.g. to Jaeger, Tempo or Langfuse. Returns the provider so callers can flush."""
    from livekit.agents.telemetry import set_tracer_provider
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces"))
    )
    set_tracer_provider(provider)
    return provider
