# voice-orchestrator

Real-time voice agent built on LiveKit Agents: Silero VAD → faster-whisper STT → LLM (OpenRouter /
OpenAI) → Kokoro TTS, with RNNoise suppression in front of VAD/STT, focused on noise robustness, interruption handling, and English/Hindi/Hinglish callers.

## Quickstart

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"      # add ,indic for IndicConformer Hindi STT
cp .env.example .env            # add VO_OPENROUTER_API_KEY
./scripts/download_models.sh
python -m vo.agent console      # talk through your mic, no server needed
```

Against a room:

```bash
docker compose up -d            # local LiveKit server (devkey/secret)
python -m vo.agent dev
```

Join with the [Agents Playground](https://agents-playground.livekit.io) using `ws://localhost:7880`.

Phone calls (SIP), see [TELEPHONY.md](docs/TELEPHONY.md):

```bash
docker compose -f docker-compose.sip.yml up -d
lk sip inbound create telephony/inbound-trunk.json
lk sip dispatch create telephony/dispatch-rule.json
python -m vo.agent dev
```

## Benchmarks

```bash
python scripts/bench_noise.py   # denoiser effect on SNR and VAD under synthetic noise
```

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Noise suppression](docs/NOISE.md)
- [Turn-taking & interruptions](docs/TURN_TAKING.md)
- [Hindi / Hinglish](docs/MULTILINGUAL.md)
- [Phone calls & tools](docs/TELEPHONY.md)
- [Latency & robustness](docs/LATENCY.md)
- [Roadmap](docs/ROADMAP.md)
