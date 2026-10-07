# voice-orchestrator

Real-time voice agent built on LiveKit Agents: Silero VAD → faster-whisper STT → LLM (OpenRouter /
OpenAI) → Kokoro TTS, with a focus on noise robustness and interruption handling.

## Quickstart

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
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

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
