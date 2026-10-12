# Phone calls (SIP) and call tools

## Topology

```
PSTN caller ─► SIP provider (Exotel / Plivo / Twilio / Telnyx …)
                   │  SIP INVITE :5060, RTP 10000-20000 (G.711 μ-law/A-law, 8 kHz)
                   ▼
             livekit-sip ──redis── livekit-server ──► room "call-<random>"
                                                      │  caller joins as a SIP participant
                                                      ▼
                                              voice-orchestrator worker
```

LiveKit's SIP bridge decodes G.711 and republishes the caller as an ordinary audio track. The agent
receives it at the 48 kHz it requests, as band-limited 300–3400 Hz content. There is no separate
8 kHz code path. Resampling happens once, in the bridge, and the narrowband character is handled
by the SIP profile below.

## Run it locally

```bash
docker compose -f docker-compose.sip.yml up -d       # redis + livekit-server + livekit-sip
lk sip inbound create telephony/inbound-trunk.json   # put your DID in "numbers"
lk sip dispatch create telephony/dispatch-rule.json  # one room per call, prefix "call-"
python -m vo.agent dev
```

Point the provider's SIP trunk at `<public-ip>:5060`. The compose file uses host networking, as
the upstream `livekit/sip` example does, so this setup runs on Linux.

**Dispatch.** The rule above relies on automatic dispatch: an unnamed worker joins every new room.
For explicit dispatch, for example several agents on one LiveKit, set `VO_AGENT_NAME=vo-agent` and
add `"roomConfig": {"agents": [{"agentName": "vo-agent"}]}` to the dispatch rule.

## Call lifecycle

1. The worker connects and waits for the caller. `CallInfo` reads `sip.phoneNumber`,
   `sip.trunkPhoneNumber` and `sip.callID` from the SIP participant's attributes.
2. The session is built *after* the caller is known, so a phone call gets the telephony profile
   and a browser call gets the WebRTC defaults.
3. The agent greets first. Callers expect the callee to speak.
4. The call ends in one of four ways:
   - the LLM calls `end_call`;
   - the caller is transferred;
   - the silence policy hangs up;
   - the caller hangs up, which closes the session through LiveKit's `close_on_disconnect`.

   Usage and turn-event counts are logged against the caller identity at shutdown.

## Telephony profile

| setting | WebRTC | SIP | why |
|---|---|---|---|
| VAD activation threshold | 0.5 | **0.35** | Measured: narrowband μ-law speech scores lower in Silero |
| barge-in `min_duration` | 0.4 s | **0.5 s** | Handset/line echo of the agent's own voice, plus line noise |
| endpointing `min_delay` | 0.4 s | **0.5 s** | The carrier jitter buffer stretches pauses |

**VAD threshold.** `vo.eval.noise.simulate_telephony` approximates a PSTN leg: a 300–3400 Hz
band, 8 kHz sampling and G.711 μ-law. On a male Kokoro voice, Silero's window recall against the
clean signal drops to 0.83 at threshold 0.5. At 0.35 it is 0.89, or 0.94 after RNNoise, with no
false triggers in the silent padding. `tests/test_telephony.py` pins this behaviour. The more
sensitive VAD is paired with a longer barge-in minimum: speech is detected earlier, but it takes
more of it to cut the agent off.

## Tools

Defined in `src/vo/telephony/tools.py`. They receive a `CallControl`, so they can be tested
without LiveKit.

| tool | behaviour |
|---|---|
| `end_call()` | Waits for the current speech (the goodbye the LLM says in the same reply) to finish playing, then deletes the room, which sends BYE to the carrier |
| `transfer_to_human(reason)` | Waits for "connecting you now" to play, then sends a SIP REFER to `VO_TRANSFER_NUMBER`. If no number is set, or the call is WebRTC, it returns a `ToolError` that the LLM relays as "I can't transfer you" |
| `lookup_order(order_id)` | Reads `data/orders.json`. Spoken IDs are normalised: "order number a one two three four" becomes `A1234`. Unknown IDs return a `ToolError` that asks the caller to repeat it |

`wait_for_playout()` matters here. Without it, the room is deleted while the goodbye or
transfer notice is still in the audio buffer, and the caller hears dead air.

## Silence policy

`user_away_timeout` (12 s) marks the caller `away`. `AwayHandler` asks once whether they are
still there. A second `away` without speech in between makes the agent say a fixed goodbye and
hang up. Any speech resets the counter.

## Configuration

| env | default |
|---|---|
| `VO_TRANSFER_NUMBER` | unset (transfers disabled) |
| `VO_ORDERS_PATH` | `data/orders.json` |
| `VO_USER_AWAY_TIMEOUT` | 12 |
| `VO_SIP_VAD_ACTIVATION` / `VO_SIP_INTERRUPT_MIN_DURATION` / `VO_SIP_ENDPOINTING_MIN_DELAY` | 0.35 / 0.5 / 0.5 |
| `VO_AGENT_NAME` | empty (automatic dispatch) |

## Not verified here

The build environment has no SIP provider or Docker. The tools, profile, call-info parsing and
PSTN-channel VAD behaviour are unit-tested. The compose file and trunk/dispatch JSON follow
LiveKit's published examples but have not been run end to end against a carrier.
