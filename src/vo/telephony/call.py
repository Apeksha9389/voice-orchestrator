"""Call identity and control for SIP (PSTN) and WebRTC participants."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from livekit import api, rtc

log = logging.getLogger("vo.call")


@dataclass(frozen=True)
class CallInfo:
    identity: str
    is_sip: bool
    caller_number: str | None = None
    trunk_number: str | None = None
    call_id: str | None = None

    @classmethod
    def from_participant(cls, p: rtc.RemoteParticipant) -> CallInfo:
        attrs = p.attributes or {}
        return cls(
            identity=p.identity,
            is_sip=p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
            caller_number=attrs.get("sip.phoneNumber"),
            trunk_number=attrs.get("sip.trunkPhoneNumber"),
            call_id=attrs.get("sip.callID"),
        )


class CallControl(Protocol):
    async def hangup(self) -> None: ...

    async def transfer(self, to: str) -> None: ...


class LiveKitCallControl:
    def __init__(self, lk: api.LiveKitAPI, room_name: str, call: CallInfo) -> None:
        self._lk, self._room, self._call = lk, room_name, call

    async def hangup(self) -> None:
        # deleting the room disconnects every participant; for SIP that sends BYE to the carrier
        log.info("hangup room=%s call_id=%s", self._room, self._call.call_id)
        await self._lk.room.delete_room(api.DeleteRoomRequest(room=self._room))

    async def transfer(self, to: str) -> None:
        if not self._call.is_sip:
            raise RuntimeError("transfer is only possible on phone (SIP) calls")
        uri = to if to.startswith(("tel:", "sip:")) else f"tel:{to}"
        log.info("transfer call_id=%s to=%s", self._call.call_id, uri)
        await self._lk.sip.transfer_sip_participant(api.TransferSIPParticipantRequest(
            room_name=self._room, participant_identity=self._call.identity, transfer_to=uri,
            play_dialtone=False,
        ))
