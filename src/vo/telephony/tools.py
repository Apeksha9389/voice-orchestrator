"""LLM function tools for call control and order lookup."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from livekit.agents import RunContext, ToolError, function_tool
from livekit.agents.llm import FunctionTool

from vo.telephony.call import CallControl

log = logging.getLogger("vo.tools")

_DIGIT_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}


def normalize_order_id(raw: str) -> str:
    """'order # a-12 34', 'A one two three four' -> 'A1234'. Spoken IDs arrive as words/spaces."""
    tokens = re.findall(r"[a-z]+|\d+", raw.lower())
    out = [_DIGIT_WORDS.get(t, t) for t in tokens if t not in {"order", "number", "no", "id"}]
    return "".join(out).upper()


class OrderStore:
    def __init__(self, orders: dict[str, dict[str, Any]]) -> None:
        self._orders = {normalize_order_id(k): v for k, v in orders.items()}

    @classmethod
    def load(cls, path: str | Path | None) -> OrderStore:
        if not path or not Path(path).exists():
            return cls({})
        return cls(json.loads(Path(path).read_text()))

    def get(self, order_id: str) -> dict[str, Any] | None:
        return self._orders.get(normalize_order_id(order_id))


def make_call_tools(control: CallControl, orders: OrderStore,
                    transfer_to: str | None = None) -> list[FunctionTool]:
    @function_tool
    async def end_call(context: RunContext) -> None:
        """End the call. Use when the user says goodbye or confirms they need nothing else.
        Say a short goodbye in the same reply before calling this."""
        await context.wait_for_playout()  # let the goodbye finish before hanging up
        await control.hangup()

    @function_tool
    async def transfer_to_human(context: RunContext, reason: str) -> str:
        """Transfer the caller to a human agent. Use when the user asks for a person or the
        request is outside what you can handle. Tell the user you are transferring them first.

        Args:
            reason: One short sentence on why the caller needs a human.
        """
        if not transfer_to:
            raise ToolError("Transfer is not available. Offer to help further instead.")
        log.info("transfer requested: %s", reason)
        await context.wait_for_playout()
        try:
            await control.transfer(transfer_to)
        except RuntimeError as e:
            raise ToolError(str(e)) from e
        return "Transfer started."

    @function_tool
    async def lookup_order(order_id: str) -> str:
        """Look up an order's status and delivery estimate.

        Args:
            order_id: The order number as the user said it, e.g. "A1234" or "a one two three".
        """
        order = orders.get(order_id)
        if order is None:
            raise ToolError(f"No order found for '{order_id}'. Ask the user to repeat it.")
        return json.dumps(order, ensure_ascii=False)

    return [end_call, transfer_to_human, lookup_order]


class AwayHandler:
    """Silence policy: after the session marks the user `away`, prompt once, then hang up."""

    def __init__(self, max_prompts: int = 1) -> None:
        self._max, self._prompts = max_prompts, 0

    def on_user_state(self, state: str) -> str | None:
        if state == "speaking":
            self._prompts = 0
            return None
        if state != "away":
            return None
        if self._prompts < self._max:
            self._prompts += 1
            return "prompt"
        return "hangup"
