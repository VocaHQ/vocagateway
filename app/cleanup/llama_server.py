"""The `llama.cpp` server protocol adapter.

Everything model-specific about talking to a pinned `llama-server` lives here:
the endpoints used, the tokenizer that supplies real token counts, and the
handful of ways a chat completion can come back that this package refuses to
trust. It executes nothing — no tool calls, no agent loop, no shell — and it
never sees a client's bearer token.

The exact request fields and template behaviour of any given llama.cpp release
must be tested against that release; the checks below are written so an
unexpected answer becomes a rejection rather than a silent acceptance.
"""

from __future__ import annotations

import json
import time
from typing import Any

from app.cleanup import prompts, transport
from app.cleanup.base import (
    MAXIMUM_INPUT_TOKENS,
    MAXIMUM_OUTPUT_BYTES,
    CleanupReason,
    CleanupRejected,
    CleanupUnavailable,
)

HEALTH_PATH = "/health"
PROPS_PATH = "/props"
TOKENIZE_PATH = "/tokenize"
COMPLETIONS_PATH = "/v1/chat/completions"

HEALTH_TIMEOUT_SECONDS = 1.0
TOKENIZE_TIMEOUT_SECONDS = 2.0
# A step that starts with its deadline already spent still gets a moment to
# fail properly rather than being handed a negative timeout.
MINIMUM_STEP_SECONDS = 0.25
TRUNCATED_FINISH_REASON = "length"
CONTEXT_SIZE_FIELD = "n_ctx"


def remaining(deadline: float, ceiling: float) -> float:
    return max(MINIMUM_STEP_SECONDS, min(ceiling, deadline - time.monotonic()))


class LlamaServerRuntime:
    """One `llama-server` reachable at a fixed local address.

    The address is supplied by the manager that owns the process (or by an
    operator running their own server for evaluation). It is never taken from a
    request, so there is no per-request URL to redirect or rebind.
    """

    def __init__(
        self,
        endpoint: transport.Endpoint,
        *,
        model_id: str,
        model_name: str = "cleanup",
    ) -> None:
        self.endpoint = endpoint
        self._model_id = model_id
        self._model_name = model_name

    @property
    def model_id(self) -> str | None:
        return self._model_id

    async def available(self) -> bool:
        try:
            reply = await transport.get_json(
                self.endpoint, HEALTH_PATH, budget=HEALTH_TIMEOUT_SECONDS
            )
        except (transport.TransportError, TimeoutError):
            return False
        return reply.status == transport.HTTP_OK

    async def context_tokens(self) -> int:
        """The context window the running server actually has, or 0 when unknown."""
        try:
            reply = await transport.get_json(
                self.endpoint, PROPS_PATH, budget=HEALTH_TIMEOUT_SECONDS
            )
        except (transport.TransportError, TimeoutError):
            return 0
        if reply.status != transport.HTTP_OK:
            return 0
        return _context_size(reply.json())

    async def count_tokens(self, text: str, *, budget: float) -> int:
        """Token count from the pinned runtime's own tokenizer.

        A character estimate is not a substitute: the same sentence costs wildly
        different numbers of tokens across scripts, and the ceiling exists to
        keep a prompt inside the context window.
        """
        reply = await transport.post_json(
            self.endpoint, TOKENIZE_PATH, {"content": text}, budget=budget
        )
        if reply.status != transport.HTTP_OK:
            raise CleanupUnavailable("The cleanup runtime rejected the tokenize request.")
        tokens = reply.json().get("tokens") if isinstance(reply.json(), dict) else None
        if not isinstance(tokens, list):
            raise CleanupUnavailable("The cleanup runtime returned an unexpected token list.")
        return len(tokens)

    async def clean(self, transcript: str, language: str, *, budget_seconds: float) -> str:
        deadline = time.monotonic() + budget_seconds
        await self._check_length(transcript, deadline)
        reply = await transport.post_json(
            self.endpoint,
            COMPLETIONS_PATH,
            prompts.chat_request(transcript, language, model=self._model_name),
            budget=remaining(deadline, budget_seconds),
        )
        if reply.status != transport.HTTP_OK:
            raise CleanupUnavailable("The cleanup runtime rejected the request.")
        return _decode_completion(_parsed(reply))

    async def _check_length(self, transcript: str, deadline: float) -> None:
        counted = await self.count_tokens(
            transcript, budget=remaining(deadline, TOKENIZE_TIMEOUT_SECONDS)
        )
        if counted > MAXIMUM_INPUT_TOKENS:
            raise CleanupRejected(CleanupReason.INPUT_TOO_LONG)


def _parsed(reply: transport.Reply) -> Any:
    """A body that is not JSON is a bad *answer*, not an unreachable runtime.

    The distinction matters to the operator: one says the model is misbehaving,
    the other says the process is gone, and they have different fixes.
    """
    try:
        return reply.json()
    except transport.TransportError as error:
        raise CleanupRejected(
            CleanupReason.INVALID_OUTPUT, "The answer was not a JSON document."
        ) from error


def _context_size(props: Any) -> int:
    if not isinstance(props, dict):
        return 0
    settings = props.get("default_generation_settings")
    if isinstance(settings, dict) and isinstance(settings.get(CONTEXT_SIZE_FIELD), int):
        return int(settings[CONTEXT_SIZE_FIELD])
    top_level = props.get(CONTEXT_SIZE_FIELD)
    return int(top_level) if isinstance(top_level, int) else 0


def _decode_completion(document: Any) -> str:
    """Pull the corrected text out, refusing every other shape an answer can take."""
    choice = _single_choice(document)
    if choice.get("finish_reason") == TRUNCATED_FINISH_REASON:
        raise CleanupRejected(CleanupReason.INVALID_OUTPUT, "The generation was truncated.")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise CleanupRejected(CleanupReason.INVALID_OUTPUT, "The answer carried no message.")
    _reject_side_channels(message)
    return _decode_text(message.get("content"))


def _single_choice(document: Any) -> dict[str, Any]:
    choices = document.get("choices") if isinstance(document, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise CleanupRejected(CleanupReason.INVALID_OUTPUT, "The answer had no single choice.")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise CleanupRejected(CleanupReason.INVALID_OUTPUT, "The choice was not an object.")
    return choice


def _reject_side_channels(message: dict[str, Any]) -> None:
    """Refuse tool calls and reasoning output rather than discarding them.

    Either one means the runtime is not in the non-thinking, tool-free mode this
    package asked for, and a transcript produced under different rules than the
    ones that were validated is not a transcript worth inserting.
    """
    if message.get("tool_calls"):
        raise CleanupRejected(CleanupReason.INVALID_OUTPUT, "The answer requested a tool call.")
    if message.get("reasoning_content"):
        raise CleanupRejected(CleanupReason.INVALID_OUTPUT, "The answer leaked reasoning output.")


def _decode_text(answer: Any) -> str:
    if not isinstance(answer, str):
        raise CleanupRejected(CleanupReason.INVALID_OUTPUT, "The answer carried no text.")
    if len(answer.encode("utf-8")) > MAXIMUM_OUTPUT_BYTES:
        raise CleanupRejected(CleanupReason.INVALID_OUTPUT, "The answer exceeded the size ceiling.")
    return _unwrap_object(answer)


def _unwrap_object(answer: str) -> str:
    try:
        document = json.loads(answer)
    except json.JSONDecodeError as error:
        raise CleanupRejected(
            CleanupReason.INVALID_OUTPUT, "The answer was not a JSON object."
        ) from error
    if not isinstance(document, dict) or set(document) != {"text"}:
        raise CleanupRejected(CleanupReason.INVALID_OUTPUT, "The answer had unexpected fields.")
    text = document["text"]
    if not isinstance(text, str):
        raise CleanupRejected(CleanupReason.INVALID_OUTPUT, "The answer's text was not a string.")
    return text
