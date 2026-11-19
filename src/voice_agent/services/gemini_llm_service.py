"""Google Gemini chat completion adapter using the official Gen AI SDK."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from google import genai
from google.genai import types

from .llm_service import (
    ConversationContext,
    Intent,
    LLMConfig,
    LLMResponse,
    LLMService,
)

logger = logging.getLogger(__name__)


class GeminiLLMService(LLMService):
    """Async Gemini Flash adapter preserving the project's LLM interface."""

    def __init__(self, config: LLMConfig, api_key: str, max_retries: int = 2):
        super().__init__(config)
        if config.provider != "gemini":
            raise ValueError(f"Gemini service requires provider='gemini', got '{config.provider}'")
        if not api_key or not api_key.strip():
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(config.timeout * 1000)),
        )
        self.max_retries = max(1, max_retries)
        self._model_validated = False
        self._session_interactions: dict[str, str] = {}

    async def aclose(self) -> None:
        """Release the SDK's async HTTP transport."""
        await self.client.aio.aclose()
        self._session_interactions.clear()

    async def list_available_models(self) -> list[str]:
        """Discover models enabled for this API key, without logging their data."""
        pager = await self.client.aio.models.list()
        names: list[str] = []
        async for model in pager:
            name = getattr(model, "name", None)
            if name:
                names.append(name.removeprefix("models/"))
        return names

    async def validate_model(self) -> None:
        """Discover and validate the configured model when explicit validation is needed."""
        if self._model_validated:
            return
        try:
            available = await asyncio.wait_for(
                self.list_available_models(), timeout=self.config.timeout
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not discover Gemini models ({type(exc).__name__})"
            ) from exc
        selected = self.config.model.removeprefix("models/")
        if selected not in available:
            raise RuntimeError(
                f"Configured Gemini model '{selected}' is not available to this API key"
            )
        self._model_validated = True

    def _system_instruction(self) -> str:
        base = self.config.system_prompt or "You are a helpful voice assistant."
        return (
            f"{base}\n\nSpeak naturally and concisely. Avoid markdown and lists. "
            "Keep responses suitable for spoken conversation, usually no more than three sentences."
        )

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if code in {408, 429, 500, 502, 503, 504}:
            return True
        name = type(exc).__name__.casefold()
        return any(word in name for word in ("timeout", "connection", "temporar"))

    async def _collect_stream(
        self, transcript: str, context: ConversationContext
    ) -> tuple[str, str | None, Any]:
        session_id = context.session_id
        previous_id = self._session_interactions.get(session_id)
        stream = await self.client.aio.interactions.create(
            model=self.config.model,
            input=transcript,
            system_instruction=self._system_instruction(),
            previous_interaction_id=previous_id,
            generation_config={
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_output_tokens": self.config.max_tokens,
            },
            stream=True,
            timeout=self.config.timeout,
        )
        chunks: list[str] = []
        interaction_id: str | None = None
        usage: Any = None
        try:
            async for event in stream:
                event_type = getattr(event, "event_type", None)
                interaction = getattr(event, "interaction", None)
                if interaction is not None:
                    interaction_id = getattr(interaction, "id", None) or interaction_id
                    if interaction_id:
                        self._session_interactions[context.session_id] = interaction_id
                    usage = getattr(interaction, "usage", None) or usage
                if event_type == "error":
                    raise RuntimeError("Gemini interaction stream returned an error event")
                if event_type != "step.delta":
                    continue
                delta = getattr(event, "delta", None)
                if getattr(delta, "type", None) == "text":
                    value = getattr(delta, "text", None)
                    if value:
                        chunks.append(value)
        finally:
            close_stream = getattr(stream, "aclose", None)
            if callable(close_stream):
                await close_stream()
        return "".join(chunks).strip(), interaction_id, usage

    async def generate_response(
        self,
        transcript: str,
        context: ConversationContext,
    ) -> LLMResponse:
        """Generate a voice-ready response, including up to five prior turns."""
        if not transcript.strip():
            raise ValueError("Transcript cannot be empty")
        last_error: Exception | None = None
        initial_interaction_id = self._session_interactions.get(context.session_id)
        for attempt in range(self.max_retries):
            try:
                raw_text, interaction_id, usage = await asyncio.wait_for(
                    self._collect_stream(transcript, context), timeout=self.config.timeout
                )
                if not raw_text:
                    raise RuntimeError("Gemini returned no text content")
                if interaction_id:
                    self._session_interactions[context.session_id] = interaction_id
                text = self.format_for_voice(raw_text)
                intent = self.classify_intent(transcript)
                parameters = self.extract_parameters(transcript, intent)
                intent = Intent(
                    name=intent.name,
                    confidence=intent.confidence,
                    parameters=parameters,
                    requires_tool=intent.requires_tool,
                    tool_name=intent.tool_name,
                )
                return LLMResponse(
                    text=text,
                    intent=intent,
                    parameters=parameters,
                    confidence=intent.confidence,
                    emotional_tags=self._extract_emotional_tags(raw_text),
                    metadata={
                        "provider": "gemini",
                        "model": self.config.model,
                        "usage": {
                            "prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
                            "response_tokens": getattr(usage, "output_tokens", 0) or 0,
                            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                        },
                    },
                )
            except Exception as exc:
                last_error = exc
                # Retrying after an interaction has begun could duplicate a user turn.
                interaction_started = (
                    self._session_interactions.get(context.session_id) != initial_interaction_id
                )
                if (
                    attempt + 1 >= self.max_retries
                    or not self._is_retryable(exc)
                    or interaction_started
                ):
                    break
                await asyncio.sleep(0.25 * (attempt + 1))
        logger.error("Gemini generation failed (%s)", type(last_error).__name__ if last_error else "unknown")
        raise RuntimeError(
            f"Gemini response generation failed ({type(last_error).__name__ if last_error else 'unknown error'})"
        ) from last_error

    def _extract_emotional_tags(self, text: str) -> list[str]:
        lowered = text.casefold()
        tags: list[str] = []
        if any(word in lowered for word in ("sorry", "apologize", "unfortunately")):
            tags.append("empathetic")
        if any(word in lowered for word in ("great", "wonderful", "excellent", "perfect")):
            tags.append("enthusiastic")
        if "?" in text:
            tags.append("questioning")
        return tags
