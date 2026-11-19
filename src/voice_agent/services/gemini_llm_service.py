"""Google Gemini chat completion adapter using the official Gen AI SDK."""

from __future__ import annotations

import asyncio
import logging
import re
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
        self._api_key = api_key
        self.max_retries = max(1, max_retries)
        self._model_validated = False

    async def aclose(self) -> None:
        """Release the SDK's async HTTP transport."""
        await self.client.aio.aclose()

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
        """Confirm the configured model is listed and supports text generation."""
        if self._model_validated:
            return
        try:
            pager = await self.client.aio.models.list()
            selected = self.config.model.removeprefix("models/")
            selected_model = None
            async for model in pager:
                if getattr(model, "name", "").removeprefix("models/") == selected:
                    selected_model = model
                    break
        except Exception as exc:
            raise RuntimeError(
                f"Could not discover Gemini models ({type(exc).__name__})"
            ) from exc
        if selected_model is None:
            raise RuntimeError(
                f"Configured Gemini model '{selected}' is not available to this API key"
            )
        actions = getattr(selected_model, "supported_actions", []) or []
        if "generateContent" not in actions:
            raise RuntimeError(
                f"Configured Gemini model '{selected}' does not support generateContent"
            )
        self._model_validated = True

    def _system_instruction(self) -> str:
        base = self.config.system_prompt or "You are a helpful voice assistant."
        return (
            f"{base}\n\nSpeak naturally and concisely. Avoid markdown and lists. "
            "Keep responses suitable for spoken conversation, usually no more than three sentences."
        )

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        """Extract an HTTP status from SDK exception fields or its safe message."""
        response = getattr(exc, "response", None)
        for value in (
            getattr(exc, "status_code", None),
            getattr(exc, "code", None),
            getattr(response, "status_code", None),
        ):
            try:
                status = int(value)
            except (TypeError, ValueError):
                continue
            if 400 <= status <= 599:
                return status
        match = re.search(r"\b([45]\d{2})\b", str(exc))
        return int(match.group(1)) if match else None

    @classmethod
    def _is_retryable(cls, exc: Exception) -> bool:
        if cls._status_code(exc) in {408, 429, 500, 502, 503, 504}:
            return True
        name = type(exc).__name__.casefold()
        return any(word in name for word in ("timeout", "connection", "temporar"))

    def _error_summary(self, exc: Exception) -> str:
        """Give callers actionable provider errors while redacting credentials."""
        status = self._status_code(exc)
        detail = str(exc).replace(self._api_key, "[REDACTED]")
        detail = re.sub(r"(?i)([?&]key=)[^&\s'\"}]+", r"\1[REDACTED]", detail)
        detail = re.sub(r"AIza[0-9A-Za-z_-]{30,}", "[REDACTED]", detail)
        lowered = detail.casefold()
        if status == 503 and "high demand" in lowered:
            return "Google Gemini HTTP 503 UNAVAILABLE: the selected model is experiencing high demand"
        if status == 504:
            return "Google Gemini HTTP 504 DEADLINE_EXCEEDED: generation did not finish before Google's deadline"
        if status == 429 and "insufficient_quota" in lowered:
            return "Google Gemini HTTP 429: account quota is insufficient"
        if status == 429:
            return "Google Gemini HTTP 429: request rate or quota limit reached"
        if status in {401, 403}:
            return f"Google Gemini HTTP {status}: authentication or model permission was rejected"
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            return f"Google Gemini request timed out after {self.config.timeout:g} seconds"
        if status:
            return f"Google Gemini HTTP {status}: {detail[:240]}"
        return f"{type(exc).__name__}: {detail[:240]}"

    def _build_contents(self, transcript: str, context: ConversationContext) -> list[types.Content]:
        """Map local conversation history to Gemini's supported content roles."""
        contents: list[types.Content] = []
        history = context.conversation_history[-10:]
        for turn in history:
            role = turn.get("role", "user")
            text = turn.get("content", "")
            if not text or role not in {"user", "assistant", "model"}:
                continue
            contents.append(
                types.Content(
                    role="model" if role == "assistant" else "user",
                    parts=[types.Part.from_text(text=text)],
                )
            )
        if not history or history[-1].get("role") != "user" or history[-1].get("content") != transcript:
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text=transcript)])
            )
        return contents

    async def _collect_stream(
        self, transcript: str, context: ConversationContext, request_state: dict[str, bool]
    ) -> tuple[str, Any]:
        stream = await self.client.aio.models.generate_content_stream(
            model=self.config.model,
            contents=self._build_contents(transcript, context),
            config=types.GenerateContentConfig(
                system_instruction=self._system_instruction(),
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_output_tokens=self.config.max_tokens,
            ),
        )
        request_state["stream_opened"] = True
        chunks: list[str] = []
        usage: Any = None
        try:
            async for chunk in stream:
                value = getattr(chunk, "text", None)
                if value:
                    chunks.append(value)
                    request_state["text_emitted"] = True
                usage = getattr(chunk, "usage_metadata", None) or usage
        finally:
            close_stream = getattr(stream, "aclose", None)
            if callable(close_stream):
                await close_stream()
        return "".join(chunks).strip(), usage

    async def generate_response(
        self,
        transcript: str,
        context: ConversationContext,
    ) -> LLMResponse:
        """Stream a voice-ready response with local multi-turn conversation history."""
        if not transcript.strip():
            raise ValueError("Transcript cannot be empty")
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            request_state = {"stream_opened": False, "text_emitted": False}
            try:
                raw_text, usage = await asyncio.wait_for(
                    self._collect_stream(transcript, context, request_state),
                    timeout=self.config.timeout,
                )
                if not raw_text:
                    raise RuntimeError("Gemini returned no text content")
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
                if (
                    attempt + 1 >= self.max_retries
                    or not self._is_retryable(exc)
                    or request_state["stream_opened"]
                    or request_state["text_emitted"]
                ):
                    break
                await asyncio.sleep(0.25 * (attempt + 1))
        summary = self._error_summary(last_error) if last_error else "unknown error"
        logger.error("Gemini generation failed: %s", summary)
        raise RuntimeError(
            f"Gemini response generation failed: {summary}"
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
