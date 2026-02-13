"""
vLLM LLM Service implementation.

This module provides a vLLM-based implementation of the LLM service,
supporting high-performance local LLM deployment with PagedAttention,
continuous batching, and GPU resource management for production workloads.

**Validates: Requirements 13.3, 9.4, 13.5**
"""

import logging
from datetime import datetime, timezone

import aiohttp

from .llm_service import (
    ConversationContext,
    LLMConfig,
    LLMResponse,
    LLMService,
)

logger = logging.getLogger(__name__)


class vLLMLLMService(LLMService):
    """
    vLLM-based LLM service implementation.

    Provides high-performance local LLM deployment with PagedAttention for
    efficient memory management, continuous batching for high throughput,
    and GPU resource management for production workloads.

    Key features:
    - PagedAttention: Efficient KV cache management
    - Continuous batching: High throughput request processing
    - GPU resource management: Configurable GPU allocation
    - OpenAI-compatible API: Easy integration
    """

    def __init__(
        self,
        config: LLMConfig,
        base_url: str = "http://localhost:8000",
        gpu_memory_utilization: float = 0.9,
        max_num_seqs: int = 256,
        max_model_len: int | None = None,
        fallback_service: LLMService | None = None,
    ):
        """
        Initialize vLLM LLM service.

        Args:
            config: LLM configuration
            base_url: vLLM server base URL (OpenAI-compatible API)
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            max_num_seqs: Maximum number of sequences in continuous batching
            max_model_len: Maximum model context length (None = model default)
            fallback_service: Fallback LLM service if vLLM is unavailable
        """
        super().__init__(config)

        if config.provider != "vllm":
            raise ValueError(f"vLLM service requires provider='vllm', got '{config.provider}'")

        self.base_url = base_url.rstrip("/")
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.fallback_service = fallback_service
        self._session: aiohttp.ClientSession | None = None
        self._server_available = None  # Cache server availability

        logger.info(
            "vLLM LLM service initialized (model: %s, GPU mem: %.1f%%, max_seqs: %d, base_url: %s)",
            config.model,
            gpu_memory_utilization * 100,
            max_num_seqs,
            base_url,
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create aiohttp session.

        Returns:
            Active aiohttp session
        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def check_server_availability(self) -> bool:
        """
        Check if vLLM server is available and responsive.

        Returns:
            True if server is available, False otherwise
        """
        # Return cached result if available
        if self._server_available is not None:
            return self._server_available

        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])

                    # Check if our model is available
                    model_ids = [m.get("id", "") for m in models]
                    self._server_available = any(
                        self.config.model in model_id for model_id in model_ids
                    )

                    if not self._server_available:
                        logger.warning(
                            "Model '%s' not found in vLLM server. Available models: %s",
                            self.config.model,
                            model_ids,
                        )

                    return self._server_available
                else:
                    logger.error("Failed to check vLLM models: HTTP %d", response.status)
                    self._server_available = False
                    return False

        except Exception as e:
            logger.error("Error checking vLLM server availability: %s", str(e))
            self._server_available = False
            return False

    async def generate_response(
        self,
        transcript: str,
        context: ConversationContext,
    ) -> LLMResponse:
        """
        Generate response from user transcript using vLLM.

        Leverages PagedAttention and continuous batching for efficient
        high-throughput inference. Falls back to alternative service if
        vLLM server is unavailable.

        Args:
            transcript: User's transcribed speech
            context: Conversation context

        Returns:
            Complete LLM response with intent, parameters, and text

        Raises:
            RuntimeError: If both vLLM and fallback service fail
        """
        # Check if server is available
        is_available = await self.check_server_availability()

        if not is_available:
            logger.warning("vLLM server unavailable, attempting fallback")
            if self.fallback_service:
                try:
                    return await self.fallback_service.generate_response(transcript, context)
                except Exception as e:
                    logger.error("Fallback service also failed: %s", str(e))
                    raise RuntimeError(
                        f"Both vLLM and fallback service failed: {str(e)}"
                    ) from e
            else:
                raise RuntimeError(
                    "vLLM server is not available and no fallback configured"
                )

        try:
            # Build prompt with voice-optimized instructions
            prompt = self._build_prompt(transcript, context)

            # Call vLLM API (OpenAI-compatible)
            response_text = await self._call_vllm(prompt)

            # Parse response and extract intent/parameters
            return self._parse_response(response_text, transcript)

        except Exception as e:
            logger.error("vLLM API error: %s", str(e))

            # Try fallback if available
            if self.fallback_service:
                logger.info("Attempting fallback service after vLLM error")
                try:
                    return await self.fallback_service.generate_response(transcript, context)
                except Exception as fallback_error:
                    logger.error("Fallback service failed: %s", str(fallback_error))

            raise RuntimeError(f"Failed to generate LLM response: {str(e)}") from e

    async def _call_vllm(self, prompt: str) -> str:
        """
        Call vLLM API to generate response.

        Uses OpenAI-compatible API endpoint with vLLM-specific optimizations
        including PagedAttention and continuous batching.

        Args:
            prompt: Complete prompt including system instructions

        Returns:
            Generated response text

        Raises:
            RuntimeError: If API call fails
        """
        session = await self._get_session()

        # Build request payload (OpenAI-compatible format)
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self._build_voice_optimized_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "stream": False,  # Non-streaming for simplicity (can be enhanced later)
        }

        # Add vLLM-specific parameters if configured
        if self.max_model_len is not None:
            payload["max_model_len"] = self.max_model_len

        try:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            ) as response:
                if response.status == 200:
                    data = await response.json()

                    # Extract response text from OpenAI-compatible format
                    choices = data.get("choices", [])
                    if choices:
                        message = choices[0].get("message", {})
                        return message.get("content", "")
                    else:
                        raise RuntimeError("No choices in vLLM response")
                else:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"vLLM API returned status {response.status}: {error_text}"
                    )

        except aiohttp.ClientError as e:
            raise RuntimeError(f"Network error calling vLLM: {str(e)}") from e

    def _build_prompt(
        self,
        transcript: str,
        context: ConversationContext,
    ) -> str:
        """
        Build prompt for vLLM with voice-optimized instructions.

        Args:
            transcript: Current user transcript
            context: Conversation context

        Returns:
            Complete prompt string
        """
        # Add conversation history
        history_text = ""
        for turn in context.conversation_history[-5:]:  # Last 5 turns
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                history_text += f"\nUser: {content}"
            elif role == "assistant":
                history_text += f"\nAssistant: {content}"

        # Build complete prompt
        prompt = f"""{history_text}

User: {transcript}
Assistant:"""

        return prompt

    def _build_voice_optimized_system_prompt(self) -> str:
        """
        Build voice-optimized system prompt.

        Incorporates requirements for voice-only responses:
        - No markdown, tables, or bullet points
        - Maximum 3 sentences per response
        - Empathetic, conversational language
        - Explicit knowledge boundaries

        Returns:
            Voice-optimized system prompt
        """
        base_prompt = self.config.system_prompt or "You are a helpful voice assistant."

        voice_instructions = """

CRITICAL VOICE RESPONSE RULES:
1. NEVER use markdown formatting (no **, *, `, #, [], or |)
2. NEVER use bullet points, numbered lists, or tables
3. Keep responses to 3 sentences or fewer
4. Use natural, conversational language as if speaking
5. Use empathetic phrasing with natural inflections
6. If you don't know something, explicitly say so - never make up information
7. Format all responses for voice-only consumption
8. Use short, clear sentences that sound natural when spoken aloud
9. Include emotional context when appropriate (e.g., "I understand that must be frustrating")
10. Avoid technical jargon unless specifically requested

Remember: Your responses will be spoken aloud by a text-to-speech system. Write as you would speak naturally in a conversation."""

        return base_prompt + voice_instructions

    def _parse_response(
        self,
        response_text: str,
        original_transcript: str,
    ) -> LLMResponse:
        """
        Parse vLLM response into LLMResponse format.

        Extracts intent and parameters from the response text and
        original transcript using the base class methods.

        Args:
            response_text: Generated response from vLLM
            original_transcript: Original user transcript

        Returns:
            Parsed LLM response
        """
        # Classify intent from original transcript
        intent = self.classify_intent(original_transcript)

        # Extract parameters from transcript
        parameters = self.extract_parameters(original_transcript, intent)

        # Determine if tool should be used
        tool_call = self.should_use_tool(intent, parameters)

        # If we have a tool call, add timestamp
        if tool_call:
            tool_call.timestamp = datetime.now(timezone.utc).isoformat()
            logger.info("Tool call detected: %s with params: %s", tool_call.tool_name, parameters)

        # Apply voice optimization to response
        formatted_text = self.format_for_voice(response_text)

        # Extract emotional tags
        emotional_tags = self._extract_emotional_tags(response_text)

        return LLMResponse(
            text=formatted_text,
            intent=intent,
            parameters=parameters,
            tool_call=tool_call,
            confidence=0.88,  # Slightly lower confidence for local models
            emotional_tags=emotional_tags,
            metadata={
                "model": self.config.model,
                "provider": "vllm",
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_num_seqs": self.max_num_seqs,
            },
        )

    def _extract_emotional_tags(self, text: str) -> list[str]:
        """
        Extract emotional tags for TTS from response text.

        Looks for emotional indicators in the text that can be
        used by TTS services for more natural speech synthesis.

        Args:
            text: Response text

        Returns:
            List of emotional tags
        """
        tags = []
        text_lower = text.lower()

        # Detect emotional context
        if any(word in text_lower for word in ["sorry", "apologize", "unfortunately"]):
            tags.append("empathetic")
        if any(word in text_lower for word in ["great", "wonderful", "excellent", "perfect"]):
            tags.append("enthusiastic")
        if any(word in text_lower for word in ["understand", "i see", "i hear you"]):
            tags.append("understanding")
        if "?" in text:
            tags.append("questioning")

        return tags

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
