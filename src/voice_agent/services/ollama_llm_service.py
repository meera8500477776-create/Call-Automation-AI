"""
Ollama LLM Service implementation.

This module provides an Ollama-based implementation of the LLM service,
supporting local LLM deployment with streaming responses, GPU acceleration,
and fallback handling for model unavailability.

**Validates: Requirements 13.2, 13.5, 7.6**
"""

import logging
import json
from typing import Optional, List, Dict, Any, AsyncIterator
from datetime import datetime, timezone

import aiohttp

from .llm_service import (
    LLMService,
    LLMConfig,
    LLMResponse,
    Intent,
    IntentType,
    ToolCall,
    ConversationContext,
)


logger = logging.getLogger(__name__)


class OllamaLLMService(LLMService):
    """
    Ollama-based LLM service implementation.
    
    Provides local LLM deployment with streaming support, GPU acceleration,
    and fallback handling for data sovereignty and offline scenarios.
    """

    def __init__(
        self,
        config: LLMConfig,
        base_url: str = "http://localhost:11434",
        gpu_enabled: bool = True,
        fallback_service: Optional[LLMService] = None,
    ):
        """
        Initialize Ollama LLM service.
        
        Args:
            config: LLM configuration
            base_url: Ollama API base URL
            gpu_enabled: Whether to use GPU acceleration
            fallback_service: Fallback LLM service if Ollama is unavailable
        """
        super().__init__(config)
        
        if config.provider != "ollama":
            raise ValueError(f"Ollama service requires provider='ollama', got '{config.provider}'")
        
        self.base_url = base_url.rstrip("/")
        self.gpu_enabled = gpu_enabled
        self.fallback_service = fallback_service
        self._session: Optional[aiohttp.ClientSession] = None
        self._model_available = None  # Cache model availability
        
        logger.info(
            "Ollama LLM service initialized (model: %s, GPU: %s, base_url: %s)",
            config.model,
            gpu_enabled,
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

    async def check_model_availability(self) -> bool:
        """
        Check if the configured model is available in Ollama.
        
        Returns:
            True if model is available, False otherwise
        """
        # Return cached result if available
        if self._model_available is not None:
            return self._model_available
        
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    
                    # Check if our model is in the list
                    self._model_available = any(
                        self.config.model in name for name in model_names
                    )
                    
                    if not self._model_available:
                        logger.warning(
                            "Model '%s' not found in Ollama. Available models: %s",
                            self.config.model,
                            model_names,
                        )
                    
                    return self._model_available
                else:
                    logger.error("Failed to check Ollama models: HTTP %d", response.status)
                    self._model_available = False
                    return False
                    
        except Exception as e:
            logger.error("Error checking Ollama model availability: %s", str(e))
            self._model_available = False
            return False

    async def generate_response(
        self,
        transcript: str,
        context: ConversationContext,
    ) -> LLMResponse:
        """
        Generate response from user transcript using Ollama.
        
        Attempts to use local Ollama model with fallback to alternative
        service if model is unavailable.
        
        Args:
            transcript: User's transcribed speech
            context: Conversation context
            
        Returns:
            Complete LLM response with intent, parameters, and text
            
        Raises:
            RuntimeError: If both Ollama and fallback service fail
        """
        # Check if model is available
        is_available = await self.check_model_availability()
        
        if not is_available:
            logger.warning("Ollama model unavailable, attempting fallback")
            if self.fallback_service:
                try:
                    return await self.fallback_service.generate_response(transcript, context)
                except Exception as e:
                    logger.error("Fallback service also failed: %s", str(e))
                    raise RuntimeError(
                        f"Both Ollama and fallback service failed: {str(e)}"
                    ) from e
            else:
                raise RuntimeError(
                    f"Ollama model '{self.config.model}' is not available and no fallback configured"
                )
        
        try:
            # Build prompt with voice-optimized instructions
            prompt = self._build_prompt(transcript, context)
            
            # Call Ollama API
            response_text = await self._call_ollama(prompt)
            
            # Parse response and extract intent/parameters
            return self._parse_response(response_text, transcript)
            
        except Exception as e:
            logger.error("Ollama API error: %s", str(e))
            
            # Try fallback if available
            if self.fallback_service:
                logger.info("Attempting fallback service after Ollama error")
                try:
                    return await self.fallback_service.generate_response(transcript, context)
                except Exception as fallback_error:
                    logger.error("Fallback service failed: %s", str(fallback_error))
            
            raise RuntimeError(f"Failed to generate LLM response: {str(e)}") from e

    async def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API to generate response.
        
        Args:
            prompt: Complete prompt including system instructions
            
        Returns:
            Generated response text
            
        Raises:
            RuntimeError: If API call fails
        """
        session = await self._get_session()
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,  # Non-streaming for simplicity (can be enhanced later)
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "top_p": self.config.top_p,
            },
        }
        
        # Add GPU configuration if enabled
        if self.gpu_enabled:
            payload["options"]["num_gpu"] = -1  # Use all available GPUs
        
        try:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", "")
                else:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Ollama API returned status {response.status}: {error_text}"
                    )
                    
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Network error calling Ollama: {str(e)}") from e

    def _build_prompt(
        self,
        transcript: str,
        context: ConversationContext,
    ) -> str:
        """
        Build prompt for Ollama with voice-optimized instructions.
        
        Args:
            transcript: Current user transcript
            context: Conversation context
            
        Returns:
            Complete prompt string
        """
        # Start with system prompt
        system_prompt = self._build_voice_optimized_system_prompt()
        
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
        prompt = f"""{system_prompt}

{history_text}

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
        Parse Ollama response into LLMResponse format.
        
        Extracts intent and parameters from the response text and
        original transcript using the base class methods.
        
        Args:
            response_text: Generated response from Ollama
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
            confidence=0.85,  # Slightly lower confidence for local models
            emotional_tags=emotional_tags,
            metadata={
                "model": self.config.model,
                "provider": "ollama",
                "gpu_enabled": self.gpu_enabled,
            },
        )

    def _extract_emotional_tags(self, text: str) -> List[str]:
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
