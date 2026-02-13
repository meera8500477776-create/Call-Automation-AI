"""
OpenAI LLM Service implementation.

This module provides an OpenAI-based implementation of the LLM service,
supporting streaming responses, function calling for tool execution,
and voice-optimized prompt engineering.

**Validates: Requirements 13.1, 6.1, 6.2, 6.3**
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from .llm_service import (
    ConversationContext,
    Intent,
    IntentType,
    LLMConfig,
    LLMResponse,
    LLMService,
    ToolCall,
)

logger = logging.getLogger(__name__)


class OpenAILLMService(LLMService):
    """
    OpenAI-based LLM service implementation.
    
    Provides streaming support, function calling for tool execution,
    and voice-optimized prompt engineering for natural conversations.
    """

    def __init__(self, config: LLMConfig, api_key: str | None = None):
        """
        Initialize OpenAI LLM service.
        
        Args:
            config: LLM configuration
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        super().__init__(config)

        if config.provider != "openai":
            raise ValueError(f"OpenAI service requires provider='openai', got '{config.provider}'")

        self.client = AsyncOpenAI(api_key=api_key)
        self._function_definitions = self._create_function_definitions()

        logger.info("OpenAI LLM service initialized with model: %s", config.model)

    async def generate_response(
        self,
        transcript: str,
        context: ConversationContext,
    ) -> LLMResponse:
        """
        Generate response from user transcript using OpenAI.
        
        Uses function calling to determine tool execution and applies
        voice-optimized prompt engineering for natural responses.
        
        Args:
            transcript: User's transcribed speech
            context: Conversation context
            
        Returns:
            Complete LLM response with intent, parameters, and text
            
        Raises:
            RuntimeError: If OpenAI API request fails
        """
        try:
            # Build messages with voice-optimized system prompt
            messages = self._build_messages(transcript, context)

            # Call OpenAI with function calling support
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                functions=self._function_definitions,
                function_call="auto",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                timeout=self.config.timeout,
            )

            # Parse response
            return self._parse_response(response, transcript)

        except OpenAIError as e:
            logger.error("OpenAI API error: %s", str(e))
            raise RuntimeError(f"Failed to generate LLM response: {str(e)}") from e
        except Exception as e:
            logger.error("Unexpected error in generate_response: %s", str(e))
            raise RuntimeError(f"Unexpected error: {str(e)}") from e

    def _build_messages(
        self,
        transcript: str,
        context: ConversationContext,
    ) -> list[ChatCompletionMessageParam]:
        """
        Build message list for OpenAI API with voice-optimized prompts.
        
        Args:
            transcript: Current user transcript
            context: Conversation context
            
        Returns:
            List of messages for OpenAI API
        """
        messages: list[ChatCompletionMessageParam] = []

        # System prompt with voice optimization instructions
        system_prompt = self._build_voice_optimized_system_prompt()
        messages.append({
            "role": "system",
            "content": system_prompt,
        })

        # Add conversation history
        for turn in context.conversation_history[-5:]:  # Last 5 turns for context
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ["user", "assistant"]:
                messages.append({
                    "role": role,
                    "content": content,
                })

        # Add current user message
        messages.append({
            "role": "user",
            "content": transcript,
        })

        return messages

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

    def _create_function_definitions(self) -> list[dict[str, Any]]:
        """
        Create OpenAI function definitions for tool execution.
        
        Defines available tools that the LLM can call for appointment
        management and other operations.
        
        Returns:
            List of function definitions in OpenAI format
        """
        return [
            {
                "name": "book_appointment",
                "description": "Book a new appointment for the user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "The date for the appointment (e.g., 'tomorrow', 'Monday', 'January 15th')",
                        },
                        "time": {
                            "type": "string",
                            "description": "The time for the appointment (e.g., '2pm', '14:30', 'morning')",
                        },
                        "reason": {
                            "type": "string",
                            "description": "The reason for the appointment (e.g., 'checkup', 'consultation')",
                        },
                        "provider": {
                            "type": "string",
                            "description": "Preferred healthcare provider or specialist",
                        },
                    },
                    "required": ["date", "time"],
                },
            },
            {
                "name": "reschedule_appointment",
                "description": "Reschedule an existing appointment to a new date/time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "appointment_id": {
                            "type": "string",
                            "description": "The ID of the appointment to reschedule",
                        },
                        "new_date": {
                            "type": "string",
                            "description": "The new date for the appointment",
                        },
                        "new_time": {
                            "type": "string",
                            "description": "The new time for the appointment",
                        },
                    },
                    "required": ["appointment_id", "new_date", "new_time"],
                },
            },
            {
                "name": "cancel_appointment",
                "description": "Cancel an existing appointment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "appointment_id": {
                            "type": "string",
                            "description": "The ID of the appointment to cancel",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Reason for cancellation (optional)",
                        },
                    },
                    "required": ["appointment_id"],
                },
            },
            {
                "name": "search_appointments",
                "description": "Search for available appointment slots or existing appointments",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date to search for appointments",
                        },
                        "provider": {
                            "type": "string",
                            "description": "Specific provider to search for",
                        },
                        "appointment_type": {
                            "type": "string",
                            "description": "Type of appointment (e.g., 'checkup', 'consultation')",
                        },
                    },
                    "required": ["date"],
                },
            },
        ]

    def _parse_response(
        self,
        response: ChatCompletion,
        original_transcript: str,
    ) -> LLMResponse:
        """
        Parse OpenAI response into LLMResponse format.
        
        Handles both regular text responses and function calls,
        extracting intent, parameters, and tool calls.
        
        Args:
            response: OpenAI API response
            original_transcript: Original user transcript
            
        Returns:
            Parsed LLM response
        """
        choice = response.choices[0]
        message = choice.message

        # Check for function call
        function_call = message.function_call
        tool_call = None
        intent_name = IntentType.GENERAL_INQUIRY.value
        parameters = {}

        if function_call:
            # Parse function call
            tool_name = str(function_call.name)
            try:
                tool_params = json.loads(function_call.arguments)
            except json.JSONDecodeError:
                logger.warning("Failed to parse function arguments: %s", function_call.arguments)
                tool_params = {}

            # Map function name to intent
            intent_name = self._map_function_to_intent(tool_name)
            parameters = tool_params

            # Create tool call
            tool_call = ToolCall(
                tool_name=tool_name,
                parameters=tool_params,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            logger.info("Function call detected: %s with params: %s", tool_name, tool_params)

        # Get response text
        response_text = message.content or ""

        # If we have a function call but no response text, generate a default response
        if function_call and not response_text:
            response_text = self._generate_default_response(tool_call.tool_name, parameters)

        # Apply voice optimization
        formatted_text = self.format_for_voice(response_text)

        # Create intent
        intent = Intent(
            name=intent_name,
            confidence=0.9,  # High confidence for OpenAI responses
            parameters=parameters,
            requires_tool=tool_call is not None,
            tool_name=tool_call.tool_name if tool_call else None,
        )

        # Extract emotional tags (if any)
        emotional_tags = self._extract_emotional_tags(response_text)

        return LLMResponse(
            text=formatted_text,
            intent=intent,
            parameters=parameters,
            tool_call=tool_call,
            confidence=0.9,
            emotional_tags=emotional_tags,
            metadata={
                "model": response.model,
                "finish_reason": choice.finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            },
        )

    def _map_function_to_intent(self, function_name: str) -> str:
        """
        Map OpenAI function name to intent type.
        
        Args:
            function_name: Name of the called function
            
        Returns:
            Intent type string
        """
        mapping = {
            "book_appointment": IntentType.SCHEDULE_APPOINTMENT.value,
            "reschedule_appointment": IntentType.RESCHEDULE_APPOINTMENT.value,
            "cancel_appointment": IntentType.CANCEL_APPOINTMENT.value,
            "search_appointments": IntentType.CHECK_APPOINTMENT.value,
        }
        return mapping.get(function_name, IntentType.GENERAL_INQUIRY.value)

    def _generate_default_response(self, tool_name: str, parameters: dict[str, Any]) -> str:
        """
        Generate default response when function call has no accompanying text.
        
        Args:
            tool_name: Name of the tool being called
            parameters: Tool parameters
            
        Returns:
            Default response text
        """
        if tool_name == "book_appointment":
            date = parameters.get("date", "that date")
            time = parameters.get("time", "that time")
            return f"I'll help you schedule an appointment for {date} at {time}."
        elif tool_name == "reschedule_appointment":
            new_date = parameters.get("new_date", "the new date")
            new_time = parameters.get("new_time", "the new time")
            return f"I'll reschedule your appointment to {new_date} at {new_time}."
        elif tool_name == "cancel_appointment":
            return "I'll cancel that appointment for you."
        elif tool_name == "search_appointments":
            date = parameters.get("date", "that date")
            return f"Let me check available appointments for {date}."
        else:
            return "I'll help you with that."

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
