"""
LLM Service interface and base implementation.

This module provides an abstract interface for Large Language Model services,
supporting intent classification, parameter extraction, tool execution decisions,
and voice-optimized response generation.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Common intent types for voice agent interactions."""
    SCHEDULE_APPOINTMENT = "schedule_appointment"
    RESCHEDULE_APPOINTMENT = "reschedule_appointment"
    CANCEL_APPOINTMENT = "cancel_appointment"
    CHECK_APPOINTMENT = "check_appointment"
    GENERAL_INQUIRY = "general_inquiry"
    TRANSFER_TO_HUMAN = "transfer_to_human"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    """
    Represents a classified user intent.

    Attributes:
        name: Intent identifier (e.g., "schedule_appointment")
        confidence: Confidence score (0.0-1.0)
        parameters: Extracted parameters from user utterance
        requires_tool: Whether this intent requires tool execution
        tool_name: Name of tool to execute if requires_tool is True
    """
    name: str
    confidence: float
    parameters: dict[str, Any] = field(default_factory=dict)
    requires_tool: bool = False
    tool_name: str | None = None

    def __post_init__(self):
        """Validate intent data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.requires_tool and not self.tool_name:
            raise ValueError("tool_name must be provided when requires_tool is True")


@dataclass
class ToolCall:
    """
    Represents a tool execution request.

    Attributes:
        tool_name: Name of the tool to execute
        parameters: Parameters to pass to the tool
        timestamp: When the tool call was created (ISO format)
    """
    tool_name: str
    parameters: dict[str, Any]
    timestamp: str | None = None


@dataclass
class LLMResponse:
    """
    Complete response from LLM service.

    Attributes:
        text: Generated response text (voice-optimized)
        intent: Classified user intent
        parameters: Extracted parameters from user input
        tool_call: Tool to execute (if applicable)
        confidence: Overall confidence score
        emotional_tags: SSML/audio tags for emotional context
        metadata: Additional response metadata
    """
    text: str
    intent: Intent
    parameters: dict[str, Any] = field(default_factory=dict)
    tool_call: ToolCall | None = None
    confidence: float = 1.0
    emotional_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate response data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class LLMConfig:
    """
    Configuration for LLM service.

    Attributes:
        provider: LLM provider ("openai", "ollama", "vllm")
        model: Model identifier
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
        system_prompt: System/persona prompt
        top_p: Nucleus sampling parameter
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
        presence_penalty: Presence penalty (-2.0 to 2.0)
        timeout: Request timeout in seconds
    """
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 150
    system_prompt: str = ""
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: float = 30.0

    def __post_init__(self):
        """Validate configuration."""
        if self.provider not in ["openai", "ollama", "vllm"]:
            raise ValueError(f"Invalid provider: {self.provider}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")


@dataclass
class ConversationContext:
    """
    Conversation context for LLM processing.

    Attributes:
        session_id: Unique session identifier
        conversation_history: List of previous turns
        extracted_parameters: Parameters extracted so far
        current_intent: Current intent being processed
        tenant_config: Tenant-specific configuration
        metadata: Additional context metadata
    """
    session_id: str
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    extracted_parameters: dict[str, Any] = field(default_factory=dict)
    current_intent: str | None = None
    tenant_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMService(ABC):
    """
    Abstract base class for LLM service implementations.

    Provides interface for intent classification, parameter extraction,
    response generation, and voice optimization. Concrete implementations
    should subclass this for specific providers (OpenAI, Ollama, vLLM).
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM service.

        Args:
            config: LLM configuration
        """
        self.config = config
        logger.info(
            "LLM service initialized (provider: %s, model: %s)",
            config.provider,
            config.model,
        )

    @abstractmethod
    async def generate_response(
        self,
        transcript: str,
        context: ConversationContext,
    ) -> LLMResponse:
        """
        Generate response from user transcript.

        Args:
            transcript: User's transcribed speech
            context: Conversation context

        Returns:
            Complete LLM response with intent, parameters, and text

        Raises:
            RuntimeError: If LLM request fails
        """
        pass

    def classify_intent(self, transcript: str) -> Intent:
        """
        Classify user intent from transcript.

        This is a basic implementation that can be overridden by subclasses
        for more sophisticated intent classification.

        Args:
            transcript: User's transcribed speech

        Returns:
            Classified intent with confidence score
        """
        transcript_lower = transcript.lower()

        # Simple keyword-based classification (order matters - check specific intents first)
        if any(word in transcript_lower for word in ["speak to", "talk to", "human", "agent", "person"]):
            return Intent(
                name=IntentType.TRANSFER_TO_HUMAN.value,
                confidence=0.9,
                requires_tool=False,
            )
        elif any(word in transcript_lower for word in ["reschedule", "change", "move"]) and ("appointment" in transcript_lower or any(w in transcript_lower for w in ["reschedule", "change appointment", "move appointment"])):
            return Intent(
                name=IntentType.RESCHEDULE_APPOINTMENT.value,
                confidence=0.8,
                requires_tool=True,
                tool_name="reschedule_appointment",
            )
        elif any(word in transcript_lower for word in ["cancel", "delete"]):
            # Cancel is likely about appointments in this context
            return Intent(
                name=IntentType.CANCEL_APPOINTMENT.value,
                confidence=0.8,
                requires_tool=True,
                tool_name="cancel_appointment",
            )
        elif any(word in transcript_lower for word in ["check", "view", "see", "show"]) and "appointment" in transcript_lower:
            return Intent(
                name=IntentType.CHECK_APPOINTMENT.value,
                confidence=0.7,
                requires_tool=True,
                tool_name="search_appointments",
            )
        elif any(word in transcript_lower for word in ["schedule", "book", "make"]) and "appointment" in transcript_lower:
            return Intent(
                name=IntentType.SCHEDULE_APPOINTMENT.value,
                confidence=0.8,
                requires_tool=True,
                tool_name="book_appointment",
            )
        else:
            return Intent(
                name=IntentType.GENERAL_INQUIRY.value,
                confidence=0.6,
                requires_tool=False,
            )

    def extract_parameters(self, transcript: str, intent: Intent) -> dict[str, Any]:
        """
        Extract structured parameters from transcript.

        This is a basic implementation using regex patterns. Subclasses
        should override for more sophisticated extraction using LLM.

        Args:
            transcript: User's transcribed speech
            intent: Classified intent

        Returns:
            Dictionary of extracted parameters
        """
        parameters = {}

        # Extract date patterns
        date_patterns = [
            r"(?:on\s+)?(\w+day)",  # Monday, Tuesday, etc.
            r"(?:on\s+)?(\w+\s+\d{1,2}(?:st|nd|rd|th)?)",  # January 15th
            r"(tomorrow|today|next week)",
        ]
        for pattern in date_patterns:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                parameters["date"] = match.group(1)
                break

        # Extract time patterns
        time_patterns = [
            r"(\d{1,2}:\d{2}\s*(?:am|pm)?)",  # 2:30pm
            r"(\d{1,2}\s*(?:am|pm))",  # 2pm
            r"(morning|afternoon|evening)",
        ]
        for pattern in time_patterns:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                parameters["time"] = match.group(1)
                break

        # Extract reason/service type
        if "for" in transcript.lower():
            parts = transcript.lower().split("for", 1)
            if len(parts) > 1:
                reason = parts[1].strip().split()[0:5]  # Take first few words
                parameters["reason"] = " ".join(reason)

        return parameters

    def should_use_tool(self, intent: Intent, parameters: dict[str, Any]) -> ToolCall | None:
        """
        Determine if tool execution is needed.

        Args:
            intent: Classified intent
            parameters: Extracted parameters

        Returns:
            ToolCall if tool should be executed, None otherwise
        """
        if not intent.requires_tool:
            return None

        # Check if we have sufficient parameters for tool execution
        self._get_required_parameters(intent.tool_name)

        # For now, create tool call even with partial parameters
        # The tool executor will handle validation
        if intent.tool_name:
            return ToolCall(
                tool_name=intent.tool_name,
                parameters=parameters,
            )

        return None

    def format_for_voice(self, response: str) -> str:
        """
        Optimize response text for voice output.

        Removes markdown, limits sentence count, and ensures
        conversational language suitable for TTS.

        Args:
            response: Raw response text

        Returns:
            Voice-optimized response text
        """
        # Remove markdown formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', response)  # Bold
        text = re.sub(r'\*(.+?)\*', r'\1', text)  # Italic
        text = re.sub(r'`(.+?)`', r'\1', text)  # Code
        text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)  # Headers (with optional leading whitespace)
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)  # Links

        # Remove bullet points and list markers
        text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

        # Remove tables (remove lines containing pipes)
        lines = text.split('\n')
        text = '\n'.join(line for line in lines if '|' not in line)

        # Split into sentences - match sentence-ending punctuation followed by space and capital letter
        # or sentence-ending punctuation at end of string
        # This avoids splitting on decimal points like "3.14" or abbreviations
        sentences = re.split(r'(?<=[.!?])(?:\s+(?=[A-Z])|$)', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Limit to 3 sentences for voice brevity
        if len(sentences) > 3:
            sentences = sentences[:3]

        # Rejoin sentences with space (they already have punctuation)
        result = ' '.join(sentences)

        # Add period only if there's no ending punctuation
        if result and not result.endswith(('.', '!', '?')):
            result += '.'

        # Final check: ensure we don't have more than 3 sentence-ending punctuation marks total
        # This handles edge cases like "00000000.0. 00000000.0." which has 4 periods
        # Count ALL periods, exclamation marks, and question marks
        punctuation_count = result.count('.') + result.count('!') + result.count('?')
        if punctuation_count > 3:
            # Find the position of the 3rd punctuation mark and truncate there
            count = 0
            for i, char in enumerate(result):
                if char in '.!?':
                    count += 1
                    if count == 3:
                        result = result[:i+1].rstrip()
                        break

        return result

    def _get_required_parameters(self, tool_name: str | None) -> list[str]:
        """
        Get required parameters for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            List of required parameter names
        """
        # Define required parameters per tool
        tool_params = {
            "book_appointment": ["date", "time"],
            "reschedule_appointment": ["appointment_id", "new_date", "new_time"],
            "cancel_appointment": ["appointment_id"],
            "search_appointments": ["date"],
        }

        return tool_params.get(tool_name, [])
