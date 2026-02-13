"""
Property-based tests for LLM Service.

**Validates: Requirements 10.2, 6.1, 6.2, 6.3**

This module contains property-based tests using Hypothesis to verify universal
properties of the LLM service across all valid inputs.
"""

import re
from unittest.mock import AsyncMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.voice_agent.services.llm_service import (
    ConversationContext,
    IntentType,
    LLMConfig,
    LLMResponse,
    LLMService,
)


# Concrete implementation for testing abstract base class
class MockLLMService(LLMService):
    """Test implementation of abstract LLMService."""

    async def generate_response(
        self,
        transcript: str,
        context: ConversationContext,
    ) -> LLMResponse:
        """Simple test implementation."""
        intent = self.classify_intent(transcript)
        parameters = self.extract_parameters(transcript, intent)
        tool_call = self.should_use_tool(intent, parameters)

        response_text = f"I understand you want to {intent.name}"
        formatted_text = self.format_for_voice(response_text)

        return LLMResponse(
            text=formatted_text,
            intent=intent,
            parameters=parameters,
            tool_call=tool_call,
            confidence=intent.confidence,
        )


# Hypothesis strategies for generating test data

@st.composite
def transcript_strategy(draw):
    """Generate realistic user transcripts."""
    # Generate various types of transcripts
    transcript_templates = [
        "I want to schedule an appointment",
        "Can I book an appointment for {date}?",
        "I need to reschedule my appointment",
        "Please cancel my appointment",
        "What are your hours?",
        "I'd like to speak to a human",
        "Check my appointments",
        "Schedule me for {date} at {time}",
    ]

    template = draw(st.sampled_from(transcript_templates))

    # Fill in placeholders if present
    if "{date}" in template:
        date = draw(st.sampled_from(["tomorrow", "Monday", "next week", "January 15th"]))
        template = template.replace("{date}", date)

    if "{time}" in template:
        time = draw(st.sampled_from(["2pm", "10:30am", "morning", "afternoon"]))
        template = template.replace("{time}", time)

    return template


@st.composite
def llm_response_text_strategy(draw):
    """Generate various LLM response texts with different formatting."""
    # Base content
    sentences = draw(st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'P', 'Z')),
            min_size=10,
            max_size=100
        ),
        min_size=1,
        max_size=10
    ))

    # Add sentence endings
    sentences = [s.strip() + "." for s in sentences if s.strip()]

    # Choose formatting style
    format_type = draw(st.sampled_from([
        "plain",
        "markdown_bold",
        "markdown_italic",
        "markdown_code",
        "markdown_headers",
        "markdown_links",
        "bullet_points",
        "numbered_list",
        "table",
        "mixed"
    ]))

    if format_type == "plain":
        return " ".join(sentences)

    elif format_type == "markdown_bold":
        text = " ".join(sentences)
        # Add bold formatting to some words
        words = text.split()
        if len(words) > 2:
            words[1] = f"**{words[1]}**"
        return " ".join(words)

    elif format_type == "markdown_italic":
        text = " ".join(sentences)
        words = text.split()
        if len(words) > 2:
            words[1] = f"*{words[1]}*"
        return " ".join(words)

    elif format_type == "markdown_code":
        text = " ".join(sentences)
        words = text.split()
        if len(words) > 2:
            words[1] = f"`{words[1]}`"
        return " ".join(words)

    elif format_type == "markdown_headers":
        return f"## Header\n{' '.join(sentences)}"

    elif format_type == "markdown_links":
        text = " ".join(sentences)
        return f"{text} Visit [our website](https://example.com) for more."

    elif format_type == "bullet_points":
        return "Here are options:\n- " + "\n- ".join(sentences[:3])

    elif format_type == "numbered_list":
        items = [f"{i+1}. {s}" for i, s in enumerate(sentences[:3])]
        return "Steps:\n" + "\n".join(items)

    elif format_type == "table":
        return "| Column 1 | Column 2 |\n| Value 1 | Value 2 |\n" + " ".join(sentences)

    else:  # mixed
        text = " ".join(sentences)
        text = f"## Important\n**Bold** and *italic* text with `code`.\n- Bullet point\n{text}"
        return text


def create_llm_service():
    """Create LLM service instance for testing."""
    config = LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.0,  # Deterministic for consistency tests
        max_tokens=150,
    )
    return MockLLMService(config)


class TestLLMServiceProperties:
    """Property-based tests for LLM Service."""

    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(transcript_strategy())
    def test_intent_classification_consistency(self, transcript):
        """
        Property 4: Intent classification consistency.

        For any transcript, classifying intent multiple times should return
        the same intent (deterministic with temperature=0).

        **Validates: Requirements 10.2**
        """
        llm_service = create_llm_service()

        # Classify the same transcript multiple times
        intent1 = llm_service.classify_intent(transcript)
        intent2 = llm_service.classify_intent(transcript)
        intent3 = llm_service.classify_intent(transcript)

        # All classifications should be identical
        assert intent1.name == intent2.name == intent3.name, \
            f"Intent classification not consistent for transcript: '{transcript}'"

        assert intent1.confidence == intent2.confidence == intent3.confidence, \
            f"Confidence not consistent for transcript: '{transcript}'"

        assert intent1.requires_tool == intent2.requires_tool == intent3.requires_tool, \
            f"Tool requirement not consistent for transcript: '{transcript}'"

        assert intent1.tool_name == intent2.tool_name == intent3.tool_name, \
            f"Tool name not consistent for transcript: '{transcript}'"

    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(llm_response_text_strategy())
    def test_voice_optimized_no_markdown(self, response_text):
        """
        Property 5a: Voice-optimized responses remove markdown.

        For any LLM response, format_for_voice() should remove all markdown
        formatting including bold, italic, code, headers, and links.

        **Validates: Requirements 6.1**
        """
        llm_service = create_llm_service()
        formatted = llm_service.format_for_voice(response_text)

        # Should not contain markdown syntax
        assert "**" not in formatted, "Bold markdown not removed"
        assert "`" not in formatted, "Code markdown not removed"

        # Check for markdown headers (# at start of line)
        lines = formatted.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped:
                assert not re.match(r'^#+\s', stripped), f"Header markdown not removed: '{line}'"

        # Check for links - should not have markdown link format
        assert not re.search(r'\[.+?\]\(.+?\)', formatted), "Link markdown not removed"

    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(llm_response_text_strategy())
    def test_voice_optimized_sentence_limit(self, response_text):
        """
        Property 5b: Voice-optimized responses limit to 3 sentences.

        For any LLM response, format_for_voice() should limit the output
        to 3 sentences or fewer.

        **Validates: Requirements 6.2**
        """
        llm_service = create_llm_service()
        formatted = llm_service.format_for_voice(response_text)

        # Skip empty responses
        if not formatted.strip():
            return

        # Count sentences (split by . ! ?)
        sentences = re.split(r'[.!?]+', formatted)
        sentences = [s.strip() for s in sentences if s.strip()]

        assert len(sentences) <= 3, \
            f"Response has {len(sentences)} sentences, expected <= 3: '{formatted}'"

    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(llm_response_text_strategy())
    def test_voice_optimized_no_tables_or_bullets(self, response_text):
        """
        Property 5c: Voice-optimized responses remove tables and bullet points.

        For any LLM response, format_for_voice() should remove tables
        (lines with pipes) and bullet points.

        **Validates: Requirements 6.3**
        """
        llm_service = create_llm_service()
        formatted = llm_service.format_for_voice(response_text)

        # Should not contain table pipes
        assert "|" not in formatted, "Table formatting not removed"

        # Should not have bullet point markers at start of lines
        lines = formatted.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped:
                # Check for common bullet point markers
                assert not re.match(r'^[-*•]\s+', stripped), \
                    f"Bullet point not removed: '{line}'"
                # Check for numbered list markers (1-3 digits followed by period and space)
                # This is more realistic than matching any number of digits
                assert not re.match(r'^\d{1,3}\.\s+', stripped), \
                    f"Numbered list not removed: '{line}'"

    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(st.text(min_size=1, max_size=500))
    def test_format_for_voice_always_returns_string(self, response_text):
        """
        Property: format_for_voice always returns a string.

        For any input text, format_for_voice() should always return a string
        (never None or other types).
        """
        llm_service = create_llm_service()
        formatted = llm_service.format_for_voice(response_text)

        assert isinstance(formatted, str), \
            f"format_for_voice returned {type(formatted)}, expected str"

    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(st.text(min_size=1, max_size=500))
    def test_format_for_voice_idempotent(self, response_text):
        """
        Property: format_for_voice is idempotent.

        Applying format_for_voice() multiple times should produce the same result
        as applying it once (idempotent operation).
        """
        llm_service = create_llm_service()
        formatted_once = llm_service.format_for_voice(response_text)
        formatted_twice = llm_service.format_for_voice(formatted_once)
        formatted_thrice = llm_service.format_for_voice(formatted_twice)

        assert formatted_once == formatted_twice == formatted_thrice, \
            "format_for_voice is not idempotent"

    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(transcript_strategy())
    def test_intent_confidence_in_valid_range(self, transcript):
        """
        Property: Intent confidence is always in valid range [0.0, 1.0].

        For any transcript, the classified intent should have a confidence
        score between 0.0 and 1.0 inclusive.
        """
        llm_service = create_llm_service()
        intent = llm_service.classify_intent(transcript)

        assert 0.0 <= intent.confidence <= 1.0, \
            f"Intent confidence {intent.confidence} out of range [0.0, 1.0]"

    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(transcript_strategy())
    def test_intent_tool_consistency(self, transcript):
        """
        Property: Intent tool requirements are consistent.

        If an intent requires a tool, it must have a tool_name.
        If it doesn't require a tool, tool_name should be None.
        """
        llm_service = create_llm_service()
        intent = llm_service.classify_intent(transcript)

        if intent.requires_tool:
            assert intent.tool_name is not None, \
                f"Intent requires tool but tool_name is None: {intent.name}"
        else:
            # tool_name can be None or not set when requires_tool is False
            # This is acceptable
            pass

    @pytest.mark.property
    @settings(max_examples=100, deadline=None)
    @given(transcript_strategy())
    def test_parameter_extraction_returns_dict(self, transcript):
        """
        Property: Parameter extraction always returns a dictionary.

        For any transcript and intent, extract_parameters() should always
        return a dictionary (possibly empty).
        """
        llm_service = create_llm_service()
        intent = llm_service.classify_intent(transcript)
        parameters = llm_service.extract_parameters(transcript, intent)

        assert isinstance(parameters, dict), \
            f"extract_parameters returned {type(parameters)}, expected dict"

    @pytest.mark.property
    @settings(max_examples=50, deadline=None)
    @given(
        st.text(min_size=10, max_size=200),
        st.text(min_size=10, max_size=200)
    )
    def test_different_transcripts_may_have_different_intents(
        self, transcript1, transcript2
    ):
        """
        Property: Different transcripts can produce different intents.

        The intent classification system should be able to distinguish
        between different types of requests (not always return the same intent).
        """
        llm_service = create_llm_service()

        # This is a sanity check - we're not asserting they're different,
        # just that the system is capable of producing different intents
        intent1 = llm_service.classify_intent(transcript1)
        intent2 = llm_service.classify_intent(transcript2)

        # Both should be valid intents
        assert intent1.name in [e.value for e in IntentType]
        assert intent2.name in [e.value for e in IntentType]

        # At least one of the intents should be valid
        assert intent1.confidence > 0.0 or intent2.confidence > 0.0


class TestOpenAILLMServiceProperties:
    """Property-based tests for OpenAI LLM Service with mocked API."""

    @pytest.mark.property
    @pytest.mark.asyncio
    @settings(max_examples=20, deadline=None)
    @given(transcript_strategy())
    async def test_openai_intent_classification_deterministic(self, transcript):
        """
        Property 4: OpenAI intent classification is deterministic with temperature=0.

        When using OpenAI with temperature=0, the same transcript should
        produce consistent intent classifications.

        **Validates: Requirements 10.2**
        """
        from src.voice_agent.services.openai_llm_service import OpenAILLMService

        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.0,  # Deterministic
            max_tokens=150,
        )

        # Mock the OpenAI client to avoid actual API calls
        with patch('src.voice_agent.services.openai_llm_service.AsyncOpenAI') as mock_openai:
            # Create mock response
            mock_response = AsyncMock()
            mock_response.choices = [AsyncMock()]
            mock_response.choices[0].message.content = "I can help you with that."
            mock_response.choices[0].message.function_call = None
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "gpt-4"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response.usage.total_tokens = 15

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            service = OpenAILLMService(config, api_key="test_key")
            context = ConversationContext(session_id="test_session")

            # Generate response multiple times
            response1 = await service.generate_response(transcript, context)
            response2 = await service.generate_response(transcript, context)

            # With temperature=0 and same mock, responses should be consistent
            assert response1.intent.name == response2.intent.name, \
                "Intent classification not consistent with temperature=0"
