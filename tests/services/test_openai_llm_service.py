"""
Unit tests for OpenAI LLM Service.

Tests OpenAI integration, streaming support, function calling,
voice-optimized prompt engineering, and error handling.

**Validates: Requirements 13.1, 6.1, 6.2, 6.3**
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from openai import OpenAIError
from openai.types.chat import ChatCompletion

from src.voice_agent.services.llm_service import (
    ConversationContext,
    IntentType,
    LLMConfig,
    LLMResponse,
)
from src.voice_agent.services.openai_llm_service import OpenAILLMService


@pytest.fixture
def openai_config():
    """Create OpenAI LLM configuration."""
    return LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful medical receptionist.",
    )


@pytest.fixture
def conversation_context():
    """Create test conversation context."""
    return ConversationContext(
        session_id="test_session_123",
        conversation_history=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help you today?"},
        ],
        extracted_parameters={},
    )


@pytest.fixture
def openai_service(openai_config):
    """Create OpenAI LLM service instance."""
    return OpenAILLMService(openai_config, api_key="test_api_key")


class TestOpenAILLMServiceInitialization:
    """Test OpenAI LLM service initialization."""

    def test_initialization_with_valid_config(self, openai_config):
        """Test service initializes with valid OpenAI configuration."""
        service = OpenAILLMService(openai_config, api_key="test_key")

        assert service.config == openai_config
        assert service.config.provider == "openai"
        assert service.config.model == "gpt-4"
        assert service.client is not None

    def test_initialization_with_invalid_provider(self):
        """Test initialization fails with non-OpenAI provider."""
        config = LLMConfig(
            provider="ollama",  # Wrong provider
            model="llama2",
        )

        with pytest.raises(ValueError, match="OpenAI service requires provider='openai'"):
            OpenAILLMService(config, api_key="test_key")

    def test_initialization_without_api_key(self, openai_config):
        """Test initialization without explicit API key (uses env var)."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env_api_key"}):
            service = OpenAILLMService(openai_config)
            assert service.client is not None

    def test_function_definitions_created(self, openai_service):
        """Test function definitions are created on initialization."""
        assert len(openai_service._function_definitions) > 0

        # Check for expected functions
        function_names = [f["name"] for f in openai_service._function_definitions]
        assert "book_appointment" in function_names
        assert "reschedule_appointment" in function_names
        assert "cancel_appointment" in function_names
        assert "search_appointments" in function_names


class TestVoiceOptimizedSystemPrompt:
    """
    Test voice-optimized system prompt generation.

    **Validates: Requirements 6.1, 6.2, 6.3**
    """

    def test_system_prompt_includes_base_prompt(self, openai_service):
        """Test system prompt includes configured base prompt."""
        prompt = openai_service._build_voice_optimized_system_prompt()

        assert "helpful medical receptionist" in prompt

    def test_system_prompt_includes_voice_rules(self, openai_service):
        """Test system prompt includes voice optimization rules."""
        prompt = openai_service._build_voice_optimized_system_prompt()

        # Check for key voice rules
        assert "markdown" in prompt.lower()
        assert "3 sentences" in prompt.lower()
        assert "conversational" in prompt.lower()
        assert "voice" in prompt.lower()

    def test_system_prompt_forbids_markdown(self, openai_service):
        """Test system prompt explicitly forbids markdown."""
        prompt = openai_service._build_voice_optimized_system_prompt()

        assert "NEVER use markdown" in prompt or "no markdown" in prompt.lower()

    def test_system_prompt_forbids_lists(self, openai_service):
        """Test system prompt forbids bullet points and lists."""
        prompt = openai_service._build_voice_optimized_system_prompt()

        assert "bullet" in prompt.lower() or "list" in prompt.lower()

    def test_system_prompt_requires_brevity(self, openai_service):
        """Test system prompt requires brief responses."""
        prompt = openai_service._build_voice_optimized_system_prompt()

        assert "3 sentences" in prompt.lower() or "brief" in prompt.lower()

    def test_system_prompt_with_empty_base(self):
        """Test system prompt generation with empty base prompt."""
        config = LLMConfig(provider="openai", model="gpt-4", system_prompt="")
        service = OpenAILLMService(config, api_key="test_key")

        prompt = service._build_voice_optimized_system_prompt()

        # Should still have voice instructions
        assert "voice" in prompt.lower()
        assert len(prompt) > 0


class TestMessageBuilding:
    """Test message building for OpenAI API."""

    def test_build_messages_includes_system_prompt(self, openai_service, conversation_context):
        """Test messages include voice-optimized system prompt."""
        messages = openai_service._build_messages("Hello", conversation_context)

        assert len(messages) > 0
        assert messages[0]["role"] == "system"
        assert "voice" in messages[0]["content"].lower()

    def test_build_messages_includes_conversation_history(self, openai_service, conversation_context):
        """Test messages include conversation history."""
        messages = openai_service._build_messages("New message", conversation_context)

        # Should have system + history + current message
        assert len(messages) >= 3

        # Check history is included
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) >= 1

    def test_build_messages_limits_history(self, openai_service):
        """Test message building limits conversation history to last 5 turns."""
        # Create context with many history items
        context = ConversationContext(
            session_id="test",
            conversation_history=[
                {"role": "user", "content": f"Message {i}"}
                for i in range(10)
            ],
        )

        messages = openai_service._build_messages("Current message", context)

        # Should have system + last 5 history + current = 7 messages
        assert len(messages) <= 7

    def test_build_messages_includes_current_transcript(self, openai_service, conversation_context):
        """Test messages include current user transcript."""
        transcript = "I want to book an appointment"
        messages = openai_service._build_messages(transcript, conversation_context)

        # Last message should be current transcript
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == transcript

    def test_build_messages_with_empty_history(self, openai_service):
        """Test message building with empty conversation history."""
        context = ConversationContext(session_id="test", conversation_history=[])
        messages = openai_service._build_messages("Hello", context)

        # Should have system + current message
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


class TestFunctionDefinitions:
    """Test OpenAI function definitions for tool calling."""

    def test_book_appointment_function_definition(self, openai_service):
        """Test book_appointment function is properly defined."""
        functions = openai_service._function_definitions
        book_func = next(f for f in functions if f["name"] == "book_appointment")

        assert book_func["description"]
        assert "parameters" in book_func
        assert "date" in book_func["parameters"]["properties"]
        assert "time" in book_func["parameters"]["properties"]
        assert "date" in book_func["parameters"]["required"]
        assert "time" in book_func["parameters"]["required"]

    def test_reschedule_appointment_function_definition(self, openai_service):
        """Test reschedule_appointment function is properly defined."""
        functions = openai_service._function_definitions
        reschedule_func = next(f for f in functions if f["name"] == "reschedule_appointment")

        assert reschedule_func["description"]
        assert "appointment_id" in reschedule_func["parameters"]["properties"]
        assert "new_date" in reschedule_func["parameters"]["properties"]
        assert "new_time" in reschedule_func["parameters"]["properties"]

    def test_cancel_appointment_function_definition(self, openai_service):
        """Test cancel_appointment function is properly defined."""
        functions = openai_service._function_definitions
        cancel_func = next(f for f in functions if f["name"] == "cancel_appointment")

        assert cancel_func["description"]
        assert "appointment_id" in cancel_func["parameters"]["properties"]
        assert "appointment_id" in cancel_func["parameters"]["required"]

    def test_search_appointments_function_definition(self, openai_service):
        """Test search_appointments function is properly defined."""
        functions = openai_service._function_definitions
        search_func = next(f for f in functions if f["name"] == "search_appointments")

        assert search_func["description"]
        assert "date" in search_func["parameters"]["properties"]
        assert "date" in search_func["parameters"]["required"]

    def test_all_functions_have_required_fields(self, openai_service):
        """Test all function definitions have required fields."""
        for func in openai_service._function_definitions:
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert "type" in func["parameters"]
            assert "properties" in func["parameters"]


class TestResponseParsing:
    """Test parsing OpenAI API responses."""

    def test_parse_text_response_without_function_call(self, openai_service):
        """Test parsing simple text response without function call."""
        # Create mock OpenAI response
        mock_response = Mock(spec=ChatCompletion)
        mock_response.model = "gpt-4"
        mock_response.choices = [
            Mock(
                message=Mock(
                    content="I can help you with that.",
                    function_call=None,
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = Mock(
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
        )

        result = openai_service._parse_response(mock_response, "Hello")

        assert isinstance(result, LLMResponse)
        assert "help you" in result.text
        assert result.tool_call is None
        assert result.intent.name == IntentType.GENERAL_INQUIRY.value

    def test_parse_response_with_function_call(self, openai_service):
        """Test parsing response with function call."""
        # Create mock OpenAI response with function call
        mock_function_call = Mock()
        mock_function_call.name = "book_appointment"
        mock_function_call.arguments = '{"date": "tomorrow", "time": "2pm"}'

        mock_response = Mock(spec=ChatCompletion)
        mock_response.model = "gpt-4"
        mock_response.choices = [
            Mock(
                message=Mock(
                    content="I'll book that appointment for you.",
                    function_call=mock_function_call,
                ),
                finish_reason="function_call",
            )
        ]
        mock_response.usage = Mock(
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
        )

        result = openai_service._parse_response(
            mock_response,
            "I want to book an appointment for tomorrow at 2pm"
        )

        assert isinstance(result, LLMResponse)
        assert result.tool_call is not None
        assert result.tool_call.tool_name == "book_appointment"
        assert result.tool_call.parameters["date"] == "tomorrow"
        assert result.tool_call.parameters["time"] == "2pm"
        assert result.intent.name == IntentType.SCHEDULE_APPOINTMENT.value
        assert result.intent.requires_tool is True

    def test_parse_response_with_function_call_no_text(self, openai_service):
        """Test parsing response with function call but no accompanying text."""
        mock_function_call = Mock()
        mock_function_call.name = "book_appointment"
        mock_function_call.arguments = '{"date": "Monday", "time": "3pm"}'

        mock_response = Mock(spec=ChatCompletion)
        mock_response.model = "gpt-4"
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=None,  # No text response
                    function_call=mock_function_call,
                ),
                finish_reason="function_call",
            )
        ]
        mock_response.usage = Mock(
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
        )

        result = openai_service._parse_response(mock_response, "Book appointment")

        # Should generate default response
        assert len(result.text) > 0
        assert "schedule" in result.text.lower() or "appointment" in result.text.lower()

    def test_parse_response_with_invalid_function_arguments(self, openai_service):
        """Test parsing response with malformed function arguments."""
        mock_function_call = Mock()
        mock_function_call.name = "book_appointment"
        mock_function_call.arguments = 'invalid json{'  # Invalid JSON

        mock_response = Mock(spec=ChatCompletion)
        mock_response.model = "gpt-4"
        mock_response.choices = [
            Mock(
                message=Mock(
                    content="I'll help with that.",
                    function_call=mock_function_call,
                ),
                finish_reason="function_call",
            )
        ]
        mock_response.usage = Mock(
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
        )

        result = openai_service._parse_response(mock_response, "Book appointment")

        # Should handle gracefully with empty parameters
        assert result.tool_call is not None
        assert result.tool_call.parameters == {}

    def test_parse_response_includes_metadata(self, openai_service):
        """Test parsed response includes usage metadata."""
        mock_response = Mock(spec=ChatCompletion)
        mock_response.model = "gpt-4"
        mock_response.choices = [
            Mock(
                message=Mock(content="Response", function_call=None),
                finish_reason="stop",
            )
        ]
        mock_response.usage = Mock(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        result = openai_service._parse_response(mock_response, "Test")

        assert "model" in result.metadata
        assert "usage" in result.metadata
        assert result.metadata["usage"]["prompt_tokens"] == 100
        assert result.metadata["usage"]["completion_tokens"] == 50
        assert result.metadata["usage"]["total_tokens"] == 150


class TestFunctionToIntentMapping:
    """Test mapping OpenAI function names to intent types."""

    def test_map_book_appointment(self, openai_service):
        """Test mapping book_appointment to schedule intent."""
        intent = openai_service._map_function_to_intent("book_appointment")
        assert intent == IntentType.SCHEDULE_APPOINTMENT.value

    def test_map_reschedule_appointment(self, openai_service):
        """Test mapping reschedule_appointment to reschedule intent."""
        intent = openai_service._map_function_to_intent("reschedule_appointment")
        assert intent == IntentType.RESCHEDULE_APPOINTMENT.value

    def test_map_cancel_appointment(self, openai_service):
        """Test mapping cancel_appointment to cancel intent."""
        intent = openai_service._map_function_to_intent("cancel_appointment")
        assert intent == IntentType.CANCEL_APPOINTMENT.value

    def test_map_search_appointments(self, openai_service):
        """Test mapping search_appointments to check intent."""
        intent = openai_service._map_function_to_intent("search_appointments")
        assert intent == IntentType.CHECK_APPOINTMENT.value

    def test_map_unknown_function(self, openai_service):
        """Test mapping unknown function defaults to general inquiry."""
        intent = openai_service._map_function_to_intent("unknown_function")
        assert intent == IntentType.GENERAL_INQUIRY.value


class TestDefaultResponseGeneration:
    """Test default response generation for function calls."""

    def test_default_response_for_book_appointment(self, openai_service):
        """Test default response for booking."""
        response = openai_service._generate_default_response(
            "book_appointment",
            {"date": "tomorrow", "time": "2pm"}
        )

        assert "schedule" in response.lower() or "appointment" in response.lower()
        assert "tomorrow" in response.lower()
        assert "2pm" in response.lower()

    def test_default_response_for_reschedule(self, openai_service):
        """Test default response for rescheduling."""
        response = openai_service._generate_default_response(
            "reschedule_appointment",
            {"new_date": "Friday", "new_time": "10am"}
        )

        assert "reschedule" in response.lower()
        assert "friday" in response.lower()
        assert "10am" in response.lower()

    def test_default_response_for_cancel(self, openai_service):
        """Test default response for cancellation."""
        response = openai_service._generate_default_response(
            "cancel_appointment",
            {"appointment_id": "123"}
        )

        assert "cancel" in response.lower()

    def test_default_response_for_search(self, openai_service):
        """Test default response for search."""
        response = openai_service._generate_default_response(
            "search_appointments",
            {"date": "next week"}
        )

        assert "check" in response.lower() or "search" in response.lower() or "available" in response.lower()

    def test_default_response_with_missing_parameters(self, openai_service):
        """Test default response handles missing parameters gracefully."""
        response = openai_service._generate_default_response(
            "book_appointment",
            {}  # No parameters
        )

        assert len(response) > 0
        assert "appointment" in response.lower()


class TestEmotionalTagExtraction:
    """Test extraction of emotional tags from responses."""

    def test_extract_empathetic_tags(self, openai_service):
        """Test extracting empathetic emotional tags."""
        text = "I'm sorry to hear that. I apologize for the inconvenience."
        tags = openai_service._extract_emotional_tags(text)

        assert "empathetic" in tags

    def test_extract_enthusiastic_tags(self, openai_service):
        """Test extracting enthusiastic emotional tags."""
        text = "That's wonderful! Great choice, this will be perfect for you."
        tags = openai_service._extract_emotional_tags(text)

        assert "enthusiastic" in tags

    def test_extract_understanding_tags(self, openai_service):
        """Test extracting understanding emotional tags."""
        text = "I understand your concern. I hear you."
        tags = openai_service._extract_emotional_tags(text)

        assert "understanding" in tags

    def test_extract_questioning_tags(self, openai_service):
        """Test extracting questioning emotional tags."""
        text = "Would you like me to help with that?"
        tags = openai_service._extract_emotional_tags(text)

        assert "questioning" in tags

    def test_extract_multiple_tags(self, openai_service):
        """Test extracting multiple emotional tags."""
        text = "I understand that's frustrating. Would you like me to help?"
        tags = openai_service._extract_emotional_tags(text)

        assert len(tags) >= 2
        assert "understanding" in tags
        assert "questioning" in tags

    def test_no_emotional_tags(self, openai_service):
        """Test text with no emotional indicators."""
        text = "The appointment is scheduled for tomorrow."
        tags = openai_service._extract_emotional_tags(text)

        assert isinstance(tags, list)


@pytest.mark.asyncio
class TestGenerateResponse:
    """
    Test complete response generation flow.

    **Validates: Requirements 13.1, 6.1, 6.2, 6.3**
    """

    async def test_generate_response_success(self, openai_service, conversation_context):
        """Test successful response generation."""
        # Mock OpenAI API call
        mock_response = Mock(spec=ChatCompletion)
        mock_response.model = "gpt-4"
        mock_response.choices = [
            Mock(
                message=Mock(
                    content="I can help you schedule an appointment.",
                    function_call=None,
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = Mock(
            prompt_tokens=50,
            completion_tokens=15,
            total_tokens=65,
        )

        openai_service.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await openai_service.generate_response(
            "I need an appointment",
            conversation_context
        )

        assert isinstance(result, LLMResponse)
        assert len(result.text) > 0
        assert result.confidence > 0

    async def test_generate_response_with_function_call(self, openai_service, conversation_context):
        """Test response generation with function calling."""
        mock_function_call = Mock()
        mock_function_call.name = "book_appointment"
        mock_function_call.arguments = '{"date": "tomorrow", "time": "2pm", "reason": "checkup"}'

        mock_response = Mock(spec=ChatCompletion)
        mock_response.model = "gpt-4"
        mock_response.choices = [
            Mock(
                message=Mock(
                    content="I'll book that for you.",
                    function_call=mock_function_call,
                ),
                finish_reason="function_call",
            )
        ]
        mock_response.usage = Mock(
            prompt_tokens=60,
            completion_tokens=25,
            total_tokens=85,
        )

        openai_service.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await openai_service.generate_response(
            "Book an appointment for tomorrow at 2pm for a checkup",
            conversation_context
        )

        assert result.tool_call is not None
        assert result.tool_call.tool_name == "book_appointment"
        assert result.intent.requires_tool is True

    async def test_generate_response_api_error(self, openai_service, conversation_context):
        """Test handling OpenAI API errors."""
        openai_service.client.chat.completions.create = AsyncMock(
            side_effect=OpenAIError("API error")
        )

        with pytest.raises(RuntimeError, match="Failed to generate LLM response"):
            await openai_service.generate_response(
                "Test transcript",
                conversation_context
            )

    async def test_generate_response_uses_config_parameters(self, openai_service, conversation_context):
        """Test response generation uses configuration parameters."""
        mock_response = Mock(spec=ChatCompletion)
        mock_response.model = "gpt-4"
        mock_response.choices = [
            Mock(
                message=Mock(content="Response", function_call=None),
                finish_reason="stop",
            )
        ]
        mock_response.usage = Mock(
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
        )

        create_mock = AsyncMock(return_value=mock_response)
        openai_service.client.chat.completions.create = create_mock

        await openai_service.generate_response("Test", conversation_context)

        # Verify config parameters were used
        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 150
        assert call_kwargs["timeout"] == 30.0

    async def test_generate_response_applies_voice_optimization(self, openai_service, conversation_context):
        """Test response applies voice optimization."""
        mock_response = Mock(spec=ChatCompletion)
        mock_response.model = "gpt-4"
        mock_response.choices = [
            Mock(
                message=Mock(
                    content="Here are **three** options:\n- Option 1\n- Option 2\n- Option 3",
                    function_call=None,
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = Mock(
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
        )

        openai_service.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await openai_service.generate_response("Test", conversation_context)

        # Voice optimization should remove markdown and lists
        assert "**" not in result.text
        assert "-" not in result.text or result.text.count("-") < 3

    async def test_generate_response_unexpected_error(self, openai_service, conversation_context):
        """Test handling unexpected errors."""
        openai_service.client.chat.completions.create = AsyncMock(
            side_effect=Exception("Unexpected error")
        )

        with pytest.raises(RuntimeError, match="Unexpected error"):
            await openai_service.generate_response("Test", conversation_context)


class TestVoiceOptimizationIntegration:
    """
    Test voice optimization is properly integrated.

    **Validates: Requirements 6.1, 6.2, 6.3**
    """

    @pytest.mark.asyncio
    async def test_response_has_no_markdown(self, openai_service, conversation_context):
        """Test generated responses contain no markdown."""
        mock_response = Mock(spec=ChatCompletion)
        mock_response.model = "gpt-4"
        mock_response.choices = [
            Mock(
                message=Mock(
                    content="This has **bold** and *italic* text with `code`.",
                    function_call=None,
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = Mock(prompt_tokens=50, completion_tokens=15, total_tokens=65)

        openai_service.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await openai_service.generate_response("Test", conversation_context)

        assert "**" not in result.text
        assert "*" not in result.text
        assert "`" not in result.text

    @pytest.mark.asyncio
    async def test_response_limited_to_three_sentences(self, openai_service, conversation_context):
        """Test responses are limited to 3 sentences."""
        long_response = ". ".join([f"Sentence {i}" for i in range(10)]) + "."

        mock_response = Mock(spec=ChatCompletion)
        mock_response.model = "gpt-4"
        mock_response.choices = [
            Mock(
                message=Mock(content=long_response, function_call=None),
                finish_reason="stop",
            )
        ]
        mock_response.usage = Mock(prompt_tokens=50, completion_tokens=50, total_tokens=100)

        openai_service.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await openai_service.generate_response("Test", conversation_context)

        # Count sentences (periods, exclamations, questions)
        sentence_count = result.text.count('.') + result.text.count('!') + result.text.count('?')
        assert sentence_count <= 3
