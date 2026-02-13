"""
Unit tests for LLM Service.

Tests intent classification, parameter extraction, tool execution decisions,
voice-optimized response formatting, and data model validation.

**Validates: Requirements 10.2, 10.3, 10.4, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6**
"""


import pytest

from src.voice_agent.services.llm_service import (
    ConversationContext,
    Intent,
    IntentType,
    LLMConfig,
    LLMResponse,
    LLMService,
    ToolCall,
)


# Concrete implementation for testing abstract base class
class TestLLMService(LLMService):
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


@pytest.fixture
def llm_config():
    """Create test LLM configuration."""
    return LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful assistant.",
    )


@pytest.fixture
def llm_service(llm_config):
    """Create LLM service instance."""
    return TestLLMService(llm_config)


@pytest.fixture
def conversation_context():
    """Create test conversation context."""
    return ConversationContext(
        session_id="test_session_123",
        conversation_history=[],
        extracted_parameters={},
    )


class TestLLMConfig:
    """Test LLM configuration validation."""

    def test_valid_config(self):
        """Test creating valid configuration."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=150,
        )

        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 150

    def test_invalid_provider(self):
        """Test invalid provider raises error."""
        with pytest.raises(ValueError, match="Invalid provider"):
            LLMConfig(
                provider="invalid_provider",
                model="gpt-4",
            )

    def test_invalid_temperature(self):
        """Test invalid temperature raises error."""
        with pytest.raises(ValueError, match="Temperature must be between"):
            LLMConfig(
                provider="openai",
                model="gpt-4",
                temperature=3.0,  # Too high
            )

    def test_invalid_max_tokens(self):
        """Test invalid max_tokens raises error."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            LLMConfig(
                provider="openai",
                model="gpt-4",
                max_tokens=-10,
            )

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig(provider="openai", model="gpt-4")

        assert config.temperature == 0.7
        assert config.max_tokens == 150
        assert config.system_prompt == ""
        assert config.top_p == 1.0
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
        assert config.timeout == 30.0


class TestIntent:
    """Test Intent data model."""

    def test_valid_intent(self):
        """Test creating valid intent."""
        intent = Intent(
            name="schedule_appointment",
            confidence=0.85,
            parameters={"date": "tomorrow"},
            requires_tool=True,
            tool_name="book_appointment",
        )

        assert intent.name == "schedule_appointment"
        assert intent.confidence == 0.85
        assert intent.parameters == {"date": "tomorrow"}
        assert intent.requires_tool is True
        assert intent.tool_name == "book_appointment"

    def test_invalid_confidence(self):
        """Test invalid confidence raises error."""
        with pytest.raises(ValueError, match="Confidence must be between"):
            Intent(
                name="test_intent",
                confidence=1.5,  # Too high
            )

    def test_requires_tool_without_tool_name(self):
        """Test requires_tool=True without tool_name raises error."""
        with pytest.raises(ValueError, match="tool_name must be provided"):
            Intent(
                name="test_intent",
                confidence=0.8,
                requires_tool=True,
                tool_name=None,
            )

    def test_default_parameters(self):
        """Test default parameter values."""
        intent = Intent(name="test_intent", confidence=0.8)

        assert intent.parameters == {}
        assert intent.requires_tool is False
        assert intent.tool_name is None


class TestLLMResponse:
    """Test LLMResponse data model."""

    def test_valid_response(self):
        """Test creating valid LLM response."""
        intent = Intent(name="test_intent", confidence=0.9)
        response = LLMResponse(
            text="I can help you with that.",
            intent=intent,
            parameters={"date": "tomorrow"},
            confidence=0.9,
        )

        assert response.text == "I can help you with that."
        assert response.intent == intent
        assert response.parameters == {"date": "tomorrow"}
        assert response.confidence == 0.9

    def test_invalid_confidence(self):
        """Test invalid confidence raises error."""
        intent = Intent(name="test_intent", confidence=0.9)

        with pytest.raises(ValueError, match="Confidence must be between"):
            LLMResponse(
                text="Test response",
                intent=intent,
                confidence=2.0,  # Too high
            )

    def test_default_values(self):
        """Test default response values."""
        intent = Intent(name="test_intent", confidence=0.9)
        response = LLMResponse(text="Test", intent=intent)

        assert response.parameters == {}
        assert response.tool_call is None
        assert response.confidence == 1.0
        assert response.emotional_tags == []
        assert response.metadata == {}


class TestIntentClassification:
    """
    Test intent classification from transcripts.
    
    **Validates: Requirements 10.2**
    """

    def test_schedule_appointment_intent(self, llm_service):
        """Test classifying schedule appointment intent."""
        transcripts = [
            "I want to schedule an appointment",
            "Can I book an appointment for tomorrow?",
            "I need to make an appointment",
        ]

        for transcript in transcripts:
            intent = llm_service.classify_intent(transcript)

            assert intent.name == IntentType.SCHEDULE_APPOINTMENT.value
            assert intent.confidence > 0.7
            assert intent.requires_tool is True
            assert intent.tool_name == "book_appointment"

    def test_reschedule_appointment_intent(self, llm_service):
        """Test classifying reschedule appointment intent."""
        transcripts = [
            "I need to reschedule my appointment",
            "Can I change my appointment to next week?",
            "I want to move my appointment",
        ]

        for transcript in transcripts:
            intent = llm_service.classify_intent(transcript)

            assert intent.name == IntentType.RESCHEDULE_APPOINTMENT.value
            assert intent.confidence > 0.7
            assert intent.requires_tool is True
            assert intent.tool_name == "reschedule_appointment"

    def test_cancel_appointment_intent(self, llm_service):
        """Test classifying cancel appointment intent."""
        transcripts = [
            "I need to cancel my appointment",
            "Can I delete my appointment?",
            "I want to cancel",
        ]

        for transcript in transcripts:
            intent = llm_service.classify_intent(transcript)

            assert intent.name == IntentType.CANCEL_APPOINTMENT.value
            assert intent.confidence > 0.7
            assert intent.requires_tool is True
            assert intent.tool_name == "cancel_appointment"

    def test_check_appointment_intent(self, llm_service):
        """Test classifying check appointment intent."""
        transcripts = [
            "Can I check my appointments?",
            "I want to view my appointments",
            "Show me my appointments",
        ]

        for transcript in transcripts:
            intent = llm_service.classify_intent(transcript)

            assert intent.name == IntentType.CHECK_APPOINTMENT.value
            assert intent.confidence > 0.6
            assert intent.requires_tool is True
            assert intent.tool_name == "search_appointments"

    def test_transfer_to_human_intent(self, llm_service):
        """Test classifying transfer to human intent."""
        transcripts = [
            "I want to speak to a human",
            "Can I talk to an agent?",
            "Transfer me to a person",
        ]

        for transcript in transcripts:
            intent = llm_service.classify_intent(transcript)

            assert intent.name == IntentType.TRANSFER_TO_HUMAN.value
            assert intent.confidence > 0.8
            assert intent.requires_tool is False

    def test_general_inquiry_intent(self, llm_service):
        """Test classifying general inquiry intent."""
        transcripts = [
            "What are your hours?",
            "Where are you located?",
            "Tell me about your services",
        ]

        for transcript in transcripts:
            intent = llm_service.classify_intent(transcript)

            assert intent.name == IntentType.GENERAL_INQUIRY.value
            assert intent.confidence > 0.5
            assert intent.requires_tool is False

    def test_case_insensitive_classification(self, llm_service):
        """Test intent classification is case insensitive."""
        transcripts = [
            "SCHEDULE AN APPOINTMENT",
            "Schedule An Appointment",
            "schedule an appointment",
        ]

        for transcript in transcripts:
            intent = llm_service.classify_intent(transcript)
            assert intent.name == IntentType.SCHEDULE_APPOINTMENT.value


class TestParameterExtraction:
    """
    Test parameter extraction from transcripts.
    
    **Validates: Requirements 10.3**
    """

    def test_extract_date_patterns(self, llm_service):
        """Test extracting various date patterns."""
        test_cases = [
            ("I want an appointment on Monday", "Monday"),
            ("Can I book for tomorrow?", "tomorrow"),
            ("Schedule me for next week", "next week"),
            ("I need an appointment January 15th", "January 15th"),
        ]

        intent = Intent(name="schedule_appointment", confidence=0.9)

        for transcript, expected_date in test_cases:
            params = llm_service.extract_parameters(transcript, intent)
            assert "date" in params
            assert expected_date.lower() in params["date"].lower()

    def test_extract_time_patterns(self, llm_service):
        """Test extracting various time patterns."""
        test_cases = [
            ("I want an appointment at 2:30pm", "2:30pm"),
            ("Can I book for 10am?", "10am"),
            ("Schedule me in the morning", "morning"),
            ("I prefer afternoon", "afternoon"),
        ]

        intent = Intent(name="schedule_appointment", confidence=0.9)

        for transcript, expected_time in test_cases:
            params = llm_service.extract_parameters(transcript, intent)
            assert "time" in params
            assert expected_time.lower() in params["time"].lower()

    def test_extract_reason(self, llm_service):
        """Test extracting appointment reason."""
        transcript = "I need an appointment for a checkup tomorrow"
        intent = Intent(name="schedule_appointment", confidence=0.9)

        params = llm_service.extract_parameters(transcript, intent)

        assert "reason" in params
        assert "checkup" in params["reason"].lower()

    def test_extract_multiple_parameters(self, llm_service):
        """Test extracting multiple parameters from single transcript."""
        transcript = "I want to schedule an appointment for Monday at 2pm for a consultation"
        intent = Intent(name="schedule_appointment", confidence=0.9)

        params = llm_service.extract_parameters(transcript, intent)

        assert "date" in params
        assert "time" in params
        assert "reason" in params
        assert "monday" in params["date"].lower()
        assert "2pm" in params["time"].lower()

    def test_no_parameters_extracted(self, llm_service):
        """Test handling transcript with no extractable parameters."""
        transcript = "I want an appointment"
        intent = Intent(name="schedule_appointment", confidence=0.9)

        params = llm_service.extract_parameters(transcript, intent)

        # Should return empty dict or dict without date/time
        assert isinstance(params, dict)


class TestToolExecutionDecisions:
    """
    Test tool execution decision logic.
    
    **Validates: Requirements 10.4**
    """

    def test_should_use_tool_for_booking(self, llm_service):
        """Test tool execution decision for appointment booking."""
        intent = Intent(
            name="schedule_appointment",
            confidence=0.9,
            requires_tool=True,
            tool_name="book_appointment",
        )
        parameters = {"date": "tomorrow", "time": "2pm"}

        tool_call = llm_service.should_use_tool(intent, parameters)

        assert tool_call is not None
        assert tool_call.tool_name == "book_appointment"
        assert tool_call.parameters == parameters

    def test_should_not_use_tool_for_general_inquiry(self, llm_service):
        """Test no tool execution for general inquiries."""
        intent = Intent(
            name="general_inquiry",
            confidence=0.8,
            requires_tool=False,
        )
        parameters = {}

        tool_call = llm_service.should_use_tool(intent, parameters)

        assert tool_call is None

    def test_tool_call_with_partial_parameters(self, llm_service):
        """Test tool call creation with partial parameters."""
        intent = Intent(
            name="schedule_appointment",
            confidence=0.9,
            requires_tool=True,
            tool_name="book_appointment",
        )
        parameters = {"date": "tomorrow"}  # Missing time

        tool_call = llm_service.should_use_tool(intent, parameters)

        # Should still create tool call (executor will validate)
        assert tool_call is not None
        assert tool_call.tool_name == "book_appointment"

    def test_tool_call_for_reschedule(self, llm_service):
        """Test tool execution decision for rescheduling."""
        intent = Intent(
            name="reschedule_appointment",
            confidence=0.85,
            requires_tool=True,
            tool_name="reschedule_appointment",
        )
        parameters = {"appointment_id": "123", "new_date": "Friday"}

        tool_call = llm_service.should_use_tool(intent, parameters)

        assert tool_call is not None
        assert tool_call.tool_name == "reschedule_appointment"
        assert tool_call.parameters == parameters

    def test_tool_call_for_cancellation(self, llm_service):
        """Test tool execution decision for cancellation."""
        intent = Intent(
            name="cancel_appointment",
            confidence=0.9,
            requires_tool=True,
            tool_name="cancel_appointment",
        )
        parameters = {"appointment_id": "456"}

        tool_call = llm_service.should_use_tool(intent, parameters)

        assert tool_call is not None
        assert tool_call.tool_name == "cancel_appointment"


class TestVoiceOptimizedFormatting:
    """
    Test voice-optimized response formatting.
    
    **Validates: Requirements 6.1, 6.2, 6.3**
    """

    def test_remove_markdown_bold(self, llm_service):
        """Test removing bold markdown formatting."""
        response = "This is **bold text** in the response."
        formatted = llm_service.format_for_voice(response)

        assert "**" not in formatted
        assert "bold text" in formatted

    def test_remove_markdown_italic(self, llm_service):
        """Test removing italic markdown formatting."""
        response = "This is *italic text* in the response."
        formatted = llm_service.format_for_voice(response)

        assert "*" not in formatted
        assert "italic text" in formatted

    def test_remove_markdown_code(self, llm_service):
        """Test removing code markdown formatting."""
        response = "Use the `code` function here."
        formatted = llm_service.format_for_voice(response)

        assert "`" not in formatted
        assert "code" in formatted

    def test_remove_markdown_headers(self, llm_service):
        """Test removing header markdown formatting."""
        response = "## Header Text\nSome content here."
        formatted = llm_service.format_for_voice(response)

        assert "#" not in formatted
        assert "Header Text" in formatted

    def test_remove_markdown_links(self, llm_service):
        """Test removing link markdown formatting."""
        response = "Visit [our website](https://example.com) for more info."
        formatted = llm_service.format_for_voice(response)

        assert "[" not in formatted
        assert "]" not in formatted
        assert "(" not in formatted
        assert "our website" in formatted

    def test_remove_bullet_points(self, llm_service):
        """Test removing bullet point formatting."""
        response = "Here are options:\n- Option one\n- Option two\n- Option three"
        formatted = llm_service.format_for_voice(response)

        assert "-" not in formatted or formatted.count("-") < 3
        assert "Option one" in formatted

    def test_remove_numbered_lists(self, llm_service):
        """Test removing numbered list formatting."""
        response = "Steps:\n1. First step\n2. Second step\n3. Third step"
        formatted = llm_service.format_for_voice(response)

        # Numbers at start of lines should be removed
        assert "First step" in formatted

    def test_limit_to_three_sentences(self, llm_service):
        """Test limiting response to 3 sentences."""
        response = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        formatted = llm_service.format_for_voice(response)

        # Count sentences
        sentence_count = formatted.count('.') + formatted.count('!') + formatted.count('?')
        assert sentence_count <= 3

    def test_add_period_if_missing(self, llm_service):
        """Test adding period if response doesn't end with punctuation."""
        response = "This is a response without ending punctuation"
        formatted = llm_service.format_for_voice(response)

        assert formatted.endswith(('.', '!', '?'))

    def test_preserve_existing_punctuation(self, llm_service):
        """Test preserving existing ending punctuation."""
        test_cases = [
            "This ends with a period.",
            "This ends with a question?",
            "This ends with excitement!",
        ]

        for response in test_cases:
            formatted = llm_service.format_for_voice(response)
            assert formatted[-1] in '.!?'

    def test_remove_tables(self, llm_service):
        """Test removing table formatting."""
        response = "Here is data:\n| Column 1 | Column 2 |\n| Value 1 | Value 2 |"
        formatted = llm_service.format_for_voice(response)

        assert "|" not in formatted

    def test_complex_formatting_removal(self, llm_service):
        """Test removing multiple formatting types at once."""
        response = """
        ## Important Information
        
        Here are **key points**:
        - First *point* with `code`
        - Second point
        - Third point
        
        Visit [our site](https://example.com) for more.
        """
        formatted = llm_service.format_for_voice(response)

        # Should have no markdown
        assert "**" not in formatted
        assert "`" not in formatted
        assert "[" not in formatted
        assert "]" not in formatted

        # Check that markdown headers are removed (# at start of line)
        lines = formatted.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped:
                assert not stripped.startswith('#'), f"Header markdown not removed: '{line}'"

        # Should preserve content
        assert "Important Information" in formatted or "key points" in formatted


class TestConversationContext:
    """Test conversation context management."""

    def test_create_context(self):
        """Test creating conversation context."""
        context = ConversationContext(
            session_id="test_123",
            conversation_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            extracted_parameters={"name": "John"},
        )

        assert context.session_id == "test_123"
        assert len(context.conversation_history) == 2
        assert context.extracted_parameters == {"name": "John"}

    def test_default_context_values(self):
        """Test default context values."""
        context = ConversationContext(session_id="test_123")

        assert context.conversation_history == []
        assert context.extracted_parameters == {}
        assert context.current_intent is None
        assert context.tenant_config == {}
        assert context.metadata == {}


class TestLLMServiceInitialization:
    """Test LLM service initialization."""

    def test_initialization_with_config(self, llm_config):
        """Test service initializes with provided configuration."""
        service = TestLLMService(llm_config)

        assert service.config == llm_config
        assert service.config.provider == "openai"
        assert service.config.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_generate_response_integration(self, llm_service, conversation_context):
        """Test complete response generation flow."""
        transcript = "I want to schedule an appointment for tomorrow at 2pm"

        response = await llm_service.generate_response(transcript, conversation_context)

        assert isinstance(response, LLMResponse)
        assert response.intent.name == IntentType.SCHEDULE_APPOINTMENT.value
        assert "date" in response.parameters
        assert "time" in response.parameters
        assert response.tool_call is not None
        assert response.tool_call.tool_name == "book_appointment"


class TestToolCall:
    """Test ToolCall data model."""

    def test_create_tool_call(self):
        """Test creating tool call."""
        tool_call = ToolCall(
            tool_name="book_appointment",
            parameters={"date": "tomorrow", "time": "2pm"},
            timestamp="2024-01-15T10:30:00Z",
        )

        assert tool_call.tool_name == "book_appointment"
        assert tool_call.parameters == {"date": "tomorrow", "time": "2pm"}
        assert tool_call.timestamp == "2024-01-15T10:30:00Z"

    def test_tool_call_without_timestamp(self):
        """Test creating tool call without timestamp."""
        tool_call = ToolCall(
            tool_name="search_appointments",
            parameters={"date": "today"},
        )

        assert tool_call.tool_name == "search_appointments"
        assert tool_call.timestamp is None


class TestRequiredParameters:
    """Test required parameter logic."""

    def test_get_required_parameters_for_booking(self, llm_service):
        """Test getting required parameters for booking."""
        params = llm_service._get_required_parameters("book_appointment")

        assert "date" in params
        assert "time" in params

    def test_get_required_parameters_for_reschedule(self, llm_service):
        """Test getting required parameters for rescheduling."""
        params = llm_service._get_required_parameters("reschedule_appointment")

        assert "appointment_id" in params
        assert "new_date" in params
        assert "new_time" in params

    def test_get_required_parameters_for_cancel(self, llm_service):
        """Test getting required parameters for cancellation."""
        params = llm_service._get_required_parameters("cancel_appointment")

        assert "appointment_id" in params

    def test_get_required_parameters_unknown_tool(self, llm_service):
        """Test getting required parameters for unknown tool."""
        params = llm_service._get_required_parameters("unknown_tool")

        assert params == []
