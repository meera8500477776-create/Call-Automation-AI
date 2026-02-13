"""
Unit tests for Ollama LLM Service.

Tests Ollama integration with streaming support, GPU acceleration,
and fallback handling for local model unavailability.

**Validates: Requirements 13.2, 13.5, 7.6**
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.voice_agent.services.llm_service import (
    ConversationContext,
    Intent,
    IntentType,
    LLMConfig,
    LLMResponse,
)
from src.voice_agent.services.ollama_llm_service import OllamaLLMService


def create_mock_response(status, json_data):
    """Helper to create mock HTTP response with async context manager support."""
    mock_response = Mock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=json_data)
    mock_response.text = AsyncMock(return_value=str(json_data))

    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_cm.__aexit__ = AsyncMock(return_value=None)

    return mock_cm


@pytest.fixture
def ollama_config():
    """Create Ollama LLM configuration."""
    return LLMConfig(
        provider="ollama",
        model="llama2",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful medical receptionist.",
    )


@pytest.fixture
def conversation_context():
    """Create conversation context."""
    return ConversationContext(
        session_id="test-session-123",
        conversation_history=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help you today?"},
        ],
        extracted_parameters={},
    )


@pytest.fixture
async def ollama_service(ollama_config):
    """Create Ollama LLM service instance."""
    service = OllamaLLMService(
        config=ollama_config,
        base_url="http://localhost:11434",
        gpu_enabled=True,
    )
    yield service
    await service.close()


class TestOllamaLLMServiceInitialization:
    """Test Ollama LLM service initialization."""

    def test_initialization_with_valid_config(self, ollama_config):
        """Test service initializes with valid configuration."""
        service = OllamaLLMService(config=ollama_config)

        assert service.config.provider == "ollama"
        assert service.config.model == "llama2"
        assert service.base_url == "http://localhost:11434"
        assert service.gpu_enabled is True
        assert service.fallback_service is None

    def test_initialization_with_custom_base_url(self, ollama_config):
        """Test service initializes with custom base URL."""
        service = OllamaLLMService(
            config=ollama_config,
            base_url="http://custom-host:8080/",
        )

        assert service.base_url == "http://custom-host:8080"

    def test_initialization_with_gpu_disabled(self, ollama_config):
        """Test service initializes with GPU disabled."""
        service = OllamaLLMService(
            config=ollama_config,
            gpu_enabled=False,
        )

        assert service.gpu_enabled is False

    def test_initialization_with_fallback_service(self, ollama_config):
        """Test service initializes with fallback service."""
        fallback = MagicMock()
        service = OllamaLLMService(
            config=ollama_config,
            fallback_service=fallback,
        )

        assert service.fallback_service is fallback

    def test_initialization_rejects_wrong_provider(self):
        """Test service rejects non-ollama provider."""
        config = LLMConfig(provider="openai", model="gpt-4")

        with pytest.raises(ValueError, match="requires provider='ollama'"):
            OllamaLLMService(config=config)


class TestModelAvailability:
    """Test model availability checking."""

    @pytest.mark.asyncio
    async def test_check_model_availability_success(self, ollama_service):
        """Test successful model availability check."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "models": [
                {"name": "llama2:latest"},
                {"name": "mistral:latest"},
            ]
        })

        # Create proper async context manager mock
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.get = Mock(return_value=mock_cm)

        with patch.object(ollama_service, '_get_session', return_value=mock_session):
            is_available = await ollama_service.check_model_availability()

            assert is_available is True
            assert ollama_service._model_available is True

    @pytest.mark.asyncio
    async def test_check_model_availability_model_not_found(self, ollama_service):
        """Test model availability check when model not found."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "models": [
                {"name": "mistral:latest"},
                {"name": "codellama:latest"},
            ]
        })

        # Create proper async context manager mock
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.get = Mock(return_value=mock_cm)

        with patch.object(ollama_service, '_get_session', return_value=mock_session):
            is_available = await ollama_service.check_model_availability()

            assert is_available is False
            assert ollama_service._model_available is False

    @pytest.mark.asyncio
    async def test_check_model_availability_api_error(self, ollama_service):
        """Test model availability check with API error."""
        mock_response = Mock()
        mock_response.status = 500

        # Create proper async context manager mock
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.get = Mock(return_value=mock_cm)

        with patch.object(ollama_service, '_get_session', return_value=mock_session):
            is_available = await ollama_service.check_model_availability()

            assert is_available is False

    @pytest.mark.asyncio
    async def test_check_model_availability_caches_result(self, ollama_service):
        """Test model availability result is cached."""
        # Set cached result
        ollama_service._model_available = True

        # Should return cached result without making API call
        is_available = await ollama_service.check_model_availability()

        assert is_available is True


class TestGenerateResponse:
    """Test response generation."""

    @pytest.mark.asyncio
    async def test_generate_response_success(
        self,
        ollama_service,
        conversation_context,
    ):
        """Test successful response generation."""
        transcript = "I need to schedule an appointment for tomorrow at 2pm"

        # Mock model availability
        ollama_service._model_available = True

        # Mock Ollama API response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "response": "I'd be happy to help you schedule an appointment for tomorrow at 2pm. Let me check available slots for you.",
        })

        # Create proper async context manager mock
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(ollama_service, '_get_session', return_value=mock_session):
            response = await ollama_service.generate_response(transcript, conversation_context)

            assert isinstance(response, LLMResponse)
            assert len(response.text) > 0
            assert response.intent.name == IntentType.SCHEDULE_APPOINTMENT.value
            assert "date" in response.parameters
            assert "time" in response.parameters
            assert response.tool_call is not None
            assert response.tool_call.tool_name == "book_appointment"

    @pytest.mark.asyncio
    async def test_generate_response_with_gpu_enabled(
        self,
        ollama_service,
        conversation_context,
    ):
        """Test response generation includes GPU configuration."""
        transcript = "Hello"
        ollama_service._model_available = True

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"response": "Hi there!"})

        # Create proper async context manager mock
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(ollama_service, '_get_session', return_value=mock_session):
            await ollama_service.generate_response(transcript, conversation_context)

            # Check that GPU configuration was included
            call_args = mock_session.post.call_args
            payload = call_args[1]["json"]
            assert payload["options"]["num_gpu"] == -1

    @pytest.mark.asyncio
    async def test_generate_response_model_unavailable_with_fallback(
        self,
        ollama_config,
        conversation_context,
    ):
        """Test response generation falls back when model unavailable."""
        transcript = "Hello"

        # Create fallback service
        fallback = AsyncMock()
        fallback_response = LLMResponse(
            text="Fallback response",
            intent=Intent(name=IntentType.GENERAL_INQUIRY.value, confidence=0.9),
            confidence=0.9,
        )
        fallback.generate_response = AsyncMock(return_value=fallback_response)

        service = OllamaLLMService(
            config=ollama_config,
            fallback_service=fallback,
        )

        # Mock model as unavailable
        service._model_available = False

        response = await service.generate_response(transcript, conversation_context)

        assert response.text == "Fallback response"
        fallback.generate_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_model_unavailable_no_fallback(
        self,
        ollama_service,
        conversation_context,
    ):
        """Test response generation fails when model unavailable and no fallback."""
        transcript = "Hello"

        # Mock model as unavailable
        ollama_service._model_available = False

        with pytest.raises(RuntimeError, match="not available and no fallback"):
            await ollama_service.generate_response(transcript, conversation_context)

    @pytest.mark.asyncio
    async def test_generate_response_api_error_with_fallback(
        self,
        ollama_config,
        conversation_context,
    ):
        """Test response generation uses fallback on API error."""
        transcript = "Hello"

        # Create fallback service
        fallback = AsyncMock()
        fallback_response = LLMResponse(
            text="Fallback response",
            intent=Intent(name=IntentType.GENERAL_INQUIRY.value, confidence=0.9),
            confidence=0.9,
        )
        fallback.generate_response = AsyncMock(return_value=fallback_response)

        service = OllamaLLMService(
            config=ollama_config,
            fallback_service=fallback,
        )
        service._model_available = True

        # Mock API error
        mock_response = Mock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal server error")

        # Create proper async context manager mock
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(service, '_get_session', return_value=mock_session):
            response = await service.generate_response(transcript, conversation_context)

            assert response.text == "Fallback response"
            fallback.generate_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_api_error_no_fallback(
        self,
        ollama_service,
        conversation_context,
    ):
        """Test response generation fails on API error without fallback."""
        transcript = "Hello"
        ollama_service._model_available = True

        # Mock API error
        mock_response = Mock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal server error")

        # Create proper async context manager mock
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(ollama_service, '_get_session', return_value=mock_session):
            with pytest.raises(RuntimeError, match="Failed to generate LLM response"):
                await ollama_service.generate_response(transcript, conversation_context)


class TestVoiceOptimization:
    """Test voice-optimized response formatting."""

    @pytest.mark.asyncio
    async def test_response_is_voice_optimized(
        self,
        ollama_service,
        conversation_context,
    ):
        """Test that responses are optimized for voice output."""
        transcript = "Hello"
        ollama_service._model_available = True

        # Response with markdown and long text
        raw_response = """**Hello!** I'm here to help you.

- First point
- Second point
- Third point

Here's a table:
| Column 1 | Column 2 |
|----------|----------|
| Data     | More     |

This is a very long response with many sentences. It goes on and on. And on. And on. And on. And on."""

        mock_cm = create_mock_response(200, {"response": raw_response})
        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(ollama_service, '_get_session', return_value=mock_session):
            response = await ollama_service.generate_response(transcript, conversation_context)

            # Check voice optimization
            assert "**" not in response.text  # No markdown
            assert "*" not in response.text
            assert "|" not in response.text  # No tables
            assert "-" not in response.text or "- " not in response.text  # No bullet points

            # Check sentence limit (should be 3 or fewer)
            sentences = [s.strip() for s in response.text.split(".") if s.strip()]
            assert len(sentences) <= 3


class TestIntentClassification:
    """Test intent classification from transcripts."""

    @pytest.mark.asyncio
    async def test_classify_schedule_intent(
        self,
        ollama_service,
        conversation_context,
    ):
        """Test classification of schedule appointment intent."""
        transcript = "I want to schedule an appointment for next Monday"
        ollama_service._model_available = True

        mock_cm = create_mock_response(200, {"response": "Sure, I can help with that."})
        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(ollama_service, '_get_session', return_value=mock_session):
            response = await ollama_service.generate_response(transcript, conversation_context)

            assert response.intent.name == IntentType.SCHEDULE_APPOINTMENT.value
            assert response.intent.requires_tool is True
            assert response.intent.tool_name == "book_appointment"

    @pytest.mark.asyncio
    async def test_classify_cancel_intent(
        self,
        ollama_service,
        conversation_context,
    ):
        """Test classification of cancel appointment intent."""
        transcript = "I need to cancel my appointment"
        ollama_service._model_available = True

        mock_cm = create_mock_response(200, {"response": "I can help you cancel."})
        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(ollama_service, '_get_session', return_value=mock_session):
            response = await ollama_service.generate_response(transcript, conversation_context)

            assert response.intent.name == IntentType.CANCEL_APPOINTMENT.value
            assert response.intent.requires_tool is True


class TestParameterExtraction:
    """Test parameter extraction from transcripts."""

    @pytest.mark.asyncio
    async def test_extract_date_and_time(
        self,
        ollama_service,
        conversation_context,
    ):
        """Test extraction of date and time parameters."""
        transcript = "Book an appointment for tomorrow at 2:30pm"
        ollama_service._model_available = True

        mock_cm = create_mock_response(200, {"response": "I'll book that for you."})
        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(ollama_service, '_get_session', return_value=mock_session):
            response = await ollama_service.generate_response(transcript, conversation_context)

            # Check that date and time parameters were extracted
            assert "date" in response.parameters
            assert "time" in response.parameters
            # The exact values depend on regex matching, just verify they're present
            assert len(response.parameters["date"]) > 0
            assert len(response.parameters["time"]) > 0


class TestEmotionalTags:
    """Test emotional tag extraction."""

    @pytest.mark.asyncio
    async def test_extract_empathetic_tags(
        self,
        ollama_service,
        conversation_context,
    ):
        """Test extraction of empathetic emotional tags."""
        transcript = "Hello"
        ollama_service._model_available = True

        mock_cm = create_mock_response(200, {
            "response": "I'm sorry to hear that. I understand this must be frustrating."
        })
        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(ollama_service, '_get_session', return_value=mock_session):
            response = await ollama_service.generate_response(transcript, conversation_context)

            assert "empathetic" in response.emotional_tags
            assert "understanding" in response.emotional_tags

    @pytest.mark.asyncio
    async def test_extract_enthusiastic_tags(
        self,
        ollama_service,
        conversation_context,
    ):
        """Test extraction of enthusiastic emotional tags."""
        transcript = "Hello"
        ollama_service._model_available = True

        mock_cm = create_mock_response(200, {
            "response": "That's wonderful! I'm excited to help you with that."
        })
        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(ollama_service, '_get_session', return_value=mock_session):
            response = await ollama_service.generate_response(transcript, conversation_context)

            assert "enthusiastic" in response.emotional_tags


class TestContextManager:
    """Test async context manager support."""

    @pytest.mark.asyncio
    async def test_context_manager_closes_session(self, ollama_config):
        """Test context manager properly closes session."""
        async with OllamaLLMService(config=ollama_config) as service:
            # Get session to create it
            await service._get_session()
            assert service._session is not None

        # Session should be closed after context exit
        assert service._session is None or service._session.closed
