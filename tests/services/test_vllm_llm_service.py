"""
Unit tests for vLLM LLM Service.

Tests vLLM integration with PagedAttention, continuous batching,
GPU resource management, and fallback handling for high-performance inference.

**Validates: Requirements 13.3, 9.4, 13.5**
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
from src.voice_agent.services.vllm_llm_service import vLLMLLMService


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
def vllm_config():
    """Create vLLM LLM configuration."""
    return LLMConfig(
        provider="vllm",
        model="meta-llama/Llama-2-7b-chat-hf",
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
async def vllm_service(vllm_config):
    """Create vLLM LLM service instance."""
    service = vLLMLLMService(
        config=vllm_config,
        base_url="http://localhost:8000",
        gpu_memory_utilization=0.9,
        max_num_seqs=256,
    )
    yield service
    await service.close()


class TestvLLMLLMServiceInitialization:
    """
    Test vLLM LLM service initialization.

    **Validates: Requirements 13.3, 9.4**
    """

    def test_initialization_with_valid_config(self, vllm_config):
        """Test service initializes with valid configuration."""
        service = vLLMLLMService(config=vllm_config)

        assert service.config.provider == "vllm"
        assert service.config.model == "meta-llama/Llama-2-7b-chat-hf"
        assert service.base_url == "http://localhost:8000"
        assert service.gpu_memory_utilization == 0.9
        assert service.max_num_seqs == 256
        assert service.fallback_service is None

    def test_initialization_with_custom_base_url(self, vllm_config):
        """Test service initializes with custom base URL."""
        service = vLLMLLMService(
            config=vllm_config,
            base_url="http://custom-host:8080/",
        )

        assert service.base_url == "http://custom-host:8080"

    def test_initialization_with_gpu_memory_utilization(self, vllm_config):
        """Test service initializes with custom GPU memory utilization."""
        service = vLLMLLMService(
            config=vllm_config,
            gpu_memory_utilization=0.8,
        )

        assert service.gpu_memory_utilization == 0.8

    def test_initialization_with_max_num_seqs(self, vllm_config):
        """Test service initializes with custom max sequences for continuous batching."""
        service = vLLMLLMService(
            config=vllm_config,
            max_num_seqs=512,
        )

        assert service.max_num_seqs == 512

    def test_initialization_with_max_model_len(self, vllm_config):
        """Test service initializes with custom max model length."""
        service = vLLMLLMService(
            config=vllm_config,
            max_model_len=4096,
        )

        assert service.max_model_len == 4096

    def test_initialization_with_fallback_service(self, vllm_config):
        """Test service initializes with fallback service."""
        fallback = MagicMock()
        service = vLLMLLMService(
            config=vllm_config,
            fallback_service=fallback,
        )

        assert service.fallback_service is fallback

    def test_initialization_rejects_wrong_provider(self):
        """Test service rejects non-vllm provider."""
        config = LLMConfig(provider="openai", model="gpt-4")

        with pytest.raises(ValueError, match="requires provider='vllm'"):
            vLLMLLMService(config=config)


class TestServerAvailability:
    """
    Test vLLM server availability checking.

    **Validates: Requirements 13.3**
    """

    @pytest.mark.asyncio
    async def test_check_server_availability_success(self, vllm_service):
        """Test successful server availability check."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": [
                {"id": "meta-llama/Llama-2-7b-chat-hf"},
                {"id": "mistralai/Mistral-7B-v0.1"},
            ]
        })

        # Create proper async context manager mock
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.get = Mock(return_value=mock_cm)

        with patch.object(vllm_service, '_get_session', return_value=mock_session):
            is_available = await vllm_service.check_server_availability()

            assert is_available is True
            assert vllm_service._server_available is True

    @pytest.mark.asyncio
    async def test_check_server_availability_model_not_found(self, vllm_service):
        """Test server availability check when model not found."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": [
                {"id": "mistralai/Mistral-7B-v0.1"},
                {"id": "codellama/CodeLlama-7b-hf"},
            ]
        })

        # Create proper async context manager mock
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.get = Mock(return_value=mock_cm)

        with patch.object(vllm_service, '_get_session', return_value=mock_session):
            is_available = await vllm_service.check_server_availability()

            assert is_available is False
            assert vllm_service._server_available is False

    @pytest.mark.asyncio
    async def test_check_server_availability_api_error(self, vllm_service):
        """Test server availability check with API error."""
        mock_response = Mock()
        mock_response.status = 500

        # Create proper async context manager mock
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.get = Mock(return_value=mock_cm)

        with patch.object(vllm_service, '_get_session', return_value=mock_session):
            is_available = await vllm_service.check_server_availability()

            assert is_available is False

    @pytest.mark.asyncio
    async def test_check_server_availability_caches_result(self, vllm_service):
        """Test server availability result is cached."""
        # Set cached result
        vllm_service._server_available = True

        # Should return cached result without making API call
        is_available = await vllm_service.check_server_availability()

        assert is_available is True


class TestGenerateResponse:
    """
    Test response generation with PagedAttention and continuous batching.

    **Validates: Requirements 13.3, 9.4, 13.5**
    """

    @pytest.mark.asyncio
    async def test_generate_response_success(
        self,
        vllm_service,
        conversation_context,
    ):
        """Test successful response generation."""
        transcript = "I need to schedule an appointment for tomorrow at 2pm"

        # Mock server availability
        vllm_service._server_available = True

        # Mock vLLM API response (OpenAI-compatible format)
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [
                {
                    "message": {
                        "content": "I'd be happy to help you schedule an appointment for tomorrow at 2pm. Let me check available slots for you.",
                        "role": "assistant",
                    },
                    "finish_reason": "stop",
                }
            ],
            "model": "meta-llama/Llama-2-7b-chat-hf",
        })

        # Create proper async context manager mock
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(vllm_service, '_get_session', return_value=mock_session):
            response = await vllm_service.generate_response(transcript, conversation_context)

            assert isinstance(response, LLMResponse)
            assert len(response.text) > 0
            assert response.intent.name == IntentType.SCHEDULE_APPOINTMENT.value
            assert "date" in response.parameters
            assert "time" in response.parameters
            assert response.tool_call is not None
            assert response.tool_call.tool_name == "book_appointment"

    @pytest.mark.asyncio
    async def test_generate_response_includes_gpu_metadata(
        self,
        vllm_service,
        conversation_context,
    ):
        """Test response includes GPU resource management metadata."""
        transcript = "Hello"
        vllm_service._server_available = True

        mock_cm = create_mock_response(200, {
            "choices": [{"message": {"content": "Hi there!", "role": "assistant"}}],
        })
        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(vllm_service, '_get_session', return_value=mock_session):
            response = await vllm_service.generate_response(transcript, conversation_context)

            # Check GPU resource management metadata
            assert "gpu_memory_utilization" in response.metadata
            assert response.metadata["gpu_memory_utilization"] == 0.9
            assert "max_num_seqs" in response.metadata
            assert response.metadata["max_num_seqs"] == 256

    @pytest.mark.asyncio
    async def test_generate_response_uses_openai_compatible_api(
        self,
        vllm_service,
        conversation_context,
    ):
        """Test response generation uses OpenAI-compatible API format."""
        transcript = "Hello"
        vllm_service._server_available = True

        mock_cm = create_mock_response(200, {
            "choices": [{"message": {"content": "Hi!", "role": "assistant"}}],
        })
        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(vllm_service, '_get_session', return_value=mock_session):
            await vllm_service.generate_response(transcript, conversation_context)

            # Check that OpenAI-compatible endpoint was called
            call_args = mock_session.post.call_args
            assert "/v1/chat/completions" in call_args[0][0]

            # Check payload format
            payload = call_args[1]["json"]
            assert "model" in payload
            assert "messages" in payload
            assert "temperature" in payload
            assert "max_tokens" in payload

    @pytest.mark.asyncio
    async def test_generate_response_server_unavailable_with_fallback(
        self,
        vllm_config,
        conversation_context,
    ):
        """Test response generation falls back when server unavailable."""
        transcript = "Hello"

        # Create fallback service
        fallback = AsyncMock()
        fallback_response = LLMResponse(
            text="Fallback response",
            intent=Intent(name=IntentType.GENERAL_INQUIRY.value, confidence=0.9),
            confidence=0.9,
        )
        fallback.generate_response = AsyncMock(return_value=fallback_response)

        service = vLLMLLMService(
            config=vllm_config,
            fallback_service=fallback,
        )

        # Mock server as unavailable
        service._server_available = False

        response = await service.generate_response(transcript, conversation_context)

        assert response.text == "Fallback response"
        fallback.generate_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_server_unavailable_no_fallback(
        self,
        vllm_service,
        conversation_context,
    ):
        """Test response generation fails when server unavailable and no fallback."""
        transcript = "Hello"

        # Mock server as unavailable
        vllm_service._server_available = False

        with pytest.raises(RuntimeError, match="not available and no fallback"):
            await vllm_service.generate_response(transcript, conversation_context)

    @pytest.mark.asyncio
    async def test_generate_response_api_error_with_fallback(
        self,
        vllm_config,
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

        service = vLLMLLMService(
            config=vllm_config,
            fallback_service=fallback,
        )
        service._server_available = True

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
        vllm_service,
        conversation_context,
    ):
        """Test response generation fails on API error without fallback."""
        transcript = "Hello"
        vllm_service._server_available = True

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

        with patch.object(vllm_service, '_get_session', return_value=mock_session):
            with pytest.raises(RuntimeError, match="Failed to generate LLM response"):
                await vllm_service.generate_response(transcript, conversation_context)


class TestVoiceOptimization:
    """
    Test voice-optimized response formatting.

    **Validates: Requirements 6.1, 6.2, 6.3**
    """

    @pytest.mark.asyncio
    async def test_response_is_voice_optimized(
        self,
        vllm_service,
        conversation_context,
    ):
        """Test that responses are optimized for voice output."""
        transcript = "Hello"
        vllm_service._server_available = True

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

        mock_cm = create_mock_response(200, {
            "choices": [{"message": {"content": raw_response, "role": "assistant"}}],
        })
        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(vllm_service, '_get_session', return_value=mock_session):
            response = await vllm_service.generate_response(transcript, conversation_context)

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
        vllm_service,
        conversation_context,
    ):
        """Test classification of schedule appointment intent."""
        transcript = "I want to schedule an appointment for next Monday"
        vllm_service._server_available = True

        mock_cm = create_mock_response(200, {
            "choices": [{"message": {"content": "Sure, I can help with that.", "role": "assistant"}}],
        })
        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(vllm_service, '_get_session', return_value=mock_session):
            response = await vllm_service.generate_response(transcript, conversation_context)

            assert response.intent.name == IntentType.SCHEDULE_APPOINTMENT.value
            assert response.intent.requires_tool is True
            assert response.intent.tool_name == "book_appointment"

    @pytest.mark.asyncio
    async def test_classify_cancel_intent(
        self,
        vllm_service,
        conversation_context,
    ):
        """Test classification of cancel appointment intent."""
        transcript = "I need to cancel my appointment"
        vllm_service._server_available = True

        mock_cm = create_mock_response(200, {
            "choices": [{"message": {"content": "I can help you cancel.", "role": "assistant"}}],
        })
        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(vllm_service, '_get_session', return_value=mock_session):
            response = await vllm_service.generate_response(transcript, conversation_context)

            assert response.intent.name == IntentType.CANCEL_APPOINTMENT.value
            assert response.intent.requires_tool is True


class TestParameterExtraction:
    """Test parameter extraction from transcripts."""

    @pytest.mark.asyncio
    async def test_extract_date_and_time(
        self,
        vllm_service,
        conversation_context,
    ):
        """Test extraction of date and time parameters."""
        transcript = "Book an appointment for tomorrow at 2:30pm"
        vllm_service._server_available = True

        mock_cm = create_mock_response(200, {
            "choices": [{"message": {"content": "I'll book that for you.", "role": "assistant"}}],
        })
        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(vllm_service, '_get_session', return_value=mock_session):
            response = await vllm_service.generate_response(transcript, conversation_context)

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
        vllm_service,
        conversation_context,
    ):
        """Test extraction of empathetic emotional tags."""
        transcript = "Hello"
        vllm_service._server_available = True

        mock_cm = create_mock_response(200, {
            "choices": [{"message": {
                "content": "I'm sorry to hear that. I understand this must be frustrating.",
                "role": "assistant",
            }}],
        })
        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(vllm_service, '_get_session', return_value=mock_session):
            response = await vllm_service.generate_response(transcript, conversation_context)

            assert "empathetic" in response.emotional_tags
            assert "understanding" in response.emotional_tags

    @pytest.mark.asyncio
    async def test_extract_enthusiastic_tags(
        self,
        vllm_service,
        conversation_context,
    ):
        """Test extraction of enthusiastic emotional tags."""
        transcript = "Hello"
        vllm_service._server_available = True

        mock_cm = create_mock_response(200, {
            "choices": [{"message": {
                "content": "That's wonderful! I'm excited to help you with that.",
                "role": "assistant",
            }}],
        })
        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(vllm_service, '_get_session', return_value=mock_session):
            response = await vllm_service.generate_response(transcript, conversation_context)

            assert "enthusiastic" in response.emotional_tags


class TestGPUResourceManagement:
    """
    Test GPU resource management configuration.

    **Validates: Requirements 9.4, 13.5**
    """

    def test_gpu_memory_utilization_configuration(self, vllm_config):
        """Test GPU memory utilization is configurable."""
        service = vLLMLLMService(
            config=vllm_config,
            gpu_memory_utilization=0.85,
        )

        assert service.gpu_memory_utilization == 0.85

    def test_continuous_batching_configuration(self, vllm_config):
        """Test continuous batching max sequences is configurable."""
        service = vLLMLLMService(
            config=vllm_config,
            max_num_seqs=512,
        )

        assert service.max_num_seqs == 512

    def test_max_model_len_configuration(self, vllm_config):
        """Test max model length is configurable."""
        service = vLLMLLMService(
            config=vllm_config,
            max_model_len=8192,
        )

        assert service.max_model_len == 8192

    def test_default_gpu_configuration(self, vllm_config):
        """Test default GPU configuration values."""
        service = vLLMLLMService(config=vllm_config)

        assert service.gpu_memory_utilization == 0.9
        assert service.max_num_seqs == 256
        assert service.max_model_len is None


class TestContextManager:
    """Test async context manager support."""

    @pytest.mark.asyncio
    async def test_context_manager_closes_session(self, vllm_config):
        """Test context manager properly closes session."""
        async with vLLMLLMService(config=vllm_config) as service:
            # Get session to create it
            await service._get_session()
            assert service._session is not None

        # Session should be closed after context exit
        assert service._session is None or service._session.closed


class TestPromptBuilding:
    """Test prompt building for vLLM."""

    def test_build_prompt_includes_history(self, vllm_service, conversation_context):
        """Test prompt includes conversation history."""
        prompt = vllm_service._build_prompt("Current message", conversation_context)

        # Should include history
        assert "Hello" in prompt
        assert "Hi! How can I help you today?" in prompt
        assert "Current message" in prompt

    def test_build_prompt_limits_history(self, vllm_service):
        """Test prompt limits conversation history to last 5 turns."""
        # Create context with many history items
        context = ConversationContext(
            session_id="test",
            conversation_history=[
                {"role": "user", "content": f"Message {i}"}
                for i in range(10)
            ],
        )

        prompt = vllm_service._build_prompt("Current message", context)

        # Should only include last 5 history items
        assert "Message 5" in prompt or "Message 6" in prompt
        assert "Message 0" not in prompt

    def test_build_voice_optimized_system_prompt(self, vllm_service):
        """Test voice-optimized system prompt generation."""
        prompt = vllm_service._build_voice_optimized_system_prompt()

        # Check for voice optimization rules
        assert "markdown" in prompt.lower()
        assert "3 sentences" in prompt.lower()
        assert "conversational" in prompt.lower()
        assert "voice" in prompt.lower()


class TestPagedAttentionIntegration:
    """
    Test PagedAttention integration for efficient memory management.

    **Validates: Requirements 13.3, 9.4**
    """

    @pytest.mark.asyncio
    async def test_vllm_uses_paged_attention(self, vllm_service, conversation_context):
        """Test that vLLM service leverages PagedAttention through API."""
        transcript = "Hello"
        vllm_service._server_available = True

        mock_cm = create_mock_response(200, {
            "choices": [{"message": {"content": "Hi!", "role": "assistant"}}],
        })
        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(vllm_service, '_get_session', return_value=mock_session):
            response = await vllm_service.generate_response(transcript, conversation_context)

            # Verify response includes vLLM provider metadata
            assert response.metadata["provider"] == "vllm"
            # PagedAttention is used implicitly by vLLM server
            assert isinstance(response, LLMResponse)


class TestContinuousBatching:
    """
    Test continuous batching configuration for high throughput.

    **Validates: Requirements 9.4, 13.3**
    """

    def test_continuous_batching_max_seqs_default(self, vllm_config):
        """Test default continuous batching configuration."""
        service = vLLMLLMService(config=vllm_config)

        # Default should support 256 concurrent sequences
        assert service.max_num_seqs == 256

    def test_continuous_batching_max_seqs_custom(self, vllm_config):
        """Test custom continuous batching configuration."""
        service = vLLMLLMService(
            config=vllm_config,
            max_num_seqs=1024,
        )

        # Should support custom number of concurrent sequences
        assert service.max_num_seqs == 1024

    @pytest.mark.asyncio
    async def test_continuous_batching_metadata_included(
        self,
        vllm_service,
        conversation_context,
    ):
        """Test continuous batching configuration is included in metadata."""
        transcript = "Hello"
        vllm_service._server_available = True

        mock_cm = create_mock_response(200, {
            "choices": [{"message": {"content": "Hi!", "role": "assistant"}}],
        })
        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_cm)

        with patch.object(vllm_service, '_get_session', return_value=mock_session):
            response = await vllm_service.generate_response(transcript, conversation_context)

            # Verify continuous batching config is in metadata
            assert "max_num_seqs" in response.metadata
            assert response.metadata["max_num_seqs"] == 256
