"""
Unit tests for OpenAI TTS Service.

Tests streaming synthesis, interface consistency with ElevenLabs,
and error handling for the OpenAI TTS service wrapper.

**Validates: Requirements 5.6**
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.voice_agent.services.elevenlabs_tts_service import (
    VoiceConfig,
    VoiceSettings,
)
from src.voice_agent.services.openai_tts_service import (
    OpenAITTSConfig,
    OpenAITTSService,
)


@pytest.fixture
def voice_config():
    """Create test voice configuration for OpenAI."""
    return VoiceConfig(
        provider="openai",
        voice_id="alloy",
        stability=0.5,
        similarity_boost=0.75,
        style=0.0,
        use_speaker_boost=False,
    )


@pytest.fixture
def tts_config(voice_config):
    """Create test TTS configuration."""
    return OpenAITTSConfig(
        api_key="test_api_key",
        default_voice_config=voice_config,
        model="tts-1",
        chunk_size=4096,
    )


@pytest.fixture
def mock_openai_client():
    """Create mock OpenAI client."""
    with patch("src.voice_agent.services.openai_tts_service.AsyncOpenAI") as mock:
        client = AsyncMock()
        mock.return_value = client
        yield client


@pytest.fixture
def tts_service(tts_config, mock_openai_client):
    """Create TTS service instance with mocked client."""
    return OpenAITTSService(tts_config)


class TestOpenAITTSService:
    """Test OpenAI TTS service."""

    def test_initialization(self, tts_service, tts_config):
        """Test service initialization."""
        assert tts_service.config == tts_config
        assert tts_service.client is not None

    def test_voice_mapping(self, tts_service):
        """Test OpenAI voice mapping."""
        assert tts_service.VOICE_MAPPING["alloy"] == "alloy"
        assert tts_service.VOICE_MAPPING["echo"] == "echo"
        assert tts_service.VOICE_MAPPING["fable"] == "fable"
        assert tts_service.VOICE_MAPPING["onyx"] == "onyx"
        assert tts_service.VOICE_MAPPING["nova"] == "nova"
        assert tts_service.VOICE_MAPPING["shimmer"] == "shimmer"

    @pytest.mark.asyncio
    async def test_synthesize_streaming_success(self, tts_service, mock_openai_client):
        """Test successful streaming synthesis."""
        # Mock audio response
        mock_response = AsyncMock()

        async def mock_iter_bytes(chunk_size):
            yield b"audio_chunk_1"
            yield b"audio_chunk_2"
            yield b"audio_chunk_3"

        mock_response.iter_bytes = mock_iter_bytes

        mock_openai_client.audio.speech.create = AsyncMock(
            return_value=mock_response
        )

        # Synthesize text
        text = "Hello, this is a test."
        chunks = []
        async for chunk in tts_service.synthesize_streaming(text):
            chunks.append(chunk)

        # Verify chunks
        assert len(chunks) == 3
        assert chunks[0] == b"audio_chunk_1"
        assert chunks[1] == b"audio_chunk_2"
        assert chunks[2] == b"audio_chunk_3"

        # Verify API was called correctly
        mock_openai_client.audio.speech.create.assert_called_once()
        call_args = mock_openai_client.audio.speech.create.call_args
        assert call_args.kwargs["model"] == "tts-1"
        assert call_args.kwargs["voice"] == "alloy"
        assert call_args.kwargs["input"] == text
        assert call_args.kwargs["response_format"] == "mp3"

    @pytest.mark.asyncio
    async def test_synthesize_streaming_empty_text(self, tts_service):
        """Test synthesis with empty text raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            async for _ in tts_service.synthesize_streaming(""):
                pass

    @pytest.mark.asyncio
    async def test_synthesize_streaming_whitespace_only(self, tts_service):
        """Test synthesis with whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            async for _ in tts_service.synthesize_streaming("   "):
                pass

    @pytest.mark.asyncio
    async def test_synthesize_streaming_wrong_provider(self, tts_service):
        """Test synthesis with wrong provider raises ValueError."""
        wrong_config = VoiceConfig(
            provider="elevenlabs",
            voice_id="test_voice",
        )

        with pytest.raises(ValueError, match="OpenAI service requires provider='openai'"):
            async for _ in tts_service.synthesize_streaming("test", wrong_config):
                pass

    @pytest.mark.asyncio
    async def test_synthesize_streaming_custom_voice_config(self, tts_service, mock_openai_client):
        """Test synthesis with custom voice configuration."""
        # Mock audio response
        mock_response = AsyncMock()

        async def mock_iter_bytes(chunk_size):
            yield b"audio_chunk"

        mock_response.iter_bytes = mock_iter_bytes

        mock_openai_client.audio.speech.create = AsyncMock(
            return_value=mock_response
        )

        # Custom voice config
        custom_config = VoiceConfig(
            provider="openai",
            voice_id="nova",
        )

        # Synthesize text
        text = "Test with custom voice."
        chunks = []
        async for chunk in tts_service.synthesize_streaming(text, custom_config):
            chunks.append(chunk)

        # Verify voice was used
        call_args = mock_openai_client.audio.speech.create.call_args
        assert call_args.kwargs["voice"] == "nova"

    @pytest.mark.asyncio
    async def test_synthesize_streaming_api_error(self, tts_service, mock_openai_client):
        """Test synthesis handles API errors."""
        mock_openai_client.audio.speech.create = AsyncMock(
            side_effect=Exception("API error")
        )

        with pytest.raises(RuntimeError, match="Speech synthesis failed"):
            async for _ in tts_service.synthesize_streaming("test"):
                pass

    def test_get_openai_voice_valid(self, tts_service):
        """Test getting valid OpenAI voice."""
        assert tts_service._get_openai_voice("alloy") == "alloy"
        assert tts_service._get_openai_voice("echo") == "echo"
        assert tts_service._get_openai_voice("fable") == "fable"
        assert tts_service._get_openai_voice("onyx") == "onyx"
        assert tts_service._get_openai_voice("nova") == "nova"
        assert tts_service._get_openai_voice("shimmer") == "shimmer"

    def test_get_openai_voice_invalid_defaults_to_alloy(self, tts_service):
        """Test getting invalid voice defaults to alloy."""
        assert tts_service._get_openai_voice("invalid_voice") == "alloy"
        assert tts_service._get_openai_voice("unknown") == "alloy"

    def test_apply_emotional_context_no_tags(self, tts_service):
        """Test applying emotional context with no tags returns unchanged text."""
        text = "Hello, how are you?"
        result = tts_service.apply_emotional_context(text, [])
        assert result == text

    def test_apply_emotional_context_with_tags(self, tts_service):
        """Test applying emotional context with tags returns unchanged text."""
        text = "I understand your concern."
        result = tts_service.apply_emotional_context(text, ["empathetic", "understanding"])
        # OpenAI doesn't support tags, so text should be unchanged
        assert result == text

    @pytest.mark.asyncio
    async def test_clone_voice_not_supported(self, tts_service):
        """Test voice cloning raises NotImplementedError."""
        audio_samples = [b"audio_data_1", b"audio_data_2"]

        with pytest.raises(NotImplementedError, match="OpenAI TTS does not support voice cloning"):
            await tts_service.clone_voice(audio_samples, "test_voice")

    @pytest.mark.asyncio
    async def test_get_voice_settings_returns_defaults(self, tts_service):
        """Test get_voice_settings returns default settings."""
        settings = await tts_service.get_voice_settings("alloy")

        # Should return default settings
        assert isinstance(settings, VoiceSettings)
        assert settings.stability == 0.5
        assert settings.similarity_boost == 0.75
        assert settings.style == 0.0
        assert settings.use_speaker_boost is False

    @pytest.mark.asyncio
    async def test_get_voice_settings_any_voice(self, tts_service):
        """Test get_voice_settings works for any voice ID."""
        settings = await tts_service.get_voice_settings("nova")
        assert isinstance(settings, VoiceSettings)

        settings = await tts_service.get_voice_settings("unknown_voice")
        assert isinstance(settings, VoiceSettings)

    @pytest.mark.asyncio
    async def test_list_available_voices(self, tts_service):
        """Test listing available voices."""
        voices = await tts_service.list_available_voices()

        # Should return 6 OpenAI voices
        assert len(voices) == 6

        # Check voice structure
        voice_ids = [v["voice_id"] for v in voices]
        assert "alloy" in voice_ids
        assert "echo" in voice_ids
        assert "fable" in voice_ids
        assert "onyx" in voice_ids
        assert "nova" in voice_ids
        assert "shimmer" in voice_ids

        # Check voice details
        for voice in voices:
            assert "voice_id" in voice
            assert "name" in voice
            assert "category" in voice
            assert "description" in voice
            assert "labels" in voice

    @pytest.mark.asyncio
    async def test_synthesize_streaming_different_voices(self, tts_service, mock_openai_client):
        """Test synthesis with different OpenAI voices."""
        # Mock audio response
        mock_response = AsyncMock()

        async def mock_iter_bytes(chunk_size):
            yield b"audio_chunk"

        mock_response.iter_bytes = mock_iter_bytes

        mock_openai_client.audio.speech.create = AsyncMock(
            return_value=mock_response
        )

        # Test each voice
        for voice_id in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
            voice_config = VoiceConfig(
                provider="openai",
                voice_id=voice_id,
            )

            chunks = []
            async for chunk in tts_service.synthesize_streaming("Test", voice_config):
                chunks.append(chunk)

            assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_synthesize_streaming_long_text(self, tts_service, mock_openai_client):
        """Test synthesis with long text."""
        # Mock audio response
        mock_response = AsyncMock()

        async def mock_iter_bytes(chunk_size):
            # Simulate multiple chunks for long text
            for i in range(10):
                yield b"audio_chunk_" + str(i).encode()

        mock_response.iter_bytes = mock_iter_bytes

        mock_openai_client.audio.speech.create = AsyncMock(
            return_value=mock_response
        )

        # Long text
        text = "This is a very long text. " * 50
        chunks = []
        async for chunk in tts_service.synthesize_streaming(text):
            chunks.append(chunk)

        # Should receive multiple chunks
        assert len(chunks) == 10

    @pytest.mark.asyncio
    async def test_interface_consistency_with_elevenlabs(self, tts_service):
        """Test that OpenAI service has consistent interface with ElevenLabs."""
        # Check all required methods exist
        assert hasattr(tts_service, "synthesize_streaming")
        assert hasattr(tts_service, "apply_emotional_context")
        assert hasattr(tts_service, "clone_voice")
        assert hasattr(tts_service, "get_voice_settings")
        assert hasattr(tts_service, "list_available_voices")

        # Check methods are callable
        assert callable(tts_service.synthesize_streaming)
        assert callable(tts_service.apply_emotional_context)
        assert callable(tts_service.clone_voice)
        assert callable(tts_service.get_voice_settings)
        assert callable(tts_service.list_available_voices)


class TestOpenAITTSConfig:
    """Test OpenAI TTS configuration."""

    def test_default_config(self, voice_config):
        """Test default configuration values."""
        config = OpenAITTSConfig(
            api_key="test_key",
            default_voice_config=voice_config,
        )

        assert config.api_key == "test_key"
        assert config.default_voice_config == voice_config
        assert config.model == "tts-1"
        assert config.chunk_size == 4096

    def test_custom_config(self, voice_config):
        """Test custom configuration values."""
        config = OpenAITTSConfig(
            api_key="test_key",
            default_voice_config=voice_config,
            model="tts-1-hd",
            chunk_size=8192,
        )

        assert config.model == "tts-1-hd"
        assert config.chunk_size == 8192
