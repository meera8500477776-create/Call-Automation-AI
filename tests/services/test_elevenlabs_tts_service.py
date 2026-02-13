"""
Unit tests for ElevenLabs TTS Service.

Tests streaming synthesis, emotional context application, voice settings,
and error handling for the ElevenLabs TTS service wrapper.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.voice_agent.services.elevenlabs_tts_service import (
    AudioTag,
    ElevenLabsTTSService,
    TTSConfig,
    VoiceConfig,
)


@pytest.fixture
def voice_config():
    """Create test voice configuration."""
    return VoiceConfig(
        provider="elevenlabs",
        voice_id="test_voice_id",
        stability=0.5,
        similarity_boost=0.75,
        style=0.0,
        use_speaker_boost=True,
        model_id="eleven_flash_v2_5",
    )


@pytest.fixture
def tts_config(voice_config):
    """Create test TTS configuration."""
    return TTSConfig(
        api_key="test_api_key",
        default_voice_config=voice_config,
        chunk_size=2048,
        optimize_streaming_latency=3,
        output_format="mp3_44100_128",
    )


@pytest.fixture
def mock_elevenlabs_client():
    """Create mock ElevenLabs client."""
    with patch("src.voice_agent.services.elevenlabs_tts_service.AsyncElevenLabs") as mock:
        client = AsyncMock()
        mock.return_value = client
        yield client


@pytest.fixture
def tts_service(tts_config, mock_elevenlabs_client):
    """Create TTS service instance with mocked client."""
    return ElevenLabsTTSService(tts_config)


class TestVoiceConfig:
    """Test VoiceConfig validation."""

    def test_valid_config(self):
        """Test valid voice configuration."""
        config = VoiceConfig(
            provider="elevenlabs",
            voice_id="test_voice",
            stability=0.5,
            similarity_boost=0.75,
        )
        assert config.provider == "elevenlabs"
        assert config.voice_id == "test_voice"
        assert config.stability == 0.5
        assert config.similarity_boost == 0.75

    def test_invalid_provider(self):
        """Test invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Invalid provider"):
            VoiceConfig(
                provider="invalid",
                voice_id="test_voice",
            )

    def test_invalid_stability(self):
        """Test invalid stability raises ValueError."""
        with pytest.raises(ValueError, match="Stability must be between"):
            VoiceConfig(
                provider="elevenlabs",
                voice_id="test_voice",
                stability=1.5,
            )

    def test_invalid_similarity_boost(self):
        """Test invalid similarity boost raises ValueError."""
        with pytest.raises(ValueError, match="Similarity boost must be between"):
            VoiceConfig(
                provider="elevenlabs",
                voice_id="test_voice",
                similarity_boost=-0.1,
            )

    def test_invalid_style(self):
        """Test invalid style raises ValueError."""
        with pytest.raises(ValueError, match="Style must be between"):
            VoiceConfig(
                provider="elevenlabs",
                voice_id="test_voice",
                style=2.0,
            )


class TestElevenLabsTTSService:
    """Test ElevenLabs TTS service."""

    def test_initialization(self, tts_service, tts_config):
        """Test service initialization."""
        assert tts_service.config == tts_config
        assert tts_service.client is not None
        assert isinstance(tts_service._voice_settings_cache, dict)

    @pytest.mark.asyncio
    async def test_synthesize_streaming_success(self, tts_service, mock_elevenlabs_client):
        """Test successful streaming synthesis."""
        # Mock audio stream
        async def mock_audio_stream():
            yield b"audio_chunk_1"
            yield b"audio_chunk_2"
            yield b"audio_chunk_3"

        mock_elevenlabs_client.text_to_speech.convert = AsyncMock(
            return_value=mock_audio_stream()
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
        mock_elevenlabs_client.text_to_speech.convert.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_streaming_empty_text(self, tts_service):
        """Test synthesis with empty text raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            async for _ in tts_service.synthesize_streaming(""):
                pass

    @pytest.mark.asyncio
    async def test_synthesize_streaming_wrong_provider(self, tts_service):
        """Test synthesis with wrong provider raises ValueError."""
        wrong_config = VoiceConfig(
            provider="openai",
            voice_id="test_voice",
        )

        with pytest.raises(ValueError, match="ElevenLabs service requires provider='elevenlabs'"):
            async for _ in tts_service.synthesize_streaming("test", wrong_config):
                pass

    @pytest.mark.asyncio
    async def test_synthesize_streaming_api_error(self, tts_service, mock_elevenlabs_client):
        """Test synthesis handles API errors."""
        mock_elevenlabs_client.text_to_speech.convert = AsyncMock(
            side_effect=Exception("API error")
        )

        with pytest.raises(RuntimeError, match="Speech synthesis failed"):
            async for _ in tts_service.synthesize_streaming("test"):
                pass

    def test_apply_emotional_context_no_tags(self, tts_service):
        """Test applying emotional context with no tags."""
        text = "Hello, how are you?"
        result = tts_service.apply_emotional_context(text, [])
        assert result == text

    def test_apply_emotional_context_empathetic(self, tts_service):
        """Test applying empathetic tag."""
        text = "I understand your concern."
        result = tts_service.apply_emotional_context(text, ["empathetic"])
        assert AudioTag.EMPATHETIC.value in result

    def test_apply_emotional_context_enthusiastic(self, tts_service):
        """Test applying enthusiastic tag."""
        text = "That's wonderful news!"
        result = tts_service.apply_emotional_context(text, ["enthusiastic"])
        assert AudioTag.EXCITED.value in result

    def test_apply_emotional_context_understanding(self, tts_service):
        """Test applying understanding tag."""
        text = "I understand what you mean."
        result = tts_service.apply_emotional_context(text, ["understanding"])
        assert AudioTag.UNDERSTANDING.value in result

    def test_apply_emotional_context_sigh_for_apology(self, tts_service):
        """Test automatic sigh insertion for apologies."""
        text = "I'm sorry about that."
        result = tts_service.apply_emotional_context(text, [])
        assert AudioTag.SIGH.value in result

    def test_apply_emotional_context_multiple_tags(self, tts_service):
        """Test applying multiple emotional tags."""
        text = "I understand, that's wonderful!"
        result = tts_service.apply_emotional_context(text, ["understanding", "enthusiastic"])
        # Should contain at least one tag
        assert any(tag.value in result for tag in AudioTag)

    @pytest.mark.asyncio
    async def test_clone_voice_not_implemented(self, tts_service):
        """Test voice cloning raises NotImplementedError."""
        audio_samples = [b"audio_data_1", b"audio_data_2"]

        with pytest.raises(NotImplementedError, match="Voice cloning requires file upload"):
            await tts_service.clone_voice(audio_samples, "test_voice")

    @pytest.mark.asyncio
    async def test_clone_voice_empty_samples(self, tts_service):
        """Test voice cloning with empty samples raises ValueError."""
        with pytest.raises(ValueError, match="At least one audio sample is required"):
            await tts_service.clone_voice([], "test_voice")

    @pytest.mark.asyncio
    async def test_clone_voice_empty_name(self, tts_service):
        """Test voice cloning with empty name raises ValueError."""
        with pytest.raises(ValueError, match="Voice name cannot be empty"):
            await tts_service.clone_voice([b"audio"], "")

    @pytest.mark.asyncio
    async def test_get_voice_settings_success(self, tts_service, mock_elevenlabs_client):
        """Test successful voice settings retrieval."""
        # Mock voice response
        mock_voice = MagicMock()
        mock_voice.settings = MagicMock()
        mock_voice.settings.stability = 0.6
        mock_voice.settings.similarity_boost = 0.8
        mock_voice.settings.style = 0.2
        mock_voice.settings.use_speaker_boost = True

        mock_elevenlabs_client.voices.get = AsyncMock(return_value=mock_voice)

        # Get voice settings
        settings = await tts_service.get_voice_settings("test_voice_id")

        # Verify settings
        assert settings.stability == 0.6
        assert settings.similarity_boost == 0.8
        assert settings.style == 0.2
        assert settings.use_speaker_boost is True

        # Verify API was called
        mock_elevenlabs_client.voices.get.assert_called_once_with(voice_id="test_voice_id")

    @pytest.mark.asyncio
    async def test_get_voice_settings_cached(self, tts_service, mock_elevenlabs_client):
        """Test voice settings are cached."""
        # Mock voice response
        mock_voice = MagicMock()
        mock_voice.settings = MagicMock()
        mock_voice.settings.stability = 0.6
        mock_voice.settings.similarity_boost = 0.8
        mock_voice.settings.style = 0.2
        mock_voice.settings.use_speaker_boost = True

        mock_elevenlabs_client.voices.get = AsyncMock(return_value=mock_voice)

        # Get voice settings twice
        settings1 = await tts_service.get_voice_settings("test_voice_id")
        settings2 = await tts_service.get_voice_settings("test_voice_id")

        # Verify same settings returned
        assert settings1 == settings2

        # Verify API was called only once (cached on second call)
        assert mock_elevenlabs_client.voices.get.call_count == 1

    @pytest.mark.asyncio
    async def test_get_voice_settings_api_error(self, tts_service, mock_elevenlabs_client):
        """Test voice settings retrieval handles API errors."""
        mock_elevenlabs_client.voices.get = AsyncMock(
            side_effect=Exception("API error")
        )

        with pytest.raises(RuntimeError, match="Failed to get voice settings"):
            await tts_service.get_voice_settings("test_voice_id")

    @pytest.mark.asyncio
    async def test_list_available_voices_success(self, tts_service, mock_elevenlabs_client):
        """Test successful voice listing."""
        # Mock voices response
        mock_voice1 = MagicMock()
        mock_voice1.voice_id = "voice_1"
        mock_voice1.name = "Voice One"
        mock_voice1.category = "premade"
        mock_voice1.description = "A test voice"
        mock_voice1.labels = {"accent": "american"}

        mock_voice2 = MagicMock()
        mock_voice2.voice_id = "voice_2"
        mock_voice2.name = "Voice Two"
        mock_voice2.category = "cloned"
        mock_voice2.description = "Another test voice"
        mock_voice2.labels = {"accent": "british"}

        mock_response = MagicMock()
        mock_response.voices = [mock_voice1, mock_voice2]

        mock_elevenlabs_client.voices.get_all = AsyncMock(return_value=mock_response)

        # List voices
        voices = await tts_service.list_available_voices()

        # Verify voices
        assert len(voices) == 2
        assert voices[0]["voice_id"] == "voice_1"
        assert voices[0]["name"] == "Voice One"
        assert voices[1]["voice_id"] == "voice_2"
        assert voices[1]["name"] == "Voice Two"

    @pytest.mark.asyncio
    async def test_list_available_voices_api_error(self, tts_service, mock_elevenlabs_client):
        """Test voice listing handles API errors."""
        mock_elevenlabs_client.voices.get_all = AsyncMock(
            side_effect=Exception("API error")
        )

        with pytest.raises(RuntimeError, match="Failed to list voices"):
            await tts_service.list_available_voices()


class TestAudioTags:
    """Test audio tag enumeration."""

    def test_audio_tag_values(self):
        """Test audio tag enum values."""
        assert AudioTag.SIGH.value == "[sigh]"
        assert AudioTag.LAUGHS.value == "[laughs]"
        assert AudioTag.EXCITED.value == "<excited>"
        assert AudioTag.EMPATHETIC.value == "<empathetic>"
        assert AudioTag.QUESTIONING.value == "<questioning>"
        assert AudioTag.UNDERSTANDING.value == "<understanding>"
