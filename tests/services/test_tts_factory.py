"""
Unit tests for TTS Provider Factory.

Tests factory pattern for creating TTS service instances with health checking
and failover logic (ElevenLabs → OpenAI).

**Validates: Requirements 5.6**
"""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.voice_agent.services.elevenlabs_tts_service import (
    ElevenLabsTTSService,
    VoiceConfig,
)
from src.voice_agent.services.openai_tts_service import OpenAITTSService
from src.voice_agent.services.tts_factory import (
    ProviderHealth,
    TTSProviderFactory,
)


@pytest.fixture
def factory():
    """Create TTS provider factory for testing."""
    return TTSProviderFactory(
        elevenlabs_api_key="test_elevenlabs_key",
        openai_api_key="test_openai_key",
        enable_failover=True,
    )


@pytest.fixture
def elevenlabs_voice_config():
    """Create ElevenLabs voice configuration."""
    return VoiceConfig(
        provider="elevenlabs",
        voice_id="test_voice_id",
        stability=0.5,
        similarity_boost=0.75,
        style=0.0,
        use_speaker_boost=True,
    )


@pytest.fixture
def openai_voice_config():
    """Create OpenAI voice configuration."""
    return VoiceConfig(
        provider="openai",
        voice_id="alloy",
    )


class TestTTSProviderFactory:
    """Test suite for TTS provider factory."""

    @pytest.mark.asyncio
    async def test_placeholder_elevenlabs_voice_fails_before_provider_check(self, factory):
        voice_config = VoiceConfig(
            provider="elevenlabs",
            voice_id="elevenlabs_voice_id_here",
        )
        factory.check_provider_health = AsyncMock()

        with pytest.raises(ValueError, match="real ElevenLabs voice ID"):
            await factory.create_tts_service(voice_config)

        factory.check_provider_health.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_create_elevenlabs_service(self, factory, elevenlabs_voice_config):
        """Test creating ElevenLabs TTS service."""
        # Mock health check to return healthy
        factory._health_cache["elevenlabs"] = ProviderHealth.HEALTHY

        service = await factory.create_tts_service(elevenlabs_voice_config)

        assert isinstance(service, ElevenLabsTTSService)
        assert service.config.api_key == "test_elevenlabs_key"
        assert service.config.default_voice_config.voice_id == "test_voice_id"

    @pytest.mark.asyncio
    async def test_create_openai_service(self, factory, openai_voice_config):
        """Test creating OpenAI TTS service."""
        # Mock health check to return healthy
        factory._health_cache["openai"] = ProviderHealth.HEALTHY

        service = await factory.create_tts_service(openai_voice_config)

        assert isinstance(service, OpenAITTSService)
        assert service.config.api_key == "test_openai_key"
        assert service.config.default_voice_config.voice_id == "alloy"

    @pytest.mark.asyncio
    async def test_create_with_custom_config(self, factory, elevenlabs_voice_config):
        """Test creating service with custom provider configuration."""
        factory._health_cache["elevenlabs"] = ProviderHealth.HEALTHY

        provider_config = {
            "chunk_size": 4096,
            "optimize_streaming_latency": 4,
            "output_format": "pcm_16000",
        }

        service = await factory.create_tts_service(
            elevenlabs_voice_config,
            provider_config=provider_config,
        )

        assert isinstance(service, ElevenLabsTTSService)
        assert service.config.chunk_size == 4096
        assert service.config.optimize_streaming_latency == 4
        assert service.config.output_format == "pcm_16000"

    @pytest.mark.asyncio
    async def test_failover_elevenlabs_to_openai(self, factory, elevenlabs_voice_config):
        """Test automatic failover from ElevenLabs to OpenAI."""
        # Mock ElevenLabs as unavailable
        factory._health_cache["elevenlabs"] = ProviderHealth.UNAVAILABLE
        factory._health_cache["openai"] = ProviderHealth.HEALTHY

        service = await factory.create_tts_service(elevenlabs_voice_config)

        # Should create OpenAI service as fallback
        assert isinstance(service, OpenAITTSService)

    @pytest.mark.asyncio
    async def test_failover_with_custom_voice(self, factory, elevenlabs_voice_config):
        """Test failover with custom fallback voice configuration."""
        factory._health_cache["elevenlabs"] = ProviderHealth.UNAVAILABLE
        factory._health_cache["openai"] = ProviderHealth.HEALTHY

        provider_config = {
            "fallback_voice": "nova",
        }

        service = await factory.create_tts_service(
            elevenlabs_voice_config,
            provider_config=provider_config,
        )

        assert isinstance(service, OpenAITTSService)
        assert service.config.default_voice_config.voice_id == "nova"

    @pytest.mark.asyncio
    async def test_failover_disabled(self, elevenlabs_voice_config):
        """Test that failover can be disabled."""
        factory = TTSProviderFactory(
            elevenlabs_api_key="test_key",
            openai_api_key="test_key",
            enable_failover=False,
        )

        # Mock ElevenLabs as unavailable
        factory._health_cache["elevenlabs"] = ProviderHealth.UNAVAILABLE

        # Mock provider creation to fail
        with patch.object(factory, '_create_provider', side_effect=RuntimeError("Provider unavailable")):
            # Should raise error instead of failing over
            with pytest.raises(RuntimeError, match="Failed to create TTS provider"):
                await factory.create_tts_service(elevenlabs_voice_config)

    @pytest.mark.asyncio
    async def test_openai_no_failover(self, factory, openai_voice_config):
        """Test that OpenAI provider has no fallback."""
        # Mock OpenAI as unavailable
        factory._health_cache["openai"] = ProviderHealth.UNAVAILABLE

        # Should raise error (no fallback for OpenAI)
        with pytest.raises(RuntimeError, match="no fallback configured"):
            await factory.create_tts_service(openai_voice_config)

    @pytest.mark.asyncio
    async def test_unsupported_provider(self, factory):
        """Test error handling for unsupported provider."""
        # Create a mock config that bypasses VoiceConfig validation
        invalid_config = MagicMock()
        invalid_config.provider = "unsupported"
        invalid_config.voice_id = "test"

        factory._health_cache["unsupported"] = ProviderHealth.HEALTHY

        # Factory wraps ValueError in RuntimeError
        with pytest.raises(RuntimeError, match="Failed to create TTS provider"):
            await factory.create_tts_service(invalid_config)

    @pytest.mark.asyncio
    async def test_all_providers_fail(self, factory, elevenlabs_voice_config):
        """Test error when all providers fail."""
        # Mock both providers as unavailable
        factory._health_cache["elevenlabs"] = ProviderHealth.UNAVAILABLE
        factory._health_cache["openai"] = ProviderHealth.UNAVAILABLE

        # Mock OpenAI creation to fail
        with patch.object(factory, '_create_provider', side_effect=RuntimeError("API error")):
            with pytest.raises(RuntimeError, match="All TTS providers failed"):
                await factory.create_tts_service(elevenlabs_voice_config)

    @pytest.mark.asyncio
    async def test_check_elevenlabs_health_success(self, factory):
        """Test ElevenLabs health check when service is healthy."""
        mock_response = MagicMock()
        mock_response.status = 200

        mock_get = AsyncMock()
        mock_get.__aenter__.return_value = mock_response

        mock_session = MagicMock()
        mock_session.get.return_value = mock_get
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = AsyncMock()

        with patch('aiohttp.ClientSession', return_value=mock_session):
            health = await factory.check_provider_health("elevenlabs")

        assert health == ProviderHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_check_elevenlabs_health_auth_error(self, factory):
        """Test ElevenLabs health check with authentication error."""
        mock_response = MagicMock()
        mock_response.status = 401

        mock_get = AsyncMock()
        mock_get.__aenter__.return_value = mock_response

        mock_session = MagicMock()
        mock_session.get.return_value = mock_get
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = AsyncMock()

        with patch('aiohttp.ClientSession', return_value=mock_session):
            health = await factory.check_provider_health("elevenlabs")

        # Authentication error means service is up but degraded
        assert health == ProviderHealth.DEGRADED

    @pytest.mark.asyncio
    async def test_check_elevenlabs_health_unavailable(self, factory):
        """Test ElevenLabs health check when service is unavailable."""
        mock_request = MagicMock()
        mock_request.__aenter__ = AsyncMock(
            side_effect=aiohttp.ClientError("Connection failed")
        )
        mock_session = MagicMock()
        mock_session.get.return_value = mock_request
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            health = await factory.check_provider_health("elevenlabs")

        assert health == ProviderHealth.UNAVAILABLE

    @pytest.mark.asyncio
    async def test_check_openai_health(self, factory):
        """Test OpenAI health check (always healthy)."""
        health = await factory.check_provider_health("openai")

        # OpenAI is always considered healthy
        assert health == ProviderHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_health_cache(self, factory):
        """Test that health check results are cached."""
        # First check
        factory._health_cache["elevenlabs"] = ProviderHealth.HEALTHY
        health1 = await factory.check_provider_health("elevenlabs")

        # Second check should use cache
        health2 = await factory.check_provider_health("elevenlabs")

        assert health1 == health2 == ProviderHealth.HEALTHY

    def test_clear_health_cache(self, factory):
        """Test clearing health cache."""
        factory._health_cache["elevenlabs"] = ProviderHealth.HEALTHY
        factory._health_cache["openai"] = ProviderHealth.HEALTHY

        factory.clear_health_cache()

        assert len(factory._health_cache) == 0

    @pytest.mark.asyncio
    async def test_get_all_provider_health(self, factory):
        """Test getting health status for all providers."""
        # Mock health checks
        factory._health_cache["elevenlabs"] = ProviderHealth.HEALTHY
        factory._health_cache["openai"] = ProviderHealth.HEALTHY

        health_status = await factory.get_all_provider_health()

        assert health_status["elevenlabs"] == ProviderHealth.HEALTHY
        assert health_status["openai"] == ProviderHealth.HEALTHY
        assert len(health_status) == 2

    @pytest.mark.asyncio
    async def test_unknown_provider_health(self, factory):
        """Test health check for unknown provider."""
        health = await factory.check_provider_health("unknown_provider")

        assert health == ProviderHealth.UNAVAILABLE
