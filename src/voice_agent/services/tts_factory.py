"""
TTS Provider Factory.

This module provides a factory pattern for creating TTS service instances
based on tenant configuration. Includes provider health checking and
failover logic (ElevenLabs → OpenAI) for high availability.

**Validates: Requirements 5.6**
"""

import logging
from enum import Enum
from typing import Any

import aiohttp

from .elevenlabs_tts_service import (
    ElevenLabsTTSService,
    TTSConfig,
    VoiceConfig,
)
from .openai_tts_service import OpenAITTSConfig, OpenAITTSService

logger = logging.getLogger(__name__)


class ProviderHealth(Enum):
    """Provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class TTSProviderFactory:
    """
    Factory for creating TTS service instances with health checking and failover.
    
    Provides a consistent interface for creating TTS providers based on
    tenant configuration, with automatic health checking and failover from
    ElevenLabs to OpenAI when primary provider is unavailable.
    """

    def __init__(
        self,
        elevenlabs_api_key: str | None = None,
        openai_api_key: str | None = None,
        enable_failover: bool = True,
    ):
        """
        Initialize TTS provider factory.
        
        Args:
            elevenlabs_api_key: ElevenLabs API key (if None, uses ELEVENLABS_API_KEY env var)
            openai_api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            enable_failover: Enable automatic failover from ElevenLabs to OpenAI
        """
        self.elevenlabs_api_key = elevenlabs_api_key
        self.openai_api_key = openai_api_key
        self.enable_failover = enable_failover

        # Cache for provider health status
        self._health_cache: dict[str, ProviderHealth] = {}

        logger.info(
            "TTS provider factory initialized (failover: %s)",
            "enabled" if enable_failover else "disabled",
        )

    async def create_tts_service(
        self,
        voice_config: VoiceConfig,
        provider_config: dict[str, Any] | None = None,
    ) -> ElevenLabsTTSService | OpenAITTSService:
        """
        Create TTS service instance based on configuration.
        
        Selects the appropriate provider implementation based on the
        voice_config.provider field and creates an instance with proper
        configuration. If failover is enabled and ElevenLabs is unavailable,
        automatically creates OpenAI fallback service.
        
        Args:
            voice_config: Voice configuration specifying provider and voice settings
            provider_config: Optional provider-specific configuration
            
        Returns:
            Configured TTS service instance
            
        Raises:
            ValueError: If provider is not supported
            RuntimeError: If provider creation fails and no fallback available
        """
        provider_config = provider_config or {}

        # Check provider health
        health = await self.check_provider_health(voice_config.provider)

        if health == ProviderHealth.UNAVAILABLE and self.enable_failover:
            logger.warning(
                "Primary provider '%s' is unavailable, attempting failover",
                voice_config.provider,
            )
            return await self._create_with_failover(voice_config, provider_config)

        # Create primary provider
        try:
            service = await self._create_provider(voice_config, provider_config)
            logger.info(
                "Created TTS service for provider '%s'",
                voice_config.provider,
            )
            return service

        except Exception as e:
            logger.error("Failed to create provider '%s': %s", voice_config.provider, str(e))

            if self.enable_failover and voice_config.provider == "elevenlabs":
                logger.info("Attempting failover after provider creation failure")
                return await self._create_with_failover(voice_config, provider_config)
            else:
                raise RuntimeError(
                    f"Failed to create TTS provider '{voice_config.provider}': {str(e)}"
                ) from e

    async def _create_provider(
        self,
        voice_config: VoiceConfig,
        provider_config: dict[str, Any],
    ) -> ElevenLabsTTSService | OpenAITTSService:
        """
        Create provider instance without failover logic.
        
        Args:
            voice_config: Voice configuration
            provider_config: Provider-specific configuration
            
        Returns:
            TTS service instance
            
        Raises:
            ValueError: If provider is not supported
        """
        provider = voice_config.provider.lower()

        if provider == "elevenlabs":
            tts_config = TTSConfig(
                api_key=provider_config.get("api_key", self.elevenlabs_api_key),
                default_voice_config=voice_config,
                chunk_size=provider_config.get("chunk_size", 2048),
                optimize_streaming_latency=provider_config.get("optimize_streaming_latency", 3),
                output_format=provider_config.get("output_format", "mp3_44100_128"),
            )
            return ElevenLabsTTSService(config=tts_config)

        elif provider == "openai":
            openai_config = OpenAITTSConfig(
                api_key=provider_config.get("api_key", self.openai_api_key),
                default_voice_config=voice_config,
                model=provider_config.get("model", "tts-1"),
                chunk_size=provider_config.get("chunk_size", 4096),
            )
            return OpenAITTSService(config=openai_config)

        else:
            raise ValueError(f"Unsupported TTS provider: {provider}")

    async def _create_with_failover(
        self,
        voice_config: VoiceConfig,
        provider_config: dict[str, Any],
    ) -> ElevenLabsTTSService | OpenAITTSService:
        """
        Create service with automatic failover to OpenAI.
        
        Failover strategy: ElevenLabs → OpenAI
        
        Args:
            voice_config: Original voice configuration
            provider_config: Provider-specific configuration
            
        Returns:
            TTS service instance (primary or fallback)
            
        Raises:
            RuntimeError: If all providers fail
        """
        primary_provider = voice_config.provider.lower()

        # Only ElevenLabs has failover to OpenAI
        if primary_provider == "elevenlabs":
            try:
                logger.info("Attempting OpenAI fallback for ElevenLabs")

                # Create fallback voice config with OpenAI provider
                fallback_voice_config = VoiceConfig(
                    provider="openai",
                    voice_id=provider_config.get("fallback_voice", "alloy"),
                    # OpenAI doesn't use these settings, but keep for consistency
                    stability=voice_config.stability,
                    similarity_boost=voice_config.similarity_boost,
                    style=voice_config.style,
                    use_speaker_boost=voice_config.use_speaker_boost,
                )

                return await self._create_provider(fallback_voice_config, provider_config)

            except Exception as e:
                logger.error("OpenAI fallback also failed: %s", str(e))
                raise RuntimeError(
                    "All TTS providers failed. Primary: ElevenLabs, Fallback: OpenAI"
                ) from e

        else:
            # OpenAI has no fallback
            raise RuntimeError(
                f"Primary provider '{primary_provider}' is unavailable and no fallback configured"
            )

    async def check_provider_health(self, provider: str) -> ProviderHealth:
        """
        Check health status of a provider.
        
        Performs lightweight health check to determine if provider is
        available and responsive. Results are cached to avoid excessive
        health checks.
        
        Args:
            provider: Provider name ("elevenlabs" or "openai")
            
        Returns:
            Provider health status
        """
        provider = provider.lower()

        # Return cached health if available
        if provider in self._health_cache:
            return self._health_cache[provider]

        try:
            if provider == "elevenlabs":
                health = await self._check_elevenlabs_health()

            elif provider == "openai":
                # OpenAI is always considered healthy (cloud service)
                # Could add actual API check if needed
                health = ProviderHealth.HEALTHY

            else:
                logger.warning("Unknown provider '%s', assuming unavailable", provider)
                health = ProviderHealth.UNAVAILABLE

            # Cache result
            self._health_cache[provider] = health

            logger.debug("Provider '%s' health: %s", provider, health.value)
            return health

        except Exception as e:
            logger.error("Error checking provider '%s' health: %s", provider, str(e))
            health = ProviderHealth.UNAVAILABLE
            self._health_cache[provider] = health
            return health

    async def _check_elevenlabs_health(self) -> ProviderHealth:
        """
        Check ElevenLabs API health.
        
        Returns:
            Provider health status
        """
        try:
            # Check ElevenLabs API availability
            timeout = aiohttp.ClientTimeout(total=5.0)
            headers = {}
            if self.elevenlabs_api_key:
                headers["xi-api-key"] = self.elevenlabs_api_key

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    "https://api.elevenlabs.io/v1/voices",
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        return ProviderHealth.HEALTHY
                    elif response.status in [401, 403]:
                        # Authentication issue, but service is up
                        logger.warning("ElevenLabs authentication issue (status: %d)", response.status)
                        return ProviderHealth.DEGRADED
                    else:
                        return ProviderHealth.DEGRADED

        except Exception as e:
            logger.debug("ElevenLabs health check failed: %s", str(e))
            return ProviderHealth.UNAVAILABLE

    def clear_health_cache(self):
        """Clear the provider health cache."""
        self._health_cache.clear()
        logger.debug("Provider health cache cleared")

    async def get_all_provider_health(self) -> dict[str, ProviderHealth]:
        """
        Get health status for all supported providers.
        
        Returns:
            Dictionary mapping provider names to health status
        """
        providers = ["elevenlabs", "openai"]
        health_status = {}

        for provider in providers:
            health_status[provider] = await self.check_provider_health(provider)

        return health_status
