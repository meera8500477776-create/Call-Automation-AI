"""
LLM Provider Factory.

This module provides a factory pattern for creating LLM service instances
based on tenant configuration. Includes provider health checking and
failover logic for high availability.

**Validates: Requirements 13.4, 13.6, 13.7**
"""

import logging
from enum import Enum
from typing import Any

import aiohttp

from ..config.settings import Settings
from .llm_service import LLMConfig, LLMService
from .ollama_llm_service import OllamaLLMService
from .openai_llm_service import OpenAILLMService
from .vllm_llm_service import vLLMLLMService

logger = logging.getLogger(__name__)


class ProviderHealth(Enum):
    """Provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class LLMProviderFactory:
    """
    Factory for creating LLM service instances with health checking and failover.

    Provides a consistent interface for creating LLM providers based on
    tenant configuration, with automatic health checking and failover to
    alternative providers when primary provider is unavailable.
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        ollama_base_url: str = "http://localhost:11434",
        vllm_base_url: str = "http://localhost:8000",
        enable_failover: bool = True,
    ):
        """
        Initialize LLM provider factory.

        Args:
            openai_api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            ollama_base_url: Ollama server base URL
            vllm_base_url: vLLM server base URL
            enable_failover: Enable automatic failover to alternative providers
        """
        self.openai_api_key = (
            openai_api_key if openai_api_key is not None else Settings().openai_api_key
        )
        self.ollama_base_url = ollama_base_url
        self.vllm_base_url = vllm_base_url
        self.enable_failover = enable_failover

        # Cache for provider health status
        self._health_cache: dict[str, ProviderHealth] = {}

        logger.info(
            "LLM provider factory initialized (failover: %s)",
            "enabled" if enable_failover else "disabled",
        )

    async def create_llm_service(
        self,
        config: LLMConfig,
        provider_config: dict[str, Any] | None = None,
    ) -> LLMService:
        """
        Create LLM service instance based on configuration.

        Selects the appropriate provider implementation based on the
        config.provider field and creates an instance with proper
        configuration. If failover is enabled and the primary provider
        is unavailable, automatically creates a fallback service.

        Args:
            config: LLM configuration specifying provider and model
            provider_config: Optional provider-specific configuration

        Returns:
            Configured LLM service instance

        Raises:
            ValueError: If provider is not supported
            RuntimeError: If provider creation fails and no fallback available
        """
        provider_config = provider_config or {}

        # Check provider health
        health = await self.check_provider_health(config.provider)

        if health == ProviderHealth.UNAVAILABLE and self.enable_failover:
            logger.warning(
                "Primary provider '%s' is unavailable, attempting failover",
                config.provider,
            )
            return await self._create_with_failover(config, provider_config)

        # Create primary provider
        try:
            service = await self._create_provider(config, provider_config)

            # If failover is enabled, add fallback service
            if self.enable_failover:
                fallback = await self._create_fallback_service(config, provider_config)
                if fallback:
                    # Inject fallback into provider if supported
                    if hasattr(service, 'fallback_service'):
                        service.fallback_service = fallback
                        logger.info(
                            "Configured fallback service for provider '%s'",
                            config.provider,
                        )

            return service

        except Exception as e:
            logger.error("Failed to create provider '%s': %s", config.provider, str(e))

            if self.enable_failover:
                logger.info("Attempting failover after provider creation failure")
                return await self._create_with_failover(config, provider_config)
            else:
                raise RuntimeError(
                    f"Failed to create LLM provider '{config.provider}': {str(e)}"
                ) from e

    async def _create_provider(
        self,
        config: LLMConfig,
        provider_config: dict[str, Any],
    ) -> LLMService:
        """
        Create provider instance without failover logic.

        Args:
            config: LLM configuration
            provider_config: Provider-specific configuration

        Returns:
            LLM service instance

        Raises:
            ValueError: If provider is not supported
        """
        provider = config.provider.lower()

        if provider == "openai":
            return OpenAILLMService(
                config=config,
                api_key=provider_config.get("api_key", self.openai_api_key),
            )

        elif provider == "ollama":
            return OllamaLLMService(
                config=config,
                base_url=provider_config.get("base_url", self.ollama_base_url),
                gpu_enabled=provider_config.get("gpu_enabled", True),
                fallback_service=None,  # Will be set by create_llm_service if needed
            )

        elif provider == "vllm":
            return vLLMLLMService(
                config=config,
                base_url=provider_config.get("base_url", self.vllm_base_url),
                gpu_memory_utilization=provider_config.get("gpu_memory_utilization", 0.9),
                max_num_seqs=provider_config.get("max_num_seqs", 256),
                max_model_len=provider_config.get("max_model_len"),
                fallback_service=None,  # Will be set by create_llm_service if needed
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    async def _create_fallback_service(
        self,
        primary_config: LLMConfig,
        provider_config: dict[str, Any],
    ) -> LLMService | None:
        """
        Create fallback service for the primary provider.

        Fallback strategy:
        - Ollama/vLLM -> OpenAI (cloud fallback for local providers)
        - OpenAI -> None (cloud provider has no fallback)

        Args:
            primary_config: Primary provider configuration
            provider_config: Provider-specific configuration

        Returns:
            Fallback service instance or None if no fallback available
        """
        primary_provider = primary_config.provider.lower()

        # Local providers fall back to OpenAI
        if primary_provider in ["ollama", "vllm"]:
            try:
                # Create OpenAI fallback with same model settings
                fallback_config = LLMConfig(
                    provider="openai",
                    model=provider_config.get("fallback_model", "gpt-4"),
                    temperature=primary_config.temperature,
                    max_tokens=primary_config.max_tokens,
                    system_prompt=primary_config.system_prompt,
                    top_p=primary_config.top_p,
                    frequency_penalty=primary_config.frequency_penalty,
                    presence_penalty=primary_config.presence_penalty,
                    timeout=primary_config.timeout,
                )

                fallback = OpenAILLMService(
                    config=fallback_config,
                    api_key=provider_config.get("api_key", self.openai_api_key),
                )

                logger.info(
                    "Created OpenAI fallback service for primary provider '%s'",
                    primary_provider,
                )
                return fallback

            except Exception as e:
                logger.warning("Failed to create fallback service: %s", str(e))
                return None

        # OpenAI has no fallback
        return None

    async def _create_with_failover(
        self,
        config: LLMConfig,
        provider_config: dict[str, Any],
    ) -> LLMService:
        """
        Create service with automatic failover to alternative provider.

        Args:
            config: Original LLM configuration
            provider_config: Provider-specific configuration

        Returns:
            LLM service instance (primary or fallback)

        Raises:
            RuntimeError: If all providers fail
        """
        primary_provider = config.provider.lower()

        # Try fallback providers in order
        if primary_provider in ["ollama", "vllm"]:
            # Try OpenAI as fallback
            try:
                logger.info("Attempting OpenAI fallback for '%s'", primary_provider)
                fallback_config = LLMConfig(
                    provider="openai",
                    model=provider_config.get("fallback_model", "gpt-4"),
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    system_prompt=config.system_prompt,
                    top_p=config.top_p,
                    frequency_penalty=config.frequency_penalty,
                    presence_penalty=config.presence_penalty,
                    timeout=config.timeout,
                )

                return await self._create_provider(fallback_config, provider_config)

            except Exception as e:
                logger.error("OpenAI fallback also failed: %s", str(e))
                raise RuntimeError(
                    f"All LLM providers failed. Primary: {primary_provider}, Fallback: OpenAI"
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
            provider: Provider name ("openai", "ollama", "vllm")

        Returns:
            Provider health status
        """
        provider = provider.lower()

        # Return cached health if available (cache for 60 seconds)
        if provider in self._health_cache:
            return self._health_cache[provider]

        try:
            if provider == "openai":
                # OpenAI is always considered healthy (cloud service)
                # Could add actual API check if needed
                health = ProviderHealth.HEALTHY

            elif provider == "ollama":
                # Check Ollama server availability
                health = await self._check_ollama_health()

            elif provider == "vllm":
                # Check vLLM server availability
                health = await self._check_vllm_health()

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

    async def _check_ollama_health(self) -> ProviderHealth:
        """
        Check Ollama server health.

        Returns:
            Provider health status
        """
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.ollama_base_url}/api/tags") as response:
                    if response.status == 200:
                        return ProviderHealth.HEALTHY
                    else:
                        return ProviderHealth.DEGRADED

        except Exception as e:
            logger.debug("Ollama health check failed: %s", str(e))
            return ProviderHealth.UNAVAILABLE

    async def _check_vllm_health(self) -> ProviderHealth:
        """
        Check vLLM server health.

        Returns:
            Provider health status
        """
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.vllm_base_url}/v1/models") as response:
                    if response.status == 200:
                        return ProviderHealth.HEALTHY
                    else:
                        return ProviderHealth.DEGRADED

        except Exception as e:
            logger.debug("vLLM health check failed: %s", str(e))
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
        providers = ["openai", "ollama", "vllm"]
        health_status = {}

        for provider in providers:
            health_status[provider] = await self.check_provider_health(provider)

        return health_status
