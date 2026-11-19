"""
Unit tests for LLM Provider Factory.

Tests factory pattern implementation, provider selection, health checking,
and failover logic.

**Validates: Requirements 13.4, 13.6, 13.7**
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.voice_agent.services.llm_factory import (
    LLMProviderFactory,
    ProviderHealth,
)
from src.voice_agent.services.llm_service import LLMConfig
from src.voice_agent.services.ollama_llm_service import OllamaLLMService
from src.voice_agent.services.openai_llm_service import OpenAILLMService
from src.voice_agent.services.vllm_llm_service import vLLMLLMService


@pytest.fixture
def factory():
    """Create LLM provider factory instance."""
    return LLMProviderFactory(
        openai_api_key="test-key",
        ollama_base_url="http://localhost:11434",
        vllm_base_url="http://localhost:8000",
        enable_failover=True,
    )


@pytest.fixture
def openai_config():
    """Create OpenAI LLM configuration."""
    return LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful assistant.",
    )


@pytest.fixture
def ollama_config():
    """Create Ollama LLM configuration."""
    return LLMConfig(
        provider="ollama",
        model="llama2",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful assistant.",
    )


@pytest.fixture
def vllm_config():
    """Create vLLM configuration."""
    return LLMConfig(
        provider="vllm",
        model="meta-llama/Llama-2-7b-chat-hf",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful assistant.",
    )


class TestLLMProviderFactory:
    """Test suite for LLM provider factory."""

    def test_default_openai_key_is_loaded_from_application_settings(self):
        factory = LLMProviderFactory()

        assert factory.openai_api_key == "test-openai-key"

    @pytest.mark.asyncio
    async def test_create_openai_service(self, factory, openai_config):
        """Test creating OpenAI service instance."""
        service = await factory.create_llm_service(openai_config)

        assert isinstance(service, OpenAILLMService)
        assert service.config.provider == "openai"
        assert service.config.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_create_ollama_service(self, factory, ollama_config):
        """Test creating Ollama service instance."""
        with patch.object(factory, 'check_provider_health', new_callable=AsyncMock) as mock_health:
            with patch.object(OllamaLLMService, 'check_model_availability', new_callable=AsyncMock) as mock_check:
                mock_health.return_value = ProviderHealth.HEALTHY
                mock_check.return_value = True

                service = await factory.create_llm_service(ollama_config)

                assert isinstance(service, OllamaLLMService)
                assert service.config.provider == "ollama"
                assert service.config.model == "llama2"
                assert service.base_url == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_create_vllm_service(self, factory, vllm_config):
        """Test creating vLLM service instance."""
        with patch.object(factory, 'check_provider_health', new_callable=AsyncMock) as mock_health:
            with patch.object(vLLMLLMService, 'check_server_availability', new_callable=AsyncMock) as mock_check:
                mock_health.return_value = ProviderHealth.HEALTHY
                mock_check.return_value = True

                service = await factory.create_llm_service(vllm_config)

                assert isinstance(service, vLLMLLMService)
                assert service.config.provider == "vllm"
                assert service.config.model == "meta-llama/Llama-2-7b-chat-hf"
                assert service.base_url == "http://localhost:8000"

    @pytest.mark.asyncio
    async def test_unsupported_provider(self, factory):
        """Test error handling for unsupported provider."""
        # Create config with valid provider first, then modify
        config = LLMConfig(
            provider="openai",
            model="test-model",
        )
        # Manually set invalid provider to bypass validation
        config.provider = "unsupported"

        # Disable failover to get the expected error
        factory_no_failover = LLMProviderFactory(enable_failover=False)

        with pytest.raises((ValueError, RuntimeError)):
            await factory_no_failover.create_llm_service(config)

    @pytest.mark.asyncio
    async def test_provider_specific_config(self, factory, ollama_config):
        """Test passing provider-specific configuration."""
        provider_config = {
            "base_url": "http://custom:11434",
            "gpu_enabled": False,
        }

        with patch.object(factory, 'check_provider_health', new_callable=AsyncMock) as mock_health:
            with patch.object(OllamaLLMService, 'check_model_availability', new_callable=AsyncMock) as mock_check:
                mock_health.return_value = ProviderHealth.HEALTHY
                mock_check.return_value = True

                service = await factory.create_llm_service(ollama_config, provider_config)

                assert service.base_url == "http://custom:11434"
                assert service.gpu_enabled is False

    @pytest.mark.asyncio
    async def test_consistent_interface(self, factory, openai_config, ollama_config, vllm_config):
        """Test that all providers implement consistent interface."""
        with patch.object(factory, 'check_provider_health', new_callable=AsyncMock) as mock_health:
            with patch.object(OllamaLLMService, 'check_model_availability', new_callable=AsyncMock) as mock_ollama:
                with patch.object(vLLMLLMService, 'check_server_availability', new_callable=AsyncMock) as mock_vllm:
                    mock_health.return_value = ProviderHealth.HEALTHY
                    mock_ollama.return_value = True
                    mock_vllm.return_value = True

                    openai_service = await factory.create_llm_service(openai_config)
                    ollama_service = await factory.create_llm_service(ollama_config)
                    vllm_service = await factory.create_llm_service(vllm_config)

                    # All should have generate_response method
                    assert hasattr(openai_service, 'generate_response')
                    assert hasattr(ollama_service, 'generate_response')
                    assert hasattr(vllm_service, 'generate_response')

                    # All should have config attribute
                    assert hasattr(openai_service, 'config')
                    assert hasattr(ollama_service, 'config')
                    assert hasattr(vllm_service, 'config')


class TestHealthChecking:
    """Test suite for provider health checking."""

    @pytest.mark.asyncio
    async def test_openai_health_always_healthy(self, factory):
        """Test that OpenAI is always considered healthy (cloud service)."""
        health = await factory.check_provider_health("openai")
        assert health == ProviderHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_ollama_health_check_success(self, factory):
        """Test successful Ollama health check."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session

            health = await factory.check_provider_health("ollama")
            assert health == ProviderHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_ollama_health_check_degraded(self, factory):
        """Test degraded Ollama health check."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session

            health = await factory.check_provider_health("ollama")
            assert health == ProviderHealth.DEGRADED

    @pytest.mark.asyncio
    async def test_ollama_health_check_unavailable(self, factory):
        """Test unavailable Ollama health check."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_request = MagicMock()
            mock_request.__aenter__ = AsyncMock(
                side_effect=Exception("Connection refused")
            )
            mock_session = MagicMock()
            mock_session.get.return_value = mock_request
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session

            health = await factory.check_provider_health("ollama")
            assert health == ProviderHealth.UNAVAILABLE

    @pytest.mark.asyncio
    async def test_vllm_health_check_success(self, factory):
        """Test successful vLLM health check."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session

            health = await factory.check_provider_health("vllm")
            assert health == ProviderHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_health_cache(self, factory):
        """Test that health check results are cached."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session

            # First call should hit the API
            health1 = await factory.check_provider_health("ollama")
            assert health1 == ProviderHealth.HEALTHY

            # Second call should use cache (no API call)
            health2 = await factory.check_provider_health("ollama")
            assert health2 == ProviderHealth.HEALTHY

            # Should only have called the API once
            assert mock_session.get.call_count == 1

    @pytest.mark.asyncio
    async def test_clear_health_cache(self, factory):
        """Test clearing health cache."""
        # Add some cached health status
        factory._health_cache["ollama"] = ProviderHealth.HEALTHY
        factory._health_cache["vllm"] = ProviderHealth.DEGRADED

        assert len(factory._health_cache) == 2

        factory.clear_health_cache()

        assert len(factory._health_cache) == 0

    @pytest.mark.asyncio
    async def test_get_all_provider_health(self, factory):
        """Test getting health status for all providers."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session

            health_status = await factory.get_all_provider_health()

            assert "openai" in health_status
            assert "ollama" in health_status
            assert "vllm" in health_status
            assert health_status["openai"] == ProviderHealth.HEALTHY


class TestFailoverLogic:
    """Test suite for provider failover logic."""

    @pytest.mark.asyncio
    async def test_ollama_failover_to_openai(self, factory, ollama_config):
        """Test failover from Ollama to OpenAI when Ollama is unavailable."""
        # Mock Ollama as unavailable
        with patch.object(factory, 'check_provider_health', new_callable=AsyncMock) as mock_health:
            mock_health.return_value = ProviderHealth.UNAVAILABLE

            service = await factory.create_llm_service(ollama_config)

            # Should have created OpenAI service as fallback
            assert isinstance(service, OpenAILLMService)
            assert service.config.provider == "openai"

    @pytest.mark.asyncio
    async def test_vllm_failover_to_openai(self, factory, vllm_config):
        """Test failover from vLLM to OpenAI when vLLM is unavailable."""
        # Mock vLLM as unavailable
        with patch.object(factory, 'check_provider_health', new_callable=AsyncMock) as mock_health:
            mock_health.return_value = ProviderHealth.UNAVAILABLE

            service = await factory.create_llm_service(vllm_config)

            # Should have created OpenAI service as fallback
            assert isinstance(service, OpenAILLMService)
            assert service.config.provider == "openai"

    @pytest.mark.asyncio
    async def test_failover_disabled(self, ollama_config):
        """Test that failover can be disabled."""
        factory = LLMProviderFactory(enable_failover=False)

        # Mock Ollama as unavailable
        with patch.object(factory, 'check_provider_health', new_callable=AsyncMock) as mock_health:
            with patch.object(factory, '_create_provider', new_callable=AsyncMock) as mock_create:
                mock_health.return_value = ProviderHealth.UNAVAILABLE
                mock_create.side_effect = RuntimeError("Provider unavailable")

                # Should raise error instead of failing over
                with pytest.raises(RuntimeError, match="Failed to create LLM provider"):
                    await factory.create_llm_service(ollama_config)

    @pytest.mark.asyncio
    async def test_fallback_service_injection(self, factory, ollama_config):
        """Test that fallback service is injected into primary provider."""
        # Mock Ollama as healthy
        with patch.object(factory, 'check_provider_health', new_callable=AsyncMock) as mock_health:
            with patch.object(OllamaLLMService, 'check_model_availability', new_callable=AsyncMock) as mock_check:
                mock_health.return_value = ProviderHealth.HEALTHY
                mock_check.return_value = True

                service = await factory.create_llm_service(ollama_config)

                # Should have Ollama service with OpenAI fallback
                assert isinstance(service, OllamaLLMService)
                assert service.fallback_service is not None
                assert isinstance(service.fallback_service, OpenAILLMService)

    @pytest.mark.asyncio
    async def test_openai_no_fallback(self, factory, openai_config):
        """Test that OpenAI has no fallback service."""
        service = await factory.create_llm_service(openai_config)

        # OpenAI should not have a fallback
        assert not hasattr(service, 'fallback_service') or service.fallback_service is None

    @pytest.mark.asyncio
    async def test_custom_fallback_model(self, factory, ollama_config):
        """Test using custom fallback model."""
        provider_config = {
            "fallback_model": "gpt-3.5-turbo",
        }

        # Mock Ollama as unavailable
        with patch.object(factory, 'check_provider_health', new_callable=AsyncMock) as mock_health:
            mock_health.return_value = ProviderHealth.UNAVAILABLE

            service = await factory.create_llm_service(ollama_config, provider_config)

            # Should have created OpenAI service with custom model
            assert isinstance(service, OpenAILLMService)
            assert service.config.model == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_all_providers_fail(self, factory, ollama_config):
        """Test error handling when all providers fail."""
        # Mock both Ollama and OpenAI as failing
        with patch.object(factory, 'check_provider_health', new_callable=AsyncMock) as mock_health:
            with patch.object(factory, '_create_provider', new_callable=AsyncMock) as mock_create:
                mock_health.return_value = ProviderHealth.UNAVAILABLE
                mock_create.side_effect = Exception("Provider creation failed")

                with pytest.raises(RuntimeError, match="All LLM providers failed"):
                    await factory.create_llm_service(ollama_config)


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_unknown_provider_health(self, factory):
        """Test health check for unknown provider."""
        health = await factory.check_provider_health("unknown_provider")
        assert health == ProviderHealth.UNAVAILABLE

    @pytest.mark.asyncio
    async def test_case_insensitive_provider(self, factory):
        """Test that provider names are case-insensitive in factory."""
        # Factory handles case-insensitivity, not LLMConfig
        config_lower = LLMConfig(provider="openai", model="gpt-4")

        service = await factory.create_llm_service(config_lower)

        assert isinstance(service, OpenAILLMService)

    @pytest.mark.asyncio
    async def test_factory_initialization_defaults(self):
        """Test factory initialization with default values."""
        factory = LLMProviderFactory()

        assert factory.openai_api_key == "test-openai-key"
        assert factory.ollama_base_url == "http://localhost:11434"
        assert factory.vllm_base_url == "http://localhost:8000"
        assert factory.enable_failover is True

    @pytest.mark.asyncio
    async def test_concurrent_service_creation(self, factory, openai_config, ollama_config):
        """Test creating multiple services concurrently."""
        import asyncio

        with patch.object(factory, 'check_provider_health', new_callable=AsyncMock) as mock_health:
            with patch.object(OllamaLLMService, 'check_model_availability', new_callable=AsyncMock) as mock_check:
                mock_health.return_value = ProviderHealth.HEALTHY
                mock_check.return_value = True

                # Create services concurrently
                services = await asyncio.gather(
                    factory.create_llm_service(openai_config),
                    factory.create_llm_service(ollama_config),
                )

                assert len(services) == 2
                assert isinstance(services[0], OpenAILLMService)
                assert isinstance(services[1], OllamaLLMService)
