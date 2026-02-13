"""Tests for application settings."""

from voice_agent.config.settings import Settings


def test_settings_from_env() -> None:
    """Test that settings load from environment variables."""
    settings = Settings()

    # These should be set by the env_setup fixture in conftest.py
    assert settings.openai_api_key == "test-openai-key"
    assert settings.deepgram_api_key == "test-deepgram-key"
    assert settings.elevenlabs_api_key == "test-elevenlabs-key"
    assert settings.twilio_account_sid == "test-twilio-sid"
    assert settings.twilio_auth_token == "test-twilio-token"
    assert settings.log_level == "DEBUG"
    assert settings.log_format == "console"


def test_settings_defaults() -> None:
    """Test default settings values."""
    settings = Settings()

    assert settings.llm_provider == "openai"
    assert settings.tts_provider == "elevenlabs"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.vllm_base_url == "http://localhost:8000"
    assert settings.langsmith_project == "voice-agent-platform"
    assert settings.enable_tls is True
