"""Tests for application settings."""

from voice_agent.config.settings import ENV_FILE, Settings


def test_env_file_is_anchored_to_repository() -> None:
    """The app must load its repository .env even when started outside the repo cwd."""
    assert Settings.model_config["env_file"] == ENV_FILE
    assert ENV_FILE.name == ".env"


def test_settings_from_env() -> None:
    """Test that settings load from environment variables."""
    settings = Settings()

    # These should be set by the env_setup fixture in conftest.py
    assert settings.openai_api_key == "test-openai-key"
    assert settings.gemini_api_key == "test-gemini-key"
    assert settings.deepgram_api_key == "test-deepgram-key"
    assert settings.elevenlabs_api_key == "test-elevenlabs-key"
    assert settings.twilio_account_sid == "test-twilio-sid"
    assert settings.twilio_auth_token == "test-twilio-token"
    assert settings.log_level == "DEBUG"
    assert settings.log_format == "console"


def test_settings_defaults() -> None:
    """Test default settings values."""
    settings = Settings()

    assert settings.llm_provider == "gemini"
    assert settings.tts_provider == "gemini"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.vllm_base_url == "http://localhost:8000"
    assert settings.langsmith_project == "voice-agent-platform"
    assert settings.enable_tls is True
