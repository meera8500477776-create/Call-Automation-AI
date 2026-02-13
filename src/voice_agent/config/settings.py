"""Application settings and environment configuration."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    openai_api_key: str = Field(default="", description="OpenAI API key")
    deepgram_api_key: str = Field(default="", description="Deepgram API key")
    elevenlabs_api_key: str = Field(default="", description="ElevenLabs API key")
    twilio_account_sid: str = Field(default="", description="Twilio account SID")
    twilio_auth_token: str = Field(default="", description="Twilio auth token")

    # LLM Configuration
    llm_provider: Literal["openai", "ollama", "vllm"] = Field(
        default="openai", description="LLM provider"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama base URL"
    )
    vllm_base_url: str = Field(default="http://localhost:8000", description="vLLM base URL")

    # TTS Configuration
    tts_provider: Literal["elevenlabs", "openai"] = Field(
        default="elevenlabs", description="TTS provider"
    )

    # Encryption
    encryption_key: str = Field(default="", description="Encryption key for AES-256")

    # Tenant Configuration
    tenant_config_dir: Path = Field(
        default=Path("./config/tenants"), description="Directory containing tenant configs"
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: Literal["json", "console"] = Field(
        default="json", description="Log output format"
    )

    # Observability
    langsmith_api_key: str = Field(default="", description="LangSmith API key")
    langsmith_project: str = Field(
        default="voice-agent-platform", description="LangSmith project name"
    )
    helicone_api_key: str = Field(default="", description="Helicone API key")

    # Database
    database_url: str = Field(
        default="sqlite:///./data/sessions.db", description="Database connection URL"
    )

    # Security
    enable_tls: bool = Field(default=True, description="Enable TLS for connections")
    tls_cert_path: Path = Field(
        default=Path("./certs/cert.pem"), description="TLS certificate path"
    )
    tls_key_path: Path = Field(default=Path("./certs/key.pem"), description="TLS key path")


# Global settings instance
settings = Settings()
