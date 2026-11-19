"""Application settings and environment configuration."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        # Anchor this to the repository instead of the caller's working directory.
        # This keeps module launches and embedded orchestrator use consistent.
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    openai_api_key: str = Field(default="", description="OpenAI API key")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    deepgram_api_key: str = Field(default="", description="Deepgram API key")
    elevenlabs_api_key: str = Field(default="", description="ElevenLabs API key")
    twilio_account_sid: str = Field(default="", description="Twilio account SID")
    twilio_auth_token: str = Field(default="", description="Twilio auth token")
    twilio_public_url: str = Field(default="", description="Public HTTPS URL for Twilio webhooks")
    voice_agent_tenant_id: str = Field(default="", description="Tenant used for inbound/local calls")
    twilio_validate_signature: bool = Field(default=True, description="Validate Twilio webhook signatures")
    server_host: str = Field(default="0.0.0.0", description="HTTP server bind address")
    server_port: int = Field(default=8000, ge=1, le=65535, description="HTTP server port")

    # LLM Configuration
    llm_provider: Literal["gemini", "openai", "ollama", "vllm"] = Field(
        default="gemini", description="LLM provider"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama base URL"
    )
    vllm_base_url: str = Field(default="http://localhost:8000", description="vLLM base URL")

    # TTS Configuration
    tts_provider: Literal["gemini", "elevenlabs", "openai"] = Field(
        default="gemini", description="TTS provider"
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
