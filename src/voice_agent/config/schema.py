"""Configuration schema data models using Pydantic."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    GEMINI = "gemini"
    OPENAI = "openai"
    OLLAMA = "ollama"
    VLLM = "vllm"


class TTSProvider(str, Enum):
    """Supported TTS providers."""

    GEMINI = "gemini"
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"


class AuthType(str, Enum):
    """Supported authentication types."""

    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    BEARER = "bearer"


class ToolType(str, Enum):
    """Supported tool types."""

    FHIR = "fhir"
    DMS = "dms"
    CUSTOM_API = "custom_api"


class PersonaConfig(BaseModel):
    """Persona configuration for the voice agent."""

    name: str = Field(..., description="Name of the voice agent persona")
    voice_id: str = Field(..., description="Voice ID for TTS provider")
    system_prompt: str = Field(
        ..., description="System prompt defining agent behavior and personality"
    )

    model_config = {"extra": "forbid"}


class VoiceSettings(BaseModel):
    """Voice synthesis settings for TTS."""

    stability: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Voice stability (0.0-1.0)"
    )
    similarity_boost: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Similarity boost (0.0-1.0)"
    )
    style: float = Field(default=0.5, ge=0.0, le=1.0, description="Style intensity (0.0-1.0)")
    use_speaker_boost: bool = Field(
        default=True, description="Enable speaker boost for clarity"
    )

    model_config = {"extra": "forbid"}


class TTSConfig(BaseModel):
    """Text-to-Speech configuration."""

    provider: TTSProvider = Field(..., description="TTS provider")
    model: str | None = Field(default=None, description="Provider-specific TTS model")
    voice_name: str | None = Field(default=None, description="Provider-specific voice name")
    style_prompt: str | None = Field(default=None, description="Natural-language delivery guidance")
    voice_settings: VoiceSettings | None = Field(
        default=None, description="Voice synthesis settings (ElevenLabs only)"
    )

    model_config = {"extra": "forbid"}


class LLMConfig(BaseModel):
    """LLM configuration."""

    provider: LLMProvider = Field(..., description="LLM provider")
    model: str = Field(..., description="Model name/identifier")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature (0.0-2.0)"
    )
    max_tokens: int = Field(
        default=150, gt=0, le=4096, description="Maximum tokens in response"
    )
    timeout: float = Field(
        default=30.0, gt=0, le=300, description="Provider request timeout in seconds"
    )

    model_config = {"extra": "forbid"}


class ToolDefinition(BaseModel):
    """Tool definition for external API integration."""

    name: str = Field(..., description="Unique tool identifier")
    type: ToolType = Field(..., description="Tool type")
    description: str = Field(..., description="Human-readable tool description")
    endpoint: str = Field(..., description="API endpoint URL")
    auth_type: AuthType = Field(..., description="Authentication type")
    scopes: list[str] | None = Field(
        default=None, description="OAuth2 scopes (if auth_type is oauth2)"
    )
    api_key_name: str | None = Field(
        default=None, description="API key parameter name (if auth_type is api_key)"
    )
    parameters_schema: dict[str, Any] = Field(
        ..., description="JSON schema for tool parameters"
    )

    @model_validator(mode="after")
    def validate_auth_requirements(self) -> "ToolDefinition":
        """Validate authentication-specific requirements."""
        if self.auth_type == AuthType.OAUTH2 and not self.scopes:
            raise ValueError("scopes are required when auth_type is oauth2")
        if self.auth_type == AuthType.API_KEY and not self.api_key_name:
            raise ValueError("api_key_name is required when auth_type is api_key")
        return self

    model_config = {"extra": "forbid"}


class KnowledgeBaseConfig(BaseModel):
    """Knowledge base configuration."""

    type: str = Field(..., description="Knowledge base type (e.g., vector_db, document)")
    source: str = Field(..., description="Source path or identifier")
    description: str | None = Field(default=None, description="Knowledge base description")

    model_config = {"extra": "forbid"}


class ConstraintsConfig(BaseModel):
    """Operational constraints for the voice agent."""

    max_call_duration: int = Field(
        default=600, gt=0, description="Maximum call duration in seconds"
    )
    response_brevity: int = Field(
        default=3, gt=0, le=10, description="Maximum sentences per response"
    )
    enable_barge_in: bool = Field(
        default=True, description="Allow user to interrupt agent"
    )
    transfer_to_human_on_error: bool = Field(
        default=True, description="Transfer to human agent on errors"
    )

    model_config = {"extra": "forbid"}


class TenantConfig(BaseModel):
    """Complete tenant configuration."""

    tenant_id: str = Field(..., description="Unique tenant identifier")
    persona: PersonaConfig = Field(..., description="Persona configuration")
    llm: LLMConfig = Field(..., description="LLM configuration")
    tts: TTSConfig | None = Field(default=None, description="TTS configuration")
    tools: list[ToolDefinition] = Field(
        default_factory=list, description="Available tools for the agent"
    )
    knowledge_bases: list[KnowledgeBaseConfig] = Field(
        default_factory=list, description="Knowledge bases for the agent"
    )
    constraints: ConstraintsConfig = Field(
        default_factory=ConstraintsConfig, description="Operational constraints"
    )

    @field_validator("tools")
    @classmethod
    def validate_unique_tool_names(cls, v: list[ToolDefinition]) -> list[ToolDefinition]:
        """Validate that tool names are unique."""
        tool_names = [tool.name for tool in v]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Tool names must be unique within a tenant configuration")
        return v

    model_config = {"extra": "forbid"}
