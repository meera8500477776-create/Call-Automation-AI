"""Unit tests for configuration schema models."""

import pytest
from pydantic import ValidationError

from voice_agent.config.schema import (
    AuthType,
    ConstraintsConfig,
    KnowledgeBaseConfig,
    LLMConfig,
    LLMProvider,
    PersonaConfig,
    TenantConfig,
    ToolDefinition,
    ToolType,
    VoiceSettings,
)


class TestPersonaConfig:
    """Tests for PersonaConfig model."""

    def test_valid_persona_config(self):
        """Test creating a valid persona configuration."""
        persona = PersonaConfig(
            name="Sarah",
            voice_id="elevenlabs_voice_123",
            system_prompt="You are a helpful assistant.",
        )
        assert persona.name == "Sarah"
        assert persona.voice_id == "elevenlabs_voice_123"
        assert persona.system_prompt == "You are a helpful assistant."

    def test_missing_required_fields(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            PersonaConfig(name="Sarah")

        errors = exc_info.value.errors()
        error_fields = {e["loc"][0] for e in errors}
        assert "voice_id" in error_fields
        assert "system_prompt" in error_fields

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError) as exc_info:
            PersonaConfig(
                name="Sarah",
                voice_id="voice_123",
                system_prompt="Prompt",
                extra_field="not_allowed",
            )

        errors = exc_info.value.errors()
        assert any("extra_field" in str(e["loc"]) for e in errors)


class TestVoiceSettings:
    """Tests for VoiceSettings model."""

    def test_valid_voice_settings(self):
        """Test creating valid voice settings."""
        settings = VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.6,
            use_speaker_boost=True,
        )
        assert settings.stability == 0.5
        assert settings.similarity_boost == 0.75
        assert settings.style == 0.6
        assert settings.use_speaker_boost is True

    def test_default_values(self):
        """Test that default values are applied."""
        settings = VoiceSettings()
        assert settings.stability == 0.5
        assert settings.similarity_boost == 0.75
        assert settings.style == 0.5
        assert settings.use_speaker_boost is True

    def test_value_range_validation(self):
        """Test that values are constrained to 0.0-1.0 range."""
        with pytest.raises(ValidationError):
            VoiceSettings(stability=1.5)

        with pytest.raises(ValidationError):
            VoiceSettings(similarity_boost=-0.1)


class TestLLMConfig:
    """Tests for LLMConfig model."""

    def test_valid_llm_config(self):
        """Test creating a valid LLM configuration."""
        llm = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            temperature=0.7,
            max_tokens=150,
        )
        assert llm.provider == LLMProvider.OPENAI
        assert llm.model == "gpt-4"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 150

    def test_default_values(self):
        """Test that default values are applied."""
        llm = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        assert llm.temperature == 0.7
        assert llm.max_tokens == 150

    def test_temperature_range_validation(self):
        """Test that temperature is constrained to valid range."""
        with pytest.raises(ValidationError):
            LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4", temperature=2.5)

        with pytest.raises(ValidationError):
            LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4", temperature=-0.1)

    def test_max_tokens_validation(self):
        """Test that max_tokens must be positive."""
        with pytest.raises(ValidationError):
            LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4", max_tokens=0)

        with pytest.raises(ValidationError):
            LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4", max_tokens=-10)


class TestToolDefinition:
    """Tests for ToolDefinition model."""

    def test_valid_oauth2_tool(self):
        """Test creating a valid OAuth2 tool definition."""
        tool = ToolDefinition(
            name="search_appointments",
            type=ToolType.FHIR,
            description="Search for appointments",
            endpoint="https://api.example.com/fhir",
            auth_type=AuthType.OAUTH2,
            scopes=["patient/Appointment.read"],
            parameters_schema={"type": "object", "properties": {}},
        )
        assert tool.name == "search_appointments"
        assert tool.type == ToolType.FHIR
        assert tool.auth_type == AuthType.OAUTH2
        assert tool.scopes == ["patient/Appointment.read"]

    def test_oauth2_requires_scopes(self):
        """Test that OAuth2 authentication requires scopes."""
        with pytest.raises(ValidationError) as exc_info:
            ToolDefinition(
                name="test_tool",
                type=ToolType.FHIR,
                description="Test",
                endpoint="https://api.example.com",
                auth_type=AuthType.OAUTH2,
                parameters_schema={"type": "object"},
            )

        errors = exc_info.value.errors()
        assert any("scopes" in str(e) for e in errors)

    def test_valid_api_key_tool(self):
        """Test creating a valid API key tool definition."""
        tool = ToolDefinition(
            name="custom_api",
            type=ToolType.CUSTOM_API,
            description="Custom API",
            endpoint="https://api.example.com",
            auth_type=AuthType.API_KEY,
            api_key_name="X-API-Key",
            parameters_schema={"type": "object"},
        )
        assert tool.auth_type == AuthType.API_KEY
        assert tool.api_key_name == "X-API-Key"

    def test_api_key_requires_key_name(self):
        """Test that API key authentication requires api_key_name."""
        with pytest.raises(ValidationError) as exc_info:
            ToolDefinition(
                name="test_tool",
                type=ToolType.CUSTOM_API,
                description="Test",
                endpoint="https://api.example.com",
                auth_type=AuthType.API_KEY,
                parameters_schema={"type": "object"},
            )

        errors = exc_info.value.errors()
        assert any("api_key_name" in str(e) for e in errors)


class TestConstraintsConfig:
    """Tests for ConstraintsConfig model."""

    def test_valid_constraints(self):
        """Test creating valid constraints configuration."""
        constraints = ConstraintsConfig(
            max_call_duration=600,
            response_brevity=3,
            enable_barge_in=True,
            transfer_to_human_on_error=True,
        )
        assert constraints.max_call_duration == 600
        assert constraints.response_brevity == 3
        assert constraints.enable_barge_in is True
        assert constraints.transfer_to_human_on_error is True

    def test_default_values(self):
        """Test that default values are applied."""
        constraints = ConstraintsConfig()
        assert constraints.max_call_duration == 600
        assert constraints.response_brevity == 3
        assert constraints.enable_barge_in is True
        assert constraints.transfer_to_human_on_error is True

    def test_max_call_duration_validation(self):
        """Test that max_call_duration must be positive."""
        with pytest.raises(ValidationError):
            ConstraintsConfig(max_call_duration=0)

        with pytest.raises(ValidationError):
            ConstraintsConfig(max_call_duration=-100)

    def test_response_brevity_validation(self):
        """Test that response_brevity is constrained to valid range."""
        with pytest.raises(ValidationError):
            ConstraintsConfig(response_brevity=0)

        with pytest.raises(ValidationError):
            ConstraintsConfig(response_brevity=11)


class TestTenantConfig:
    """Tests for TenantConfig model."""

    def test_valid_tenant_config(self):
        """Test creating a valid tenant configuration."""
        config = TenantConfig(
            tenant_id="test_tenant_001",
            persona=PersonaConfig(
                name="Sarah",
                voice_id="voice_123",
                system_prompt="You are helpful.",
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4",
            ),
        )
        assert config.tenant_id == "test_tenant_001"
        assert config.persona.name == "Sarah"
        assert config.llm.provider == LLMProvider.OPENAI
        assert len(config.tools) == 0
        assert len(config.knowledge_bases) == 0

    def test_tenant_config_with_tools(self):
        """Test tenant configuration with tools."""
        config = TenantConfig(
            tenant_id="test_tenant_001",
            persona=PersonaConfig(
                name="Sarah",
                voice_id="voice_123",
                system_prompt="You are helpful.",
            ),
            llm=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4"),
            tools=[
                ToolDefinition(
                    name="tool1",
                    type=ToolType.FHIR,
                    description="Tool 1",
                    endpoint="https://api.example.com",
                    auth_type=AuthType.OAUTH2,
                    scopes=["read"],
                    parameters_schema={"type": "object"},
                ),
                ToolDefinition(
                    name="tool2",
                    type=ToolType.CUSTOM_API,
                    description="Tool 2",
                    endpoint="https://api.example.com",
                    auth_type=AuthType.API_KEY,
                    api_key_name="X-API-Key",
                    parameters_schema={"type": "object"},
                ),
            ],
        )
        assert len(config.tools) == 2
        assert config.tools[0].name == "tool1"
        assert config.tools[1].name == "tool2"

    def test_duplicate_tool_names_rejected(self):
        """Test that duplicate tool names are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TenantConfig(
                tenant_id="test_tenant_001",
                persona=PersonaConfig(
                    name="Sarah",
                    voice_id="voice_123",
                    system_prompt="You are helpful.",
                ),
                llm=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4"),
                tools=[
                    ToolDefinition(
                        name="duplicate_tool",
                        type=ToolType.FHIR,
                        description="Tool 1",
                        endpoint="https://api.example.com",
                        auth_type=AuthType.OAUTH2,
                        scopes=["read"],
                        parameters_schema={"type": "object"},
                    ),
                    ToolDefinition(
                        name="duplicate_tool",
                        type=ToolType.CUSTOM_API,
                        description="Tool 2",
                        endpoint="https://api.example.com",
                        auth_type=AuthType.API_KEY,
                        api_key_name="X-API-Key",
                        parameters_schema={"type": "object"},
                    ),
                ],
            )

        errors = exc_info.value.errors()
        assert any("unique" in str(e).lower() for e in errors)

    def test_tenant_config_with_knowledge_bases(self):
        """Test tenant configuration with knowledge bases."""
        config = TenantConfig(
            tenant_id="test_tenant_001",
            persona=PersonaConfig(
                name="Sarah",
                voice_id="voice_123",
                system_prompt="You are helpful.",
            ),
            llm=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4"),
            knowledge_bases=[
                KnowledgeBaseConfig(
                    type="vector_db",
                    source="policies.pdf",
                    description="Company policies",
                ),
            ],
        )
        assert len(config.knowledge_bases) == 1
        assert config.knowledge_bases[0].type == "vector_db"

    def test_tenant_config_with_constraints(self):
        """Test tenant configuration with custom constraints."""
        config = TenantConfig(
            tenant_id="test_tenant_001",
            persona=PersonaConfig(
                name="Sarah",
                voice_id="voice_123",
                system_prompt="You are helpful.",
            ),
            llm=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4"),
            constraints=ConstraintsConfig(
                max_call_duration=300,
                response_brevity=2,
            ),
        )
        assert config.constraints.max_call_duration == 300
        assert config.constraints.response_brevity == 2
