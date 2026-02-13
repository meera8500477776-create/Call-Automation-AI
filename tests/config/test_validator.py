"""Unit tests for configuration validator."""

from pathlib import Path

import pytest

from voice_agent.config.schema import LLMProvider, TenantConfig
from voice_agent.config.validator import ConfigValidator, ValidationResult


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_valid_result(self):
        """Test creating a valid validation result."""
        config = TenantConfig(
            tenant_id="test",
            persona={
                "name": "Sarah",
                "voice_id": "voice_123",
                "system_prompt": "Prompt",
            },
            llm={"provider": "openai", "model": "gpt-4"},
        )
        result = ValidationResult(is_valid=True, config=config)
        
        assert result.is_valid
        assert result.config == config
        assert len(result.errors) == 0
        assert bool(result) is True

    def test_invalid_result(self):
        """Test creating an invalid validation result."""
        errors = ["Error 1", "Error 2"]
        result = ValidationResult(is_valid=False, errors=errors)
        
        assert not result.is_valid
        assert result.config is None
        assert result.errors == errors
        assert bool(result) is False

    def test_repr_valid(self):
        """Test string representation of valid result."""
        config = TenantConfig(
            tenant_id="test_tenant",
            persona={
                "name": "Sarah",
                "voice_id": "voice_123",
                "system_prompt": "Prompt",
            },
            llm={"provider": "openai", "model": "gpt-4"},
        )
        result = ValidationResult(is_valid=True, config=config)
        
        repr_str = repr(result)
        assert "valid=True" in repr_str
        assert "test_tenant" in repr_str

    def test_repr_invalid(self):
        """Test string representation of invalid result."""
        result = ValidationResult(is_valid=False, errors=["Error 1", "Error 2"])
        
        repr_str = repr(result)
        assert "valid=False" in repr_str
        assert "errors=2" in repr_str


class TestConfigValidator:
    """Tests for ConfigValidator class."""

    def test_validate_dict_valid_config(self):
        """Test validating a valid configuration dictionary."""
        config_dict = {
            "tenant_id": "test_tenant_001",
            "persona": {
                "name": "Sarah",
                "voice_id": "elevenlabs_voice_123",
                "system_prompt": "You are a helpful assistant.",
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 150,
            },
        }
        
        result = ConfigValidator.validate_dict(config_dict)
        
        assert result.is_valid
        assert result.config is not None
        assert result.config.tenant_id == "test_tenant_001"
        assert result.config.persona.name == "Sarah"
        assert result.config.llm.provider == LLMProvider.OPENAI
        assert len(result.errors) == 0

    def test_validate_dict_missing_required_fields(self):
        """Test validating a configuration with missing required fields."""
        config_dict = {
            "tenant_id": "test_tenant_001",
            "persona": {
                "name": "Sarah",
                # Missing voice_id and system_prompt
            },
        }
        
        result = ConfigValidator.validate_dict(config_dict)
        
        assert not result.is_valid
        assert result.config is None
        assert len(result.errors) > 0
        assert any("voice_id" in error for error in result.errors)
        assert any("system_prompt" in error for error in result.errors)

    def test_validate_dict_invalid_field_types(self):
        """Test validating a configuration with invalid field types."""
        config_dict = {
            "tenant_id": "test_tenant_001",
            "persona": {
                "name": "Sarah",
                "voice_id": "voice_123",
                "system_prompt": "Prompt",
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": "invalid",  # Should be float
            },
        }
        
        result = ConfigValidator.validate_dict(config_dict)
        
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_validate_dict_invalid_enum_value(self):
        """Test validating a configuration with invalid enum value."""
        config_dict = {
            "tenant_id": "test_tenant_001",
            "persona": {
                "name": "Sarah",
                "voice_id": "voice_123",
                "system_prompt": "Prompt",
            },
            "llm": {
                "provider": "invalid_provider",  # Invalid enum value
                "model": "gpt-4",
            },
        }
        
        result = ConfigValidator.validate_dict(config_dict)
        
        assert not result.is_valid
        assert any("provider" in error for error in result.errors)

    def test_validate_dict_with_tools(self):
        """Test validating a configuration with tools."""
        config_dict = {
            "tenant_id": "test_tenant_001",
            "persona": {
                "name": "Sarah",
                "voice_id": "voice_123",
                "system_prompt": "Prompt",
            },
            "llm": {"provider": "openai", "model": "gpt-4"},
            "tools": [
                {
                    "name": "search_appointments",
                    "type": "fhir",
                    "description": "Search appointments",
                    "endpoint": "https://api.example.com",
                    "auth_type": "oauth2",
                    "scopes": ["patient/Appointment.read"],
                    "parameters_schema": {"type": "object"},
                }
            ],
        }
        
        result = ConfigValidator.validate_dict(config_dict)
        
        assert result.is_valid
        assert len(result.config.tools) == 1
        assert result.config.tools[0].name == "search_appointments"

    def test_validate_dict_duplicate_tool_names(self):
        """Test validating a configuration with duplicate tool names."""
        config_dict = {
            "tenant_id": "test_tenant_001",
            "persona": {
                "name": "Sarah",
                "voice_id": "voice_123",
                "system_prompt": "Prompt",
            },
            "llm": {"provider": "openai", "model": "gpt-4"},
            "tools": [
                {
                    "name": "duplicate_tool",
                    "type": "fhir",
                    "description": "Tool 1",
                    "endpoint": "https://api.example.com",
                    "auth_type": "oauth2",
                    "scopes": ["read"],
                    "parameters_schema": {"type": "object"},
                },
                {
                    "name": "duplicate_tool",
                    "type": "custom_api",
                    "description": "Tool 2",
                    "endpoint": "https://api.example.com",
                    "auth_type": "api_key",
                    "api_key_name": "X-API-Key",
                    "parameters_schema": {"type": "object"},
                },
            ],
        }
        
        result = ConfigValidator.validate_dict(config_dict)
        
        assert not result.is_valid
        assert any("unique" in error.lower() for error in result.errors)

    def test_validate_file_valid_yaml(self, tmp_path):
        """Test validating a valid YAML configuration file."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
tenant_id: test_tenant_001
persona:
  name: Sarah
  voice_id: voice_123
  system_prompt: You are helpful.
llm:
  provider: openai
  model: gpt-4
"""
        )
        
        result = ConfigValidator.validate_file(config_file)
        
        assert result.is_valid
        assert result.config.tenant_id == "test_tenant_001"

    def test_validate_file_invalid_yaml(self, tmp_path):
        """Test validating an invalid YAML configuration file."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
tenant_id: test_tenant_001
persona:
  name: Sarah
  # Missing required fields
llm:
  provider: openai
  model: gpt-4
"""
        )
        
        result = ConfigValidator.validate_file(config_file)
        
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_validate_file_nonexistent(self):
        """Test validating a nonexistent file."""
        result = ConfigValidator.validate_file(Path("/nonexistent/config.yaml"))
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("not found" in error.lower() for error in result.errors)

    def test_load_and_validate_success(self, tmp_path):
        """Test load_and_validate with valid configuration."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
tenant_id: test_tenant_001
persona:
  name: Sarah
  voice_id: voice_123
  system_prompt: You are helpful.
llm:
  provider: openai
  model: gpt-4
"""
        )
        
        config = ConfigValidator.load_and_validate(config_file)
        
        assert isinstance(config, TenantConfig)
        assert config.tenant_id == "test_tenant_001"

    def test_load_and_validate_failure(self, tmp_path):
        """Test load_and_validate with invalid configuration."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
tenant_id: test_tenant_001
persona:
  name: Sarah
  # Missing required fields
"""
        )
        
        with pytest.raises(ValueError) as exc_info:
            ConfigValidator.load_and_validate(config_file)
        
        assert "Invalid configuration" in str(exc_info.value)

    def test_validate_complete_healthcare_config(self, tmp_path):
        """Test validating a complete healthcare configuration."""
        config_file = tmp_path / "healthcare.yaml"
        config_file.write_text(
            """
tenant_id: healthcare_clinic_001
persona:
  name: Sarah
  voice_id: elevenlabs_voice_id
  system_prompt: You are a medical receptionist.
llm:
  provider: openai
  model: gpt-4
  temperature: 0.7
  max_tokens: 150
tts:
  provider: elevenlabs
  voice_settings:
    stability: 0.5
    similarity_boost: 0.75
    style: 0.5
    use_speaker_boost: true
tools:
  - name: search_appointments
    type: fhir
    description: Search for appointments
    endpoint: https://fhir.epic.com/api/FHIR/R4
    auth_type: oauth2
    scopes:
      - patient/Appointment.read
    parameters_schema:
      type: object
      required:
        - patient_id
      properties:
        patient_id:
          type: string
knowledge_bases:
  - type: vector_db
    source: clinic_policies.pdf
    description: Clinic policies
constraints:
  max_call_duration: 600
  response_brevity: 3
  enable_barge_in: true
  transfer_to_human_on_error: true
"""
        )
        
        result = ConfigValidator.validate_file(config_file)
        
        assert result.is_valid
        assert result.config.tenant_id == "healthcare_clinic_001"
        assert len(result.config.tools) == 1
        assert len(result.config.knowledge_bases) == 1
        assert result.config.tts is not None
        assert result.config.tts.provider.value == "elevenlabs"
