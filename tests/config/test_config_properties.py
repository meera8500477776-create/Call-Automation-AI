"""Property-based tests for configuration system.

**Validates: Requirements 1.1, 1.2**

This module contains property-based tests using Hypothesis to verify universal
properties of the configuration system across all valid inputs.
"""

import json
from typing import Any, Dict

import pytest
import yaml
from hypothesis import HealthCheck, given, settings, strategies as st

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
    TTSConfig,
    TTSProvider,
    VoiceSettings,
)


# Hypothesis strategies for generating valid configuration data

@st.composite
def persona_config_strategy(draw):
    """Generate valid PersonaConfig instances."""
    return PersonaConfig(
        name=draw(st.text(min_size=1, max_size=20, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ")),
        voice_id=draw(st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-")),
        system_prompt=draw(st.text(min_size=10, max_size=100)),
    )


@st.composite
def voice_settings_strategy(draw):
    """Generate valid VoiceSettings instances."""
    return VoiceSettings(
        stability=draw(st.floats(min_value=0.0, max_value=1.0)),
        similarity_boost=draw(st.floats(min_value=0.0, max_value=1.0)),
        style=draw(st.floats(min_value=0.0, max_value=1.0)),
        use_speaker_boost=draw(st.booleans()),
    )


@st.composite
def tts_config_strategy(draw):
    """Generate valid TTSConfig instances."""
    provider = draw(st.sampled_from(list(TTSProvider)))
    voice_settings = draw(st.one_of(st.none(), voice_settings_strategy()))
    return TTSConfig(provider=provider, voice_settings=voice_settings)


@st.composite
def llm_config_strategy(draw):
    """Generate valid LLMConfig instances."""
    return LLMConfig(
        provider=draw(st.sampled_from(list(LLMProvider))),
        model=draw(st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_.")),
        temperature=draw(st.floats(min_value=0.0, max_value=2.0)),
        max_tokens=draw(st.integers(min_value=1, max_value=4096)),
    )


@st.composite
def tool_definition_strategy(draw):
    """Generate valid ToolDefinition instances."""
    name = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-"
    )))
    tool_type = draw(st.sampled_from(list(ToolType)))
    description = draw(st.text(min_size=1, max_size=50))
    # Use a simpler endpoint generation strategy
    domain = draw(st.text(min_size=3, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"))
    endpoint = f"https://{domain}.example.com/api"
    auth_type = draw(st.sampled_from(list(AuthType)))
    
    # Generate auth-specific fields based on auth_type
    scopes = None
    api_key_name = None
    if auth_type == AuthType.OAUTH2:
        scopes = draw(st.lists(st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz"), min_size=1, max_size=3))
    elif auth_type == AuthType.API_KEY:
        api_key_name = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_"
        )))
    
    parameters_schema = draw(st.fixed_dictionaries({
        "type": st.just("object"),
        "properties": st.dictionaries(
            st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
            st.fixed_dictionaries({"type": st.sampled_from(["string", "number", "boolean"])}),
            max_size=3
        )
    }))
    
    return ToolDefinition(
        name=name,
        type=tool_type,
        description=description,
        endpoint=endpoint,
        auth_type=auth_type,
        scopes=scopes,
        api_key_name=api_key_name,
        parameters_schema=parameters_schema,
    )


@st.composite
def knowledge_base_config_strategy(draw):
    """Generate valid KnowledgeBaseConfig instances."""
    return KnowledgeBaseConfig(
        type=draw(st.sampled_from(["vector_db", "document", "api"])),
        source=draw(st.text(min_size=1, max_size=50)),
        description=draw(st.one_of(st.none(), st.text(min_size=1, max_size=50))),
    )


@st.composite
def constraints_config_strategy(draw):
    """Generate valid ConstraintsConfig instances."""
    return ConstraintsConfig(
        max_call_duration=draw(st.integers(min_value=1, max_value=3600)),
        response_brevity=draw(st.integers(min_value=1, max_value=10)),
        enable_barge_in=draw(st.booleans()),
        transfer_to_human_on_error=draw(st.booleans()),
    )


@st.composite
def tenant_config_strategy(draw):
    """Generate valid TenantConfig instances."""
    # Generate unique tool names
    num_tools = draw(st.integers(min_value=0, max_value=3))
    tools = []
    tool_names = set()
    
    for _ in range(num_tools):
        tool = draw(tool_definition_strategy())
        # Ensure unique tool names
        while tool.name in tool_names:
            tool = draw(tool_definition_strategy())
        tool_names.add(tool.name)
        tools.append(tool)
    
    return TenantConfig(
        tenant_id=draw(st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-")),
        persona=draw(persona_config_strategy()),
        llm=draw(llm_config_strategy()),
        tts=draw(st.one_of(st.none(), tts_config_strategy())),
        tools=tools,
        knowledge_bases=draw(st.lists(knowledge_base_config_strategy(), max_size=2)),
        constraints=draw(constraints_config_strategy()),
    )


@pytest.mark.property
class TestConfigurationRoundTrip:
    """Property 1: Configuration round-trip.
    
    **Validates: Requirements 1.1, 1.2**
    
    For any valid TenantConfig, serializing then deserializing should produce
    an equivalent configuration.
    """

    @settings(
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=50,
        deadline=None,
    )
    @given(tenant_config_strategy())
    def test_json_round_trip(self, config: TenantConfig):
        """Test that TenantConfig survives JSON serialization round-trip.
        
        **Validates: Requirements 1.1, 1.2**
        """
        # Serialize to JSON
        json_str = config.model_dump_json()
        
        # Deserialize from JSON
        config_dict = json.loads(json_str)
        restored_config = TenantConfig.model_validate(config_dict)
        
        # Verify equivalence
        assert restored_config == config
        assert restored_config.model_dump() == config.model_dump()

    @settings(
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=50,
        deadline=None,
    )
    @given(tenant_config_strategy())
    def test_dict_round_trip(self, config: TenantConfig):
        """Test that TenantConfig survives dict conversion round-trip.
        
        **Validates: Requirements 1.1, 1.2**
        """
        # Convert to dict
        config_dict = config.model_dump()
        
        # Restore from dict
        restored_config = TenantConfig.model_validate(config_dict)
        
        # Verify equivalence
        assert restored_config == config
        assert restored_config.model_dump() == config.model_dump()

    @settings(
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=50,
        deadline=None,
    )
    @given(tenant_config_strategy())
    def test_yaml_round_trip(self, config: TenantConfig):
        """Test that TenantConfig survives YAML serialization round-trip.
        
        **Validates: Requirements 1.1, 1.2**
        """
        # Serialize to YAML
        config_dict = config.model_dump(mode="json")
        yaml_str = yaml.dump(config_dict, default_flow_style=False)
        
        # Deserialize from YAML
        restored_dict = yaml.safe_load(yaml_str)
        restored_config = TenantConfig.model_validate(restored_dict)
        
        # Verify equivalence
        assert restored_config == config
        assert restored_config.model_dump() == config.model_dump()

    @settings(
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=50,
        deadline=None,
    )
    @given(tenant_config_strategy())
    def test_nested_model_preservation(self, config: TenantConfig):
        """Test that nested models are preserved through serialization.
        
        **Validates: Requirements 1.1, 1.2**
        """
        # Serialize and deserialize
        config_dict = config.model_dump()
        restored_config = TenantConfig.model_validate(config_dict)
        
        # Verify nested models
        assert restored_config.persona == config.persona
        assert restored_config.llm == config.llm
        assert restored_config.tts == config.tts
        assert restored_config.constraints == config.constraints
        
        # Verify lists of nested models
        assert len(restored_config.tools) == len(config.tools)
        for original_tool, restored_tool in zip(config.tools, restored_config.tools):
            assert restored_tool == original_tool
        
        assert len(restored_config.knowledge_bases) == len(config.knowledge_bases)
        for original_kb, restored_kb in zip(config.knowledge_bases, restored_config.knowledge_bases):
            assert restored_kb == original_kb

    @settings(
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=50,
        deadline=None,
    )
    @given(tenant_config_strategy())
    def test_enum_preservation(self, config: TenantConfig):
        """Test that enum values are preserved through serialization.
        
        **Validates: Requirements 1.1, 1.2**
        """
        # Serialize and deserialize
        config_dict = config.model_dump()
        restored_config = TenantConfig.model_validate(config_dict)
        
        # Verify enum values
        assert restored_config.llm.provider == config.llm.provider
        if config.tts:
            assert restored_config.tts.provider == config.tts.provider
        
        for original_tool, restored_tool in zip(config.tools, restored_config.tools):
            assert restored_tool.type == original_tool.type
            assert restored_tool.auth_type == original_tool.auth_type

    @settings(
        suppress_health_check=[HealthCheck.too_slow],
        max_examples=50,
        deadline=None,
    )
    @given(tenant_config_strategy())
    def test_optional_fields_preservation(self, config: TenantConfig):
        """Test that optional fields (None values) are preserved.
        
        **Validates: Requirements 1.1, 1.2**
        """
        # Serialize and deserialize
        config_dict = config.model_dump()
        restored_config = TenantConfig.model_validate(config_dict)
        
        # Verify optional fields
        assert (restored_config.tts is None) == (config.tts is None)
        
        for original_tool, restored_tool in zip(config.tools, restored_config.tools):
            assert (restored_tool.scopes is None) == (original_tool.scopes is None)
            assert (restored_tool.api_key_name is None) == (original_tool.api_key_name is None)
        
        for original_kb, restored_kb in zip(config.knowledge_bases, restored_config.knowledge_bases):
            assert (restored_kb.description is None) == (original_kb.description is None)
