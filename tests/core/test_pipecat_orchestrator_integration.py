"""
Integration tests for Pipecat Orchestrator.

Tests pipeline initialization with various tenant configurations, call routing,
session lifecycle management, and component integration.

**Validates: Requirements 1.3, 1.4, 11.5**
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml

from src.voice_agent.config.registry import TenantRegistry
from src.voice_agent.core.pipecat_orchestrator import (
    CallMetadata,
    PipecatOrchestrator,
)
from src.voice_agent.core.session_manager import SessionManager
from src.voice_agent.core.voice_pipeline import VoicePipeline


@pytest.fixture
def temp_config_dir():
    """Create temporary directory for tenant configurations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "tenants"
        config_dir.mkdir()
        yield config_dir


@pytest.fixture
def healthcare_tenant_config(temp_config_dir):
    """Create healthcare tenant configuration file."""
    config = {
        "tenant_id": "healthcare_clinic_001",
        "persona": {
            "name": "Sarah",
            "voice_id": "healthcare_voice_123",
            "system_prompt": "You are Sarah, a friendly medical receptionist.",
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 150,
        },
        "tts": {
            "provider": "elevenlabs",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
            },
        },
        "tools": [],
        "knowledge_bases": [],
        "constraints": {
            "max_call_duration": 600,
            "response_brevity": 3,
            "enable_barge_in": True,
        },
    }

    config_file = temp_config_dir / "healthcare_clinic_001.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    return config_file


@pytest.fixture
def automotive_tenant_config(temp_config_dir):
    """Create automotive tenant configuration file."""
    config = {
        "tenant_id": "automotive_dealer_001",
        "persona": {
            "name": "Mike",
            "voice_id": "automotive_voice_456",
            "system_prompt": "You are Mike, a helpful automotive service advisor.",
        },
        "llm": {
            "provider": "ollama",
            "model": "llama2",
            "temperature": 0.6,
            "max_tokens": 200,
        },
        "tts": {
            "provider": "openai",
            "voice_settings": {
                "stability": 0.6,
            },
        },
        "tools": [],
        "knowledge_bases": [],
        "constraints": {
            "max_call_duration": 600,
            "response_brevity": 3,
            "enable_barge_in": True,
        },
    }

    config_file = temp_config_dir / "automotive_dealer_001.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    return config_file


@pytest.fixture
def caller_mapping():
    """Create caller ID to tenant mapping."""
    return {
        "+15551234567": "healthcare_clinic_001",
        "+15559876543": "automotive_dealer_001",
        "+15555555555": "healthcare_clinic_001",
    }


@pytest.fixture
def tenant_registry(temp_config_dir, healthcare_tenant_config, automotive_tenant_config, caller_mapping):
    """Create tenant registry with test configurations."""
    registry = TenantRegistry(
        config_dir=temp_config_dir,
        caller_mapping=caller_mapping,
    )
    return registry


@pytest.fixture
def session_manager():
    """Create session manager for testing."""
    return SessionManager()


@pytest.fixture
def orchestrator(tenant_registry, session_manager):
    """Create orchestrator with real dependencies."""
    return PipecatOrchestrator(
        tenant_registry=tenant_registry,
        session_manager=session_manager,
    )


class TestPipelineInitializationWithVariousConfigs:
    """Test pipeline initialization with different tenant configurations."""

    @pytest.mark.asyncio
    async def test_initialize_pipeline_with_healthcare_config(
        self, orchestrator, tenant_registry
    ):
        """
        Test pipeline initialization with healthcare tenant configuration.
        
        Validates that pipeline is correctly initialized with healthcare-specific
        settings including persona, LLM provider, and voice configuration.
        
        **Validates: Requirements 1.3, 1.4**
        """
        # Load healthcare tenant config
        tenant_config = tenant_registry.load_tenant_config("healthcare_clinic_001")

        # Mock LLM service creation
        with patch.object(orchestrator._llm_factory, 'create_llm_service', new_callable=AsyncMock) as mock_llm:
            mock_llm_service = Mock()
            mock_llm.return_value = mock_llm_service

            # Initialize pipeline (async method)
            pipeline = await orchestrator.initialize_pipeline(
                session_id="test_session_healthcare",
                tenant_config=tenant_config,
            )

        # Verify pipeline was created
        assert pipeline is not None
        assert isinstance(pipeline, VoicePipeline)

        # Verify pipeline configuration
        assert pipeline.config.session_id == "test_session_healthcare"
        assert pipeline.config.llm_service == mock_llm_service

        # Verify STT configuration (uses defaults since not in tenant config)
        assert pipeline.config.stt_config.model == "nova-2"
        assert pipeline.config.stt_config.language == "en-US"

        # Verify voice configuration
        assert pipeline.config.voice_config.voice_id == "healthcare_voice_123"
        assert pipeline.config.voice_config.provider == "elevenlabs"

        # Verify VAD configuration (uses defaults)
        assert pipeline.config.vad_config.threshold == 0.5
        assert pipeline.config.vad_config.min_speech_duration_ms == 250

        # Verify barge-in is enabled
        assert pipeline.config.enable_barge_in is True

    @pytest.mark.asyncio
    async def test_initialize_pipeline_with_automotive_config(
        self, orchestrator, tenant_registry
    ):
        """
        Test pipeline initialization with automotive tenant configuration.
        
        Validates that pipeline is correctly initialized with automotive-specific
        settings including different LLM provider (Ollama) and TTS provider (OpenAI).
        
        **Validates: Requirements 1.3, 1.4, 4.2**
        """
        # Load automotive tenant config
        tenant_config = tenant_registry.load_tenant_config("automotive_dealer_001")

        # Mock LLM service creation
        with patch.object(orchestrator._llm_factory, 'create_llm_service', new_callable=AsyncMock) as mock_llm:
            mock_llm_service = Mock()
            mock_llm.return_value = mock_llm_service

            # Initialize pipeline
            pipeline = await orchestrator.initialize_pipeline(
                session_id="test_session_automotive",
                tenant_config=tenant_config,
            )

        # Verify pipeline was created
        assert pipeline is not None
        assert isinstance(pipeline, VoicePipeline)

        # Verify pipeline configuration
        assert pipeline.config.session_id == "test_session_automotive"

        # Verify voice configuration (OpenAI TTS)
        assert pipeline.config.voice_config.voice_id == "automotive_voice_456"
        assert pipeline.config.voice_config.provider == "openai"

        # Verify VAD configuration (uses defaults)
        assert pipeline.config.vad_config.threshold == 0.5
        assert pipeline.config.vad_config.min_speech_duration_ms == 250

    @pytest.mark.asyncio
    async def test_initialize_multiple_pipelines_concurrently(
        self, orchestrator, tenant_registry
    ):
        """
        Test initializing multiple pipelines concurrently for different tenants.
        
        Validates that the orchestrator can handle concurrent pipeline
        initialization for multiple tenants without conflicts.
        
        **Validates: Requirements 1.4**
        """
        healthcare_config = tenant_registry.load_tenant_config("healthcare_clinic_001")
        automotive_config = tenant_registry.load_tenant_config("automotive_dealer_001")

        # Mock LLM service creation
        with patch.object(orchestrator._llm_factory, 'create_llm_service', new_callable=AsyncMock) as mock_llm:
            mock_llm_service = Mock()
            mock_llm.return_value = mock_llm_service

            # Initialize multiple pipelines (async calls)
            pipeline1 = await orchestrator.initialize_pipeline(
                session_id="session_1",
                tenant_config=healthcare_config,
            )
            pipeline2 = await orchestrator.initialize_pipeline(
                session_id="session_2",
                tenant_config=automotive_config,
            )
            pipeline3 = await orchestrator.initialize_pipeline(
                session_id="session_3",
                tenant_config=healthcare_config,
            )

            pipelines = [pipeline1, pipeline2, pipeline3]

        # Verify all pipelines were created
        assert len(pipelines) == 3
        assert all(isinstance(p, VoicePipeline) for p in pipelines)

        # Verify each has unique session ID
        session_ids = [p.config.session_id for p in pipelines]
        assert len(set(session_ids)) == 3

        # Verify correct configurations
        assert pipelines[0].config.voice_config.voice_id == "healthcare_voice_123"
        assert pipelines[1].config.voice_config.voice_id == "automotive_voice_456"
        assert pipelines[2].config.voice_config.voice_id == "healthcare_voice_123"


class TestCallRouting:
    """Test call routing to correct tenant."""

    def test_route_call_to_healthcare_tenant(self, orchestrator):
        """
        Test routing call to healthcare tenant based on caller ID.
        
        Validates that calls from healthcare-mapped numbers are routed
        to the correct healthcare tenant configuration.
        
        **Validates: Requirements 1.3**
        """
        call_metadata = CallMetadata(
            caller_id="+15551234567",
            call_sid="CA_healthcare_001",
            from_number="+15551234567",
            to_number="+15551111111",
        )

        tenant_config = orchestrator.route_call(
            caller_id=call_metadata.caller_id,
            call_metadata=call_metadata,
        )

        # Verify routed to healthcare tenant
        assert tenant_config.tenant_id == "healthcare_clinic_001"
        assert tenant_config.persona.name == "Sarah"
        assert tenant_config.llm.provider == "openai"

    def test_route_call_to_automotive_tenant(self, orchestrator):
        """
        Test routing call to automotive tenant based on caller ID.
        
        Validates that calls from automotive-mapped numbers are routed
        to the correct automotive tenant configuration.
        
        **Validates: Requirements 1.3, 4.2**
        """
        call_metadata = CallMetadata(
            caller_id="+15559876543",
            call_sid="CA_automotive_001",
            from_number="+15559876543",
            to_number="+15552222222",
        )

        tenant_config = orchestrator.route_call(
            caller_id=call_metadata.caller_id,
            call_metadata=call_metadata,
        )

        # Verify routed to automotive tenant
        assert tenant_config.tenant_id == "automotive_dealer_001"
        assert tenant_config.persona.name == "Mike"
        assert tenant_config.llm.provider == "ollama"

    def test_route_multiple_callers_to_same_tenant(self, orchestrator):
        """
        Test routing multiple caller IDs to the same tenant.
        
        Validates that multiple caller IDs can map to the same tenant.
        
        **Validates: Requirements 1.3**
        """
        caller1 = "+15551234567"
        caller2 = "+15555555555"

        call_metadata1 = CallMetadata(
            caller_id=caller1,
            call_sid="CA_001",
            from_number=caller1,
        )
        call_metadata2 = CallMetadata(
            caller_id=caller2,
            call_sid="CA_002",
            from_number=caller2,
        )

        config1 = orchestrator.route_call(caller1, call_metadata1)
        config2 = orchestrator.route_call(caller2, call_metadata2)

        # Both should route to same tenant
        assert config1.tenant_id == config2.tenant_id == "healthcare_clinic_001"

    def test_route_call_with_unknown_caller(self, orchestrator):
        """
        Test routing call with unknown caller ID raises error.
        
        Validates that calls from unmapped numbers are rejected.
        
        **Validates: Requirements 1.3**
        """
        call_metadata = CallMetadata(
            caller_id="+15559999999",  # Not in mapping
            call_sid="CA_unknown",
            from_number="+15559999999",
        )

        with pytest.raises(KeyError):
            orchestrator.route_call(
                caller_id=call_metadata.caller_id,
                call_metadata=call_metadata,
            )


class TestSessionLifecycleManagement:
    """Test session lifecycle management."""

    @pytest.mark.asyncio
    async def test_complete_session_lifecycle(self, orchestrator):
        """
        Test complete session lifecycle from start to stop.
        
        Validates that a session can be started, used, and stopped cleanly
        with proper resource management.
        
        **Validates: Requirements 11.5, 11.6**
        """
        call_metadata = CallMetadata(
            caller_id="+15551234567",
            call_sid="CA_lifecycle_test",
            from_number="+15551234567",
            to_number="+15551111111",
        )

        # Mock pipeline initialization
        with patch.object(orchestrator, 'initialize_pipeline', new_callable=AsyncMock) as mock_init:
            mock_pipeline = AsyncMock(spec=VoicePipeline)
            mock_pipeline.initialize = AsyncMock()
            mock_pipeline.shutdown = AsyncMock()
            mock_init.return_value = mock_pipeline

            # Start session
            pipeline = await orchestrator.start_session(
                session_id="lifecycle_test_session",
                caller_id=call_metadata.caller_id,
                call_metadata=call_metadata,
            )

            # Verify session was started
            assert pipeline is not None
            assert "lifecycle_test_session" in orchestrator._active_pipelines
            assert orchestrator.get_session_count() == 1

            # Verify pipeline was initialized
            mock_pipeline.initialize.assert_called_once()

            # Stop session
            summary = await orchestrator.stop_session("lifecycle_test_session", persist=True)

            # Verify session was stopped
            assert "lifecycle_test_session" not in orchestrator._active_pipelines
            assert orchestrator.get_session_count() == 0

            # Verify pipeline was shutdown
            mock_pipeline.shutdown.assert_called_once()

            # Verify summary was returned
            assert summary is not None
            assert "session_id" in summary

    @pytest.mark.asyncio
    async def test_multiple_concurrent_sessions(self, orchestrator):
        """
        Test managing multiple concurrent sessions.
        
        Validates that the orchestrator can handle multiple active sessions
        simultaneously without interference.
        
        **Validates: Requirements 1.4, 11.5**
        """
        # Mock pipeline initialization
        with patch.object(orchestrator, 'initialize_pipeline', new_callable=AsyncMock) as mock_init:
            mock_pipeline1 = AsyncMock(spec=VoicePipeline)
            mock_pipeline1.initialize = AsyncMock()
            mock_pipeline1.shutdown = AsyncMock()

            mock_pipeline2 = AsyncMock(spec=VoicePipeline)
            mock_pipeline2.initialize = AsyncMock()
            mock_pipeline2.shutdown = AsyncMock()

            mock_pipeline3 = AsyncMock(spec=VoicePipeline)
            mock_pipeline3.initialize = AsyncMock()
            mock_pipeline3.shutdown = AsyncMock()

            mock_init.side_effect = [mock_pipeline1, mock_pipeline2, mock_pipeline3]

            # Start multiple sessions
            call1 = CallMetadata(caller_id="+15551234567", call_sid="CA_001", from_number="+15551234567")
            call2 = CallMetadata(caller_id="+15559876543", call_sid="CA_002", from_number="+15559876543")
            call3 = CallMetadata(caller_id="+15555555555", call_sid="CA_003", from_number="+15555555555")

            pipeline1 = await orchestrator.start_session("session_1", "+15551234567", call1)
            pipeline2 = await orchestrator.start_session("session_2", "+15559876543", call2)
            pipeline3 = await orchestrator.start_session("session_3", "+15555555555", call3)

            # Verify all sessions are active
            assert orchestrator.get_session_count() == 3
            active_ids = orchestrator.get_active_session_ids()
            assert "session_1" in active_ids
            assert "session_2" in active_ids
            assert "session_3" in active_ids

            # Verify each pipeline is accessible
            assert orchestrator.get_active_pipeline("session_1") == mock_pipeline1
            assert orchestrator.get_active_pipeline("session_2") == mock_pipeline2
            assert orchestrator.get_active_pipeline("session_3") == mock_pipeline3

            # Stop sessions in different order
            await orchestrator.stop_session("session_2")
            assert orchestrator.get_session_count() == 2
            assert "session_2" not in orchestrator.get_active_session_ids()

            await orchestrator.stop_session("session_1")
            assert orchestrator.get_session_count() == 1

            await orchestrator.stop_session("session_3")
            assert orchestrator.get_session_count() == 0

    @pytest.mark.asyncio
    async def test_session_cleanup_on_start_failure(self, orchestrator):
        """
        Test session cleanup when start fails.
        
        Validates that if session start fails, resources are properly cleaned up.
        
        **Validates: Requirements 11.5, 11.6**
        """
        call_metadata = CallMetadata(
            caller_id="+15551234567",
            call_sid="CA_fail_test",
            from_number="+15551234567",
        )

        # Mock pipeline initialization to fail
        with patch.object(orchestrator, 'initialize_pipeline', new_callable=AsyncMock) as mock_init:
            mock_pipeline = AsyncMock(spec=VoicePipeline)
            mock_pipeline.initialize = AsyncMock(side_effect=RuntimeError("Init failed"))
            mock_init.return_value = mock_pipeline

            # Attempt to start session (should fail)
            with pytest.raises(RuntimeError, match="Init failed"):
                await orchestrator.start_session(
                    session_id="fail_test_session",
                    caller_id=call_metadata.caller_id,
                    call_metadata=call_metadata,
                )

            # Verify session was cleaned up
            session = orchestrator._session_manager.get_session("fail_test_session")
            assert session is None or session.end_time is not None

    @pytest.mark.asyncio
    async def test_stop_session_with_pipeline_error(self, orchestrator):
        """
        Test stopping session when pipeline shutdown fails.
        
        Validates that session is still cleaned up even if pipeline
        shutdown encounters an error.
        
        **Validates: Requirements 11.5, 11.6**
        """
        call_metadata = CallMetadata(
            caller_id="+15551234567",
            call_sid="CA_shutdown_error",
            from_number="+15551234567",
        )

        # Mock pipeline with failing shutdown
        with patch.object(orchestrator, 'initialize_pipeline', new_callable=AsyncMock) as mock_init:
            mock_pipeline = AsyncMock(spec=VoicePipeline)
            mock_pipeline.initialize = AsyncMock()
            mock_pipeline.shutdown = AsyncMock(side_effect=RuntimeError("Shutdown failed"))
            mock_init.return_value = mock_pipeline

            # Start session
            await orchestrator.start_session(
                session_id="error_test_session",
                caller_id=call_metadata.caller_id,
                call_metadata=call_metadata,
            )

            assert orchestrator.get_session_count() == 1

            # Stop session (should not raise despite shutdown error)
            summary = await orchestrator.stop_session("error_test_session")

            # Verify session was still cleaned up
            assert "error_test_session" not in orchestrator._active_pipelines
            assert orchestrator.get_session_count() == 0
            assert summary is not None


class TestComponentIntegration:
    """Test component integration and data flow."""

    @pytest.mark.asyncio
    async def test_tenant_registry_to_pipeline_integration(self, orchestrator, tenant_registry):
        """
        Test data flow from tenant registry to pipeline initialization.
        
        Validates that tenant configuration is correctly passed through
        the orchestrator to create a properly configured pipeline.
        
        **Validates: Requirements 1.3, 1.4**
        """
        # Load tenant config
        tenant_config = tenant_registry.load_tenant_config("healthcare_clinic_001")

        # Mock LLM service
        with patch.object(orchestrator._llm_factory, 'create_llm_service', new_callable=AsyncMock) as mock_llm:
            mock_llm_service = Mock()
            mock_llm.return_value = mock_llm_service

            # Initialize pipeline
            pipeline = await orchestrator.initialize_pipeline(
                session_id="integration_test",
                tenant_config=tenant_config,
            )

            # Verify LLM factory was called with correct config
            mock_llm.assert_called_once()
            llm_config_arg = mock_llm.call_args[0][0]
            assert llm_config_arg.provider == "openai"
            assert llm_config_arg.model == "gpt-4"
            assert llm_config_arg.temperature == 0.7

            # Verify pipeline received correct configuration
            assert pipeline.config.llm_service == mock_llm_service
            assert pipeline.config.stt_config.model == "nova-2"
            assert pipeline.config.voice_config.voice_id == "healthcare_voice_123"

    @pytest.mark.asyncio
    async def test_session_manager_integration(self, orchestrator):
        """
        Test integration between orchestrator and session manager.
        
        Validates that session manager correctly creates and manages
        sessions initiated by the orchestrator.
        
        **Validates: Requirements 11.5**
        """
        call_metadata = CallMetadata(
            caller_id="+15551234567",
            call_sid="CA_session_mgr_test",
            from_number="+15551234567",
            to_number="+15551111111",
            additional_data={"source": "twilio"},
        )

        # Mock pipeline
        with patch.object(orchestrator, 'initialize_pipeline', new_callable=AsyncMock) as mock_init:
            mock_pipeline = AsyncMock(spec=VoicePipeline)
            mock_pipeline.initialize = AsyncMock()
            mock_pipeline.shutdown = AsyncMock()
            mock_init.return_value = mock_pipeline

            # Start session
            await orchestrator.start_session(
                session_id="session_mgr_test",
                caller_id=call_metadata.caller_id,
                call_metadata=call_metadata,
            )

            # Verify session was created in session manager
            session = orchestrator._session_manager.get_session("session_mgr_test")
            assert session is not None
            assert session.session_id == "session_mgr_test"
            assert session.tenant_id == "healthcare_clinic_001"
            assert session.caller_id == "+15551234567"
            assert session.metadata["call_sid"] == "CA_session_mgr_test"
            assert session.metadata["additional_data"]["source"] == "twilio"

            # Stop session
            await orchestrator.stop_session("session_mgr_test")

            # Verify session was ended (not deleted, but ended)
            session = orchestrator._session_manager.get_session("session_mgr_test")
            # Session should be ended (has end_time) but still retrievable
            assert session is not None
            assert session.end_time is not None

    @pytest.mark.asyncio
    async def test_end_to_end_call_flow(self, orchestrator):
        """
        Test complete end-to-end call flow through all components.
        
        Validates the complete data flow:
        Call → Route → Create Session → Initialize Pipeline → Active → Stop → Cleanup
        
        **Validates: Requirements 1.3, 1.4, 11.5, 11.6**
        """
        call_metadata = CallMetadata(
            caller_id="+15551234567",
            call_sid="CA_e2e_test",
            from_number="+15551234567",
            to_number="+15551111111",
        )

        # Mock pipeline
        with patch.object(orchestrator, 'initialize_pipeline', new_callable=AsyncMock) as mock_init:
            mock_pipeline = AsyncMock(spec=VoicePipeline)
            mock_pipeline.initialize = AsyncMock()
            mock_pipeline.shutdown = AsyncMock()
            mock_init.return_value = mock_pipeline

            # Step 1: Route call
            tenant_config = orchestrator.route_call(
                caller_id=call_metadata.caller_id,
                call_metadata=call_metadata,
            )
            assert tenant_config.tenant_id == "healthcare_clinic_001"

            # Step 2: Start session (creates session + initializes pipeline)
            pipeline = await orchestrator.start_session(
                session_id="e2e_test_session",
                caller_id=call_metadata.caller_id,
                call_metadata=call_metadata,
            )

            # Verify session is active
            assert orchestrator.get_session_count() == 1
            assert orchestrator.get_active_pipeline("e2e_test_session") == mock_pipeline

            # Verify session exists in session manager
            session = orchestrator._session_manager.get_session("e2e_test_session")
            assert session is not None
            assert session.tenant_id == "healthcare_clinic_001"

            # Step 3: Simulate active call (pipeline would process audio here)
            # In real scenario, pipeline would be processing audio chunks

            # Step 4: Stop session
            summary = await orchestrator.stop_session("e2e_test_session", persist=True)

            # Verify complete cleanup
            assert orchestrator.get_session_count() == 0
            assert orchestrator.get_active_pipeline("e2e_test_session") is None

            # Session should be ended (has end_time) but still retrievable
            session = orchestrator._session_manager.get_session("e2e_test_session")
            assert session is not None
            assert session.end_time is not None

            # Verify summary contains session info
            assert summary is not None
            assert "session_id" in summary

    @pytest.mark.asyncio
    async def test_concurrent_sessions_different_tenants(self, orchestrator):
        """
        Test concurrent sessions for different tenants.
        
        Validates that multiple tenants can have active sessions simultaneously
        with proper isolation.
        
        **Validates: Requirements 1.4, 11.5**
        """
        # Mock pipeline
        with patch.object(orchestrator, 'initialize_pipeline', new_callable=AsyncMock) as mock_init:
            mock_pipeline1 = AsyncMock(spec=VoicePipeline)
            mock_pipeline1.initialize = AsyncMock()
            mock_pipeline1.shutdown = AsyncMock()

            mock_pipeline2 = AsyncMock(spec=VoicePipeline)
            mock_pipeline2.initialize = AsyncMock()
            mock_pipeline2.shutdown = AsyncMock()

            mock_init.side_effect = [mock_pipeline1, mock_pipeline2]

            # Start healthcare session
            healthcare_call = CallMetadata(
                caller_id="+15551234567",
                call_sid="CA_healthcare",
                from_number="+15551234567",
            )

            healthcare_pipeline = await orchestrator.start_session(
                session_id="healthcare_session",
                caller_id="+15551234567",
                call_metadata=healthcare_call,
            )

            # Start automotive session
            automotive_call = CallMetadata(
                caller_id="+15559876543",
                call_sid="CA_automotive",
                from_number="+15559876543",
            )

            automotive_pipeline = await orchestrator.start_session(
                session_id="automotive_session",
                caller_id="+15559876543",
                call_metadata=automotive_call,
            )

            # Verify both sessions are active
            assert orchestrator.get_session_count() == 2

            # Verify sessions are isolated
            healthcare_session = orchestrator._session_manager.get_session("healthcare_session")
            automotive_session = orchestrator._session_manager.get_session("automotive_session")

            assert healthcare_session.tenant_id == "healthcare_clinic_001"
            assert automotive_session.tenant_id == "automotive_dealer_001"

            # Verify pipelines are different
            assert healthcare_pipeline != automotive_pipeline

            # Cleanup
            await orchestrator.stop_session("healthcare_session")
            await orchestrator.stop_session("automotive_session")

            assert orchestrator.get_session_count() == 0
